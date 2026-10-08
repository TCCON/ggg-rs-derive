use darling::{FromDeriveInput, FromField, ast};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

use crate::common_netcdf::{
    DimList, Direction, FloatTy, MapFns, MapSpec, NcKind, OnFill, UnpackSpec, apply_rename_all,
    float_element_type, infer_kind, kind_from_str, nc_name, not_float_message, resolve_unpack,
    validate_dims,
};

/// What unpacking a single field needs, once the container default, the field
/// override, and the field's element type have all been taken into account.
/// `None` rather than this struct means "emit no unpacking code at all".
#[derive(Debug, Clone, Copy)]
struct UnpackPlan {
    float_ty: FloatTy,
    scale: bool,
    offset: bool,
    on_fill: OnFill,
}

pub(crate) fn expand(input: TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(input as syn::DeriveInput);
    let parsed = match ReadNetcdfInput::from_derive_input(&derive_input) {
        Ok(p) => p,
        Err(e) => return e.write_errors().into(),
    };
    let name = &parsed.ident;
    let rename_all = parsed.rename_all.as_deref();
    let fields = parsed
        .data
        .take_struct()
        .expect("darling's supports(struct_named) guarantees this")
        .fields;

    // Validate `rename_all` once up front (with an empty field name) so an
    // unknown rule is reported even on an empty struct.
    let mut errors = darling::Error::accumulator();
    if let Some(rule) = rename_all
        && let Err(e) = apply_rename_all("x", rule)
    {
        errors.push(e);
    }

    let mut helpers = HelpersNeeded::default();
    let mut field_blocks: Vec<TokenStream2> = Vec::new();
    let mut field_names: Vec<syn::Ident> = Vec::new();

    for f in &fields {
        // Resolve the read strategy and the netCDF name, collecting errors so
        // every problem is reported in one compile.
        let Some((read_kind, map_fns, unpack)) =
            errors.handle(resolve_field(f, parsed.unpack.as_ref(), parsed.on_fill))
        else {
            continue;
        };
        let fieldname = f.ident.as_ref().unwrap().clone();

        // Unlike writing, every field has to be initialized to build `Self`, so
        // a skipped field falls back to `Default` instead of dropping out of the
        // generated code entirely.
        if read_kind == NcKind::Skip {
            field_blocks.push(quote! {
                let #fieldname = ::core::default::Default::default();
            });
            field_names.push(fieldname);
            continue;
        }

        let Some(name) = errors.handle(nc_name(&fieldname, f.rename.as_deref(), rename_all)) else {
            continue;
        };

        helpers.set_needed_helpers_for_kind(read_kind);
        if map_fns.map_units.is_some() {
            helpers.get_units = true;
        }
        if f.dimensions.is_some() {
            helpers.check_dims = true;
        }
        if let Some(plan) = unpack {
            helpers.set_needed_unpack(plan.float_ty);
        }
        field_blocks.push(field_block(f, read_kind, &name, map_fns, unpack));
        field_names.push(fieldname);
    }

    if let Err(e) = errors.finish() {
        return e.write_errors().into();
    }

    let helper_fns = helpers.tokens();
    let (impl_generics, ty_generics, where_clause) = parsed.generics.split_for_impl();
    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            /// Read this struct's fields from netCDF variables or attributes under `grp`.
            /// Pass a file root (`File::root()`) or a subgroup.
            pub fn read_netcdf(grp: &::netcdf::Group<'_>) -> ::netcdf::Result<Self> {
                #helper_fns
                #(#field_blocks)*
                Ok(Self { #(#field_names),* })
            }
        }
    };

    TokenStream::from(expanded)
}

#[derive(FromDeriveInput)]
#[darling(attributes(netcdf), supports(struct_named))]
struct ReadNetcdfInput {
    ident: syn::Ident,
    generics: syn::Generics,
    data: ast::Data<darling::util::Ignored, ReadNetcdfField>,

    /// serde-style case convention applied to every field's netCDF name.
    #[darling(default)]
    rename_all: Option<String>,

    /// Default unpacking behaviour for every field: `unpack` / `unpack = false` /
    /// `unpack(scale = .., offset = ..)`. Individual fields can override it.
    /// Unpacking is on by default, so this is only needed to turn something off.
    #[darling(default)]
    unpack: Option<UnpackSpec>,

    /// Default handling of fill and missing values: `"nan"` (default), `"keep"`,
    /// `"ignore"`, or `"error"`.
    #[darling(default)]
    on_fill: Option<OnFill>,

    /// Accepted and ignored: `describe` is a `WriteNetcdf` option. Both derives
    /// read the same `netcdf` attribute, so each has to tolerate the other's
    /// keys — otherwise a struct deriving both could not be annotated at all.
    /// Declaring them explicitly rather than using `allow_unknown_fields` keeps
    /// genuine typos an error.
    #[darling(default)]
    #[allow(dead_code)]
    describe: Option<darling::util::Ignored>,
}

#[derive(FromField)]
#[darling(attributes(netcdf), forward_attrs(doc))]
struct ReadNetcdfField {
    ident: Option<syn::Ident>,
    ty: syn::Type,

    /// Do not write this field at all.
    #[darling(default)]
    skip: bool,

    /// Override the netCDF name for this field (takes precedence over the
    /// container-level `rename_all`).
    #[darling(default)]
    rename: Option<String>,

    /// Force a read strategy: `"values"`, `"array"`, or `"string"`. Required
    /// when `map`/`map_units` is present (the output type can't be inferred).
    #[darling(default)]
    kind: Option<String>,

    /// Expected dimension names for this variable, not checked if omitted
    #[darling(default)]
    dimensions: Option<DimList>,

    /// Write the field as a *group attribute* instead of a variable.
    #[darling(default)]
    attribute: bool,

    /// Map the field value after reading. The direction is mandatory, since
    /// writing needs a different function: `map(read = path::to::fn)`, where the
    /// function is `fn(NcTy, &str) -> netcdf::Result<Out>`. Its arguments are
    /// the value read from the netCDF file (by value) and the struct field name.
    /// Reading is fallible — the data comes from a file this program did not
    /// necessarily write — so the function returns a `netcdf::Result`. A
    /// `write = ...` may also be present for `WriteNetcdf`; it is ignored here.
    #[darling(default)]
    map: Option<MapSpec>,

    /// Like `map`, but the read function also accepts the value of the `units`
    /// attribute as a third argument:
    /// `fn(NcTy, &str, Option<&str>) -> netcdf::Result<Out>`.
    /// If the variable has no `units`, the third argument is `None`.
    #[darling(default)]
    map_units: Option<MapSpec>,

    /// Override the container's unpacking behaviour for this field. Replaces the
    /// container setting wholesale rather than merging with it, the same way
    /// `rename` replaces `rename_all`.
    #[darling(default)]
    unpack: Option<UnpackSpec>,

    /// Override the container's fill/missing-value handling for this field.
    #[darling(default)]
    on_fill: Option<OnFill>,

    /// Accepted and ignored: on read, `scale_factor` and `add_offset` come from
    /// the file, not from the annotation. Parsed with the real type rather than
    /// `Ignored` so that a malformed `scale_offset(...)` is still reported even
    /// on a struct that only derives `ReadNetcdf`. See `unpack` to control
    /// whether the file's values are applied.
    #[darling(default)]
    #[allow(dead_code)]
    scale_offset: Option<crate::common_netcdf::ScaleOffsetSpec>,

    /// Accepted and ignored: `attr`, `describe`, and `no_describe` are
    /// `WriteNetcdf` options. See the note on `ReadNetcdfInput::describe`.
    #[darling(default, multiple, rename = "attr")]
    #[allow(dead_code)]
    var_attrs: Vec<darling::util::Ignored>,

    #[darling(default)]
    #[allow(dead_code)]
    describe: Option<darling::util::Ignored>,

    #[darling(default)]
    #[allow(dead_code)]
    no_describe: bool,
}

#[derive(Debug, Default)]
struct HelpersNeeded {
    /// true if a field needs a helper to confirm that its dimensions
    /// match what is expected
    check_dims: bool,

    /// true if a field needs a helper to read a vector of values
    values: bool,

    /// true if a field needs a helper to read an array of values
    /// (possibly with multiple dimensions)
    array: bool,

    /// true if a field needs a helper to read a sequence of strings
    /// into a vector
    strings: bool,

    /// true if a field needs a helper to read a scalar numeric value
    scalar: bool,

    /// true if a field needs a helper to read a single string value
    string_scalar: bool,

    /// true if a field needs a helper to be read from an attribute
    attribute: bool,

    /// true if we need to read a units attribute to pass to a map_units function.
    get_units: bool,

    /// true if some `f32` field applies scale/offset or checks for fill values
    unpack_f32: bool,

    /// true if some `f64` field does the same
    unpack_f64: bool,
}

impl HelpersNeeded {
    /// Note that a field needs the unpack helper for its float type. The
    /// `__OnFill` enum and the fill/missing lookups come along with it.
    fn set_needed_unpack(&mut self, float_ty: FloatTy) {
        match float_ty {
            FloatTy::F32 => self.unpack_f32 = true,
            FloatTy::F64 => self.unpack_f64 = true,
        }
    }

    /// True if either unpack helper is in play, and therefore the shared
    /// `__OnFill` enum and fill-value lookups need emitting too.
    fn any_unpack(&self) -> bool {
        self.unpack_f32 || self.unpack_f64
    }

    fn set_needed_helpers_for_kind(&mut self, read_kind: NcKind) {
        match read_kind {
            NcKind::Values => self.values = true,
            NcKind::Array => self.array = true,
            NcKind::StringScalar => self.string_scalar = true,
            NcKind::StringVec => self.strings = true,
            NcKind::Scalar => self.scalar = true,
            NcKind::Attribute => self.attribute = true,
            NcKind::Skip => (),
        }
    }

    fn tokens(&self) -> TokenStream2 {
        let mut out = TokenStream2::new();
        // Tip: to help write/edit a function, make a regular function annotated with `#[cfg(test)]`
        // so it resolved netcdf and ndarray, work on it outside the quote!, then paste it into the quote!.
        // Alternatively, if this code is added in a test, the IDE will pick up error (but not autocomplete).
        if self.check_dims {
            out.extend(quote! {
                fn __check_dims<'g>(
                    g: &'g ::netcdf::Group<'_>,
                    name: &str,
                    dims: &[&str],
                ) -> ::netcdf::Result<()> {
                    let var = g
                        .variable(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: '{name}'")))?;

                    let vardims: Vec<String> = var.dimensions().iter().map(|d| d.name()).collect();
                    let matches = vardims.len() == dims.len() && vardims.iter().zip(dims).all(|(got, want)| got.as_str() == *want);
                    if matches {
                        Ok(())
                    } else {
                        let expected = dims.join(", ");
                        let actual = vardims.join(", ");
                        Err(::netcdf::Error::Str(format!(
                            "Variable '{name}' had wrong dimension number or names: expected [{expected}], got [{actual}]"
                        )))
                    }
                }
            });
        }
        // The getters below only read. Applying `scale_factor` / `add_offset` and
        // masking fill values is the job of the `__unpack_*` helpers, which are
        // emitted only for float fields that need them. Keeping the two apart is
        // what lets these stay at `T: NcTypeDescriptor + Copy`: an earlier version
        // scaled in place here, which forced `MulAssign + AddAssign +
        // TryFrom<AttributeValue>` onto every field of the same kind and silently
        // truncated a fractional scale factor to 0 on integer fields.
        if self.values {
            out.extend(quote! {
                fn __get_values<'g, T: ::netcdf::NcTypeDescriptor + Copy>(
                    g: &'g ::netcdf::Group<'_>,
                    name: &str,
                ) -> ::netcdf::Result<Vec<T>> {
                    let var = g
                        .variable(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: {name}")))?;
                    // `..` is the only extent that means "the whole variable";
                    // `()` would select zero dimensions.
                    let values = var.get_values::<T, _>(..)?;
                    Ok(values)
                }
            });
        }

        if self.array {
            out.extend(quote! {
                fn __get_array<'g, T: ::netcdf::NcTypeDescriptor + Copy, D: ::ndarray::Dimension>(
                    g: &'g ::netcdf::Group<'_>,
                    name: &str,
                ) -> ::netcdf::Result<::ndarray::Array<T, D>> {
                    let var = g
                        .variable(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: {name}")))?;
                    let array = var.get::<T, _>(..)?.into_dimensionality().map_err(|_| {
                        ::netcdf::Error::Str(format!(
                            "Variable '{name}' could not be converted into a {}D array",
                            D::NDIM.unwrap() // should only be None for dynamically dimensioned arrays, which should never have this error
                        ))
                    })?;
                    Ok(array)
                }
            });
        }

        if self.strings {
            out.extend(quote! {
                fn __get_strings<'g>(g: &'g ::netcdf::Group<'_>, name: &str) -> ::netcdf::Result<Vec<String>> {
                    let var = g
                        .variable(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: {name}")))?;

                    let var_dims = var.dimensions();
                    if var_dims.len() != 1 {
                        return Err(::netcdf::Error::Str(format!(
                            "string variable '{name}' expected to have 1 dimension, instead had {}",
                            var_dims.len()
                        )));
                    }
                    let n = var_dims[0].len();
                    let mut out = Vec::with_capacity(n);

                    for i in 0..n {
                        let s = var.get_string([i])?;
                        out.push(s);
                    }

                    Ok(out)
                }
            });
        }

        if self.scalar {
            out.extend(quote! {
                fn __get_scalar<'g, T: ::netcdf::NcTypeDescriptor + Copy>(
                    g: &'g ::netcdf::Group<'_>,
                    name: &str,
                ) -> ::netcdf::Result<T> {
                    let var = g
                        .variable(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: '{name}'")))?;
                    let value = var.get_value::<T, _>(())?;
                    Ok(value)
                }
            });
        }

        if self.string_scalar {
            out.extend(quote! {
                fn __get_scalar_string<'g>(g: &'g ::netcdf::Group<'_>, name: &str) -> ::netcdf::Result<String> {
                    let var = g
                        .variable(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: {name}")))?;

                    let s = var.get_string(())?;
                    Ok(s)
                }
            });
        }

        if self.attribute {
            out.extend(quote! {
                fn __get_attribute<'g, T: TryFrom<::netcdf::AttributeValue>>(
                    g: &'g ::netcdf::Group<'_>,
                    name: &str,
                ) -> ::netcdf::Result<T>
                where
                    ::netcdf::Error: From<<T as TryFrom<::netcdf::AttributeValue>>::Error>,
                {
                    let attr_val = g
                        .attribute_value(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("group attribute: '{name}'")))??;
                    let val: T = attr_val.try_into()?;
                    Ok(val)
                }
            });
        }

        if self.get_units {
            out.extend(quote! {
                fn __get_units<'g>(
                    g: &'g ::netcdf::Group<'_>,
                    varname: &str,
                    unit_attr: &str,
                ) -> ::netcdf::Result<Option<String>> {
                    let var = g
                        .variable(varname)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: '{varname}'")))?;
                    let unit_attr_val = var.attribute_value(unit_attr).transpose()?;
                    match unit_attr_val {
                        Some(::netcdf::AttributeValue::Str(s)) => Ok(Some(s)),
                        Some(::netcdf::AttributeValue::Strs(vs)) => Ok(Some(vs.join(" "))),
                        None => Ok(None),
                        _ => Err(::netcdf::Error::Str(format!(
                            "Expected unit-like attribute '{unit_attr}' on variable '{varname}' to be string like, but it was not"
                        ))),
                    }
                }
            });
        }

        if self.any_unpack() {
            out.extend(quote! {
                // Not every variant is constructed by every struct, so silence
                // dead-code warnings in the caller's crate.
                #[allow(dead_code)]
                #[derive(Clone, Copy, PartialEq, Eq)]
                enum __OnFill { Nan, Keep, Ignore, Error }

                // netCDF reports a *default* fill value for variables that never
                // declared `_FillValue`, and we deliberately honour it — this
                // matches netCDF4-python, whose `_toma` falls back to
                // `default_fillvals` when the attribute is absent. The exception is
                // the byte types, where Unidata advises against assuming a default
                // because the range is too narrow to spare a value; there we only
                // mask when `_FillValue` was written explicitly.
                fn __fill_value_as_f64(var: &::netcdf::Variable<'_>) -> ::netcdf::Result<Option<f64>> {
                    use ::netcdf::types::{FloatType, IntType, NcVariableType};
                    let declared = var.attribute("_FillValue").is_some();
                    Ok(match var.vartype() {
                        NcVariableType::Int(IntType::I8 | IntType::U8) if !declared => None,
                        NcVariableType::Int(IntType::I8) => var.fill_value::<i8>()?.map(|v| v as f64),
                        NcVariableType::Int(IntType::U8) => var.fill_value::<u8>()?.map(|v| v as f64),
                        NcVariableType::Int(IntType::I16) => var.fill_value::<i16>()?.map(|v| v as f64),
                        NcVariableType::Int(IntType::U16) => var.fill_value::<u16>()?.map(|v| v as f64),
                        NcVariableType::Int(IntType::I32) => var.fill_value::<i32>()?.map(|v| v as f64),
                        NcVariableType::Int(IntType::U32) => var.fill_value::<u32>()?.map(|v| v as f64),
                        NcVariableType::Int(IntType::I64) => var.fill_value::<i64>()?.map(|v| v as f64),
                        NcVariableType::Int(IntType::U64) => var.fill_value::<u64>()?.map(|v| v as f64),
                        NcVariableType::Float(FloatType::F32) => var.fill_value::<f32>()?.map(|v| v as f64),
                        NcVariableType::Float(FloatType::F64) => var.fill_value::<f64>()?,
                        // Strings, chars, compound types: no numeric fill value.
                        _ => None,
                    })
                }

                // `missing_value` is a pure attribute convention with no netCDF-level
                // support, and it is allowed to be a vector of several values.
                fn __missing_values_as_f64(var: &::netcdf::Variable<'_>) -> ::netcdf::Result<Vec<f64>> {
                    use ::netcdf::AttributeValue as __Av;
                    let Some(val) = var.attribute_value("missing_value") else {
                        return Ok(Vec::new());
                    };
                    Ok(match val? {
                        __Av::Uchar(v) => vec![v as f64],
                        __Av::Uchars(vs) => vs.into_iter().map(|v| v as f64).collect(),
                        __Av::Schar(v) => vec![v as f64],
                        __Av::Schars(vs) => vs.into_iter().map(|v| v as f64).collect(),
                        __Av::Ushort(v) => vec![v as f64],
                        __Av::Ushorts(vs) => vs.into_iter().map(|v| v as f64).collect(),
                        __Av::Short(v) => vec![v as f64],
                        __Av::Shorts(vs) => vs.into_iter().map(|v| v as f64).collect(),
                        __Av::Uint(v) => vec![v as f64],
                        __Av::Uints(vs) => vs.into_iter().map(|v| v as f64).collect(),
                        __Av::Int(v) => vec![v as f64],
                        __Av::Ints(vs) => vs.into_iter().map(|v| v as f64).collect(),
                        __Av::Ulonglong(v) => vec![v as f64],
                        __Av::Ulonglongs(vs) => vs.into_iter().map(|v| v as f64).collect(),
                        __Av::Longlong(v) => vec![v as f64],
                        __Av::Longlongs(vs) => vs.into_iter().map(|v| v as f64).collect(),
                        __Av::Float(v) => vec![v as f64],
                        __Av::Floats(vs) => vs.into_iter().map(|v| v as f64).collect(),
                        __Av::Double(v) => vec![v],
                        __Av::Doubles(vs) => vs,
                        __Av::Str(_) | __Av::Strs(_) => return Err(::netcdf::Error::Str(format!(
                            "the 'missing_value' attribute of variable '{}' is a string, \
                             which cannot be compared against numeric data",
                            var.name()
                        ))),
                    })
                }
            });
        }

        // Two concrete functions rather than one generic: `NaN` cannot be produced
        // from a generic `T` without a numeric trait, and the only crate that
        // provides one (`num-traits`) would have to be a dependency of the *caller*,
        // which would break the self-contained-codegen property.
        for float_ty in [FloatTy::F32, FloatTy::F64] {
            let needed = match float_ty {
                FloatTy::F32 => self.unpack_f32,
                FloatTy::F64 => self.unpack_f64,
            };
            if !needed {
                continue;
            }
            let fn_name = float_ty.unpack_fn();
            let ty = syn::Ident::new(float_ty.name(), proc_macro2::Span::call_site());
            out.extend(quote! {
                // Masking happens before scaling, because `_FillValue` is stored in
                // the *packed* representation: once a fill value has been multiplied
                // by the scale factor it is indistinguishable from real data.
                // netCDF4-python orders it the same way (`_toma` then scale).
                fn #fn_name<'a>(
                    g: &::netcdf::Group<'_>,
                    name: &str,
                    vals: impl IntoIterator<Item = &'a mut #ty>,
                    apply_scale: bool,
                    apply_offset: bool,
                    on_fill: __OnFill,
                ) -> ::netcdf::Result<()> {
                    let var = g
                        .variable(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: '{name}'")))?;

                    let (fill, missing) = if on_fill == __OnFill::Ignore {
                        (None, Vec::new())
                    } else {
                        (__fill_value_as_f64(&var)?, __missing_values_as_f64(&var)?)
                    };

                    let scale: Option<#ty> = if apply_scale {
                        match var.attribute_value("scale_factor") {
                            Some(v) => Some(f64::try_from(v?)? as #ty),
                            None => None,
                        }
                    } else {
                        None
                    };
                    let offset: Option<#ty> = if apply_offset {
                        match var.attribute_value("add_offset") {
                            Some(v) => Some(f64::try_from(v?)? as #ty),
                            None => None,
                        }
                    } else {
                        None
                    };

                    for (idx, val) in vals.into_iter().enumerate() {
                        if on_fill != __OnFill::Ignore {
                            // Compare in f64: widening the raw value and the fill
                            // value to f64 is exact for every on-disk type that can
                            // round-trip through this field.
                            let raw = *val as f64;
                            let matches = |target: f64| {
                                if target.is_nan() { raw.is_nan() } else { raw == target }
                            };
                            if fill.is_some_and(&matches) || missing.iter().copied().any(matches) {
                                match on_fill {
                                    __OnFill::Nan => *val = #ty::NAN,
                                    __OnFill::Keep => {}
                                    __OnFill::Error => {
                                        return Err(::netcdf::Error::Str(format!(
                                            "variable '{name}' contains a fill or missing value \
                                             at index {idx}, and this field is declared \
                                             on_fill = \"error\""
                                        )));
                                    }
                                    __OnFill::Ignore => {}
                                }
                                // Deliberately unscaled: a masked element is not data.
                                continue;
                            }
                        }
                        if let Some(s) = scale {
                            *val *= s;
                        }
                        if let Some(o) = offset {
                            *val += o;
                        }
                    }

                    Ok(())
                }
            });
        }

        out
    }
}

/// Resolve the netCDF kind of the field (array, string, skipped, etc.), the map
/// functions that apply when reading, and the unpacking plan, checking the
/// annotations for errors.
fn resolve_field<'f>(
    field: &'f ReadNetcdfField,
    container_unpack: Option<&UnpackSpec>,
    container_on_fill: Option<OnFill>,
) -> Result<(NcKind, MapFns<'f>, Option<UnpackPlan>), darling::Error> {
    let ident = field.ident.as_ref().unwrap();
    let err = |msg: &str| darling::Error::custom(msg).with_span(ident);

    let has_dims = field.dimensions.as_ref().is_some_and(|d| !d.0.is_empty());

    // `skip` conflicts with every other option
    if field.skip {
        if field.rename.is_some()
            || field.kind.is_some()
            || field.attribute
            || field.map.is_some()
            || field.map_units.is_some()
            || field.unpack.is_some()
            || field.on_fill.is_some()
        {
            return Err(err(
                "`skip` cannot be combined with any other `netcdf` option",
            ));
        }
        return Ok((NcKind::Skip, MapFns::default(), None));
    }

    // `map` and `map_units` are mutually exclusive
    if field.map.is_some() && field.map_units.is_some() {
        return Err(err("`map` and `map_units` cannot be used together"));
    }

    // Narrow each `map(...)` spec down to the function this derive needs.
    let map_fns = MapFns::resolve(
        field.map.as_ref(),
        field.map_units.as_ref(),
        Direction::Read,
        ident,
    )?;

    // A field written as a group attribute is not a variable, so variable-only
    // options don't apply.
    if field.attribute {
        if field.kind.is_some() || field.map_units.is_some() || has_dims {
            return Err(err(
                "`attribute` is incompatible with `dimensions`, `kind`, `attr`, `map_units`, and `describe`",
            ));
        }
        if field.unpack.is_some() || field.on_fill.is_some() {
            return Err(err(
                "`unpack`/`on_fill` do not apply to an `attribute` field: scale_factor, \
                 add_offset, and _FillValue are variable attributes",
            ));
        }
        return Ok((NcKind::Attribute, map_fns, None));
    }

    // With a `map`/`map_units` the read type is opaque, so `kind` is required.
    let rk = if map_fns.any() {
        let Some(kind) = &field.kind else {
            return Err(err(
                "`map`/`map_units` requires an explicit `kind = \"values\"|\"array\"|\"string\"`",
            ));
        };
        kind_from_str(kind, &field.ty).map_err(|e| e.with_span(ident))?
    } else if let Some(kind) = &field.kind {
        // Explicit `kind`, else infer from the type.
        kind_from_str(kind, &field.ty).map_err(|e| e.with_span(ident))?
    } else {
        infer_kind(&field.ty).map_err(|e| e.with_span(ident))?
    };
    validate_dims(rk, has_dims, &field.dimensions, ident)?;

    let unpack = resolve_unpack_plan(field, rk, container_unpack, container_on_fill)?;
    Ok((rk, map_fns, unpack))
}

/// Work out whether and how this field unpacks.
///
/// Returns `None` — meaning "emit nothing" — when the field holds strings, when
/// its elements are not a recognizable `f32`/`f64`, or when every part of
/// unpacking is switched off. A *field-level* opt-in on a non-float type is an
/// error; a container-level one is not, because a struct that mixes float and
/// integer fields has to be able to set a container default without every
/// integer field failing to compile.
fn resolve_unpack_plan(
    field: &ReadNetcdfField,
    read_kind: NcKind,
    container_unpack: Option<&UnpackSpec>,
    container_on_fill: Option<OnFill>,
) -> Result<Option<UnpackPlan>, darling::Error> {
    let ident = field.ident.as_ref().unwrap();
    let flags = resolve_unpack(
        container_unpack,
        field.unpack.as_ref(),
        container_on_fill,
        field.on_fill,
    );

    // Strings have no numeric packing; `Attribute` and `Skip` returned earlier.
    if matches!(read_kind, NcKind::StringScalar | NcKind::StringVec) {
        if flags.explicit_field_optin {
            return Err(darling::Error::custom(
                "`unpack`/`on_fill` do not apply to a string field",
            )
            .with_span(ident));
        }
        return Ok(None);
    }

    // With a `map`, the field's declared type is the mapper's *output*, which
    // says nothing about what came out of the file, so the float check can't be
    // made. Unpacking still runs on the physical values before the mapper, so
    // trust the annotation and use the widest float.
    let mapped = field.map.is_some() || field.map_units.is_some();

    let float_ty = if mapped {
        if !flags.explicit_field_optin {
            return Ok(None);
        }
        FloatTy::F64
    } else {
        match float_element_type(&field.ty) {
            Some(ft) => ft,
            None if flags.explicit_field_optin => {
                return Err(
                    darling::Error::custom(not_float_message("unpack", &field.ty)).with_span(ident),
                );
            }
            // Integer (or opaque) field inheriting a container default: skip.
            None => return Ok(None),
        }
    };

    if !flags.any() {
        return Ok(None);
    }

    Ok(Some(UnpackPlan {
        float_ty,
        scale: flags.scale,
        offset: flags.offset,
        on_fill: flags.on_fill,
    }))
}

/// Return code that assigns a variable named with the field's name to the result
/// of reading the variable from the netCDF file.
fn field_block(
    field: &ReadNetcdfField,
    read_kind: NcKind,
    varname: &str,
    map_fns: MapFns<'_>,
    unpack: Option<UnpackPlan>,
) -> TokenStream2 {
    let fieldname = field.ident.as_ref().unwrap();
    // `map`/`map_units` receive the *struct field* name, not the netCDF variable
    // name. `String` tokenizes as a string literal; note that quote! does not
    // interpolate inside a literal, so `"#fieldname"` would emit the text
    // `#fieldname` verbatim.
    let fieldname_str = fieldname.to_string();

    // Generate the call to read in the values from the netcdf file. `grp` will be a &netcdf::Group<'_>.
    let call = match read_kind {
        NcKind::Values => quote! {
            __get_values(grp, #varname)
        },
        NcKind::Array => quote! {
            __get_array(grp, #varname)
        },
        NcKind::StringScalar => quote! {
            __get_scalar_string(grp, #varname)
        },
        NcKind::StringVec => quote! {
            __get_strings(grp, #varname)
        },
        NcKind::Scalar => quote! {
            __get_scalar(grp, #varname)
        },
        NcKind::Attribute => quote! {
            __get_attribute(grp, #varname)
        },
        NcKind::Skip => unreachable!("skip fields are filtered out before codegen"),
    };

    // If we need to map values, include that here. Remember that read map functions
    // have a signature of (netcdf_value, field_name) -> netcdf::Result<Out> and read
    // map_unit functions have (netcdf_value, field_name, unit_string) -> netcdf::Result<Out>.
    // Both are fallible, so their calls take a `?`.
    let remap = if let Some(map_fxn) = map_fns.map {
        quote! {
            let __var = grp
                .variable(#varname)
                .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: '{}'", #varname)))?;
            #map_fxn(__val, #fieldname_str, &__var)?
        }
    } else if let Some((map_fxn, unit_attr)) = map_fns.map_units {
        quote! {
            let __var = grp
                .variable(#varname)
                .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: '{}'", #varname)))?;
            let __units = __get_units(grp, #varname, #unit_attr)?;
            #map_fxn(__val, #fieldname_str, &__var, __units.as_deref())?
        }
    } else {
        quote!(__val)
    };

    // Unpacking mutates the values read from the file, so it happens between the
    // read and the `map` call — the map function receives physical values.
    let (binding, unpack_stmt) = match unpack {
        None => (quote!(let __val), TokenStream2::new()),
        Some(plan) => {
            let unpack_fn = plan.float_ty.unpack_fn();
            let scale = plan.scale;
            let offset = plan.offset;
            let on_fill = plan.on_fill.variant();
            // A scalar has no `iter_mut`, so hand over a one-element slice.
            let vals = match read_kind {
                NcKind::Scalar => quote!(::core::slice::from_mut(&mut __val)),
                _ => quote!(__val.iter_mut()),
            };
            (
                quote!(let mut __val),
                quote! {
                    #unpack_fn(grp, #varname, #vals, #scale, #offset, __OnFill::#on_fill)?;
                },
            )
        }
    };

    let read_and_remap = quote! {
        let #fieldname = {
            #binding = #call?;
            #unpack_stmt
            #remap
        };
    };

    // If we are to check dimensions, include that before we read the variable
    if let Some(dims) = field.dimensions.as_ref() {
        let expected: Vec<&str> = dims.0.iter().map(|d| d.as_str()).collect();
        quote! {
            __check_dims(grp, #varname, &[#(#expected),*])?;
            #read_and_remap
        }
    } else {
        read_and_remap
    }
}
