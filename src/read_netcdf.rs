use darling::{FromDeriveInput, FromField, ast};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

use crate::common_netcdf::{
    DimList, Direction, MapFns, MapSpec, NcKind, apply_rename_all, infer_kind, kind_from_str,
    nc_name, validate_dims,
};

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
        let Some((read_kind, map_fns)) = errors.handle(resolve_field(f)) else {
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
        field_blocks.push(field_block(f, read_kind, &name, map_fns));
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
}

#[derive(Debug, Default)]
struct HelpersNeeded {
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
}

impl HelpersNeeded {
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

        if self.values {
            out.extend(quote! {
                // Return a vector of values, 
                fn __get_values<
                    'g,
                    T: ::netcdf::NcTypeDescriptor + Copy + ::std::ops::MulAssign + ::std::ops::AddAssign + TryFrom<::netcdf::AttributeValue>,
                >(
                    g: &'g ::netcdf::Group<'_>,
                    name: &str,
                ) -> ::netcdf::Result<Vec<T>>
                where
                    ::netcdf::Error: From<<T as TryFrom<::netcdf::AttributeValue>>::Error>,
                {
                    let var = g
                        .variable(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: {name}")))?;
                    // TODO: omit the scale factor/offset if field is annotated not to apply them
                    let scale_factor = var.attribute_value("scale_factor").transpose()?;
                    let offset = var.attribute_value("add_offset").transpose()?;
                    // `..` is the only extent that means "the whole variable";
                    // `()` would select zero dimensions.
                    let mut values = var.get_values::<T, _>(..)?;

                    // I decided to do the scaling and offset addition in if let blocks rather than using unwrap_or (a) to
                    // avoid even more complicated trait bounds and (b) to avoid the calculation of multiplying by 1 or adding
                    // 0 if it isn't needed.
                    if let Some(scale) = scale_factor {
                        let scale: T = scale.try_into()?;
                        for val in values.iter_mut() {
                            *val *= scale;
                        }
                    }

                    if let Some(offset) = offset {
                        let offset: T = offset.try_into()?;
                        for val in values.iter_mut() {
                            *val += offset;
                        }
                    }

                    Ok(values)
                }
            });
        }

        if self.array {
            out.extend(quote! {
                fn __get_array<
                    'g,
                    T: ::netcdf::NcTypeDescriptor + Copy + ::std::ops::MulAssign + ::std::ops::AddAssign + TryFrom<::netcdf::AttributeValue>,
                    D: ::ndarray::Dimension,
                >(
                    g: &'g ::netcdf::Group<'_>,
                    name: &str,
                ) -> ::netcdf::Result<::ndarray::Array<T, D>>
                where
                    ::netcdf::Error: From<<T as TryFrom<::netcdf::AttributeValue>>::Error>,
                {
                    let var = g
                        .variable(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: {name}")))?;
                    // TODO: omit the scale factor/offset if field is annotated not to apply them
                    let scale_factor = var.attribute_value("scale_factor").transpose()?;
                    let offset = var.attribute_value("add_offset").transpose()?;
                    let mut array = var.get::<T, _>(..)?.into_dimensionality().map_err(|_| {
                        ::netcdf::Error::Str(format!(
                            "Variable '{name}' could not be converted into a {}D array",
                            D::NDIM.unwrap() // should only be None for dynamically dimensioned arrays, which should never have this error
                        ))
                    })?;

                    // I decided to do the scaling and offset addition in if let blocks rather than using unwrap_or (a) to
                    // avoid even more complicated trait bounds and (b) to avoid the calculation of multiplying by 1 or adding
                    // 0 if it isn't needed.
                    if let Some(scale) = scale_factor {
                        let scale: T = scale.try_into()?;
                        for val in array.iter_mut() {
                            *val *= scale;
                        }
                    }

                    if let Some(offset) = offset {
                        let offset: T = offset.try_into()?;
                        for val in array.iter_mut() {
                            *val += offset;
                        }
                    }

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
                fn __get_scalar<
                    'g,
                    T: ::netcdf::NcTypeDescriptor + Copy + ::std::ops::MulAssign + ::std::ops::AddAssign + TryFrom<::netcdf::AttributeValue>,
                >(
                    g: &'g ::netcdf::Group<'_>,
                    name: &str,
                ) -> ::netcdf::Result<T>
                where
                    ::netcdf::Error: From<<T as TryFrom<::netcdf::AttributeValue>>::Error>,
                {
                    let var = g
                        .variable(name)
                        .ok_or_else(|| ::netcdf::Error::NotFound(format!("variable: '{name}'")))?;
                    // TODO: omit the scale factor/offset if field is annotated not to apply them
                    let scale_factor = var.attribute_value("scale_factor").transpose()?;
                    let offset = var.attribute_value("add_offset").transpose()?;
                    let mut value = var.get_value::<T, _>(())?;
                    if let Some(scale) = scale_factor {
                        let scale: T = scale.try_into()?;
                        value *= scale;
                    }
                    if let Some(offset) = offset {
                        let offset = offset.try_into()?;
                        value += offset;
                    }
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

                    // TODO: check dimensionality - not sure if a scalar string variable will have 0 or 1 dimension(s)
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

        out
    }
}

/// Resolve the netCDF kind of the field (array, string, skipped, etc.) and the
/// map functions that apply when reading, checking the annotations for errors.
fn resolve_field(field: &ReadNetcdfField) -> Result<(NcKind, MapFns<'_>), darling::Error> {
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
        {
            return Err(err(
                "`skip` cannot be combined with any other `netcdf` option",
            ));
        }
        return Ok((NcKind::Skip, MapFns::default()));
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
        if field.kind.is_some() || field.map_units.is_some() {
            return Err(err(
                "`attribute` is incompatible with `dimensions`, `kind`, `attr`, `map_units`, and `describe`",
            ));
        }
        return Ok((NcKind::Attribute, map_fns));
    }

    // With a `map`/`map_units` the read type is opaque, so `kind` is required.
    if map_fns.any() {
        let Some(kind) = &field.kind else {
            return Err(err(
                "`map`/`map_units` requires an explicit `kind = \"values\"|\"array\"|\"string\"`",
            ));
        };
        let nck = kind_from_str(kind, &field.ty).map_err(|e| e.with_span(ident))?;
        return validate_dims(nck, has_dims, &field.dimensions, ident).map(|()| (nck, map_fns));
    }

    // Explicit `kind`, else infer from the type.
    let rk = if let Some(kind) = &field.kind {
        kind_from_str(kind, &field.ty).map_err(|e| e.with_span(ident))?
    } else {
        infer_kind(&field.ty).map_err(|e| e.with_span(ident))?
    };
    validate_dims(rk, has_dims, &field.dimensions, ident).map(|()| (rk, map_fns))
}

/// Return code that assigns a variable named with the field's name to the result
/// of reading the variable from the netCDF file.
fn field_block(
    field: &ReadNetcdfField,
    read_kind: NcKind,
    varname: &str,
    map_fns: MapFns<'_>,
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
            #map_fxn(__val, #fieldname_str)?
        }
    } else if let Some((map_fxn, unit_attr)) = map_fns.map_units {
        quote! {
            let __units = __get_units(grp, #varname, #unit_attr)?;
            #map_fxn(__val, #fieldname_str, __units.as_deref())?
        }
    } else {
        quote!(__val)
    };

    quote! {
        let #fieldname = {
            let __val = #call?;
            #remap
        };
    }
}
