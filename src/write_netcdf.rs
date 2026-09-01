//! Implementation of the `#[derive(WriteNetcdf)]` macro.
//!
//! The derive generates an inherent method
//!
//! ```ignore
//! pub fn write_netcdf(&self, grp: &mut ::netcdf::GroupMut<'_>) -> ::netcdf::Result<()>
//! ```
//!
//! that writes each field of the struct into a netCDF group (either the file
//! root, obtained with `FileMut::root_mut()`, or a subgroup created with
//! `FileMut::add_group()`). Every field becomes either a netCDF *variable* or a
//! *group attribute*, according to its type and the `#[netcdf(...)]` field
//! attributes.
//!
//! ## How the generated code is structured
//!
//! netCDF's `add_variable::<T>(name, dims)` needs the element type `T` and the
//! dimension *names* up front, and a variable's dimensions must already exist in
//! the file. Rather than try to recover the element type syntactically (which
//! breaks the moment a `map` function changes the type), the generated method
//! begins by defining a handful of **local helper `fn`s**. Their type
//! parameters are inferred from their arguments, so `map`ped and un-`map`ped
//! fields flow through the exact same code path. The helpers reference only
//! `::netcdf` and `::ndarray`, so no runtime support crate is required — the
//! whole thing is self-contained like the sibling `Subset` derive.
//!
//! Only the helpers actually needed by the struct's fields are emitted. This
//! matters for two reasons: it avoids dead-code warnings, and it means a struct
//! made only of `Vec` fields never references `::ndarray` (which such a crate
//! may not even depend on).

use darling::ast::NestedMeta;
use darling::util::Override;
use darling::{FromDeriveInput, FromField, FromMeta, ast};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Type;

// ---------------------------------------------------------------------------
// darling input models
// ---------------------------------------------------------------------------

/// A list of dimension names written as `#[netcdf(dimensions("time", "level"))]`.
///
/// darling implements `FromMeta` for `Vec<T>` only for a few concrete element
/// types (not `String`), so we wrap the list in a newtype and parse each entry
/// ourselves via `FromMeta::from_list`.
#[derive(Debug, Clone, Default)]
struct DimList(Vec<String>);

impl FromMeta for DimList {
    fn from_list(items: &[NestedMeta]) -> darling::Result<Self> {
        // Accumulate so that *all* malformed entries are reported at once.
        let mut errors = darling::Error::accumulator();
        let mut names = Vec::with_capacity(items.len());
        for item in items {
            if let Some(name) = errors.handle(String::from_nested_meta(item)) {
                names.push(name);
            }
        }
        errors.finish()?;
        Ok(DimList(names))
    }
}

/// A single `#[netcdf(attr(name = "...", value = ...))]` or
/// `#[netcdf(attr(name = "...", func = path::to::fn))]` entry, attaching an
/// attribute to the variable created for the field.
///
/// Exactly one of `value` / `func` must be given (validated in [`resolve_field`]).
/// `with` adapter: keep a string literal verbatim (see the `value` field).
fn keep_expr(meta: &syn::Meta) -> darling::Result<Option<syn::Expr>> {
    darling::util::parse_expr::preserve_str_literal(meta).map(Some)
}

#[derive(Debug, FromMeta)]
struct NcVarAttr {
    /// The netCDF attribute name.
    name: String,
    /// A constant expression to use as the attribute value. Must be
    /// `Into<netcdf::AttributeValue>` (e.g. a string, number, or `Vec` thereof).
    ///
    /// Parsed with `preserve_str_literal` so `value = "K"` stays the string
    /// literal `"K"` rather than being re-parsed as the identifier `K`.
    #[darling(default, with = keep_expr)]
    value: Option<syn::Expr>,
    /// A function `fn(&Self, &FieldTy, &str) -> V` (with `V: Into<AttributeValue>`)
    /// producing the attribute value from the whole struct, the field value, and
    /// the field's netCDF name.
    #[darling(default)]
    func: Option<syn::Path>,
}

#[derive(FromField)]
#[darling(attributes(netcdf), forward_attrs(doc))]
struct WriteNetcdfField {
    ident: Option<syn::Ident>,
    ty: syn::Type,

    /// Do not write this field at all.
    #[darling(default)]
    skip: bool,

    /// Override the netCDF name for this field (takes precedence over the
    /// container-level `rename_all`).
    #[darling(default)]
    rename: Option<String>,

    /// Force a write strategy: `"values"`, `"array"`, or `"string"`. Required
    /// when `map`/`map_units` is present (the output type can't be inferred).
    #[darling(default)]
    kind: Option<String>,

    /// Dimension names for this variable, auto-created if missing.
    #[darling(default)]
    dimensions: Option<DimList>,

    /// Write the field as a *group attribute* instead of a variable.
    #[darling(default)]
    attribute: bool,

    /// Map the field value before writing: `fn(&Self, &FieldTy, &str) -> Out`.
    #[darling(default)]
    map: Option<syn::Path>,

    /// Like `map`, but also returns the `units` attribute:
    /// `fn(&Self, &FieldTy, &str) -> (Out, String)`.
    #[darling(default)]
    map_units: Option<syn::Path>,

    /// Zero or more attributes to attach to the created variable.
    ///
    /// Not named `attrs`: darling reserves that field name on `FromField`
    /// structs for the raw `Vec<syn::Attribute>`. We map the `attr(...)` key to
    /// this field explicitly.
    #[darling(default, multiple, rename = "attr")]
    var_attrs: Vec<NcVarAttr>,

    /// The raw field attributes, filled by darling because of the magic name
    /// `attrs`. Used only to read `///` doc comments (which the compiler lowers
    /// to `#[doc = "..."]`) for the description feature.
    attrs: Vec<syn::Attribute>,

    /// Per-field override of the container-level `describe`. `#[netcdf(describe)]`
    /// turns descriptions on for this field using the inherited attribute name;
    /// `#[netcdf(describe = "notes")]` also overrides the attribute name (as with
    /// `rename`).
    #[darling(default)]
    describe: Option<Override<String>>,

    /// Disable the field's doc-comment description even when the container turns
    /// descriptions on for every field.
    #[darling(default)]
    no_describe: bool,
}

#[derive(FromDeriveInput)]
#[darling(attributes(netcdf), supports(struct_named))]
struct WriteNetcdfInput {
    ident: syn::Ident,
    generics: syn::Generics,
    data: ast::Data<darling::util::Ignored, WriteNetcdfField>,

    /// serde-style case convention applied to every field's netCDF name.
    #[darling(default)]
    rename_all: Option<String>,

    /// Write each field's `///` doc comment as a variable attribute. The word
    /// form `#[netcdf(describe)]` uses the default attribute name `"description"`;
    /// `#[netcdf(describe = "long_name")]` chooses the attribute name. Individual
    /// fields can override the name or opt out with `#[netcdf(no_describe)]`.
    #[darling(default)]
    describe: Option<Override<String>>,
}

// ---------------------------------------------------------------------------
// Field classification
// ---------------------------------------------------------------------------

/// How a field's value is written to netCDF.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WriteKind {
    /// 1-D `Vec`/slice via `VariableMut::put_values`.
    Values,
    /// ndarray `Array*` via `VariableMut::put`.
    Array,
    /// A single string via `VariableMut::put_string`.
    StringScalar,
    /// A `Vec<String>` via a loop of `VariableMut::put_string`.
    StringVec,
    /// A scalar (number/bool) via `VariableMut::put_value`.
    Scalar,
    /// Written as a group attribute rather than a variable.
    Attribute,
    /// Not written.
    Skip,
}

/// Pull the first generic argument `T` out of `Vec<T>` (used to distinguish
/// `Vec<String>` from `Vec<f64>` during inference).
fn vec_element_type(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let seg = type_path.path.segments.last()?;
    if seg.ident != "Vec" {
        return None;
    }
    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
        if let Some(syn::GenericArgument::Type(t)) = args.args.first() {
            return Some(t);
        }
    }
    None
}

/// True if `ty` is `String`, `&str`, or `&String` — i.e. a scalar string.
fn is_scalar_string(ty: &Type) -> bool {
    match ty {
        Type::Reference(r) => is_scalar_string(&r.elem),
        Type::Path(p) => p
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "String" || s.ident == "str"),
        _ => false,
    }
}

/// Map an explicit `kind = "..."` string to a [`WriteKind`], using the field's
/// declared type only to decide scalar-vs-vec for `"string"`.
fn kind_from_str(kind: &str, ty: &Type) -> Result<WriteKind, darling::Error> {
    match kind {
        "values" => Ok(WriteKind::Values),
        "array" => Ok(WriteKind::Array),
        "string" => {
            if is_scalar_string(ty) {
                Ok(WriteKind::StringScalar)
            } else {
                Ok(WriteKind::StringVec)
            }
        }
        other => Err(darling::Error::custom(format!(
            "unknown kind \"{other}\", expected \"values\", \"array\", or \"string\""
        ))),
    }
}

/// Infer a [`WriteKind`] from the field's declared type.
fn infer_kind(ty: &Type) -> Result<WriteKind, darling::Error> {
    if is_scalar_string(ty) {
        return Ok(WriteKind::StringScalar);
    }
    if let Some(elem) = vec_element_type(ty) {
        return if is_scalar_string(elem) {
            Ok(WriteKind::StringVec)
        } else {
            Ok(WriteKind::Values)
        };
    }
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            if seg.ident.to_string().starts_with("Array") {
                return Ok(WriteKind::Array);
            }
            // Numeric/bool scalars fall through to `Scalar`.
            return Ok(WriteKind::Scalar);
        }
    }
    Err(darling::Error::custom(
        "couldn't infer how to write this field — annotate with \
         #[netcdf(kind = \"values\"|\"array\"|\"string\")], #[netcdf(attribute)], \
         or #[netcdf(skip)]",
    ))
}

/// Validate a field's attributes and determine its [`WriteKind`]. All errors are
/// spanned on the field ident so the compiler points at the right place.
fn resolve_field(f: &WriteNetcdfField) -> Result<WriteKind, darling::Error> {
    let ident = f.ident.as_ref().unwrap();
    let err = |msg: &str| darling::Error::custom(msg).with_span(ident);

    let has_dims = f.dimensions.as_ref().is_some_and(|d| !d.0.is_empty());

    // `skip` is exclusive with every other option.
    if f.skip {
        if f.rename.is_some()
            || f.kind.is_some()
            || has_dims
            || f.attribute
            || f.map.is_some()
            || f.map_units.is_some()
            || !f.var_attrs.is_empty()
            || f.describe.is_some()
            || f.no_describe
        {
            return Err(err(
                "`skip` cannot be combined with any other `netcdf` option",
            ));
        }
        return Ok(WriteKind::Skip);
    }

    // `map` and `map_units` are mutually exclusive.
    if f.map.is_some() && f.map_units.is_some() {
        return Err(err("`map` and `map_units` cannot be used together"));
    }

    // `describe` and `no_describe` contradict each other.
    if f.describe.is_some() && f.no_describe {
        return Err(err("`describe` and `no_describe` cannot be used together"));
    }

    // Each `attr(...)` needs exactly one of `value` / `func`.
    for a in &f.var_attrs {
        match (&a.value, &a.func) {
            (Some(_), Some(_)) => {
                return Err(err("`attr(...)` cannot set both `value` and `func`"));
            }
            (None, None) => {
                return Err(err("`attr(...)` needs either `value` or `func`"));
            }
            _ => {}
        }
    }

    // A field written as a group attribute is not a variable, so variable-only
    // options don't apply.
    if f.attribute {
        if has_dims
            || f.kind.is_some()
            || !f.var_attrs.is_empty()
            || f.map_units.is_some()
            || f.describe.is_some()
            || f.no_describe
        {
            return Err(err(
                "`attribute` is incompatible with `dimensions`, `kind`, `attr`, `map_units`, and `describe`",
            ));
        }
        return Ok(WriteKind::Attribute);
    }

    // With a `map`/`map_units` the written type is opaque, so `kind` is required.
    if f.map.is_some() || f.map_units.is_some() {
        let Some(kind) = &f.kind else {
            return Err(err(
                "`map`/`map_units` requires an explicit `kind = \"values\"|\"array\"|\"string\"`",
            ));
        };
        let wk = kind_from_str(kind, &f.ty).map_err(|e| e.with_span(ident))?;
        return validate_dims(wk, has_dims, &f.dimensions, ident).map(|()| wk);
    }

    // Explicit `kind`, else infer from the type.
    let wk = if let Some(kind) = &f.kind {
        kind_from_str(kind, &f.ty).map_err(|e| e.with_span(ident))?
    } else {
        infer_kind(&f.ty).map_err(|e| e.with_span(ident))?
    };
    validate_dims(wk, has_dims, &f.dimensions, ident).map(|()| wk)
}

/// Enforce the dimension-count rules that depend on the write strategy.
fn validate_dims(
    wk: WriteKind,
    has_dims: bool,
    dims: &Option<DimList>,
    ident: &syn::Ident,
) -> Result<(), darling::Error> {
    let err = |msg: &str| darling::Error::custom(msg).with_span(ident);
    match wk {
        WriteKind::Values | WriteKind::StringVec => {
            if dims.as_ref().is_some_and(|d| d.0.len() > 1) {
                return Err(err("a 1-D field accepts at most one dimension name"));
            }
        }
        WriteKind::Scalar | WriteKind::StringScalar => {
            if has_dims {
                return Err(err("a scalar field cannot have dimensions"));
            }
        }
        // Arrays accept any number of names; anything else has been handled.
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// rename_all
// ---------------------------------------------------------------------------

/// Apply a serde-style `rename_all` rule to a (snake_case) Rust field name.
fn apply_rename_all(field: &str, rule: &str) -> Result<String, darling::Error> {
    let words: Vec<&str> = field.split('_').filter(|w| !w.is_empty()).collect();
    let upper_first = |w: &str| {
        let mut c = w.chars();
        match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => String::new(),
        }
    };
    Ok(match rule {
        "lowercase" => field.replace('_', "").to_lowercase(),
        "UPPERCASE" => field.replace('_', "").to_uppercase(),
        "snake_case" => field.to_string(),
        "SCREAMING_SNAKE_CASE" => field.to_uppercase(),
        "kebab-case" => field.replace('_', "-"),
        "SCREAMING-KEBAB-CASE" => field.to_uppercase().replace('_', "-"),
        "PascalCase" => words.iter().map(|w| upper_first(w)).collect(),
        "camelCase" => {
            let mut it = words.iter();
            let head = it.next().map(|w| w.to_lowercase()).unwrap_or_default();
            head + &it.map(|w| upper_first(w)).collect::<String>()
        }
        other => {
            return Err(darling::Error::custom(format!(
                "unknown rename_all rule \"{other}\""
            )));
        }
    })
}

/// Resolve a field's final netCDF name: `rename` wins, then `rename_all`, then
/// the raw field name.
fn nc_name(f: &WriteNetcdfField, rename_all: Option<&str>) -> Result<String, darling::Error> {
    if let Some(name) = &f.rename {
        return Ok(name.clone());
    }
    let field = f.ident.as_ref().unwrap().to_string();
    match rename_all {
        Some(rule) => apply_rename_all(&field, rule),
        None => Ok(field),
    }
}

// ---------------------------------------------------------------------------
// Descriptions from doc comments
// ---------------------------------------------------------------------------

/// Collect a field's `///` doc comment into a single trimmed string. The
/// compiler lowers each `///` line to a `#[doc = "..."]` attribute, so we glue
/// those back together. Returns `None` when the field has no doc comment.
fn extract_doc(attrs: &[syn::Attribute]) -> Option<String> {
    let mut lines = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &attr.meta {
            if let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) = &nv.value
            {
                let line = s.value();
                // rustdoc keeps the single space after `///`; drop it so the
                // stored text isn't uniformly indented.
                lines.push(line.strip_prefix(' ').unwrap_or(&line).to_string());
            }
        }
    }
    let text = lines.join(" ");
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Decide whether a field gets a description attribute and, if so, its name.
/// Mirrors rename resolution: the container sets the default, each field may
/// override the name or opt out.
fn resolve_describe(
    container: Option<&Override<String>>,
    field: Option<&Override<String>>,
    no_describe: bool,
) -> Option<String> {
    // The container default is a pair of (enabled, attribute name).
    let (container_on, container_name): (bool, String) = match container {
        None => (false, "description".to_string()),
        Some(Override::Inherit) => (true, "description".to_string()),
        Some(Override::Explicit(name)) => (true, name.clone()),
    };
    if no_describe {
        return None;
    }
    match field {
        None => container_on.then_some(container_name),
        Some(Override::Inherit) => Some(container_name),
        Some(Override::Explicit(name)) => Some(name.clone()),
    }
}

// ---------------------------------------------------------------------------
// Which local helpers to emit
// ---------------------------------------------------------------------------

/// Tracks which helper `fn`s the generated method needs, so we only emit (and
/// therefore only reference `::ndarray` for) what's actually used.
#[derive(Default)]
struct HelpersNeeded {
    ensure_dim: bool,
    ensure_dims: bool,
    values: bool,
    array: bool,
    strings: bool,
    scalar: bool,
    string_scalar: bool,
}

impl HelpersNeeded {
    fn note(&mut self, wk: WriteKind) {
        match wk {
            WriteKind::Values => {
                self.values = true;
                self.ensure_dim = true;
            }
            WriteKind::StringVec => {
                self.strings = true;
                self.ensure_dim = true;
            }
            WriteKind::Array => {
                self.array = true;
                self.ensure_dim = true;
                self.ensure_dims = true;
            }
            WriteKind::Scalar => self.scalar = true,
            WriteKind::StringScalar => self.string_scalar = true,
            WriteKind::Attribute | WriteKind::Skip => {}
        }
    }

    /// The token block of local helper `fn`s for the top of `write_netcdf`.
    fn tokens(&self) -> TokenStream2 {
        let mut out = TokenStream2::new();
        if self.ensure_dim {
            out.extend(quote! {
                // Create `name` with length `len` if it doesn't exist yet.
                fn __ensure_dim(g: &mut ::netcdf::GroupMut<'_>, name: &str, len: usize)
                    -> ::netcdf::Result<()>
                {
                    if g.dimension(name).is_none() {
                        g.add_dimension(name, len)?;
                        log::trace!("Created dimension {name} with length {len}");
                    }
                    Ok(())
                }
            });
        }
        if self.ensure_dims {
            out.extend(quote! {
                // Ensure one dimension per array axis, sized from the array shape.
                fn __ensure_dims(g: &mut ::netcdf::GroupMut<'_>, names: &[&str], shape: &[usize])
                    -> ::netcdf::Result<()>
                {
                    for (nm, ln) in names.iter().zip(shape.iter()) {
                        __ensure_dim(g, nm, *ln)?;
                    }
                    Ok(())
                }
            });
        }
        if self.values {
            out.extend(quote! {
                // Element type `T` is inferred from `vals`, so `map`ped and
                // un-`map`ped fields share this path.
                fn __put_values<'g, T: ::netcdf::NcTypeDescriptor>(
                    g: &'g mut ::netcdf::GroupMut<'_>, name: &str, dims: &[&str], vals: &[T],
                ) -> ::netcdf::Result<::netcdf::VariableMut<'g>> {
                    log::trace!("Writing to variable '{name}' with __put_values");
                    let mut v = g.add_variable::<T>(name, dims)?;
                    v.put_values(vals, ..)?;
                    Ok(v)
                }
            });
        }
        if self.array {
            out.extend(quote! {
                fn __put_array<'g, T: ::netcdf::NcTypeDescriptor, D: ::ndarray::Dimension>(
                    g: &'g mut ::netcdf::GroupMut<'_>, name: &str, dims: &[&str],
                    arr: ::ndarray::ArrayView<'_, T, D>,
                ) -> ::netcdf::Result<::netcdf::VariableMut<'g>> {
                    if dims.len() != arr.ndim() {
                        return Err(netcdf::Error::Str(format!(
                            "{} dimension(s) were defined for variable {}, but the array has {} dimension(s)",
                            dims.len(), name, arr.ndim()
                        )))
                    }
                    log::trace!("Writing to variable '{name}' with __put_array");
                    let mut v = g.add_variable::<T>(name, dims)?;
                    v.put(arr, ..)?;
                    Ok(v)
                }
            });
        }
        if self.strings {
            out.extend(quote! {
                // netCDF writes one string at a time, so loop over the slice.
                fn __put_strings<'g>(
                    g: &'g mut ::netcdf::GroupMut<'_>, name: &str, dims: &[&str], vals: &[String],
                ) -> ::netcdf::Result<::netcdf::VariableMut<'g>> {
                    log::trace!("Writing to variable '{name}' with __put_strings");
                    let mut v = g.add_string_variable(name, dims)?;
                    for (i, s) in vals.iter().enumerate() {
                        v.put_string(s, [i])?;
                    }
                    Ok(v)
                }
            });
        }
        if self.scalar {
            out.extend(quote! {
                fn __put_scalar<'g, T: ::netcdf::NcTypeDescriptor>(
                    g: &'g mut ::netcdf::GroupMut<'_>, name: &str, val: T,
                ) -> ::netcdf::Result<::netcdf::VariableMut<'g>> {
                    log::trace!("Writing to variable '{name}' with __put_scalar");
                    let mut v = g.add_variable::<T>(name, &[])?;
                    v.put_value(val, ())?;
                    Ok(v)
                }
            });
        }
        if self.string_scalar {
            out.extend(quote! {
                fn __put_string_scalar<'g>(
                    g: &'g mut ::netcdf::GroupMut<'_>, name: &str, s: &str,
                ) -> ::netcdf::Result<::netcdf::VariableMut<'g>> {
                    log::trace!("Writing to variable '{name}' with __put_string_scalar");
                    let mut v = g.add_string_variable(name, &[])?;
                    v.put_string(s, ())?;
                    Ok(v)
                }
            });
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Per-field code generation
// ---------------------------------------------------------------------------

/// Build the `{ ... }` statement block that writes one field. `description`, if
/// present, is the `(attribute name, text)` to attach to the created variable.
fn field_block(
    f: &WriteNetcdfField,
    wk: WriteKind,
    name: &str,
    description: Option<(String, String)>,
) -> TokenStream2 {
    let ident = f.ident.as_ref().unwrap();
    let mapped = f.map.is_some() || f.map_units.is_some();

    // `prelude` computes the value to write; `place` names it afterwards.
    // Without a map we operate directly on `self.<field>`.
    let (prelude, place): (TokenStream2, TokenStream2) = if let Some(m) = &f.map {
        (
            quote! { let __val = #m(self, &self.#ident, #name); },
            quote!(__val),
        )
    } else if let Some(m) = &f.map_units {
        (
            quote! { let (__val, __units) = #m(self, &self.#ident, #name); },
            quote!(__val),
        )
    } else {
        (quote! {}, quote!(self.#ident))
    };

    // A scalar/attribute value must be owned; clone only when we can't move.
    let owned = if mapped {
        quote!(__val)
    } else {
        quote!(self.#ident.clone())
    };

    // Attribute statements applied to the created variable (`__var`).
    let mut attr_stmts = TokenStream2::new();
    // The doc-comment description is written first so an explicit `attr(...)`
    // could intentionally shadow it if desired.
    if let Some((desc_attr, desc_text)) = &description {
        attr_stmts.extend(quote! { __var.put_attribute(#desc_attr, #desc_text)?; });
    }
    for a in &f.var_attrs {
        let an = &a.name;
        if let Some(v) = &a.value {
            attr_stmts.extend(quote! { __var.put_attribute(#an, #v)?; });
        } else if let Some(func) = &a.func {
            attr_stmts
                .extend(quote! { __var.put_attribute(#an, #func(self, &self.#ident, #name))?; });
        }
    }
    if f.map_units.is_some() {
        attr_stmts.extend(quote! { __var.put_attribute("units", __units)?; });
    }
    let needs_var = !attr_stmts.is_empty();

    // The dimension-name list literal (`["time", "level"]`), if given explicitly.
    let dim_names: Vec<String> = f
        .dimensions
        .as_ref()
        .map(|d| d.0.clone())
        .unwrap_or_default();

    // Produce the `helper(...)` call that creates + writes the variable.
    let call: TokenStream2 = match wk {
        WriteKind::Values => {
            let d0 = dim_names
                .first()
                .cloned()
                .unwrap_or_else(|| name.to_string());
            quote! {{
                __ensure_dim(grp, #d0, #place.len())?;
                __put_values(grp, #name, &[#d0], & #place)
            }}
        }
        WriteKind::StringVec => {
            let d0 = dim_names
                .first()
                .cloned()
                .unwrap_or_else(|| name.to_string());
            quote! {{
                __ensure_dim(grp, #d0, #place.len())?;
                __put_strings(grp, #name, &[#d0], & #place)
            }}
        }
        WriteKind::Array => {
            if dim_names.is_empty() {
                // No names given: derive one per axis at runtime as `name_0`, ...
                quote! {{
                    let __shape = #place.shape().to_vec();
                    let __names: ::std::vec::Vec<String> =
                        (0..__shape.len()).map(|__i| format!("{}_{}", #name, __i)).collect();
                    let __nrefs: ::std::vec::Vec<&str> =
                        __names.iter().map(|__s| __s.as_str()).collect();
                    __ensure_dims(grp, &__nrefs, &__shape)?;
                    __put_array(grp, #name, &__nrefs, #place.view())
                }}
            } else {
                quote! {{
                    __ensure_dims(grp, &[#(#dim_names),*], #place.shape())?;
                    __put_array(grp, #name, &[#(#dim_names),*], #place.view())
                }}
            }
        }
        WriteKind::StringScalar => quote! { __put_string_scalar(grp, #name, & #place) },
        WriteKind::Scalar => quote! { __put_scalar(grp, #name, #owned) },
        WriteKind::Attribute => {
            // A group attribute: no variable, no dimensions.
            return quote! {{
                #prelude
                grp.add_attribute(#name, #owned)?;
            }};
        }
        WriteKind::Skip => unreachable!("skip fields are filtered out before codegen"),
    };

    // Bind `__var` only when there are attributes to attach (avoids unused
    // variable/`mut` warnings); the block scope releases the borrow of `grp`
    // before the next field runs.
    if needs_var {
        quote! {{
            #prelude
            let mut __var = #call?;
            #attr_stmts
        }}
    } else {
        quote! {{
            #prelude
            #call?;
        }}
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn expand(input: TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(input as syn::DeriveInput);
    let parsed = match WriteNetcdfInput::from_derive_input(&derive_input) {
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
    if let Some(rule) = rename_all {
        if let Err(e) = apply_rename_all("x", rule) {
            errors.push(e);
        }
    }

    let mut helpers = HelpersNeeded::default();
    let mut field_blocks: Vec<TokenStream2> = Vec::new();

    for f in &fields {
        // Resolve the write strategy and the netCDF name, collecting errors so
        // every problem is reported in one compile.
        let Some(wk) = errors.handle(resolve_field(f)) else {
            continue;
        };
        if wk == WriteKind::Skip {
            continue;
        }
        let Some(name) = errors.handle(nc_name(f, rename_all)) else {
            continue;
        };
        // Descriptions only apply to variables, not to group-attribute fields.
        let description = if wk == WriteKind::Attribute {
            None
        } else {
            resolve_describe(parsed.describe.as_ref(), f.describe.as_ref(), f.no_describe)
                .and_then(|attr| extract_doc(&f.attrs).map(|doc| (attr, doc)))
        };
        helpers.note(wk);
        field_blocks.push(field_block(f, wk, &name, description));
    }

    if let Err(e) = errors.finish() {
        return e.write_errors().into();
    }

    let helper_fns = helpers.tokens();
    let (impl_generics, ty_generics, where_clause) = parsed.generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            /// Write this struct's fields into `grp` as netCDF variables and
            /// group attributes. Pass a file root (`FileMut::root_mut()`) or a
            /// subgroup. Missing dimensions are created automatically.
            pub fn write_netcdf(&self, grp: &mut ::netcdf::GroupMut<'_>) -> ::netcdf::Result<()> {
                #helper_fns
                #(#field_blocks)*
                Ok(())
            }
        }
    };

    TokenStream::from(expanded)
}
