use syn::Type;

// ---------------------------------------------------------------------------
// rename_all
// ---------------------------------------------------------------------------

/// Apply a serde-style `rename_all` rule to a (snake_case) Rust field name.
pub(crate) fn apply_rename_all(field: &str, rule: &str) -> Result<String, darling::Error> {
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
pub(crate) fn nc_name(
    ident: &syn::Ident,
    rename: Option<&str>,
    rename_all: Option<&str>,
) -> Result<String, darling::Error> {
    if let Some(name) = rename {
        return Ok(name.to_string());
    }
    let field = ident.to_string();
    match rename_all {
        Some(rule) => apply_rename_all(&field, rule),
        None => Ok(field),
    }
}

// --------------------
// Field classification
// --------------------

/// How a field's value is written to or read from netCDF.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum NcKind {
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
pub(crate) fn vec_element_type(ty: &Type) -> Option<&Type> {
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
pub(crate) fn is_scalar_string(ty: &Type) -> bool {
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
pub(crate) fn kind_from_str(kind: &str, ty: &Type) -> Result<NcKind, darling::Error> {
    match kind {
        "values" => Ok(NcKind::Values),
        "array" => Ok(NcKind::Array),
        "string" => {
            if is_scalar_string(ty) {
                Ok(NcKind::StringScalar)
            } else {
                Ok(NcKind::StringVec)
            }
        }
        other => Err(darling::Error::custom(format!(
            "unknown kind \"{other}\", expected \"values\", \"array\", or \"string\""
        ))),
    }
}

/// Infer a [`WriteKind`] from the field's declared type.
pub(crate) fn infer_kind(ty: &Type) -> Result<NcKind, darling::Error> {
    if is_scalar_string(ty) {
        return Ok(NcKind::StringScalar);
    }
    if let Some(elem) = vec_element_type(ty) {
        return if is_scalar_string(elem) {
            Ok(NcKind::StringVec)
        } else {
            Ok(NcKind::Values)
        };
    }
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            if seg.ident.to_string().starts_with("Array") {
                return Ok(NcKind::Array);
            }
            // Numeric/bool scalars fall through to `Scalar`.
            return Ok(NcKind::Scalar);
        }
    }
    Err(darling::Error::custom(
        "couldn't infer how to write this field — annotate with \
         #[netcdf(kind = \"values\"|\"array\"|\"string\")], #[netcdf(attribute)], \
         or #[netcdf(skip)]",
    ))
}

/// A list of dimension names written as `#[netcdf(dimensions("time", "level"))]`.
///
/// darling implements `FromMeta` for `Vec<T>` only for a few concrete element
/// types (not `String`), so we wrap the list in a newtype and parse each entry
/// ourselves via `FromMeta::from_list`.
#[derive(Debug, Clone, Default)]
pub(crate) struct DimList(pub(crate) Vec<String>);

/// Enforce the dimension-count rules for types with obvious dimension limits (scalars, vectors, etc.).
pub(crate) fn validate_dims(
    wk: NcKind,
    has_dims: bool,
    dims: &Option<DimList>,
    ident: &syn::Ident,
) -> Result<(), darling::Error> {
    let err = |msg: &str| darling::Error::custom(msg).with_span(ident);
    match wk {
        NcKind::Values | NcKind::StringVec => {
            if dims.as_ref().is_some_and(|d| d.0.len() > 1) {
                return Err(err("a 1-D field accepts at most one dimension name"));
            }
        }
        NcKind::Scalar | NcKind::StringScalar => {
            if has_dims {
                return Err(err("a scalar field cannot have dimensions"));
            }
        }
        // Arrays accept any number of names; anything else has been handled.
        _ => {}
    }
    Ok(())
}
