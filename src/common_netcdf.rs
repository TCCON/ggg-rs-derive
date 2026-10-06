use darling::FromMeta;
use darling::ast::NestedMeta;
use syn::Type;

// ---------------------------------------------------------------------------
// map / map_units
// ---------------------------------------------------------------------------

/// Which direction a derive is generating code for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    Read,
    Write,
}

impl Direction {
    /// The key used inside `map(...)` for this direction.
    fn key(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
        }
    }

    /// The derive that generates this direction, for error messages.
    fn derive_name(self) -> &'static str {
        match self {
            Self::Read => "ReadNetcdf",
            Self::Write => "WriteNetcdf",
        }
    }
}

const MAP_FORM_HELP: &str = "`map` and `map_units` need an explicit direction, because reading \
     and writing require different functions: `map(write = path::to::fn)`, \
     `map(read = path::to::fn)`, or both";

const MAP_UNITS_HELP: &str =
    "`map()` cannot contain a `units_attr` field, as only a `map_units` attribute uses units";

const DEFAULT_UNIT_ATTR: &str = "units";

/// The function pair behind `#[netcdf(map(...))]` / `#[netcdf(map_units(...))]`.
///
/// The two directions are *not* interchangeable — they are inverses of each
/// other, they have different signatures (the read side has no `&Self`, since
/// it is still building it), and only the read side is fallible — so the
/// direction is always named explicitly. Either half may be omitted when only
/// one of the two derives is in use; whichever derive needs the missing half
/// reports it (see [`MapFns::resolve`]).
#[derive(Debug, Clone)]
pub(crate) struct MapSpec {
    pub(crate) write: Option<syn::Path>,
    pub(crate) read: Option<syn::Path>,
    pub(crate) units_attr: Option<String>,
}

/// The `write = ...` / `read = ...` body of a `map(...)`, parsed by darling.
/// Split out from [`MapSpec`] so that `MapSpec`'s own `FromMeta` impl can reject
/// the non-directional forms with a useful message.
#[derive(FromMeta)]
struct MapSpecFields {
    #[darling(default)]
    write: Option<syn::Path>,
    #[darling(default)]
    read: Option<syn::Path>,
    units_attr: Option<String>,
}

impl FromMeta for MapSpec {
    /// The only accepted form: `map(write = ..., read = ...)`.
    fn from_list(items: &[NestedMeta]) -> darling::Result<Self> {
        let fields = MapSpecFields::from_list(items)?;
        if fields.write.is_none() && fields.read.is_none() {
            return Err(darling::Error::custom(MAP_FORM_HELP));
        }
        Ok(Self {
            write: fields.write,
            read: fields.read,
            units_attr: fields.units_attr,
        })
    }

    /// Rejects a bare `map`.
    fn from_word() -> darling::Result<Self> {
        Err(darling::Error::custom(MAP_FORM_HELP))
    }

    /// Rejects the old non-directional `map = path::to::fn` form.
    fn from_expr(_expr: &syn::Expr) -> darling::Result<Self> {
        Err(darling::Error::custom(MAP_FORM_HELP))
    }
}

/// The map functions that apply to one direction, resolved from the field's
/// `map(...)` / `map_units(...)` specs.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MapFns<'a> {
    pub(crate) map: Option<&'a syn::Path>,
    pub(crate) map_units: Option<(&'a syn::Path, &'a str)>,
}

impl<'a> MapFns<'a> {
    /// Pick out the `dir` half of each spec. A spec that exists but does not
    /// cover `dir` is an error: the field would otherwise be silently read or
    /// written unmapped, which is almost never what a half-specified `map`
    /// means. Fields of a struct that derives only one direction are unaffected,
    /// since the other derive never runs.
    pub(crate) fn resolve(
        map: Option<&'a MapSpec>,
        map_units: Option<&'a MapSpec>,
        dir: Direction,
        ident: &syn::Ident,
    ) -> Result<Self, darling::Error> {
        if map.is_some_and(|m| m.units_attr.is_some()) {
            return Err(darling::Error::custom(MAP_UNITS_HELP));
        }

        let map_fxn = direction_fn(map, "map", dir, ident)?;
        let map_units_fxn = direction_fn(map_units, "map_units", dir, ident)?;
        let map_units_tuple = map_units.map(|m| {
            (
                map_units_fxn.unwrap(),
                m.units_attr.as_deref().unwrap_or(DEFAULT_UNIT_ATTR),
            )
        });

        Ok(Self {
            map: map_fxn,
            map_units: map_units_tuple,
        })
    }

    /// True if this direction maps the field at all.
    pub(crate) fn any(&self) -> bool {
        self.map.is_some() || self.map_units.is_some()
    }
}

/// Take one direction's function out of an optional spec, erroring when the
/// spec is present but silent about this direction.
fn direction_fn<'a>(
    spec: Option<&'a MapSpec>,
    attr_name: &str,
    dir: Direction,
    ident: &syn::Ident,
) -> Result<Option<&'a syn::Path>, darling::Error> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    let chosen = match dir {
        Direction::Read => spec.read.as_ref(),
        Direction::Write => spec.write.as_ref(),
    };
    match chosen {
        Some(path) => Ok(Some(path)),
        None => Err(darling::Error::custom(format!(
            "`{attr_name}` on this field has no `{} = ...`, which `{}` requires",
            dir.key(),
            dir.derive_name()
        ))
        .with_span(ident)),
    }
}

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
    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
        && let Some(syn::GenericArgument::Type(t)) = args.args.first()
    {
        return Some(t);
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
    if let Type::Path(p) = ty
        && let Some(seg) = p.path.segments.last()
    {
        if seg.ident.to_string().starts_with("Array") {
            return Ok(NcKind::Array);
        }
        // Numeric/bool scalars fall through to `Scalar`.
        return Ok(NcKind::Scalar);
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
