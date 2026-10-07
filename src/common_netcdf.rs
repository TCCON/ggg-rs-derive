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

/// Pull the first angle-bracketed type argument out of a path type, so
/// `Vec<T>`, `Array2<T>`, and `Array<T, D>` all yield `T`. Returns `None` for
/// types with no generic arguments and for anything that isn't a path.
fn first_type_arg(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let seg = type_path.path.segments.last()?;
    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
        && let Some(syn::GenericArgument::Type(t)) = args.args.first()
    {
        return Some(t);
    }
    None
}

/// Pull the first generic argument `T` out of `Vec<T>` (used to distinguish
/// `Vec<String>` from `Vec<f64>` during inference).
pub(crate) fn vec_element_type(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.path.segments.last()?.ident != "Vec" {
        return None;
    }
    first_type_arg(ty)
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

// ---------------------------------------------------------------------------
// Packing and unpacking (scale_factor / add_offset / fill values)
// ---------------------------------------------------------------------------

/// The Rust float type a field's elements are stored in. Packing and unpacking
/// only apply to these two: the netCDF attribute conventions put the *packed*
/// data in a byte or short variable and the unpacked data in a float or double,
/// and scaling an integer field in place would truncate the scale factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloatTy {
    F32,
    F64,
}

impl FloatTy {
    /// The type's name, for error messages and for naming the `__unpack_*` helper.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// The identifier of the generated unpack helper for this type.
    pub(crate) fn unpack_fn(self) -> syn::Ident {
        syn::Ident::new(
            match self {
                Self::F32 => "__unpack_f32",
                Self::F64 => "__unpack_f64",
            },
            proc_macro2::Span::call_site(),
        )
    }
}

/// Decide whether a field's elements are `f32` or `f64`.
///
/// Looks at the first generic argument for container types (`Vec<f64>`,
/// `Array2<f32>`, `Array<f64, Ix2>`) and at the type itself for bare scalars.
/// Deliberately conservative: a type alias such as `type Matrix = Array2<f32>`
/// is opaque to the macro and yields `None`, which means "don't pack/unpack" by
/// default and an error if the field explicitly asked for it.
pub(crate) fn float_element_type(ty: &Type) -> Option<FloatTy> {
    fn as_float(ty: &Type) -> Option<FloatTy> {
        let Type::Path(p) = ty else {
            return None;
        };
        // Reject qualified or multi-segment paths other than the bare names:
        // `f64` is never written `std::f64` as a *type*.
        let seg = p.path.segments.last()?;
        if !seg.arguments.is_empty() {
            return None;
        }
        match seg.ident.to_string().as_str() {
            "f32" => Some(FloatTy::F32),
            "f64" => Some(FloatTy::F64),
            _ => None,
        }
    }

    as_float(ty).or_else(|| first_type_arg(ty).and_then(as_float))
}

/// Render a type for an error message. `quote!` puts spaces around every token,
/// so `Vec<f64>` would otherwise come out as `Vec < f64 >`.
pub(crate) fn type_name(ty: &Type) -> String {
    quote::quote!(#ty)
        .to_string()
        .replace(" <", "<")
        .replace("< ", "<")
        .replace(" >", ">")
        .replace("> ", ">")
        .replace(" ,", ",")
        .replace(" ::", "::")
        .replace(":: ", "::")
}

/// The name of a field's element type, but only when it is a primitive we
/// recognize. Used to tell "this is an `i16`, which cannot be unpacked" apart
/// from "this is a `Matrix`, and I have no idea what is inside it" — the advice
/// for each is different.
pub(crate) fn primitive_element_name(ty: &Type) -> Option<String> {
    const PRIMITIVES: [&str; 15] = [
        "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "i128", "u128", "isize", "usize",
        "f32", "f64", "bool",
    ];
    let candidate = first_type_arg(ty).unwrap_or(ty);
    let Type::Path(p) = candidate else {
        return None;
    };
    let seg = p.path.segments.last()?;
    if !seg.arguments.is_empty() {
        return None;
    }
    let name = seg.ident.to_string();
    PRIMITIVES.contains(&name.as_str()).then_some(name)
}

/// The message explaining why a field cannot be packed or unpacked, tailored to
/// whether its element type was recognizable. `attr` names the attribute being
/// rejected so the same wording serves both derives.
pub(crate) fn not_float_message(attr: &str, ty: &Type) -> String {
    match primitive_element_name(ty) {
        Some(found) => format!(
            "`{attr}` requires an f32 or f64 element type, found `{found}` — scale_factor \
             and add_offset are only meaningful for floating-point data"
        ),
        None => format!(
            "`{attr}` requires an f32 or f64 element type, but the element type of `{}` \
             could not be determined — a type alias is opaque to the macro. Use a concrete \
             `Vec<f64>` or `Array<f64, _>`, or drop the annotation.",
            type_name(ty)
        ),
    }
}

/// A netCDF storage type for packed data, named by `nc_type = "..."`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NcStorageType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl NcStorageType {
    /// The Rust type to hand `add_variable::<T>` / collect the packed values into.
    pub(crate) fn rust_ty(self) -> syn::Ident {
        let name = match self {
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        };
        syn::Ident::new(name, proc_macro2::Span::call_site())
    }

    pub(crate) fn name(self) -> String {
        self.rust_ty().to_string()
    }

    /// True for the integer types, which are the ones that need rounding and a
    /// range check when packing.
    pub(crate) fn is_integer(self) -> bool {
        !matches!(self, Self::F32 | Self::F64)
    }
}

impl FromMeta for NcStorageType {
    fn from_string(value: &str) -> darling::Result<Self> {
        Ok(match value {
            "i8" => Self::I8,
            "u8" => Self::U8,
            "i16" => Self::I16,
            "u16" => Self::U16,
            "i32" => Self::I32,
            "u32" => Self::U32,
            "i64" => Self::I64,
            "u64" => Self::U64,
            "f32" => Self::F32,
            "f64" => Self::F64,
            other => {
                return Err(darling::Error::custom(format!(
                    "unknown nc_type \"{other}\", expected one of \"i8\", \"u8\", \"i16\", \
                     \"u16\", \"i32\", \"u32\", \"i64\", \"u64\", \"f32\", \"f64\""
                )));
            }
        })
    }
}

const SCALE_OFFSET_HELP: &str = "`scale_offset` needs at least one of `scale = ..` or \
     `offset = ..`, plus the packed on-disk type, e.g. \
     `scale_offset(scale = 0.01, offset = 400.0, nc_type = \"i16\")`";

/// `#[netcdf(scale_offset(scale = .., offset = .., nc_type = ".."))]` on the
/// write side: divide the field's values down into a smaller on-disk type and
/// record the `scale_factor` / `add_offset` attributes so a reader can invert it.
#[derive(Debug, Clone)]
pub(crate) struct ScaleOffsetSpec {
    pub(crate) scale: Option<f64>,
    pub(crate) offset: Option<f64>,
    pub(crate) nc_type: NcStorageType,
}

/// The body of a `scale_offset(...)`, parsed by darling. Split out so the outer
/// `FromMeta` impl can reject the degenerate forms with a useful message.
#[derive(FromMeta)]
struct ScaleOffsetFields {
    #[darling(default)]
    scale: Option<f64>,
    #[darling(default)]
    offset: Option<f64>,
    #[darling(default)]
    nc_type: Option<NcStorageType>,
}

impl FromMeta for ScaleOffsetSpec {
    fn from_list(items: &[NestedMeta]) -> darling::Result<Self> {
        let fields = ScaleOffsetFields::from_list(items)?;
        if fields.scale.is_none() && fields.offset.is_none() {
            return Err(darling::Error::custom(SCALE_OFFSET_HELP));
        }
        // Required: without it the variable keeps the field's own float type and
        // packing saves nothing, which is the only reason to pack.
        let Some(nc_type) = fields.nc_type else {
            return Err(darling::Error::custom(
                "`scale_offset` needs `nc_type = \"i16\"` (or another numeric type) \
                 to say what the packed values are stored as",
            ));
        };
        Ok(Self {
            scale: fields.scale,
            offset: fields.offset,
            nc_type,
        })
    }

    fn from_word() -> darling::Result<Self> {
        Err(darling::Error::custom(SCALE_OFFSET_HELP))
    }

    fn from_expr(_expr: &syn::Expr) -> darling::Result<Self> {
        Err(darling::Error::custom(SCALE_OFFSET_HELP))
    }
}

/// What to do with a value that matches the variable's fill or missing value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum OnFill {
    /// Replace with `NaN`, leaving it unscaled. The default.
    #[default]
    Nan,
    /// Leave the raw stored value in place, unscaled.
    Keep,
    /// Don't look for fill values at all; scale them like any other value.
    Ignore,
    /// Report an error naming the variable and index.
    Error,
}

impl OnFill {
    /// The variant of the generated `__OnFill` enum that corresponds to this.
    pub(crate) fn variant(self) -> syn::Ident {
        syn::Ident::new(
            match self {
                Self::Nan => "Nan",
                Self::Keep => "Keep",
                Self::Ignore => "Ignore",
                Self::Error => "Error",
            },
            proc_macro2::Span::call_site(),
        )
    }
}

impl FromMeta for OnFill {
    fn from_string(value: &str) -> darling::Result<Self> {
        Ok(match value {
            "nan" => Self::Nan,
            "keep" => Self::Keep,
            "ignore" => Self::Ignore,
            "error" => Self::Error,
            other => {
                return Err(darling::Error::custom(format!(
                    "unknown on_fill \"{other}\", expected \"nan\", \"keep\", \"ignore\", \
                     or \"error\""
                )));
            }
        })
    }
}

/// `#[netcdf(unpack(...))]` on the read side. Both halves default to on, so the
/// spec only ever records a deliberate choice.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnpackSpec {
    pub(crate) scale: bool,
    pub(crate) offset: bool,
}

/// The body of an `unpack(...)`, parsed by darling.
#[derive(FromMeta)]
struct UnpackFields {
    #[darling(default)]
    scale: Option<bool>,
    #[darling(default)]
    offset: Option<bool>,
}

impl FromMeta for UnpackSpec {
    /// `unpack(scale = false)` — unlisted keys stay on.
    fn from_list(items: &[NestedMeta]) -> darling::Result<Self> {
        let fields = UnpackFields::from_list(items)?;
        Ok(Self {
            scale: fields.scale.unwrap_or(true),
            offset: fields.offset.unwrap_or(true),
        })
    }

    /// A bare `unpack` turns both on.
    fn from_word() -> darling::Result<Self> {
        Ok(Self {
            scale: true,
            offset: true,
        })
    }

    /// `unpack = false` turns both off, `unpack = true` both on.
    fn from_bool(value: bool) -> darling::Result<Self> {
        Ok(Self {
            scale: value,
            offset: value,
        })
    }
}

/// The unpacking settings in force for one field, after combining the
/// container-level default with the field's own override.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnpackFlags {
    pub(crate) scale: bool,
    pub(crate) offset: bool,
    pub(crate) on_fill: OnFill,
    /// True when *this field* asked for unpacking, as opposed to inheriting it
    /// from the container. Only a field-level request makes a non-float element
    /// type an error: a container-level `unpack` on a struct with a mix of float
    /// and integer fields has to be allowed to pass the integers by.
    pub(crate) explicit_field_optin: bool,
}

impl UnpackFlags {
    /// True if anything at all needs doing: some scaling, or a fill check.
    pub(crate) fn any(&self) -> bool {
        self.scale || self.offset || self.on_fill != OnFill::Ignore
    }
}

/// Combine container and field settings. The field's `unpack(...)` replaces the
/// container's wholesale rather than merging key by key, matching how `rename`
/// relates to `rename_all`. `on_fill` resolves the same way.
pub(crate) fn resolve_unpack(
    container_unpack: Option<&UnpackSpec>,
    field_unpack: Option<&UnpackSpec>,
    container_on_fill: Option<OnFill>,
    field_on_fill: Option<OnFill>,
) -> UnpackFlags {
    let spec = field_unpack.or(container_unpack).copied().unwrap_or(
        // Neither level said anything: unpack by default.
        UnpackSpec {
            scale: true,
            offset: true,
        },
    );
    UnpackFlags {
        scale: spec.scale,
        offset: spec.offset,
        on_fill: field_on_fill.or(container_on_fill).unwrap_or_default(),
        explicit_field_optin: field_unpack.is_some_and(|s| s.scale || s.offset)
            || field_on_fill.is_some_and(|f| f != OnFill::Ignore),
    }
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
