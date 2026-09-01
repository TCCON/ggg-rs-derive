use proc_macro::TokenStream;

mod subset;
mod write_netcdf;

/// Adds a `.subset(&[usize])` method to a struct to subset [`Vec`] or ndarray Array fields by indices.
///
/// In basic usage, annotating a structure with `#[derive(Subset)]` will implement the `.subset`
/// method. This requires that all fields be `Vec` or `ndarray` Array types. Specifically, the
/// types must be `Vec`, start with `Array`, or start with `ndarray::Array`. The default `.subset`
/// method will then subset multi-dimensional arrays along their first dimension:
///
/// ```rust
/// use ggg_rs_derive::Subset;
/// use ndarray::{Array1, array};
///
/// #[derive(Subset)]
/// struct Data {
///     v: Vec<i32>,
///     a1: Array1<f32>,
///     a2: ndarray::Array2<i64>
/// }
///
/// let orig = Data {
///     v: vec![100, 200, 300],
///     a1: array![10.0, 20.0, 30.0],
///     a2: array![[1, -1], [2, -2], [3, -3]]
/// };
///
/// let sub = orig.subset(&[0, 2]);
/// assert_eq!(sub.v, vec![100, 300]);
/// assert_eq!(sub.a1, array![10., 30.]);
/// assert_eq!(sub.a2, array![[1, -1], [3, -3]]);
/// ```
///
/// Note that `.subset()` will panic if any of the indices are out of bounds
/// for any of the fields.
///
/// # Field attributes
///
/// `subset(skip)` allows you to skip subsetting fields, either because they cannot
/// or should not be subset:
///
/// ```rust
/// use ggg_rs_derive::Subset;
///
/// #[derive(Subset)]
/// struct Data {
///     v: Vec<i32>,
///     #[subset(skip)]
///     s: &'static str
/// }
///
/// let orig = Data { v: vec![1, 2], s: "Hello, world!" };
/// let sub = orig.subset(&[1]);
/// assert_eq!(sub.v, vec![2]);
/// assert_eq!(sub.s, "Hello, world!");
/// ```
///
/// `subset(kind = "vec|array")` allows you to indicate whether the field should be subset
/// as a `Vec` or `ndarray::Array*`. This is necessary if, for example, you have a type
/// alias:
///
/// ```rust
/// use ggg_rs_derive::Subset;
/// use ndarray::array;
///
/// type NVector = Vec<f64>;
/// type Matrix = ndarray::Array2<f64>;
///
/// #[derive(Subset)]
/// struct Data {
///     #[subset(kind = "vec")]
///     v: NVector,
///
///     #[subset(kind = "array")]
///     a: Matrix
/// }
///
/// let orig = Data {
///     v: vec![100.0, 200.0],
///     a: array![[1., 10.], [2., 20.]]
/// };
/// let sub = orig.subset(&[0]);
///
/// assert_eq!(sub.v, vec![100.0]);
/// assert_eq!(sub.a, array![[1.0, 10.0]]);
/// ```
///
/// `subset(axis = i)` allows you to subset along a different axis of an
/// `ndarray` than the first:
///
/// ```rust
/// use ggg_rs_derive::Subset;
/// use ndarray::{Array2, array};
///
/// #[derive(Subset)]
/// struct Data {
///     #[subset(axis = 1)]
///     a: Array2<i32>
/// }
///
/// let orig = Data { a: array![[1, 2, 3, 4], [10, 20, 30, 40]] };
/// let sub = orig.subset(&[1, 3]);
/// assert_eq!(sub.a, array![[2, 4], [20, 40]]);
/// ```
#[proc_macro_derive(Subset, attributes(subset))]
pub fn derive_subset(input: TokenStream) -> TokenStream {
    subset::expand(input)
}

/// Adds a `write_netcdf` method that writes a struct's fields to a netCDF file.
///
/// The generated method has the signature
///
/// ```ignore
/// pub fn write_netcdf(&self, grp: &mut netcdf::GroupMut<'_>) -> netcdf::Result<()>
/// ```
///
/// It writes each field as a netCDF *variable* (or a *group attribute*) into the
/// given group. Pass the file root — `file.root_mut().unwrap()` — to write at
/// the top level, or a subgroup from `file.add_group(...)`. Any dimension named
/// by a field that does not yet exist in the file is created automatically,
/// sized from that field's length/shape.
///
/// By default the write strategy is inferred from each field's type:
/// `ndarray` `Array*` fields use [`netcdf::VariableMut::put`], `Vec`/slice fields
/// use [`netcdf::VariableMut::put_values`], `String`/`Vec<String>` fields use
/// [`netcdf::VariableMut::put_string`], and remaining scalars use
/// `put_value`. Variables are named after their fields unless renamed.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
/// use ndarray::Array2;
///
/// #[derive(WriteNetcdf)]
/// #[netcdf(rename_all = "snake_case")]
/// struct Data {
///     // 1-D data sharing the "time" dimension (created on first use).
///     #[netcdf(dimensions("time"))]
///     xco2: Vec<f64>,
///
///     // 2-D array; dimensions map to axes in order.
///     #[netcdf(dimensions("time", "level"))]
///     prior_profile: Array2<f64>,
///
///     // A per-observation string is written with put_string.
///     #[netcdf(dimensions("time"))]
///     spectrum_name: Vec<String>,
/// }
///
/// # fn main() -> netcdf::Result<()> {
/// let data = Data {
///     xco2: vec![400.0, 401.0],
///     prior_profile: Array2::zeros((2, 3)),
///     spectrum_name: vec!["pa2004".into(), "pa2005".into()],
/// };
/// let mut file = netcdf::create("out.nc")?;
/// let mut root = file.root_mut().unwrap();
/// data.write_netcdf(&mut root)?;
/// # Ok(()) }
/// ```
///
/// # Container attributes
///
/// - `#[netcdf(rename_all = "...")]` — apply a serde-style case convention
///   (`snake_case`, `SCREAMING_SNAKE_CASE`, `kebab-case`, `camelCase`,
///   `PascalCase`, `lowercase`, `UPPERCASE`, `SCREAMING-KEBAB-CASE`) to every
///   field's netCDF name.
///
/// # Field attributes
///
/// - `#[netcdf(rename = "name")]` — set an explicit variable/attribute name
///   (overrides `rename_all`).
/// - `#[netcdf(dimensions("d0", "d1", ...))]` — dimension names in axis order,
///   auto-created if absent.
/// - `#[netcdf(kind = "values" | "array" | "string")]` — force the write
///   strategy (needed for type aliases, or whenever `map`/`map_units` is used).
/// - `#[netcdf(skip)]` — do not write this field.
/// - `#[netcdf(attribute)]` — write the field as a *group attribute* rather than
///   a variable.
/// - `#[netcdf(attr(name = "units", value = "K"))]` — attach a constant attribute
///   to the variable (repeatable). Use `attr(name = "...", func = path::to::fn)`
///   to compute it via `fn(&Self, &FieldTy, &str) -> V` (`V: Into<AttributeValue>`).
/// - `#[netcdf(map = path::to::fn)]` — transform the value before writing with
///   `fn(&Self, &FieldTy, &str) -> Out`.
/// - `#[netcdf(map_units = path::to::fn)]` — like `map`, but the function returns
///   `(Out, String)` and the returned string is written as the `units` attribute.
///
/// The three-argument hooks receive the whole struct, a reference to the field
/// value, and the field's netCDF name, so a mapping can depend on sibling fields.
#[proc_macro_derive(WriteNetcdf, attributes(netcdf))]
pub fn derive_write_netcdf(input: TokenStream) -> TokenStream {
    write_netcdf::expand(input)
}
