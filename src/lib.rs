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
/// Dimensions must be specified with the `#[netcdf(dimensions(...))]` field attribute
/// as shown in the following example.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
/// use ndarray::Array2;
///
/// #[derive(WriteNetcdf)]
/// #[netcdf(rename_all = "camelCase")]
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
/// `#[netcdf(rename_all = "...")]` — apply a serde-style case convention
/// (`snake_case`, `SCREAMING_SNAKE_CASE`, `kebab-case`, `camelCase`,
/// `PascalCase`, `lowercase`, `UPPERCASE`, `SCREAMING-KEBAB-CASE`) to every
/// field's netCDF name.
///
/// # Field attributes
/// `#[netcdf(dimensions("d0", "d1", ...))]` is required. It lists the dimension
/// names that map to each dimension of the field. `Vec` types only need a single
/// dimension, while `ndarray` multidimensional arrays need one per dimension.
/// The dimensions will be created automatically if needed, but note that dimensions
/// from parent groups will be inherited. That is, if a dimension "time" in the root
/// group has length 5, and you declare that a field in the "support" subgroup needs
/// a time dimension with length 10, you will get an error.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
/// use ndarray::Array2;
///
/// #[derive(WriteNetcdf)]
/// struct Data {
///     // Vec types need 1 dimension
///     #[netcdf(dimensions("time"))]
///     v: Vec<f32>,
///
///     // So do 1D arrays, note that the type can be prefixed
///     // with `ndarray`
///     #[netcdf(dimensions("time"))]
///     a1: ndarray::Array1<f64>,
///
///     // 2D arrays need two dimensions, "frame" will be the `Axis(0)`
///     // length, "footprint" the `Axis(1)` length.
///     #[netcdf(dimensions("frame", "footprint"))]
///     a2: Array2<i64>
/// }
/// ```
///
/// `#[netcdf(rename = "name")]` allows you to set an explicit variable/attribute name.
///  It overrides `rename_all`, if that is present on the struct.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
///
/// #[derive(WriteNetcdf)]
/// #[netcdf(rename_all = "camelCase")]
/// struct Data {
///     // This data will be written to the "utc_time" variable
///     // instead of a "timestamps" variables.
///     #[netcdf(dimensions("time"), rename = "utc_time")]
///     timestamps: Vec<i64>,
///
///     // This would be written to "wmoScale" because of the
///     // struct-level `rename_all`
///     #[netcdf(dimensions("time"))]
///     wmo_scale: Vec<String>
/// }
/// ```
///  
/// `#[netcdf(kind = "values" | "array" | "string")]` sets how the field is written;
///   "values" uses the [`netcdf::VariableMut::put_values`] method, "array" the
///   [`netcdf::VariableMut::put`] method, and "string" will iterate over the elements,
///   calling [`netcdf::VariableMut::put_string`] for each. This is needed when a type
///   alias prevents inference of the type from its name, or when you use the `map` or
///   `map_units` attribute to change the type.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
///
/// type Matrix = ndarray::Array2<f32>;
///
/// #[derive(WriteNetcdf)]
/// struct Data {
///     // The macro doesn't know that `Matrix` is actually
///     // an array, so it tries to write is as a scalar by default,
///     // which fails. The `kind` attribute tells it to treat this
///     // as an array.
///     #[netcdf(dimensions("obs", "obs"), kind = "array")]
///     covariance: Matrix
/// }
/// ```
///
/// `#[netcdf(skip)]` will skip writing this field.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
///
/// #[derive(WriteNetcdf)]
/// struct Data {
///     // Only xco2 gets written, spectrum does not.
///     #[netcdf(skip)]
///     spectrum: Vec<String>,
///     #[netcdf(dimensions("time"))]
///     xco2: Vec<f64>
/// }
/// ```
///
/// `#[netcdf(attribute)]` indicates that the value of this field must be written
/// as a group attribute, not a variable. This allows you to save scalar metadata
/// as attributes, rather than scalar variables.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
///
/// #[derive(WriteNetcdf)]
/// struct Data {
///     #[netcdf(dimensions("time"))]
///     xco2: Vec<f64>,
///     // Creates a group attribute called `doi` containing this string.
///     #[netcdf(attribute)]
///     doi: String,
/// }
/// ```
///
/// `#[netcdf(attr(name = "units", value = "K"))]` will attach a constant attribute
///   to the variable. Use `attr(name = "...", func = path::to::fn)` if the attribute
///   needs to be computed. The function must take 3 arguments, a reference to the struct,
///   a reference to the field values, and the field name: `fn(&Self, &FieldTy, &str) -> V`
///  `V` must implement `Into<AttributeValue>`. You can specify multiple attributes.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
///
/// // This function only uses the field name, but it could use the values
/// // of the field or the entire structure - the latter can be useful if
/// // you need to distill an attribute from a separate field. Here we use
/// // generics without trait bounds so that this function could apply to
/// // any field.
/// fn infer_long_name<'s, D, V>(_data: &D, _values: &V, name: &'s str) -> &'s str {
///     match name {
///         "xco2" => "carbon_dioxide_column_average",
///         _ => name,
///     }
/// }
///
/// #[derive(WriteNetcdf)]
/// struct Data {
///     // spectrum will have two attributes: `description` (string) and `public` (integer)
///     #[netcdf(
///         dimensions("time"),
///         attr( name = "description", value = "name of the near-IR spectrum" ),
///         attr( name = "public", value = 0 )
///     )]
///     spectrum: Vec<String>,
///
///     // xco2 will have three attributes: `description` (string), `public` (integer), and
///     // long_name (string).
///     #[netcdf(
///         dimensions("time"),
///         attr( name = "description", value = "column average mole ratio of CO2 to dry air" ),
///         attr( name = "public", value = 1 ),
///         attr( name = "long_name", func = infer_long_name ),
///     )]
///     xco2: Vec<f64>
/// }
/// ```
///
/// `#[netcdf(map = path::to::fn)]` allows you to define a function to transform the value
///  before writing. The function must take 3 arguments, a reference to the struct,
///  a reference to the field values, and the field name: `fn(&Self, &FieldTy, &str) -> Out.
///  This allows you to work around types that cannot be directly written to netCDF files,
///  such as [`chrono::DateTime`] instances. Note that this must be combined with the "kind"
///  attribute, as the macro will not be able to infer the output type.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
/// use chrono::{DateTime, Utc};
///
/// fn datetime_to_timestamp(_data: &Data, times: &[DateTime<Utc>], _name: &str) -> Vec<i64> {
///     times.iter().map(|t| t.timestamp()).collect()
/// }
///
/// #[derive(WriteNetcdf)]
/// struct Data {
///     // `DateTime<_>` doesn't implement `NcTypeDescriptor`, so we have to change it
///     // to a type that does. Note that we also provide the "kind" attribute.
///     #[netcdf(dimensions("time"), map = datetime_to_timestamp, kind = "values")]
///     times: Vec<DateTime<Utc>>
/// }
/// ```
///
/// `#[netcdf(map_units = path::to::fn)]` is like `map`, but the function returns
/// `(Out, String)` and the returned string is written as the `units` attribute.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
/// use chrono::{DateTime, Utc};
///
/// fn datetime_to_timestamp(_data: &Data, times: &[DateTime<Utc>], _name: &str) -> (Vec<i64>, &'static str) {
///     let timestamps = times.iter().map(|t| t.timestamp()).collect();
///     (timestamps, "seconds since 1970-01-01 00:00:00")
/// }
///
/// #[derive(WriteNetcdf)]
/// struct Data {
///     // This is a better version of the above example, because the times variable
///     // now gets the right units.
///     #[netcdf(dimensions("time"), map_units = datetime_to_timestamp, kind = "values")]
///     times: Vec<DateTime<Utc>>
/// }
/// ```
#[proc_macro_derive(WriteNetcdf, attributes(netcdf))]
pub fn derive_write_netcdf(input: TokenStream) -> TokenStream {
    write_netcdf::expand(input)
}
