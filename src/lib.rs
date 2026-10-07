use proc_macro::TokenStream;

mod common_netcdf;
mod read_netcdf;
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
/// `#[netcdf(describe)]` — write each field's `///` doc comment as a variable
/// attribute. The bare word uses the attribute name `"description"`; the
/// `#[netcdf(describe = "long_name")]` form chooses a different attribute name.
/// Fields with no doc comment contribute nothing, and individual fields can
/// override the attribute name or opt out (see the field attributes below).
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
///
/// #[derive(WriteNetcdf)]
/// #[netcdf(describe = "long_name")]
/// struct Data {
///     /// Column-averaged dry-air mole fraction of CO2
///     // Writes a `long_name` attribute with the text above.
///     #[netcdf(dimensions("time"))]
///     xco2: Vec<f64>,
/// }
/// ```
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
/// `#[netcdf(describe)]` / `#[netcdf(describe = "name")]` / `#[netcdf(no_describe)]`
/// overrides the container-level `describe` for a single field, in the same way
/// `rename` overrides `rename_all`. The bare word turns descriptions on for this
/// field (using the inherited attribute name), the `= "name"` form also changes
/// the attribute name, and `no_describe` opts the field out even when the
/// container enables descriptions for everything.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
///
/// #[derive(WriteNetcdf)]
/// #[netcdf(describe = "long_name")]
/// struct Data {
///     /// Column-averaged dry-air mole fraction of CO2
///     // Written to the "long_name" attribute (inherited from the container).
///     #[netcdf(dimensions("time"))]
///     xco2: Vec<f64>,
///
///     /// Surface pressure
///     // This field overrides the attribute name to "comment".
///     #[netcdf(dimensions("time"), describe = "comment")]
///     psurf: Vec<f64>,
///
///     /// Internal bookkeeping counter
///     // No description attribute is written for this field.
///     #[netcdf(dimensions("time"), no_describe)]
///     counter: Vec<i32>,
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
/// `#[netcdf(map(write = path::to::fn))]` allows you to define a function to transform
///  the value before writing. The function must take 3 arguments, a reference to the struct,
///  a reference to the field values, and the field name: `fn(&Self, &FieldTy, &str) -> Out`.
///  This allows you to work around types that cannot be directly written to netCDF files,
///  such as [`chrono::DateTime`] instances. Note that this must be combined with the "kind"
///  attribute, as the macro will not be able to infer the output type.
///
/// The direction is part of the syntax and cannot be omitted. Reading needs the
/// *inverse* function, so `ReadNetcdf` looks for `map(read = ...)` instead — see
/// the [`ReadNetcdf`] docs. A struct that derives both puts both halves in the
/// same `map(...)`: `map(write = to_nc, read = from_nc)`.
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
///     #[netcdf(dimensions("time"), map(write = datetime_to_timestamp), kind = "values")]
///     times: Vec<DateTime<Utc>>
/// }
/// ```
///
/// `#[netcdf(map_units(write = path::to::fn))]` is like `map`, but the function returns
/// `(Out, String)` and the returned string is written as the `units` attribute.
/// This version can also accept a `units_attr` option to change which attribute the units
/// are written to; by default, they are written to "units".
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
///     #[netcdf(dimensions("time"), map_units(write = datetime_to_timestamp), kind = "values")]
///     times: Vec<DateTime<Utc>>
/// }
/// ```
///
/// `#[netcdf(scale_offset(scale = .., offset = .., nc_type = ".."))]` packs the
/// field into a smaller on-disk type, following the netCDF attribute conventions.
/// Values are stored as `(value - offset) / scale`, and the `scale_factor` and
/// `add_offset` attributes are written so that a reader — including
/// [`ReadNetcdf`], which inverts this automatically — can recover the original
/// numbers.
///
/// `scale` and `offset` are individually optional, but at least one is required.
/// `nc_type` is mandatory and names the packed type (`"i8"`, `"u8"`, `"i16"`,
/// `"u16"`, `"i32"`, `"u32"`, `"i64"`, `"u64"`, `"f32"`, `"f64"`): without it the
/// variable would keep the field's own type and packing would save nothing.
///
/// ```no_run
/// use ggg_rs_derive::WriteNetcdf;
///
/// #[derive(WriteNetcdf)]
/// struct Data {
///     // Stored as `short xco2(time)` with 0.01 ppm resolution around a 400 ppm
///     // baseline — a quarter the size of the f64 values in memory.
///     #[netcdf(
///         dimensions("time"),
///         scale_offset(scale = 0.01, offset = 400.0, nc_type = "i16")
///     )]
///     xco2: Vec<f64>,
/// }
/// ```
///
/// Three things worth knowing:
///
/// * The field's element type must be `f32` or `f64`. Packing an integer field
///   has nothing to divide, so it is a compile error, as is a type alias the
///   macro cannot see inside (use a concrete `Vec<f64>` or `Array<f64, _>`).
/// * Values are rounded with `round_ties_even`, matching `numpy.around` as used
///   by netCDF4-python's packing, and a value that does not fit the packed type
///   is a runtime error rather than a silently clamped one.
/// * `scale_offset` cannot be combined with `map`/`map_units`: a mapped value's
///   type is opaque to the macro, so it cannot check that what gets written is
///   floating-point. Pack inside the map function instead.
#[proc_macro_derive(WriteNetcdf, attributes(netcdf))]
pub fn derive_write_netcdf(input: TokenStream) -> TokenStream {
    write_netcdf::expand(input)
}

/// Adds a `read_netcdf` associated function that builds a struct from a netCDF file.
///
/// The generated function has the signature
///
/// ```ignore
/// pub fn read_netcdf(grp: &netcdf::Group<'_>) -> netcdf::Result<Self>
/// ```
///
/// It is the counterpart to [`WriteNetcdf`] and shares its field attributes:
/// `rename`, `rename_all`, `kind`, `dimensions`, `attribute`, and `skip` all
/// mean the same thing in both directions, so one set of annotations serves
/// both derives. See [`WriteNetcdf`] for definitions of those attributes.
///
/// Note that this is an *associated function*, not a method —
/// there is no `self` yet — so you call it as `Data::read_netcdf(&root)`.
///
/// Fields marked with `#[netcdf(skip)]` are filled in with [`Default::default()`],
/// thus a skipped field's type must implement [`Default`].
///
/// # Remapping values
///
/// `#[netcdf(map(read = path::to::fn))]`: transforms the value after it is read.
/// The function takes the value from the file and the struct field
/// name, and returns a [`netcdf::Result`] containing the new type (which must
/// match the structure field type): `fn(NcTy, &str) -> netcdf::Result<Out>`.
/// Note that, unlike the corresponding functions for [`WriteNetcdf`], there is no
/// `&Self` argument, because `read_netcdf` is still constructing `Self` when the
/// function runs. Further, function is fallible: the data comes from a file this
/// program did not necessarily write, so conversion could fail. The error type
/// is that from the netcdf crate.
///
/// `#[netcdf(map_units(read = path::to::fn))]`: the given function will also
/// variable's units. By default, the units are read from the "units" attribute,
/// but this can be overridden with the `units_attr` option to `map_units`, e.g.,
/// `#[netcdf(map_units(read = path::to::fn, units_attr = "long_units"))]`.
/// The signature of the function given to `read` is:
///
/// `fn(NcTy, &str, Option<&str>) -> netcdf::Result<Out>`
///
/// where the `Option<&str>` will contain the units string if the attribute
/// is present. If that attribute is not present, this will be `None`.
///
/// As with `WriteNetcdf`, `kind` is required whenever a mapping applies,
/// since the type in the file no longer matches the field's declared type.
///
/// ```no_run
/// use ggg_rs_derive::ReadNetcdf;
///
///
/// fn ppm_to_frac(values: Vec<f64>, name: &str, units: Option<&str>) -> netcdf::Result<Vec<f64>> {
///     match units {
///         Some("ppm") => Ok(values.into_iter().map(|v| v / 1.0e6).collect()),
///         other => Err(netcdf::Error::Str(
///             format!("field '{name}' expected units of 'ppm', found {other:?}")
///         )),
///     }
/// }
///
/// #[derive(ReadNetcdf)]
/// struct Data {
///     #[netcdf(
///         dimensions("time"),
///         map_units(read = ppm_to_frac),
///         kind = "values"
///     )]
///     xluft: Vec<f64>,
/// }
/// ```
///
/// A `map(...)` that names only one direction is an error in the derive that
/// needs the other half, so a half-finished mapping is caught at compile time
/// rather than silently reading or writing the field unmapped. Supplying only
/// `write = ...` is therefore fine for a struct that derives `WriteNetcdf`
/// alone, and becomes an error as soon as `ReadNetcdf` is added.
///
/// # Unpacking: `scale_factor` and `add_offset`
///
/// Float fields are unpacked automatically. If the variable carries a
/// `scale_factor` or `add_offset` attribute, the values read from the file are
/// transformed by `value * scale_factor + add_offset` — the inverse of what
/// [`WriteNetcdf`]'s `scale_offset` does — so a packed `short` variable arrives
/// as the physical `f64` quantity it represents.
///
/// `#[netcdf(unpack(...))]` turns that off, on the container or on a single
/// field. A bare `unpack` means both halves on, `unpack = false` means both off,
/// and the nested form controls each independently. A field's setting replaces
/// the container's wholesale, the same way `rename` replaces `rename_all`.
///
/// ```no_run
/// use ggg_rs_derive::ReadNetcdf;
///
/// #[derive(ReadNetcdf)]
/// #[netcdf(unpack(offset = false))]
/// struct Data {
///     // Scaled but not offset, inherited from the container.
///     #[netcdf(dimensions("time"))]
///     xco2: Vec<f64>,
///
///     // Exactly as stored in the file.
///     #[netcdf(dimensions("time"), unpack = false)]
///     raw_counts: Vec<f64>,
///
///     // Both halves, overriding the container.
///     #[netcdf(dimensions("time"), unpack)]
///     psurf: Vec<f64>,
/// }
/// ```
///
/// Unpacking only applies to `f32` and `f64` fields, because a fractional scale
/// factor cannot be applied to an integer without truncating it. Integer fields
/// are therefore passed through untouched, including under a container-level
/// `unpack` — a struct is free to mix the two. Asking for it on a specific
/// non-float field is a compile error, as is asking for it on a type alias the
/// macro cannot see inside.
///
/// Unpacking happens *before* any `map(read = ...)`, so a map function receives
/// physical values.
///
/// # Fill and missing values
///
/// A value matching the variable's fill value or its `missing_value` attribute
/// is detected *before* scaling — `_FillValue` is stored in the packed
/// representation, so once it has been multiplied by the scale factor it is
/// indistinguishable from real data. `#[netcdf(on_fill = "...")]` chooses what
/// happens to it, on the container or a single field:
///
/// | Value | Behaviour |
/// |---|---|
/// | `"nan"` (default) | replace with `NaN`, leaving it unscaled |
/// | `"keep"` | leave the raw stored value in place, unscaled |
/// | `"ignore"` | do not look for fill values; scale them like any other value |
/// | `"error"` | fail, naming the variable and the index |
///
/// ```no_run
/// use ggg_rs_derive::ReadNetcdf;
///
/// #[derive(ReadNetcdf)]
/// #[netcdf(on_fill = "error")]
/// struct Data {
///     // A gap in this variable means the file is not usable.
///     #[netcdf(dimensions("time"))]
///     xco2: Vec<f64>,
///
///     // Here gaps are expected, so they become NaN.
///     #[netcdf(dimensions("time"), on_fill = "nan")]
///     psurf: Vec<f64>,
/// }
/// ```
///
/// Two details that will eventually surprise someone:
///
/// * **An undeclared fill value still counts.** The fill value comes from
///   netCDF itself (`nc_inq_var_fill`), which reports the *type's default* —
///   `-32767` for `short`, and so on — for any variable that never wrote a
///   `_FillValue` attribute. Such values are masked. This matches
///   netCDF4-python, whose auto-masking falls back to `default_fillvals` the
///   same way. If a variable legitimately uses that value as data, set
///   `on_fill = "ignore"` for the field.
/// * **Byte variables are exempt from that fallback.** For an `i8`/`u8`
///   variable, only an explicitly written `_FillValue` is honoured, following
///   Unidata's advice that a byte range is too narrow to assume one of its
///   values means "missing".
///
/// `missing_value` is read as an attribute and may list several sentinels.
/// `valid_range`, `valid_min`, and `valid_max` are not yet honoured.
#[proc_macro_derive(ReadNetcdf, attributes(netcdf))]
pub fn derive_read_netcdf(input: TokenStream) -> TokenStream {
    read_netcdf::expand(input)
}
