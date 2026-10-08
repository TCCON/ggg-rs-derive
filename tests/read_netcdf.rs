use ggg_rs_derive::{ReadNetcdf, WriteNetcdf};
use ndarray::{Array2, array};

/// Write half of `map_units`: scale to ppm and report the units string.
fn frac_to_ppm<T>(_data: &T, values: &Vec<f64>, _name: &str) -> (Vec<f64>, String) {
    (
        values.iter().map(|v| v * 1.0e6).collect(),
        "ppm".to_string(),
    )
}

/// Read half of `map_units`: the inverse of [`frac_to_ppm`]. Reading is
/// fallible, so unexpected units in the file are reported rather than silently
/// producing wrong numbers.
fn ppm_to_frac(
    values: Vec<f64>,
    name: &str,
    _var: &netcdf::Variable<'_>,
    units: Option<&str>,
) -> netcdf::Result<Vec<f64>> {
    match units {
        Some("ppm") => Ok(values.into_iter().map(|v| v / 1.0e6).collect()),
        other => Err(netcdf::Error::Str(format!(
            "field '{name}' expected units of 'ppm', found {other:?}"
        ))),
    }
}

#[derive(Debug, ReadNetcdf, WriteNetcdf)]
#[netcdf(rename_all = "snake_case")]
struct Data {
    // 1-D variable on the shared "time" dimension, with a constant and a
    // computed attribute.
    #[netcdf(dimensions("time"))]
    xco2: Vec<f64>,

    // 2-D array over (time, level).
    #[netcdf(dimensions("time", "level"))]
    prior_profile: Array2<f64>,

    // Per-observation strings.
    #[netcdf(dimensions("time"))]
    spectrum: Vec<String>,

    // Renamed + mapped value whose units come from the map function. Both
    // directions are named, so this field round-trips through the inverse pair.
    #[netcdf(
        rename = "xluft_ppm",
        dimensions("time"),
        map_units(write = frac_to_ppm, read = ppm_to_frac),
        kind = "values"
    )]
    xluft: Vec<f64>,

    // Written as a global (group) attribute, not a variable.
    #[netcdf(attribute)]
    source: String,

    // Inferred scalar.
    n_obs: i32,

    // Never written.
    #[netcdf(skip)]
    _cache: Vec<u8>,
}

fn sample() -> Data {
    Data {
        xco2: vec![400.0, 401.0, 402.0],
        prior_profile: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        spectrum: vec![
            "oc2004".to_string(),
            "oc2005".to_string(),
            "oc2006".to_string(),
        ],
        // Exactly representable in binary, so the *1e6 / 1e6 round trip is exact.
        xluft: vec![1.0, 0.5, 0.25],
        source: "unit-test".to_string(),
        n_obs: 3,
        _cache: vec![0, 1, 2],
    }
}

/// Write `data` to a new file and read it back.
fn round_trip(data: &Data, path: &std::path::Path) -> netcdf::Result<Data> {
    {
        let mut file = netcdf::create(path)?;
        let mut root = file.root_mut().unwrap();
        data.write_netcdf(&mut root)?;
    }
    let file = netcdf::open(path)?;
    let root = file.root().unwrap();
    Data::read_netcdf(&root)
}

#[test]
fn test_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("root.nc");
    let data = sample();
    let read = round_trip(&data, &path).unwrap();

    assert_eq!(data.xco2, read.xco2);
    assert_eq!(data.prior_profile, read.prior_profile);
    assert_eq!(data.spectrum, read.spectrum);
    assert_eq!(data.source, read.source);
    assert_eq!(data.n_obs, read.n_obs);
    assert_eq!(Vec::<u8>::new(), read._cache);
    // Went out through `frac_to_ppm` and came back through `ppm_to_frac`.
    assert_eq!(data.xluft, read.xluft);
}

/// The stored variable really is in the mapped units, i.e. the write-side
/// mapping happened and the read-side assertion above is not vacuous.
#[test]
fn test_mapped_field_is_stored_in_mapped_units() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("root.nc");
    let data = sample();
    {
        let mut file = netcdf::create(&path).unwrap();
        let mut root = file.root_mut().unwrap();
        data.write_netcdf(&mut root).unwrap();
    }

    let file = netcdf::open(&path).unwrap();
    let var = file.variable("xluft_ppm").expect("xluft_ppm variable");
    let stored = var.get_values::<f64, _>(..).unwrap();
    assert_eq!(vec![1.0e6, 0.5e6, 0.25e6], stored);
}

/// A read map function returns `netcdf::Result`, so a failure inside it
/// propagates out of `read_netcdf` instead of panicking or being ignored.
#[test]
fn test_read_map_error_propagates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_units.nc");
    let data = sample();
    {
        let mut file = netcdf::create(&path).unwrap();
        let mut root = file.root_mut().unwrap();
        data.write_netcdf(&mut root).unwrap();
        // Overwrite the units that `frac_to_ppm` wrote with something
        // `ppm_to_frac` refuses to accept.
        let mut var = root.variable_mut("xluft_ppm").unwrap();
        var.put_attribute("units", "percent").unwrap();
    }

    let file = netcdf::open(&path).unwrap();
    let root = file.root().unwrap();
    let err = Data::read_netcdf(&root).expect_err("bad units should be an error");
    let msg = err.to_string();
    assert!(
        msg.contains("xluft") && msg.contains("percent"),
        "unexpected error message: {msg}"
    );
}

#[derive(Debug, ReadNetcdf, WriteNetcdf)]
struct CustUnitAttr {
    #[netcdf(
        rename = "xluft_ppm",
        dimensions("time"),
        map_units(write = frac_to_ppm, read = ppm_to_frac, units_attr = "unit_abbrev"),
        kind = "values"
    )]
    xluft: Vec<f64>,
}

#[test]
fn test_read_custom_unit_attr() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("root.nc");
    let cust = CustUnitAttr {
        xluft: vec![1.0, 0.5, 0.25],
    };

    {
        let mut file = netcdf::create(&path).unwrap();
        let mut root = file.root_mut().unwrap();
        cust.write_netcdf(&mut root).unwrap();
    }

    // Verify that the xluft values were written as x*1e6 and the unit abbrev
    // attribute exists, but "units" does not - otherwise this test could
    // pass spuriously
    let file = netcdf::open(&path).unwrap();
    let root = file.root().unwrap();
    {
        let var = root.variable("xluft_ppm").unwrap();
        assert!(var.attribute("units").is_none());
        assert!(var.attribute("unit_abbrev").is_some());
        let vals: Vec<f64> = var.get_values::<f64, _>(..).unwrap();
        assert_eq!(vals, vec![1e6, 0.5e6, 0.25e6]);
    }

    let read = CustUnitAttr::read_netcdf(&root).unwrap();
    assert_eq!(read.xluft, cust.xluft);
}

// We'll use this to read back in a CustUnitAttr but expect the wrong dimension name
#[derive(Debug, ReadNetcdf)]
struct DimNameMismatchIn {
    #[allow(unused)]
    #[netcdf(dimensions("spectrum"))]
    xluft_ppm: Vec<f64>,
}

#[test]
fn test_check_dims() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("root.nc");
    let cust = CustUnitAttr {
        xluft: vec![1.0, 0.5, 0.25],
    };

    {
        let mut file = netcdf::create(&path).unwrap();
        let mut root = file.root_mut().unwrap();
        cust.write_netcdf(&mut root).unwrap();
    }

    let file = netcdf::open(&path).unwrap();
    let root = file.root().unwrap();
    let read_res = DimNameMismatchIn::read_netcdf(&root);
    let expected_error =
        "Variable 'xluft_ppm' had wrong dimension number or names: expected [spectrum], got [time]";
    if let Err(netcdf::Error::Str(msg)) = read_res {
        assert_eq!(msg, expected_error)
    } else {
        assert!(false, "Got unexpected error type or no error");
    }
}

// ---------------------------------------------------------------------------
// scale_factor / add_offset packing, and fill-value masking
// ---------------------------------------------------------------------------

/// Packs into an i16 on disk. 0.01 ppm resolution over a 400 ppm baseline, which
/// is roughly how a real CO2 product would be stored.
#[derive(Debug, ReadNetcdf, WriteNetcdf)]
struct Packed {
    #[netcdf(
        dimensions("time"),
        scale_offset(scale = 0.01, offset = 400.0, nc_type = "i16")
    )]
    xco2: Vec<f64>,
}

/// Reads the same file, but takes the stored values verbatim.
#[derive(Debug, ReadNetcdf)]
#[netcdf(unpack = false)]
struct PackedRaw {
    #[netcdf(dimensions("time"))]
    xco2: Vec<f64>,
}

/// Reads the same file applying the scale but not the offset.
#[derive(Debug, ReadNetcdf)]
#[netcdf(unpack(offset = false))]
struct PackedScaleOnly {
    #[netcdf(dimensions("time"))]
    xco2: Vec<f64>,
}

fn write_packed(path: &std::path::Path, data: &Packed) {
    let mut file = netcdf::create(path).unwrap();
    let mut root = file.root_mut().unwrap();
    data.write_netcdf(&mut root).unwrap();
}

#[test]
fn test_packed_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.nc");
    // Chosen so (v - 400) / 0.01 is an exact integer and the round trip is exact.
    let data = Packed {
        xco2: vec![400.0, 412.34, 387.5],
    };
    write_packed(&path, &data);

    let file = netcdf::open(&path).unwrap();
    let var = file.variable("xco2").unwrap();

    // The on-disk type really is i16 — otherwise packing saved nothing and the
    // round-trip assertion below would pass for the wrong reason.
    assert_eq!(
        var.vartype(),
        netcdf::types::NcVariableType::Int(netcdf::types::IntType::I16)
    );
    // And the stored values really are packed.
    assert_eq!(var.get_values::<i16, _>(..).unwrap(), vec![0, 1234, -1250]);
    // The attributes a reader needs to invert it.
    assert_eq!(
        var.attribute("scale_factor").unwrap().value().unwrap(),
        netcdf::AttributeValue::Double(0.01)
    );
    assert_eq!(
        var.attribute("add_offset").unwrap().value().unwrap(),
        netcdf::AttributeValue::Double(400.0)
    );

    let root = file.root().unwrap();
    let read = Packed::read_netcdf(&root).unwrap();
    for (want, got) in data.xco2.iter().zip(&read.xco2) {
        assert!((want - got).abs() < 1e-9, "{want} != {got}");
    }
}

#[test]
fn test_unpack_disabled_reads_stored_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.nc");
    write_packed(
        &path,
        &Packed {
            xco2: vec![400.0, 412.34, 387.5],
        },
    );

    let file = netcdf::open(&path).unwrap();
    let root = file.root().unwrap();
    let raw = PackedRaw::read_netcdf(&root).unwrap();
    assert_eq!(raw.xco2, vec![0.0, 1234.0, -1250.0]);
}

#[test]
fn test_unpack_scale_without_offset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.nc");
    write_packed(
        &path,
        &Packed {
            xco2: vec![400.0, 412.34, 387.5],
        },
    );

    let file = netcdf::open(&path).unwrap();
    let root = file.root().unwrap();
    let scaled = PackedScaleOnly::read_netcdf(&root).unwrap();
    assert_eq!(scaled.xco2, vec![0.0, 12.34, -12.5]);
}

#[test]
fn test_packing_out_of_range_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overflow.nc");
    // (1000 - 400) / 0.01 = 60000, past i16::MAX.
    let data = Packed {
        xco2: vec![400.0, 1000.0],
    };
    let mut file = netcdf::create(&path).unwrap();
    let mut root = file.root_mut().unwrap();
    let err = data
        .write_netcdf(&mut root)
        .expect_err("60000 does not fit in i16");
    let msg = err.to_string();
    assert!(
        msg.contains("xco2") && msg.contains("i16"),
        "unexpected error message: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Fill and missing values
// ---------------------------------------------------------------------------

/// Build a file containing one packed variable of an arbitrary on-disk type,
/// plus whatever attributes the caller wants. Lets the fill-value paths be
/// tested against files this crate did not write.
fn write_raw<T: netcdf::NcTypeDescriptor + Copy>(
    path: &std::path::Path,
    varname: &str,
    values: &[T],
    attrs: &[(&str, netcdf::AttributeValue)],
) {
    let mut file = netcdf::create(path).unwrap();
    file.add_dimension("time", values.len()).unwrap();
    let mut var = file.add_variable::<T>(varname, &["time"]).unwrap();
    for (name, value) in attrs {
        var.put_attribute(name, value.clone()).unwrap();
    }
    var.put_values(values, ..).unwrap();
}

fn scale_and_offset() -> Vec<(&'static str, netcdf::AttributeValue)> {
    vec![
        ("scale_factor", netcdf::AttributeValue::Double(0.01)),
        ("add_offset", netcdf::AttributeValue::Double(400.0)),
    ]
}

#[derive(Debug, ReadNetcdf)]
struct Xco2Default {
    #[netcdf(dimensions("time"))]
    xco2: Vec<f64>,
}

#[derive(Debug, ReadNetcdf)]
#[netcdf(on_fill = "keep")]
struct Xco2Keep {
    #[netcdf(dimensions("time"))]
    xco2: Vec<f64>,
}

#[derive(Debug, ReadNetcdf)]
#[netcdf(on_fill = "error")]
struct Xco2Error {
    // Only ever read back as an error, so the field itself goes unused.
    #[allow(unused)]
    #[netcdf(dimensions("time"))]
    xco2: Vec<f64>,
}

#[derive(Debug, ReadNetcdf)]
#[netcdf(on_fill = "ignore")]
struct Xco2Ignore {
    #[netcdf(dimensions("time"))]
    xco2: Vec<f64>,
}

/// A declared `_FillValue` becomes NaN, and is not scaled on the way through.
#[test]
fn test_declared_fill_value_becomes_nan() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fill.nc");
    let mut attrs = scale_and_offset();
    attrs.push(("_FillValue", netcdf::AttributeValue::Short(-9999)));
    write_raw(&path, "xco2", &[1234i16, -9999, -1250], &attrs);

    let file = netcdf::open(&path).unwrap();
    let root = file.root().unwrap();
    let read = Xco2Default::read_netcdf(&root).unwrap();

    assert!(read.xco2[1].is_nan(), "expected NaN, got {}", read.xco2[1]);
    // Neighbours still scaled normally, so masking didn't disturb them.
    assert_eq!(read.xco2[0], 412.34);
    assert_eq!(read.xco2[2], 387.5);
}

/// netCDF reports a *default* fill value for variables that never declared one,
/// and we honour it — matching netCDF4-python, whose `_toma` falls back to
/// `default_fillvals`. This pins that behaviour, since it is the surprising one:
/// -32767 is `NC_FILL_SHORT` and gets masked even though nothing declared it.
#[test]
fn test_undeclared_default_fill_is_masked() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("default_fill.nc");
    write_raw(
        &path,
        "xco2",
        &[1234i16, -32767, -1250],
        &scale_and_offset(),
    );

    let file = netcdf::open(&path).unwrap();
    let root = file.root().unwrap();
    // Nothing wrote a _FillValue attribute...
    assert!(
        root.variable("xco2")
            .unwrap()
            .attribute("_FillValue")
            .is_none()
    );
    // ...but NC_FILL_SHORT is still treated as fill.
    let read = Xco2Default::read_netcdf(&root).unwrap();
    assert!(read.xco2[1].is_nan(), "expected NaN, got {}", read.xco2[1]);
    assert_eq!(read.xco2[0], 412.34);

    // `on_fill = "ignore"` is the escape hatch: scale it like any other value.
    let ignored = Xco2Ignore::read_netcdf(&root).unwrap();
    assert_eq!(ignored.xco2[1], -32767.0 * 0.01 + 400.0);
}

#[derive(Debug, ReadNetcdf)]
struct Flags {
    #[netcdf(dimensions("time"))]
    flags: Vec<f64>,
}

/// Byte variables are exempt from the default-fill assumption: the range is too
/// narrow to spare a value, which is Unidata's stated advice. An explicit
/// `_FillValue` is still honoured.
#[test]
fn test_byte_default_fill_is_exempt() {
    let dir = tempfile::tempdir().unwrap();

    let undeclared = dir.path().join("byte_undeclared.nc");
    write_raw(&undeclared, "flags", &[1i8, -127, 3], &[]);
    let file = netcdf::open(&undeclared).unwrap();
    let read = Flags::read_netcdf(&file.root().unwrap()).unwrap();
    assert_eq!(
        read.flags,
        vec![1.0, -127.0, 3.0],
        "NC_FILL_BYTE should not be assumed"
    );

    let declared = dir.path().join("byte_declared.nc");
    write_raw(
        &declared,
        "flags",
        &[1i8, -127, 3],
        &[("_FillValue", netcdf::AttributeValue::Schar(-127))],
    );
    let file = netcdf::open(&declared).unwrap();
    let read = Flags::read_netcdf(&file.root().unwrap()).unwrap();
    assert!(read.flags[1].is_nan(), "declared _FillValue should mask");
    assert_eq!(read.flags[0], 1.0);
}

#[test]
fn test_missing_value_scalar_and_vector() {
    let dir = tempfile::tempdir().unwrap();

    let scalar = dir.path().join("missing_scalar.nc");
    let mut attrs = scale_and_offset();
    attrs.push(("missing_value", netcdf::AttributeValue::Short(-1)));
    write_raw(&scalar, "xco2", &[1234i16, -1, -1250], &attrs);
    let file = netcdf::open(&scalar).unwrap();
    let read = Xco2Default::read_netcdf(&file.root().unwrap()).unwrap();
    assert!(read.xco2[1].is_nan());
    assert_eq!(read.xco2[0], 412.34);

    // `missing_value` is allowed to list several sentinels.
    let vector = dir.path().join("missing_vector.nc");
    let mut attrs = scale_and_offset();
    attrs.push((
        "missing_value",
        netcdf::AttributeValue::Shorts(vec![-1, -2]),
    ));
    write_raw(&vector, "xco2", &[1234i16, -1, -2], &attrs);
    let file = netcdf::open(&vector).unwrap();
    let read = Xco2Default::read_netcdf(&file.root().unwrap()).unwrap();
    assert!(read.xco2[1].is_nan() && read.xco2[2].is_nan());
    assert_eq!(read.xco2[0], 412.34);
}

#[test]
fn test_on_fill_keep_and_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fill.nc");
    let mut attrs = scale_and_offset();
    attrs.push(("_FillValue", netcdf::AttributeValue::Short(-9999)));
    write_raw(&path, "xco2", &[1234i16, -9999, -1250], &attrs);

    let file = netcdf::open(&path).unwrap();
    let root = file.root().unwrap();

    // "keep" leaves the raw stored value, deliberately unscaled.
    let kept = Xco2Keep::read_netcdf(&root).unwrap();
    assert_eq!(kept.xco2[1], -9999.0);
    assert_eq!(kept.xco2[0], 412.34);

    // "error" refuses the file, naming the variable and the index.
    let err = Xco2Error::read_netcdf(&root).expect_err("fill value should be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("xco2") && msg.contains("index 1"),
        "unexpected error message: {msg}"
    );
}

/// A container-level `unpack` must not break integer fields: it is a default,
/// and a struct mixing float and integer fields has to stay compilable. This is
/// also the regression test for the old behaviour, where unpacking ran on every
/// field and truncated a fractional scale factor to 0 for integer types.
#[derive(Debug, ReadNetcdf, WriteNetcdf)]
#[netcdf(unpack)]
struct Mixed {
    #[netcdf(dimensions("time"))]
    xco2: Vec<f64>,
    #[netcdf(dimensions("time"))]
    quality: Vec<i32>,
}

#[test]
fn test_container_unpack_leaves_integer_fields_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.nc");
    let data = Mixed {
        xco2: vec![400.0, 401.5],
        quality: vec![-7, 100],
    };
    {
        let mut file = netcdf::create(&path).unwrap();
        let mut root = file.root_mut().unwrap();
        data.write_netcdf(&mut root).unwrap();
    }

    let file = netcdf::open(&path).unwrap();
    let read = Mixed::read_netcdf(&file.root().unwrap()).unwrap();
    assert_eq!(read.xco2, data.xco2);
    assert_eq!(read.quality, data.quality);
}

/// A field's `unpack` replaces the container's wholesale rather than merging key
/// by key — the same rule `rename` follows with respect to `rename_all`. Here the
/// container disables the offset, and the field asks for both halves back.
#[derive(Debug, ReadNetcdf)]
#[netcdf(unpack(offset = false))]
struct FieldOverridesContainer {
    #[netcdf(dimensions("time"), unpack)]
    xco2: Vec<f64>,
}

#[test]
fn test_field_unpack_replaces_container_setting() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.nc");
    write_raw(&path, "xco2", &[1234i16, -1250], &scale_and_offset());

    let file = netcdf::open(&path).unwrap();
    let root = file.root().unwrap();

    // The container alone would have given scale-only: [12.34, -12.5].
    let scaled_only = PackedScaleOnly::read_netcdf(&root).unwrap();
    assert_eq!(scaled_only.xco2, vec![12.34, -12.5]);

    // The field override restores the offset too.
    let both = FieldOverridesContainer::read_netcdf(&root).unwrap();
    assert_eq!(both.xco2, vec![412.34, 387.5]);
}
