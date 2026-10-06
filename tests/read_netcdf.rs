use std::assert_eq;

use ggg_rs_derive::{ReadNetcdf, WriteNetcdf};
use ndarray::{Array2, array};

/// Write half of `map_units`: scale to ppm and report the units string.
fn frac_to_ppm(_data: &Data, values: &Vec<f64>, _name: &str) -> (Vec<f64>, String) {
    (
        values.iter().map(|v| v * 1.0e6).collect(),
        "ppm".to_string(),
    )
}

/// Read half of `map_units`: the inverse of [`frac_to_ppm`]. Reading is
/// fallible, so unexpected units in the file are reported rather than silently
/// producing wrong numbers.
fn ppm_to_frac(values: Vec<f64>, name: &str, units: Option<&str>) -> netcdf::Result<Vec<f64>> {
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
