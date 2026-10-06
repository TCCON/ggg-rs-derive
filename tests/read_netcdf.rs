use std::assert_eq;

use ggg_rs_derive::{ReadNetcdf, WriteNetcdf};
use ndarray::{Array2, array};

#[derive(ReadNetcdf, WriteNetcdf)]
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

    // Renamed + mapped value whose units come from the map function.
    // #[netcdf(rename = "xluft_ppm", dimensions("time"), map_units = frac_to_ppm, kind = "values")]
    // xluft: Vec<f64>,

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
        source: "unit-test".to_string(),
        n_obs: 3,
        _cache: vec![0, 1, 2],
    }
}

#[test]
fn test_round_trip_no_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("root.nc");
    let data = sample();
    {
        let mut file = netcdf::create(&path).unwrap();
        let mut root = file.root_mut().unwrap();
        data.write_netcdf(&mut root).unwrap();
    }

    let file = netcdf::open(&path).unwrap();
    let root = file.root().unwrap();
    let read = Data::read_netcdf(&root).unwrap();

    assert_eq!(data.xco2, read.xco2);
    assert_eq!(data.prior_profile, read.prior_profile);
    assert_eq!(data.spectrum, read.spectrum);
    assert_eq!(data.source, read.source);
    assert_eq!(data.n_obs, read.n_obs);
    assert_eq!(Vec::<u8>::new(), read._cache);
}
