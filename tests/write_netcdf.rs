//! Integration tests for `#[derive(WriteNetcdf)]`: write a struct to a real
//! netCDF file, reopen it, and assert on variables, attributes, and dimensions.

use ggg_rs_derive::WriteNetcdf;
use ndarray::{Array2, array};

/// `map_units` hook: convert a fraction to ppm and report the units string.
fn frac_to_ppm(_data: &Data, values: &Vec<f64>, _name: &str) -> (Vec<f64>, String) {
    (values.iter().map(|v| v * 1.0e6).collect(), "ppm".to_string())
}

/// `attr(func = ...)` hook: build a long_name from the field's netCDF name.
fn make_long_name(_data: &Data, _values: &Vec<f64>, name: &str) -> String {
    format!("data for {name}")
}

#[derive(WriteNetcdf)]
#[netcdf(rename_all = "snake_case")]
struct Data {
    // 1-D variable on the shared "time" dimension, with a constant and a
    // computed attribute.
    #[netcdf(
        dimensions("time"),
        attr(name = "units", value = "ppm"),
        attr(name = "long_name", func = make_long_name)
    )]
    xco2: Vec<f64>,

    // 2-D array over (time, level).
    #[netcdf(dimensions("time", "level"))]
    prior_profile: Array2<f64>,

    // Per-observation strings.
    #[netcdf(dimensions("time"))]
    spectrum: Vec<String>,

    // Renamed + mapped value whose units come from the map function.
    #[netcdf(rename = "xluft_ppm", dimensions("time"), map_units = frac_to_ppm, kind = "values")]
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
        xco2: vec![400.0, 401.0],
        prior_profile: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        spectrum: vec!["pa2004".to_string(), "pa2005".to_string()],
        xluft: vec![1.0e-6, 2.0e-6],
        source: "unit-test".to_string(),
        n_obs: 2,
        _cache: vec![0, 1, 2],
    }
}

#[test]
fn writes_variables_attributes_and_dimensions_to_root() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("root.nc");

    {
        let mut file = netcdf::create(&path).unwrap();
        let mut root = file.root_mut().unwrap();
        sample().write_netcdf(&mut root).unwrap();
    }

    let file = netcdf::open(&path).unwrap();

    // Dimensions were auto-created with the right sizes.
    assert_eq!(file.dimension("time").unwrap().len(), 2);
    assert_eq!(file.dimension("level").unwrap().len(), 3);

    // Skipped field is absent.
    assert!(file.variable("cache").is_none());
    assert!(file.variable("_cache").is_none());

    // Plain 1-D variable + its attributes.
    let xco2 = file.variable("xco2").unwrap();
    assert_eq!(xco2.get_values::<f64, _>(..).unwrap(), vec![400.0, 401.0]);
    assert_eq!(
        xco2.attribute("units").unwrap().value().unwrap(),
        netcdf::AttributeValue::Str("ppm".to_string())
    );
    assert_eq!(
        xco2.attribute("long_name").unwrap().value().unwrap(),
        netcdf::AttributeValue::Str("data for xco2".to_string())
    );

    // 2-D array preserved shape and values.
    let prior = file.variable("prior_profile").unwrap();
    assert_eq!(prior.dimensions().len(), 2);
    assert_eq!(
        prior.get::<f64, _>(..).unwrap(),
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn()
    );

    // String variable: check length and one element.
    let spectrum = file.variable("spectrum").unwrap();
    assert_eq!(spectrum.dimensions()[0].len(), 2);
    assert_eq!(spectrum.get_string([0]).unwrap(), "pa2004");
    assert_eq!(spectrum.get_string([1]).unwrap(), "pa2005");

    // Renamed + mapped variable, with units from the map function.
    let xluft = file.variable("xluft_ppm").unwrap();
    assert!(file.variable("xluft").is_none());
    assert_eq!(xluft.get_values::<f64, _>(..).unwrap(), vec![1.0, 2.0]);
    assert_eq!(
        xluft.attribute("units").unwrap().value().unwrap(),
        netcdf::AttributeValue::Str("ppm".to_string())
    );

    // Scalar variable.
    let n_obs = file.variable("n_obs").unwrap();
    assert_eq!(n_obs.get_value::<i32, _>(..).unwrap(), 2);

    // Field written as a global attribute rather than a variable.
    assert!(file.variable("source").is_none());
    assert_eq!(
        file.attribute("source").unwrap().value().unwrap(),
        netcdf::AttributeValue::Str("unit-test".to_string())
    );
}

#[test]
fn writes_into_a_subgroup() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group.nc");

    {
        let mut file = netcdf::create(&path).unwrap();
        let mut grp = file.add_group("site_pa").unwrap();
        sample().write_netcdf(&mut grp).unwrap();
    }

    let file = netcdf::open(&path).unwrap();
    let grp = file.group("site_pa").unwrap().unwrap();
    assert_eq!(grp.dimension("time").unwrap().len(), 2);
    assert_eq!(
        grp.variable("xco2").unwrap().get_values::<f64, _>(..).unwrap(),
        vec![400.0, 401.0]
    );
    assert_eq!(
        grp.attribute("source").unwrap().value().unwrap(),
        netcdf::AttributeValue::Str("unit-test".to_string())
    );
    // The root has no variables — everything went into the group.
    assert!(file.variable("xco2").is_none());
}
