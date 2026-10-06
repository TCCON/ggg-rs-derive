// map() must not contain "units_attr =" because it does not use units, so
// including that is misleading. Make sure that the map_units() version does
// allow it - that should not produce an error.
use ggg_rs_derive::WriteNetcdf;

fn ident_no_units(_s: &Bad, v: &Vec<f64>, _n: &str) -> Vec<f64> {
    v.clone()
}

fn ident_with_units(_s: &Good, v: &Vec<f64>, _n: &str) -> (Vec<f64>, String) {
    (v.clone(), "".to_string())
}

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(dimensions("time"), map(write = ident_no_units, units_attr = "long_units"), kind = "values")]
    values: Vec<f64>,
}

#[derive(WriteNetcdf)]
struct Good {
    #[netcdf(dimensions("time"), map_units(write = ident_with_units, units_attr = "long_units"), kind = "values")]
    values: Vec<f64>,
}

fn main() {}
