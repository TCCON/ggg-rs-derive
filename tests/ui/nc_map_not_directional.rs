// `map` must name the direction, because reading and writing need different
// (inverse) functions. The old non-directional `map = fn` form is rejected.
use ggg_rs_derive::WriteNetcdf;

fn ident(_s: &Bad, v: &Vec<f64>, _n: &str) -> Vec<f64> {
    v.clone()
}

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(dimensions("time"), map = ident, kind = "values")]
    values: Vec<f64>,
}

fn main() {}
