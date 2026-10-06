// A `map(...)` that only covers reading is an error for `WriteNetcdf`, which
// would otherwise silently write the field unmapped.
use ggg_rs_derive::WriteNetcdf;

fn from_nc(v: Vec<f64>, _n: &str) -> netcdf::Result<Vec<f64>> {
    Ok(v)
}

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(dimensions("time"), map(read = from_nc), kind = "values")]
    values: Vec<f64>,
}

fn main() {}
