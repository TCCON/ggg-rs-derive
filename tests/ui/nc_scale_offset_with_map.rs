// With a `map`, the declared type is the mapper's output, so the macro cannot
// check that what gets written is floating-point. The combination is refused
// rather than guessed at.
use ggg_rs_derive::WriteNetcdf;

fn ident(_s: &Bad, v: &Vec<f64>, _n: &str) -> Vec<f64> {
    v.clone()
}

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(
        dimensions("time"),
        kind = "values",
        map(write = ident),
        scale_offset(scale = 0.01, nc_type = "i16")
    )]
    xco2: Vec<f64>,
}

fn main() {}
