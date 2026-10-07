// Packing an already-integral field has nothing to divide.
use ggg_rs_derive::WriteNetcdf;

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(dimensions("time"), scale_offset(scale = 0.01, nc_type = "i16"))]
    counts: Vec<i32>,
}

fn main() {}
