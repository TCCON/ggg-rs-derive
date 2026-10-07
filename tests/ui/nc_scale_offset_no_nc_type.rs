// `nc_type` is mandatory: without it the variable would keep the field's own
// float type and packing would save nothing, which is the only reason to pack.
use ggg_rs_derive::WriteNetcdf;

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(dimensions("time"), scale_offset(scale = 0.01))]
    xco2: Vec<f64>,
}

fn main() {}
