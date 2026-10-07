// A `scale_offset` that scales by nothing and offsets by nothing is a no-op
// dressed up as an instruction.
use ggg_rs_derive::WriteNetcdf;

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(dimensions("time"), scale_offset(nc_type = "i16"))]
    xco2: Vec<f64>,
}

fn main() {}
