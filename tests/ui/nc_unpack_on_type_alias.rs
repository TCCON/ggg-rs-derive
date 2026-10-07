// A type alias is opaque to the macro, so it cannot confirm the element type is
// a float. Rather than guess, an explicit opt-in on such a type is refused.
use ggg_rs_derive::ReadNetcdf;

type Matrix = ndarray::Array2<f32>;

#[derive(ReadNetcdf)]
struct Bad {
    #[netcdf(dimensions("frame", "footprint"), kind = "array", unpack)]
    m: Matrix,
}

fn main() {}
