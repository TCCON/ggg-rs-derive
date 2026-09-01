use ggg_rs_derive::WriteNetcdf;

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(kind = "tensor")]
    values: Vec<f64>,
}

fn main() {}
