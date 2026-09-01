use ggg_rs_derive::WriteNetcdf;

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(skip, rename = "x")]
    values: Vec<f64>,
}

fn main() {}
