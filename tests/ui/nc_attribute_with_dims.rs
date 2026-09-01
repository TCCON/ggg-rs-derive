use ggg_rs_derive::WriteNetcdf;

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(attribute, dimensions("time"))]
    source: String,
}

fn main() {}
