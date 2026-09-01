use ggg_rs_derive::WriteNetcdf;

#[derive(WriteNetcdf)]
#[netcdf(rename_all = "TitleCase")]
struct Bad {
    #[netcdf(dimensions("time"))]
    values: Vec<f64>,
}

fn main() {}
