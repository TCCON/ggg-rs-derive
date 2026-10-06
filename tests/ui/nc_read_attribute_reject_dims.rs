use ggg_rs_derive::ReadNetcdf;

#[derive(ReadNetcdf)]
struct Bad {
    #[netcdf(dimensions("time"), attribute)]
    att_val: String,
}

fn main() {}
