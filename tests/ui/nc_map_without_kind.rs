use ggg_rs_derive::WriteNetcdf;

fn ident(_s: &Bad, v: &Vec<f64>, _n: &str) -> Vec<f64> {
    v.clone()
}

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(dimensions("time"), map = ident)]
    values: Vec<f64>,
}

fn main() {}
