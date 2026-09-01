use ggg_rs_derive::WriteNetcdf;

fn units(_s: &Bad, _v: &Vec<f64>, _n: &str) -> String {
    "ppm".to_string()
}

#[derive(WriteNetcdf)]
struct Bad {
    #[netcdf(dimensions("time"), attr(name = "units", value = "ppm", func = units))]
    values: Vec<f64>,
}

fn main() {}
