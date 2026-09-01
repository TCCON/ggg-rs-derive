use ggg_rs_derive::WriteNetcdf;

#[derive(WriteNetcdf)]
struct Bad {
    /// A global attribute cannot also carry a description.
    #[netcdf(attribute, describe = "long_name")]
    source: String,
}

fn main() {}
