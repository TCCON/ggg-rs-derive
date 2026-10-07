// scale_factor, add_offset, and _FillValue are *variable* attributes; a field
// stored as a group attribute has none of them.
use ggg_rs_derive::ReadNetcdf;

#[derive(ReadNetcdf)]
struct Bad {
    #[netcdf(attribute, unpack)]
    doi: f64,
}

fn main() {}
