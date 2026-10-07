// Unpacking only makes sense for floating-point data: scale_factor is
// fractional, so applying it to an integer field would truncate it. Asking for
// it explicitly on an integer field is an error (inheriting it from the
// container is not — see tests/read_netcdf.rs).
use ggg_rs_derive::ReadNetcdf;

#[derive(ReadNetcdf)]
struct Bad {
    #[netcdf(dimensions("time"), unpack)]
    flags: Vec<i16>,
}

fn main() {}
