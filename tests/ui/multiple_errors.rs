use ggg_rs_derive::Subset;

#[derive(Subset)]
struct Bad {
    #[subset(axis = 1)]
    a: Vec<u8>,
    #[subset(kind = "tensor")]
    b: Vec<u8>,
}

fn main() {}
