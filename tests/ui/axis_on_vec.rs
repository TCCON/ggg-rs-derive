use ggg_rs_derive::Subset;

#[derive(Subset)]
struct Bad {
    #[subset(axis = 1)]
    values: Vec<u8>,
}

fn main() {}
