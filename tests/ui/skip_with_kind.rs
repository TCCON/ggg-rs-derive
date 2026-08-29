use ggg_rs_derive::Subset;

#[derive(Subset)]
struct Bad {
    #[subset(skip, kind = "vec")]
    values: Vec<u8>,
}

fn main() {}
