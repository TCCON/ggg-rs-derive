use ggg_rs_derive::Subset;

#[derive(Subset)]
struct Bad {
    #[subset(kind = "tensor")]
    thing: Vec<u8>,
}

fn main() {}
