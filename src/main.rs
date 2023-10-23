use std::env;

fn main() {
    let mut args = env::args();
    // get rid of environment variable
    args.next();
}
