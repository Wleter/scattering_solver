use nalgebra::Matrix3;
use problems::Problems;
use quantum::problem_selector::{get_args, ProblemSelector};

pub mod problems;

fn main() {
    Problems::select(&mut get_args());
}
