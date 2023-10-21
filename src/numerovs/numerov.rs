use crate::types::{CMatrix, FMatrix};

use super::propagator::MultiStepPropagator;

struct RatioNumerov<T> {
    r: f64,
    dr: f64,
    psi1: T,
    psi2: T,
    psi3: T,
}


impl MultiStepPropagator for RatioNumerov<f64> {
    fn variable_step(&mut self) {

    }

    fn step(&mut self) {

    }

    fn half_step(&mut self) {

    }

    fn double_step(&mut self) {

    }
}

impl<const N: usize> MultiStepPropagator for RatioNumerov<CMatrix<N>> {
    fn variable_step(&mut self) {

    }

    fn step(&mut self) {

    }

    fn half_step(&mut self) {

    }

    fn double_step(&mut self) {

    }
}
impl<const N: usize> MultiStepPropagator for RatioNumerov<FMatrix<N>> {
    fn variable_step(&mut self) {

    }

    fn step(&mut self) {

    }

    fn half_step(&mut self) {

    }

    fn double_step(&mut self) {

    }
}