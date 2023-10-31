use dyn_clone::DynClone;

use crate::types::{CMatrix, FMatrix};

pub trait OnePotential: DynClone {
    fn value(&self, r: &f64) -> f64;

    fn asymptotic_value(&self) -> f64 {
        self.value(&f64::INFINITY)
    }
}

impl Clone for Box<dyn OnePotential> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

pub trait MultiPotential: DynClone {
    fn dim(&self) -> usize;

    fn value(&self, r: &f64) -> FMatrix;

    fn asymptotic_value(&self) -> FMatrix {
        self.value(&f64::INFINITY)
    }
}

impl Clone for Box<dyn MultiPotential> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

pub trait MultiCPotential: DynClone {
    fn dim(&self) -> usize;

    fn value(&self, r: &f64) -> CMatrix;

    fn asymptotic_value(&self) -> CMatrix {
        self.value(&f64::INFINITY)
    }
}

impl Clone for Box<dyn MultiCPotential> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}