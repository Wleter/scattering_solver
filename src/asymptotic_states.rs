use quantum::units::{energy_units::Energy, Unit};

use crate::types::{DFMatrix, FMatrix};

pub struct AsymptoticStates<const N: usize> {
    energies: Vec<f64>,
    eigenvectors: FMatrix<N>,
    entrance_channel: usize,
}

impl<const N: usize> AsymptoticStates<N> {
    pub fn new<U: Unit>(energies: Vec<Energy<U>>, eigenvectors: FMatrix<N>, entrance_channel: usize) -> Self {
        Self {
            energies: energies.iter().map(|e| e.to_au()).collect(),
            eigenvectors,
            entrance_channel,
        }
    }

    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    pub fn eigenvectors(&self) -> &FMatrix<N> {
        &self.eigenvectors
    }

    pub fn entrance_channel(&self) -> usize {
        self.entrance_channel
    }
}

pub struct DynAsymptoticStates {
    energies: Vec<f64>,
    eigenvectors: DFMatrix,
    entrance_channel: usize,
}

impl DynAsymptoticStates {
    pub fn new<U: Unit>(energies: Vec<Energy<U>>, eigenvectors: DFMatrix, entrance_channel: usize) -> Self {
        Self {
            energies: energies.iter().map(|e| e.to_au()).collect(),
            eigenvectors,
            entrance_channel,
        }
    }

    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    pub fn eigenvectors(&self) -> &DFMatrix {
        &self.eigenvectors
    }

    pub fn entrance_channel(&self) -> usize {
        self.entrance_channel
    }
}