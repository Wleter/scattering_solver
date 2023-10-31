use crate::types::FMatrix;

pub struct AsymptoticStates {
    pub energies: Vec<f64>,
    pub eigenvectors: FMatrix,
    pub entrance_channel: usize,
}
