use crate::types::FMatrix;

pub struct AsymptoticStates<const N: usize> {
    pub energies: Vec<f64>,
    pub eigenvectors: FMatrix<N>,
    pub entrance_channel: usize,
}
