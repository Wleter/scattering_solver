use crate::types::FMatrix;

pub struct AsymptoticStates<const N: usize> {
    pub energies: [f64; N],
    pub eigenvectors: FMatrix<N>,
    pub entrance_channel: usize,
}
