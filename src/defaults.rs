use crate::types::FMatrix;

pub struct SingleDefaults;
pub struct MultiDefaults;

impl SingleDefaults {
    pub fn boundary() -> (f64, f64) {
        (1.1, 1.11)
    }

    pub fn init_wave() -> f64 {
        1e-50
    }
}

impl MultiDefaults {
    pub fn boundary(dim: usize) -> (FMatrix, FMatrix) {
        (
            FMatrix::from_diagonal_element(dim, dim, 1.1),
            FMatrix::from_diagonal_element(dim, dim, 1.11),
        )
    }

    pub fn init_wave(dim: usize) -> FMatrix {
        FMatrix::from_diagonal_element(dim, dim, 1e-50)
    }
}
