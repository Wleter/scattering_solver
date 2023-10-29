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
    pub fn boundary<const N: usize>() -> (FMatrix<N>, FMatrix<N>) {
        (
            FMatrix::<N>::from_diagonal_element(1.1),
            FMatrix::<N>::from_diagonal_element(1.11),
        )
    }

    pub fn init_wave<const N: usize>() -> FMatrix<N> {
        FMatrix::<N>::from_diagonal_element(1e-50)
    }
}
