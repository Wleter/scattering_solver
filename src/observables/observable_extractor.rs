use ndarray::ArrayView2;
use ndarray_linalg::Eig;
use num::complex::Complex64;

use crate::{
    collision_params::CollisionParams,
    numerovs::propagator::NumerovResult,
    potentials::potential::Potential,
    utility::{asymptotic_bessel_j, asymptotic_bessel_n, bessel_j_ratio, bessel_n_ratio}, types::{FMatrix, CMatrix},
};

use super::s_matrix::{HasSMatrix, OneChanSMatrix};

pub struct ObservableExtractor<'a, T, P>
where
    P: Potential<Space = T>,
{
    collision_params: &'a mut CollisionParams<P>,
    result: NumerovResult<T>,
}

impl<'a, T, P> ObservableExtractor<'a, T, P>
where
    P: Potential<Space = T>,
{
    pub fn new(collision_params: &'a mut CollisionParams<P>, result: NumerovResult<T>) -> Self {
        Self {
            collision_params,
            result,
        }
    }
}

impl<'a, P> ObservableExtractor<'a, f64, P>
where
    P: Potential<Space = f64>,
{
    pub fn calculate_s_matrix(&mut self, l: usize) -> impl HasSMatrix {
        let r_last = self.result.r_last;
        let r_prev_last = self.result.r_last - self.result.dr;
        let wave_ratio = self.result.wave_ratio;

        let energy = self
            .collision_params
            .particles
            .internals
            .get_value("energy");
        let mass = self.collision_params.particles.red_mass();
        let asymptotic_potential = self.collision_params.potential.value(&r_last);

        let momentum = (2.0 * mass * (energy - asymptotic_potential)).sqrt();
        assert!(momentum.is_nan() == false, "channel is closed, no S-Matrix");

        let j_last = asymptotic_bessel_j(momentum * r_last, l);
        let j_prev_last = asymptotic_bessel_j(momentum * r_prev_last, l);
        let n_last = asymptotic_bessel_n(momentum * r_last, l);
        let n_prev_last = asymptotic_bessel_n(momentum * r_prev_last, l);

        let k_matrix = -(wave_ratio * j_prev_last - j_last) / (wave_ratio * n_prev_last - n_last);

        let s_matrix = Complex64::new(1.0, k_matrix) / Complex64::new(1.0, -k_matrix);

        OneChanSMatrix::new(s_matrix, momentum)
    }
}

impl<'a, const N: usize, P> ObservableExtractor<'a, FMatrix<N>, P>
where
    P: Potential<Space = FMatrix<N>>,
{
    pub fn calculate_s_matrix(&mut self, l: usize) { //-> impl HasSMatrix {
        let r_last = self.result.r_last;
        let r_prev_last = self.result.r_last - self.result.dr;
        let wave_ratio = self.result.wave_ratio;

        let energy = self
            .collision_params
            .particles
            .internals
            .get_value("energy");
        let mass = self.collision_params.particles.red_mass();

        let asymptotic_potential = self.collision_params.potential.value(&r_last);
        let slice = asymptotic_potential.as_slice();
        let asymptotic_potential = ArrayView2::from_shape((N, N), slice).unwrap();

        let (eigen_vals, eigen_vecs) = asymptotic_potential.eig().unwrap();
        let is_open_channel = eigen_vals
            .iter()
            .map(|val| val.re > energy)
            .collect::<Vec<bool>>();

        let momenta: Vec<f64> = eigen_vals
            .iter()
            .map(|val| (2.0 * mass * (energy - val.re).abs()).sqrt())
            .collect();
        
        let mut j_last = FMatrix::<N>::zeros(); 
        let mut j_prev_last = FMatrix::<N>::zeros();
        let mut n_last = FMatrix::<N>::zeros();
        let mut n_prev_last = FMatrix::<N>::zeros();

        for i in 0..N {
            let momentum = momenta[i];
            if is_open_channel[i] {
                j_last[(i, i)] = asymptotic_bessel_j(momentum * r_last, l);
                j_prev_last[(i, i)] = asymptotic_bessel_j(momentum * r_prev_last, l);
                n_last[(i, i)] = asymptotic_bessel_n(momentum * r_last, l);
                n_prev_last[(i, i)] = asymptotic_bessel_n(momentum * r_prev_last, l);
            } else {
                j_last[(i, i)] = bessel_j_ratio(momentum * r_last, momentum * r_prev_last);
                j_prev_last[(i, i)] = 1.0;
                n_last[(i, i)] = bessel_n_ratio(momentum * r_last, momentum * r_prev_last);
                n_prev_last[(i, i)] = 1.0;
            }
        }
        let mut transform = CMatrix::<N>::zeros();
        transform.copy_from_slice(eigen_vecs.as_slice().unwrap());

        // FMatrix::<N>::from_slice(eigen_vecs.as_slice().unwrap());

        let wave_transf = eigen_vecs.t() * wave_ratio * eigen_vecs;

        // let k_matrix = -(wave_ratio * j_prev_last - j_last) / (wave_ratio * n_prev_last - n_last);

        // let s_matrix = Complex64::new(1.0, k_matrix) / Complex64::new(1.0, -k_matrix);

        // MultiChanSMatrix::new(s_matrix, momentum)
    }
}