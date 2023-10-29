use std::rc::Rc;

use nalgebra::DMatrix;
use num::complex::Complex64;

use crate::{
    asymptotic_states::AsymptoticStates,
    collision_params::CollisionParams,
    numerovs::propagator::NumerovResult,
    potentials::potential::Potential,
    types::FMatrix,
    utility::{asymptotic_bessel_j, asymptotic_bessel_n, bessel_j_ratio, bessel_n_ratio},
};

use super::s_matrix::{MultiChanSMatrix, OneChanSMatrix};

pub struct ObservableExtractor<T, P>
where
    P: Potential<Space = T>,
{
    collision_params: Rc<CollisionParams<P>>,
    result: NumerovResult<T>,
}

impl<T, P> ObservableExtractor<T, P>
where
    P: Potential<Space = T>,
{
    pub fn new(collision_params: Rc<CollisionParams<P>>, result: NumerovResult<T>) -> Self {
        Self {
            collision_params,
            result,
        }
    }

    pub fn new_result(&mut self, result: NumerovResult<T>) {
        self.result = result;
    }
}

impl<P> ObservableExtractor<f64, P>
where
    P: Potential<Space = f64>,
{
    pub fn calculate_s_matrix(&mut self, l: usize, asymptotic: f64) -> OneChanSMatrix {
        let r_last = self.result.r_last;
        let r_prev_last = self.result.r_last - self.result.dr;
        let wave_ratio = self.result.wave_ratio;

        let energy = self
            .collision_params
            .particles
            .internals
            .get_value("energy");
        let mass = self.collision_params.particles.red_mass();

        let momentum = (2.0 * mass * (energy - asymptotic)).sqrt();
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

impl<const N: usize, P> ObservableExtractor<FMatrix<N>, P>
where
    P: Potential<Space = FMatrix<N>>,
{
    pub fn calculate_s_matrix(
        &mut self,
        l: usize,
        asymptotic: &AsymptoticStates<N>,
    ) -> MultiChanSMatrix {
        let r_last = self.result.r_last;
        let r_prev_last = self.result.r_last - self.result.dr;
        let wave_ratio = self.result.wave_ratio;

        let energy = self
            .collision_params
            .particles
            .internals
            .get_value("energy");
        let mass = self.collision_params.particles.red_mass();

        let is_open_channel = asymptotic
            .energies
            .iter()
            .map(|val| val < &energy)
            .collect::<Vec<bool>>();
        let momenta: Vec<f64> = asymptotic
            .energies
            .iter()
            .map(|val| (2.0 * mass * (energy - val).abs()).sqrt())
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
        let wave_transf = asymptotic.eigenvectors.transpose() * wave_ratio * asymptotic.eigenvectors;

        let k_matrix = -(wave_transf * n_prev_last - n_last).try_inverse().unwrap()
            * (wave_transf * j_prev_last - j_last);

        let open_channel_count = is_open_channel.iter().filter(|val| **val).count();
        let mut red_ik_matrix = DMatrix::<Complex64>::zeros(open_channel_count, open_channel_count);

        let mut i_full = 0;
        for i in 0..open_channel_count {
            while is_open_channel[i_full] == false {
                i_full += 1
            }

            let mut j_full = 0;
            for j in 0..open_channel_count {
                while is_open_channel[j_full] == false {
                    j_full += 1
                }

                red_ik_matrix[(i, j)] = Complex64::new(0.0, k_matrix[(i_full, j_full)]);
                j_full += 1;
            }
            i_full += 1;
        }
        let id = DMatrix::<Complex64>::identity(open_channel_count, open_channel_count);

        let s_matrix = (&id - &red_ik_matrix).try_inverse().unwrap() * (id + red_ik_matrix);
        let channels = is_open_channel
            .iter()
            .enumerate()
            .filter(|(_, val)| **val)
            .map(|(i, _)| i)
            .collect::<Vec<usize>>();

        MultiChanSMatrix::new(s_matrix, momenta, channels, asymptotic.entrance_channel)
    }
}
