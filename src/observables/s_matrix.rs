use std::f64::consts::PI;

use num::complex::Complex64;

use crate::types::CMatrix;

pub trait HasSMatrix {
    fn get_scattering_length(&self, channel: usize) -> Complex64;

    fn get_elastic_cross_sect(&self, channel: usize) -> f64;

    fn get_inelastic_cross_sect(&self, channel: usize) -> f64;
}

pub struct OneChanSMatrix {
    s_matrix: Complex64,
    momentum: f64,
}

impl OneChanSMatrix {
    pub fn new(s_matrix: Complex64, momentum: f64) -> Self {
        Self { s_matrix, momentum }
    }
}

impl HasSMatrix for OneChanSMatrix {
    fn get_scattering_length(&self, _channel: usize) -> Complex64 {
        1.0 / Complex64::new(0.0, self.momentum) * (1.0 - self.s_matrix) / (1.0 + self.s_matrix)
    }

    fn get_elastic_cross_sect(&self, _channel: usize) -> f64 {
        PI / self.momentum.powi(2) * (1.0 - self.s_matrix).norm_sqr()
    }

    fn get_inelastic_cross_sect(&self, _channel: usize) -> f64 {
        PI / self.momentum.powi(2) * (1.0 - self.s_matrix.norm()).powi(2)
    }
}

pub struct MultiChanSMatrix<const N: usize> {
    s_matrix: CMatrix<N>,
    momenta: Vec<f64>,
    entrance: usize,
}

impl<const N: usize> MultiChanSMatrix<N> {
    pub fn new(s_matrix: CMatrix<N>, momenta: Vec<f64>, entrance: usize) -> Self {
        Self {
            s_matrix,
            momenta,
            entrance,
        }
    }
}

impl<const N: usize> HasSMatrix for MultiChanSMatrix<N> {
    fn get_scattering_length(&self, channel: usize) -> Complex64 {
        1.0 / Complex64::new(0.0, self.momenta[channel]) * (1.0 - self.s_matrix[(channel, channel)])
            / (1.0 + self.s_matrix[(channel, channel)])
    }

    fn get_elastic_cross_sect(&self, channel: usize) -> f64 {
        PI / self.momenta[channel].powi(2) * (1.0 - self.s_matrix[(channel, channel)]).norm_sqr()
    }

    fn get_inelastic_cross_sect(&self, channel: usize) -> f64 {
        if self.entrance == channel {
            PI / self.momenta[channel].powi(2)
                * (1.0 - self.s_matrix[(channel, channel)].norm()).powi(2)
        } else {
            PI / self.momenta[channel].powi(2)
                * (self.s_matrix[(self.entrance, channel)].norm()).powi(2)
        }
    }
}
