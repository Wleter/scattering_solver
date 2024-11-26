#[cfg(feature = "faer")]
pub mod faer_backed;

// #[cfg(feature = "nalgebra")]
// pub mod static_backed;

use crate::{potentials::{dispersion_potential::Dispersion, potential::Potential}, utility::AngularSpin};

#[derive(Clone)]
pub struct MultiNumerovData<'a, F, P>
where 
    P: Potential<Space = F>
{
    pub(super) r: f64,
    pub(super) dr: f64,

    pub(super) potential: &'a P,
    pub(super) centrifugal: Option<Dispersion>,
    pub(super) mass: f64,
    pub(super) energy: f64,
    pub(super) l: AngularSpin,

    pub(super) potential_buffer: F,
    pub(super) unit: F,
    pub(super) current_g_func: F,

    pub(super) psi1: F,
    pub(super) psi2: F,
}

pub struct MultiRatioNumerovStep<T>
{
    f1: T,
    f2: T,
    f3: T,

    buffer1: T,
    buffer2: T,
}
