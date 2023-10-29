use crate::{boundary::Boundary, potentials::potential::Potential};

pub(super) trait MultiStep<P: Potential> {
    /// Performs a step with the same step size
    fn step(&mut self);

    /// Halves the step size without performing a step
    fn half_step(&mut self);

    /// Doubles the step size without performing a step
    fn double_step(&mut self);

    /// Returns the recommended step size for the current step
    fn recommended_step_size(&mut self) -> f64;

    /// Performs a step with a variable step size
    fn variable_step(&mut self);
}

/// Struct storing the result of a Numerov propagation
#[derive(Debug, Clone, Default)]
pub struct NumerovResult<T> {
    /// last position in the propagation
    pub r_last: f64,
    /// last step size
    pub dr: f64,
    /// Ratio of wave function on position r and r - dr
    pub wave_ratio: T,
}

/// A trait for Numerov propagator
pub trait Numerov<T, P: Potential<Space = T>> {
    /// Prepares the propagator for a new propagation
    /// starting from position r and with a boundary condition
    /// `psi(r) = boundary.0` and `psi(r - dr) = boundary.1`
    fn prepare(&mut self, boundary: &Boundary<T>);

    /// Propagate the wave function until position is larger than r.
    /// [`prepare`] must be called before calling this method.
    fn propagate_to(&mut self, r: f64);

    /// Propagate the wave function until position is larger than r
    /// with a initial value of a wave function `wave_init`.
    /// [`prepare`] must be called before calling this method.
    /// Return the list of positions and the corresponding wave function values
    fn propagate_values(&mut self, r: f64, wave_init: T) -> (Vec<f64>, Vec<T>);

    /// Returns the result of the propagation
    fn result(&self) -> NumerovResult<T>;
}
