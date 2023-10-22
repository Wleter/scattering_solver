use crate::potentials::potential::Potential;

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

/// A trait for the result of a Numerov propagation
pub trait NumerovResult<T> {
    /// Returns the current position in the propagation
    fn r(&self) -> f64;
    /// Returns the current step size
    fn dr(&self) -> f64;
    /// Returns ratio of wave function on position r and r - dr
    fn wave_ratio(&self) -> &T;
}

/// A trait for Numerov propagator
pub trait Numerov<T, P: Potential<Space = T>> {
    /// Prepares the propagator for a new propagation
    /// starting from position r and with a boundary condition
    /// `psi(r) = boundary.0` and `psi(r - dr) = boundary.1`
    fn prepare(&mut self, r: f64, boundary: (T, T));

    /// Propagate the wave function until position is larger than r.
    /// [`prepare`] must be called before calling this method.
    fn propagate_to(&mut self, r: f64);

    /// Propagate the wave function until position is larger than r
    /// with a initial value of a wave function `wave_init`.
    /// [`prepare`] must be called before calling this method.
    /// Return the list of a wave function values
    /// and the corresponding positions
    fn propagate_values(&mut self, r: f64, wave_init: T) -> (Vec<T>, Vec<f64>);
}
