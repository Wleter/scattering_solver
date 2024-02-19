use std::mem::take;

use crate::{boundary::Boundary, potentials::potential::Potential};

pub(super) trait MultiStep<P: Potential> {
    /// Performs a step with the same step size
    fn step(&mut self);

    /// Halves the step size without performing a step
    fn half_step(&mut self);

    /// Doubles the step size without performing a step
    fn double_step(&mut self);

    /// Returns the recommended step size for the current step
    fn recommended_step_size(&self) -> f64;

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

    /// Performs a single step of the propagation
    /// [`prepare`] must be called before calling this method.
    fn single_step(&mut self);

    /// Propagate the wave function until position is larger than r.
    /// [`prepare`] must be called before calling this method.
    fn propagate_to(&mut self, r: f64);

    /// Propagate the wave function until position is larger than r
    /// with a initial value of a wave function `wave_init`.
    /// [`prepare`] must be called before calling this method.
    /// Return the list of positions and the corresponding wave function values
    fn propagate_values(&mut self, r_stop: f64, wave_init: T, sampling: Sampling) -> (Vec<f64>, Vec<T>);

    /// Returns the result of the propagation
    fn result(&self) -> NumerovResult<T>;
}

pub(crate) struct SamplingStorage<T> {
    rs: Vec<f64>,
    waves: Vec<T>,
    sample_each: Option<usize>,
    sample_step: Option<f64>,
    counter: usize,
}

impl<T: Clone> SamplingStorage<T> {
    pub fn new(sampling: Sampling, r: &f64, wave: &T, r_stop: &f64) -> Self {
        let capacity = match sampling {
            Sampling::Uniform(capacity) => capacity,
            Sampling::Variable(capacity) => capacity,
        };

        let mut rs = Vec::with_capacity(capacity);
        let mut waves = Vec::with_capacity(capacity);
        rs.push(*r);
        waves.push(wave.clone());

        let sample_each = match sampling {
            Sampling::Uniform(_) => None,
            Sampling::Variable(_) => Some(1),
        };

        let sample_step = match sampling {
            Sampling::Uniform(_) => Some((r_stop - r).abs() / capacity as f64),
            Sampling::Variable(_) => None,
        };

        Self { 
            rs,
            waves,
            sample_each,
            sample_step,
            counter: 0,
        }
    }

    pub fn sample(&mut self, r: &f64, wave: &T) {
        if let Some(sample_each) = self.sample_each {
            self.counter += 1;
            if self.counter % sample_each == 0 {
                self.rs.push(*r);
                self.waves.push(wave.clone());
            }

            if self.rs.len() == self.rs.capacity() {
                self.sample_each = Some(2 * sample_each);
                // delete every second element
                let rs = take(&mut self.rs);
                let waves = take(&mut self.waves);

                self.rs = rs.into_iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .map(|(_, r)| r)
                    .collect();

                self.waves = waves.into_iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 1)
                    .map(|(_, w)| w)
                    .collect();
            }

        } else if let Some(sample_step) = self.sample_step {
            if (r - self.rs.last().unwrap()).abs() > sample_step {
                self.rs.push(*r);
                self.waves.push(wave.clone());
            }
        }
    }

    pub fn result(self) -> (Vec<f64>, Vec<T>) {
        (self.rs, self.waves)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Sampling {
    Uniform(usize),
    Variable(usize), // because of cap of step size it can over prioritize wrong regions todo! 
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling::Uniform(1000)
    }
}