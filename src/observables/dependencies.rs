use quantum::particles::Particles;
use rayon::prelude::*;

use crate::{
    asymptotic_states::AsymptoticStates,
    boundary::Boundary,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    potentials::potential::{PotentialCurve, PotentialSurface},
    types::{FMatrix, FNField},
};
use num::complex::Complex64;

use super::{observable_extractor::ObservableExtractor, s_matrix::HasSMatrix};

/// Struct for general scattering dependencies for single channel problems.
pub struct SingleDependencies;

impl SingleDependencies {
    /// Calculates scattering lengths calculated from wave function propagated to distances from `distances`.
    /// It is the single channel real case with default propagation settings.
    /// Propagation is starting from `boundary` and using `collision_params`.
    /// Returns vector of distances and vector of corresponding scattering lengths. 
    pub fn propagation_distance<P>(
        distances: Vec<f64>,
        particles: &Particles,
        potential: &P,
        boundary: Boundary<f64>,
    ) -> (Vec<f64>, Vec<Complex64>)
    where
        P: PotentialCurve,
    {
        let mut rs = Vec::with_capacity(distances.len());
        let mut scatterings = Vec::with_capacity(distances.len());
        let asymptotic = potential.asymptotic_value();
        let l = particles.internals.get_value("l") as usize;

        let particles_cloned = particles.clone();

        let mut numerov = RatioNumerov::new(particles, potential, 1.0);
        numerov.prepare(&boundary);
        let result = numerov.result();
        let mut observable_extractor = ObservableExtractor::new(particles, potential, result);

        for distance in distances {
            numerov.propagate_to(distance);
            let result = numerov.result();
            rs.push(result.r_last);

            observable_extractor.new_result(result);

            let s_matrix = observable_extractor.calculate_s_matrix(l, asymptotic);
            let scattering_length = s_matrix.get_scattering_length(0);
            scatterings.push(scattering_length);
        }

        (rs, scatterings)
    }

    /// Calculates scattering lengths dependence on changing the collision parameters `collision_params` with `change_function`.
    /// It is the single channel real case with default propagation settings.
    /// Propagation is starting from `boundary` and it is propagated to the distance `propagation_distance`.
    /// For parallel calculation use `params_change_par`.
    pub fn params_change<P>(
        changes: &[f64],
        change_function: impl Fn(&f64, &mut Particles),
        mut particles: Particles,
        potential: P,
        boundary: Boundary<f64>,
        propagation_distance: f64,
    ) -> Vec<Complex64>
    where
        P: PotentialCurve,
    {
        let mut scatterings = Vec::with_capacity(changes.len());
        let asymptotic = potential.asymptotic_value();
        let l = particles.internals.get_value("l") as usize;

        for change in changes {
            change_function(&change, &mut particles);

            let mut numerov = RatioNumerov::new(&particles, &potential, 1.0);
            numerov.prepare(&boundary);
            numerov.propagate_to(propagation_distance);
            let result = numerov.result();

            let mut observable_extractor = ObservableExtractor::new(&particles, &potential, result);

            let s_matrix = observable_extractor.calculate_s_matrix(l, asymptotic);
            let scattering_length = s_matrix.get_scattering_length(0);
            scatterings.push(scattering_length);
        }

        scatterings
    }

    /// Parallel calculation of scattering lengths dependence on changing the collision parameters `collision_params` with `change_function`.
    /// It is the single channel real case with default propagation settings.
    /// Propagation is starting from `boundary` and it is propagated to the distance `propagation_distance`.
    /// For sequential calculation use `params_change`.
    pub fn params_change_par<P>(
        changes: &[f64],
        change_function: impl Fn(&f64, &mut Particles) + Sync + Send,
        mut particles: Particles,
        potential: P,
        boundary: Boundary<f64>,
        propagation_distance: f64,
    ) -> Vec<Complex64>
    where
        P: PotentialCurve,
    {
        let asymptotic = potential.asymptotic_value();
        let l = particles.internals.get_value("l") as usize;

        let scatterings = changes
            .par_iter()
            .map_with((particles, potential), |(mut particles, potential), change| {
                change_function(&change, &mut particles);

                let mut numerov = RatioNumerov::new(&particles, &potential, 1.0, 0.0f64);
                numerov.prepare(&boundary);
                numerov.propagate_to(propagation_distance);
                let result = numerov.result();

                let mut observable_extractor = ObservableExtractor::new(&particles, &potential, result);

                let s_matrix = observable_extractor.calculate_s_matrix(l, asymptotic);
                s_matrix.get_scattering_length(0)
            })
            .collect();

        scatterings
    }
}

/// Struct for general scattering dependencies for multi channel problems.
pub struct MultiDependencies;

impl MultiDependencies {
    /// Calculates scattering lengths calculated from wave function propagated to distances from `distances`.
    /// It is the single channel real case with default propagation settings.
    /// Propagation is starting from `boundary` and using `collision_params`.
    /// Returns vector of distances and vector of corresponding scattering lengths. 
    pub fn propagation_distance<const N: usize, T, P>(
        distances: Vec<f64>,
        mut particles: Particles,
        potential: P,
        boundary: Boundary<FMatrix<N>>,
        asymptotic: AsymptoticStates<N>,
        entrance_channel: usize,
    ) -> (Vec<f64>, Vec<Complex64>)
    where
        P: PotentialSurface<T>,
        T: FNField<N>,
    {
        let mut rs = Vec::with_capacity(distances.len());
        let mut scatterings = Vec::with_capacity(distances.len());
        let l = particles.internals.get_value("l") as usize;

        let mut numerov = RatioNumerov::new(&particles, &potential, 1.0);
        numerov.prepare(&boundary);
        let mut result = numerov.result();
        let mut observable_extractor = ObservableExtractor::new(&particles, &potential, result);

        for distance in distances {
            numerov.propagate_to(distance);
            result = numerov.result();
            rs.push(result.r_last);

            observable_extractor.new_result(result);

            let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic);
            let scattering_length = s_matrix.get_scattering_length(entrance_channel);
            scatterings.push(scattering_length);
        }

        (rs, scatterings)
    }

    /// Calculates scattering lengths dependence on changing the collision parameters `collision_params` with `change_function`.
    /// It is the multi channel real case with default propagation settings.
    /// Propagation is starting from `boundary` and it is propagated to the distance `propagation_distance`.
    /// For parallel calculation use `params_change_par`.
    pub fn params_change<const N: usize, T, P>(
        changes: &[f64],
        change_function: impl Fn(&f64, &mut Particles),
        mut particles: Particles,
        potential: P,
        boundary: Boundary<FMatrix<N>>,
        asymptotic: AsymptoticStates<N>,
        entrance_channel: usize,
        propagation_distance: f64,
    ) -> Vec<Complex64>
    where
        P: PotentialSurface<T>,
        T: FNField<N>,
    {
        let mut scatterings = Vec::with_capacity(changes.len());
        let l = particles.internals.get_value("l") as usize;

        for change in changes {
            change_function(&change, &mut particles);

            let mut numerov = RatioNumerov::new(&particles, &potential, 1.0);
            numerov.prepare(&boundary);
            numerov.propagate_to(propagation_distance);
            let result = numerov.result();

            let mut observable_extractor = ObservableExtractor::new(&particles, &potential, result);

            let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic);
            let scattering_length = s_matrix.get_scattering_length(entrance_channel);
            scatterings.push(scattering_length);
        }

        scatterings
    }

    /// Parallel calculation of scattering lengths dependence on changing the collision parameters `collision_params` with `change_function`.
    /// It is the multi channel real case with default propagation settings.
    /// Propagation is starting from `boundary` and it is propagated to the distance `propagation_distance`.
    /// For sequential calculation use `params_change`.
    pub fn params_change_par<const N: usize, P, T>(
        changes: &[f64],
        change_function: impl Fn(&f64, &mut Particles) + Sync + Send,
        mut particles: Particles,
        potential: P,
        boundary: Boundary<FMatrix<N>>,
        asymptotic: AsymptoticStates<N>,
        entrance_channel: usize,
        propagation_distance: f64,
    ) -> Vec<Complex64>
    where
        P: PotentialSurface<T>,
        T: FNField<N>,
    {
        let l = particles.internals.get_value("l") as usize;

        let scatterings = changes
            .par_iter()
            .map_with((particles, potential), |(mut particles, potential), change| {
                change_function(&change, &mut particles);

                let mut numerov = RatioNumerov::new(&particles, &potential, 1.0);
                numerov.prepare(&boundary);
                numerov.propagate_to(propagation_distance);
                let result = numerov.result();

                let mut observable_extractor = ObservableExtractor::new(&particles, &potential, result);

                let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic);
                s_matrix.get_scattering_length(entrance_channel)
            })
            .collect();

        scatterings
    }
}
