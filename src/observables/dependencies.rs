use rayon::prelude::*;

use crate::{
    asymptotic_states::AsymptoticStates,
    boundary::Boundary,
    collision_params::CollisionParams,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    potentials::potential::Potential,
    types::FMatrix,
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
        collision_params: CollisionParams<P>,
        boundary: Boundary<f64>,
    ) -> (Vec<f64>, Vec<Complex64>)
    where
        P: Potential<Space = f64>,
    {
        let mut rs = Vec::with_capacity(distances.len());
        let mut scatterings = Vec::with_capacity(distances.len());
        let asymptotic = collision_params.potential.asymptotic_value();
        let l = collision_params.particles.internals.get_value("l") as usize;

        let collisions_cloned = collision_params.clone();

        let mut numerov = RatioNumerov::new(&collisions_cloned, 1.0);
        numerov.prepare(&boundary);
        let result = numerov.result();
        let mut observable_extractor = ObservableExtractor::new(&collision_params, result);

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
        changes: &Vec<f64>,
        change_function: impl Fn(&f64, &mut CollisionParams<P>),
        mut collision_params: CollisionParams<P>,
        boundary: Boundary<f64>,
        propagation_distance: f64,
    ) -> Vec<Complex64>
    where
        P: Potential<Space = f64>,
    {
        let mut scatterings = Vec::with_capacity(changes.len());
        let asymptotic = collision_params.potential.asymptotic_value();
        let l = collision_params.particles.internals.get_value("l") as usize;

        for change in changes {
            change_function(&change, &mut collision_params);

            let mut numerov = RatioNumerov::new(&collision_params, 1.0);
            numerov.prepare(&boundary);
            numerov.propagate_to(propagation_distance);
            let result = numerov.result();

            let mut observable_extractor = ObservableExtractor::new(&collision_params, result);

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
        changes: &Vec<f64>,
        change_function: impl Fn(&f64, &mut CollisionParams<P>) + Sync + Send,
        collision_params: CollisionParams<P>,
        boundary: Boundary<f64>,
        propagation_distance: f64,
    ) -> Vec<Complex64>
    where
        P: Potential<Space = f64>,
    {
        let asymptotic = collision_params.potential.asymptotic_value();
        let l = collision_params.particles.internals.get_value("l") as usize;

        let scatterings = changes
            .par_iter()
            .map_with(collision_params, |mut collision_params, change| {
                change_function(&change, &mut collision_params);

                let mut numerov = RatioNumerov::new(&collision_params, 1.0);
                numerov.prepare(&boundary);
                numerov.propagate_to(propagation_distance);
                let result = numerov.result();

                let mut observable_extractor = ObservableExtractor::new(&collision_params, result);

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
    pub fn propagation_distance<P, const N: usize>(
        distances: Vec<f64>,
        collision_params: CollisionParams<P>,
        boundary: Boundary<FMatrix<N>>,
        asymptotic: AsymptoticStates<N>,
        entrance_channel: usize,
    ) -> (Vec<f64>, Vec<Complex64>)
    where
        P: Potential<Space = FMatrix<N>>,
    {
        let mut rs = Vec::with_capacity(distances.len());
        let mut scatterings = Vec::with_capacity(distances.len());
        let l = collision_params.particles.internals.get_value("l") as usize;

        let collisions_cloned = collision_params.clone();

        let mut numerov = RatioNumerov::new(&collisions_cloned, 1.0);
        numerov.prepare(&boundary);
        let mut result = numerov.result();
        let mut observable_extractor = ObservableExtractor::new(&collision_params, result);

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
    pub fn params_change<P, const N: usize>(
        changes: &Vec<f64>,
        change_function: impl Fn(&f64, &mut CollisionParams<P>),
        mut collision_params: CollisionParams<P>,
        boundary: Boundary<FMatrix<N>>,
        asymptotic: AsymptoticStates<N>,
        entrance_channel: usize,
        propagation_distance: f64,
    ) -> Vec<Complex64>
    where
        P: Potential<Space = FMatrix<N>>,
    {
        let mut scatterings = Vec::with_capacity(changes.len());
        let l = collision_params.particles.internals.get_value("l") as usize;

        for change in changes {
            change_function(&change, &mut collision_params);

            let mut numerov = RatioNumerov::new(&collision_params, 1.0);
            numerov.prepare(&boundary);
            numerov.propagate_to(propagation_distance);
            let result = numerov.result();

            let mut observable_extractor = ObservableExtractor::new(&collision_params, result);

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
    pub fn params_change_par<P, const N: usize>(
        changes: &Vec<f64>,
        change_function: impl Fn(&f64, &mut CollisionParams<P>) + Sync + Send,
        collision_params: CollisionParams<P>,
        boundary: Boundary<FMatrix<N>>,
        asymptotic: AsymptoticStates<N>,
        entrance_channel: usize,
        propagation_distance: f64,
    ) -> Vec<Complex64>
    where
        P: Potential<Space = FMatrix<N>>,
    {
        let l = collision_params.particles.internals.get_value("l") as usize;

        let scatterings = changes
            .par_iter()
            .map_with(collision_params, |mut collision_params, change| {
                change_function(&change, &mut collision_params);

                let mut numerov = RatioNumerov::new(&collision_params, 1.0);
                numerov.prepare(&boundary);
                numerov.propagate_to(propagation_distance);
                let result = numerov.result();

                let mut observable_extractor = ObservableExtractor::new(&collision_params, result);

                let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic);
                s_matrix.get_scattering_length(entrance_channel)
            })
            .collect();

        scatterings
    }
}
