// use rayon::prelude::*;

use std::rc::Rc;

use crate::{
    boundary::Boundary,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    potentials::potential::{OnePotential, MultiPotential}, types::FMatrix, asymptotic_states::AsymptoticStates,
};
use num::complex::Complex64;
use quantum::particles::Particles;

use super::{observable_extractor::ObservableExtractor, s_matrix::HasSMatrix};

pub struct SingleDependencies;

impl SingleDependencies {
    pub fn propagation_distance(
        distances: Vec<f64>,
        particles: Particles,
        potential: impl OnePotential + 'static,
        boundary: Boundary<f64>,
    ) -> (Vec<f64>, Vec<Complex64>)
    {

        let mut rs = Vec::with_capacity(distances.len());
        let mut scatterings = Vec::with_capacity(distances.len());
        let asymptotic = potential.asymptotic_value();
        let l = particles.internals.get_value("l") as usize;

        let particles = Rc::new(particles);
        let potential: Rc<dyn OnePotential> = Rc::new(potential);

        let mut numerov = RatioNumerov::new_single(particles.clone(), potential.clone(), 1.0);
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

    pub fn params_change<P: OnePotential + Clone + 'static>(
        changes: &Vec<f64>,
        change_function: impl Fn(&f64, &mut Particles, &mut P),
        mut particles: Particles,
        mut potential: P,
        boundary: Boundary<f64>,
        propagation_distance: f64,
    ) -> Vec<Complex64>
    {
        let mut scatterings = Vec::with_capacity(changes.len());
        let asymptotic = potential.asymptotic_value();
        let l = particles.internals.get_value("l") as usize;

        for change in changes {
            change_function(&change, &mut particles, &mut potential);
            let rc_particles = Rc::new(particles.clone());
            let rc_potential: Rc<dyn OnePotential> = Rc::new(potential.clone());

            let mut numerov = RatioNumerov::new_single(rc_particles.clone(), rc_potential.clone(), 1.0);
            numerov.prepare(&boundary);
            numerov.propagate_to(propagation_distance);
            let result = numerov.result();

            let mut observable_extractor = ObservableExtractor::new(rc_particles, rc_potential, result);

            let s_matrix = observable_extractor.calculate_s_matrix(l, asymptotic);
            let scattering_length = s_matrix.get_scattering_length(0);
            scatterings.push(scattering_length);
        }

        scatterings
    }
}

pub struct MultiDependencies;

impl MultiDependencies {
    pub fn propagation_distance(
        distances: Vec<f64>,
        particles: Particles,
        potential: impl MultiPotential + 'static,
        boundary: Boundary<FMatrix>,
        asymptotic: AsymptoticStates,
        entrance_channel: usize,
    ) -> (Vec<f64>, Vec<Complex64>)
    {
        let mut rs = Vec::with_capacity(distances.len());
        let mut scatterings = Vec::with_capacity(distances.len());
        let l = particles.internals.get_value("l") as usize;

        let particles = Rc::new(particles);
        let potential: Rc<dyn MultiPotential> = Rc::new(potential);

        let mut numerov = RatioNumerov::new_multi(particles.clone(), potential.clone(), 1.0);
        numerov.prepare(&boundary);
        let mut result = numerov.result();
        let mut observable_extractor = ObservableExtractor::new(particles, potential, result);

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

    pub fn params_change<P: MultiPotential + Clone + 'static>(
        changes: &Vec<f64>,
        change_function: impl Fn(&f64, &mut Particles, &mut P),
        mut particles: Particles,
        mut potential: P,
        boundary: Boundary<FMatrix>,
        asymptotic: AsymptoticStates,
        entrance_channel: usize,
        propagation_distance: f64,
    ) -> Vec<Complex64>
    {
        let mut scatterings = Vec::with_capacity(changes.len());
        let l = particles.internals.get_value("l") as usize;

        for change in changes {
            change_function(&change, &mut particles, &mut potential);

            let rc_particles = Rc::new(particles.clone());
            let rc_potential: Rc<dyn MultiPotential> = Rc::new(potential.clone());

            let mut numerov = RatioNumerov::new_multi(rc_particles.clone(), rc_potential.clone(), 1.0);
            numerov.prepare(&boundary);
            numerov.propagate_to(propagation_distance);
            let result = numerov.result();

            let mut observable_extractor = ObservableExtractor::new(rc_particles, rc_potential, result);

            let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic);
            let scattering_length = s_matrix.get_scattering_length(entrance_channel);
            scatterings.push(scattering_length);
        }

        scatterings
    }
}

