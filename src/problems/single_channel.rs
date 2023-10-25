use std::{collections::VecDeque, time::Instant};

use quantum::{
    particle_factory::create_atom, particles::Particles, problem_selector::ProblemSelector,
    saving::save_vec, units::energy_units::EnergyUnit,
};
use scattering_solver::{
    collision_params::CollisionParams,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    observables::{observable_extractor::ObservableExtractor, s_matrix::HasSMatrix},
    potentials::{
        composite_potential::CompositePotential, dispersion_potential::DispersionPotential,
        potential::Potential, potential_factory::create_lj,
    },
};

pub struct SingleChannel {}

impl ProblemSelector for SingleChannel {
    const NAME: &'static str = "single channel";

    fn list() -> Vec<&'static str> {
        vec!["wave function", "scattering length"]
    }

    fn methods(number: &str, _args: &mut VecDeque<String>) {
        match number {
            "0" => Self::wave_function(),
            "1" => Self::scattering_length(),
            _ => println!("No method found for number {}", number),
        }
    }
}

impl SingleChannel {
    fn create_collision_params() -> CollisionParams<CompositePotential<DispersionPotential>> {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = EnergyUnit::Kelvin.to_au(1e-7);

        let particles = Particles::new_pair(particle1, particle2, energy);
        let potential = create_lj(0.002, 9.0, 0.0);

        CollisionParams::new(particles, potential)
    }

    fn wave_function() {
        let start = Instant::now();

        let mut collision_params = Self::create_collision_params();
        let mut numerov = RatioNumerov::new(&mut collision_params);

        let preparation = start.elapsed();

        numerov.prepare(6.5, (1.1, 1.11));
        let (rs, waves) = numerov.propagate_values(100.0, 1e-50);
        let propagation = start.elapsed() - preparation;

        let potential_values: Vec<f64> = rs
            .iter()
            .map(|r| numerov.collision_params.potential.value(r))
            .collect();

        let header = vec!["position", "wave function", "potential"];
        let data = rs
            .iter()
            .zip(waves.iter())
            .zip(potential_values.iter())
            .map(|((r, wave), v)| vec![*r, *wave, EnergyUnit::Au.to_kelvin(*v)])
            .collect();

        save_vec("single_chan/wave_function", data, header).unwrap();

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
    }

    fn scattering_length() {
        let start = Instant::now();

        let mut collision_params = Self::create_collision_params();
        let mut numerov = RatioNumerov::new(&mut collision_params);

        let preparation = start.elapsed();

        numerov.prepare(6.5, (1.1, 1.11));
        numerov.propagate_to(1000.0);
        let result = numerov.result();

        let propagation = start.elapsed() - preparation;

        let mut observable_extractor = ObservableExtractor::new(&mut collision_params, result);
        let s_matrix = observable_extractor.calculate_s_matrix(0);
        let scattering_length = s_matrix.get_scattering_length(0);

        let extraction = start.elapsed() - preparation - propagation;

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }
}
