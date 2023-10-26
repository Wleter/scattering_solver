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
        potential::Potential, potential_factory::create_lj, gaussian_coupling::GaussianCoupling, coupling_factory::couple_neighbors,
    }, types::FMatrix,
};

pub struct TwoChannel {}

impl ProblemSelector for TwoChannel {
    const NAME: &'static str = "two channel";

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

impl TwoChannel {
    fn create_collision_params() -> CollisionParams<impl Potential<Space = FMatrix<2>>> {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = EnergyUnit::Kelvin.to_au(1e-7);

        let particles = Particles::new_pair(particle1, particle2, energy);
        let potential_lj1 = create_lj(0.002, 9.0, 0.0);
        let potential_lj2 = create_lj(0.0021, 8.9, EnergyUnit::Kelvin.to_au(1.0));

        let coupling = GaussianCoupling::new(EnergyUnit::Kelvin.to_au(10.0), 11.0, 2.0);

        let coupled_potential = couple_neighbors(vec![coupling], [potential_lj1, potential_lj2]);
        CollisionParams::new(particles, coupled_potential)
    }

    fn wave_function() {
        println!("Calculating wave function...");
        let start = Instant::now();

        let mut collision_params = Self::create_collision_params();
        let mut numerov = RatioNumerov::new(&mut collision_params);

        let preparation = start.elapsed();

        numerov.prepare(6.5, (FMatrix::<2>::from_diagonal_element(1.1), FMatrix::<2>::from_diagonal_element(1.11)));
        let (rs, waves) = numerov.propagate_values(50.0, FMatrix::<2>::from_diagonal_element(1e-50));
        let propagation = start.elapsed() - preparation;

        let header = vec!["position", "channel 1", "channel 2"];
        let data = rs
            .iter()
            .zip(waves.iter())
            .map(|(r, wave)| vec![*r, wave[(0, 0)], wave[(1, 1)]])
            .collect();

        save_vec("two_chan/wave_function", data, header).unwrap();

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
    }

    fn scattering_length() {
        println!("Calculating scattering length...");
        let start = Instant::now();

        let mut collision_params = Self::create_collision_params();
        let mut numerov = RatioNumerov::new(&mut collision_params);

        let preparation = start.elapsed();

        numerov.prepare(6.5, (FMatrix::<2>::from_diagonal_element(1.1), FMatrix::<2>::from_diagonal_element(1.11)));
        numerov.propagate_to(1000.0);
        let result = numerov.result();

        let propagation = start.elapsed() - preparation;

        let mut observable_extractor = ObservableExtractor::new(&mut collision_params, result);
        // let s_matrix = observable_extractor.calculate_s_matrix(0);
        // let scattering_length = s_matrix.get_scattering_length(0);

        let extraction = start.elapsed() - preparation - propagation;

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        // println!("Scattering length: {:.2e} bohr", scattering_length);
    }
}
