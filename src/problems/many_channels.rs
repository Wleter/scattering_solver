use std::{collections::VecDeque, time::Instant};

use quantum::{
    particle_factory::create_atom, particles::Particles, problem_selector::ProblemSelector,
    units::energy_units::EnergyUnit,
};
use scattering_solver::{
    asymptotic_states::AsymptoticStates,
    boundary::Boundary,
    collision_params::CollisionParams,
    defaults::MultiDefaults,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    observables::{observable_extractor::ObservableExtractor, s_matrix::HasSMatrix},
    potentials::{
        coupling_factory::couple_neighbors, gaussian_coupling::GaussianCoupling,
        potential::Potential, potential_factory::create_lj,
    }, types::FMatrix, utility::linspace,
};

pub struct ManyChannels {}

impl ProblemSelector for ManyChannels {
    const NAME: &'static str = "large number of channels";

    fn list() -> Vec<&'static str> {
        vec!["scattering length"]
    }

    fn methods(number: &str, _args: &mut VecDeque<String>) {
        match number {
            "0" => Self::scattering_length(),
            _ => println!("No method found for number {}", number),
        }
    }
}

impl ManyChannels {
    fn create_collision_params<const N: usize>() -> CollisionParams<impl Potential<Space = FMatrix<N>>> {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = EnergyUnit::Kelvin.to_au(1e-7);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.internals.insert_value("l", 0.0);

        let wells: [f64; N] = linspace(0.0019, 0.0022, N).try_into().unwrap();
        let potentials = wells.map(|well| create_lj(well, 9.0, EnergyUnit::Kelvin.to_au(well / 0.0019 - 1.0)));

        let couplings_strength = linspace(5.0, 15.0, N-1);
        let couplings = couplings_strength
            .iter()
            .map(|c| GaussianCoupling::new(EnergyUnit::Kelvin.to_au(*c), 11.0, 2.0))
            .collect();

        let coupled_potential = couple_neighbors(couplings, potentials);
        CollisionParams::new(particles, coupled_potential)
    }

    fn scattering_length() {
        println!("Calculating scattering length...");
        let start = Instant::now();

        const N: usize = 50;
        let collision_params = Self::create_collision_params::<N>();
        let mut numerov = RatioNumerov::new(&collision_params, 1.0);

        let energies = linspace(0.0019, 0.0022, 50)
            .iter()
            .map(|well| EnergyUnit::Kelvin.to_au(well / 0.0019 - 1.0))
            .collect();

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, MultiDefaults::boundary()));
        numerov.propagate_to(1000.0);
        let result = numerov.result();

        let propagation = start.elapsed() - preparation;

        let mut observable_extractor = ObservableExtractor::new(&collision_params, result);

        let asymptotic_states = AsymptoticStates {
            energies,
            eigenvectors: FMatrix::<N>::identity(),
            entrance_channel: 0,
        };
        let l = collision_params.particles.internals.get_value("l") as usize;

        let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic_states);
        let scattering_length = s_matrix.get_scattering_length(0);

        let extraction = start.elapsed() - preparation - propagation;

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} ms", propagation.as_millis());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }
}
