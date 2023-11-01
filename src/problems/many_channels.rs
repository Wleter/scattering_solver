use std::{collections::VecDeque, time::Instant, rc::Rc};

use quantum::{
    particle_factory::create_atom, particles::Particles, problem_selector::ProblemSelector,
    units::energy_units::EnergyUnit,
};
use scattering_solver::{
    asymptotic_states::AsymptoticStates,
    boundary::Boundary,
    defaults::MultiDefaults,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    observables::{observable_extractor::ObservableExtractor, s_matrix::HasSMatrix},
    potentials::{
        coupling_factory::couple_neighbors, gaussian_coupling::GaussianCoupling,
        potential_factory::create_lj, coupled_potential::CoupledPotential, potential::{OnePotential, MultiPotential},
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
    fn create_collision_params<const N: usize>() -> (Particles, CoupledPotential) {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = EnergyUnit::Kelvin.to_au(1e-7);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.internals.insert_value("l", 0.0);

        let wells = linspace(0.0019, 0.0022, N);
        let mut potentials: Vec<Box<dyn OnePotential>> = Vec::with_capacity(N);
        for well in wells {
            potentials.push(Box::new(create_lj(well, 9.0, EnergyUnit::Kelvin.to_au(well / 0.0019 - 1.0))));
        }
        
        let couplings_strength = linspace(5.0, 15.0, N-1);
        let mut couplings: Vec<Box<dyn OnePotential>> = Vec::with_capacity(N-1);
        for cs in couplings_strength {
            couplings.push(Box::new(GaussianCoupling::new(EnergyUnit::Kelvin.to_au(cs), 11.0, 2.0)));
        }

        let coupled_potential = couple_neighbors(couplings, potentials);
        (particles, coupled_potential)
    }

    fn scattering_length() {
        println!("Calculating scattering length...");
        let start = Instant::now();

        const N: usize = 200;
        let (particles, potential) = Self::create_collision_params::<N>();
        let particles = Rc::new(particles);
        let potential: Rc<dyn MultiPotential + 'static> = Rc::new(potential);
        let mut numerov = RatioNumerov::new_multi(particles.clone(), potential.clone(), 1.0);

        let energies = linspace(0.0019, 0.0022, N)
            .iter()
            .map(|well| EnergyUnit::Kelvin.to_au(well / 0.0019 - 1.0))
            .collect();

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, MultiDefaults::boundary(N)));
        numerov.propagate_to(1000.0);
        let result = numerov.result();

        let propagation = start.elapsed() - preparation;

        let mut observable_extractor = ObservableExtractor::new(particles.clone(), potential.clone(), result);

        let asymptotic_states = AsymptoticStates {
            energies,
            eigenvectors: FMatrix::identity(N, N),
            entrance_channel: 0,
        };
        let l = particles.internals.get_value("l") as usize;

        let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic_states);
        let scattering_length = s_matrix.get_scattering_length(0);

        let extraction = start.elapsed() - preparation - propagation;

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} ms", propagation.as_millis());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }
}
