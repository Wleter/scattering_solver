use std::{collections::VecDeque, time::Instant, rc::Rc};

use quantum::{
    particle_factory::create_atom, particles::Particles, problem_selector::ProblemSelector,
    saving::{save_param_change, save_param_change_complex}, units::energy_units::EnergyUnit,
};
use scattering_solver::{
    asymptotic_states::AsymptoticStates,
    boundary::Boundary,
    defaults::MultiDefaults,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    observables::{observable_extractor::ObservableExtractor, s_matrix::HasSMatrix, dependencies::MultiDependencies},
    potentials::{
        coupling_factory::couple_neighbors, gaussian_coupling::GaussianCoupling,
        potential_factory::create_lj, potential::MultiPotential, coupled_potential::CoupledPotential,
    },
    types::FMatrix, utility::linspace,
};

pub struct TwoChannel {}

impl ProblemSelector for TwoChannel {
    const NAME: &'static str = "two channel";

    fn list() -> Vec<&'static str> {
        vec!["wave function", "scattering length", "mass scaling"]
    }

    fn methods(number: &str, _args: &mut VecDeque<String>) {
        match number {
            "0" => Self::wave_function(),
            "1" => Self::scattering_length(),
            "2" => Self::mass_scaling(),
            _ => println!("No method found for number {}", number),
        }
    }
}

impl TwoChannel {
    fn create_collision_params() -> (Particles, CoupledPotential) {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = EnergyUnit::Kelvin.to_au(1e-7);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.internals.insert_value("l", 0.0);

        let potential_lj1 = Box::new(create_lj(0.002, 9.0, 0.0));
        let potential_lj2 = Box::new(create_lj(0.0021, 8.9, EnergyUnit::Kelvin.to_au(1.0)));

        let coupling = Box::new(GaussianCoupling::new(EnergyUnit::Kelvin.to_au(10.0), 11.0, 2.0));

        let coupled_potential = couple_neighbors(vec![coupling], vec![potential_lj1, potential_lj2]);
        (particles, coupled_potential)
    }

    fn wave_function() {
        println!("Calculating wave function...");
        let start = Instant::now();

        let (particles, potential) = Self::create_collision_params();
        let particles = Rc::new(particles);
        let potential: Rc<dyn MultiPotential + 'static> = Rc::new(potential);

        let dim = potential.dim();
        let mut numerov = RatioNumerov::new_multi(particles.clone(), potential.clone(), 1.0);

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, MultiDefaults::boundary(dim)));

        let (rs, waves) = numerov.propagate_values(100.0, MultiDefaults::init_wave(dim));
        let propagation = start.elapsed() - preparation;

        let header = vec!["position", "channel 1", "channel 2"];
        let data = waves
            .iter()
            .map(|wave| vec![wave[(0, 0)], wave[(0, 1)]])
            .collect();

        save_param_change("two_chan/wave_function", rs, data, header).unwrap();

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
    }

    fn scattering_length() {
        println!("Calculating scattering length...");
        let start = Instant::now();

        let (particles, potential) = Self::create_collision_params();
        let particles = Rc::new(particles);
        let potential: Rc<dyn MultiPotential + 'static> = Rc::new(potential);

        let dim = potential.dim();
        let mut numerov = RatioNumerov::new_multi(particles.clone(), potential.clone(), 1.0);

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, MultiDefaults::boundary(dim)));
        numerov.propagate_to(1000.0);
        let result = numerov.result();

        let propagation = start.elapsed() - preparation;

        let mut observable_extractor = ObservableExtractor::new(particles.clone(), potential.clone(), result);
        let asymptotic = potential.asymptotic_value();

        let asymptotic_states = AsymptoticStates {
            energies: vec![asymptotic[(0, 0)], asymptotic[(1, 1)]],
            eigenvectors: FMatrix::identity(2, 2),
            entrance_channel: 0,
        };
        let l = particles.internals.get_value("l") as usize;

        let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic_states);
        let scattering_length = s_matrix.get_scattering_length(0);

        let extraction = start.elapsed() - preparation - propagation;

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }

    fn mass_scaling() {
        println!("Calculating scattering length vs mass scaling...");

        let (particles, potential) = Self::create_collision_params();
        let dim = potential.dim();

        let scalings = linspace(0.8, 1.2, 1000);
        fn change_function(
            scaling: &f64,
            particles: &mut Particles,
            _potential: &mut CoupledPotential,
        ) {
            particles.scale_red_mass(*scaling);
        }

        let asymptotic = potential.asymptotic_value();
        let asymptotic_states = AsymptoticStates {
            energies: vec![asymptotic[(0, 0)], asymptotic[(1, 1)]],
            eigenvectors: FMatrix::identity(dim, dim),
            entrance_channel: 0,
        };

        let scatterings = MultiDependencies::params_change(
            &scalings,
            change_function,
            particles,
            potential,
            Boundary::new(6.5, MultiDefaults::boundary(dim)),
            asymptotic_states,
            0,
            1e3,
        );

        let header = vec![
            "mass scale factor",
            "scattering length real",
            "scattering length imag",
        ];

        save_param_change_complex("two_chan/mass_scaling", scalings, scatterings, header)
            .unwrap();
    }
}
