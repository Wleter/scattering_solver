use std::{collections::VecDeque, time::Instant, rc::Rc};

use quantum::{
    particle_factory::create_atom,
    particles::Particles,
    problem_selector::ProblemSelector,
    saving::{save_param_change, save_param_change_complex},
    units::energy_units::EnergyUnit,
};
use scattering_solver::{
    boundary::Boundary,
    defaults::SingleDefaults,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    observables::{
        dependencies::SingleDependencies, observable_extractor::ObservableExtractor, s_matrix::HasSMatrix,
    },
    potentials::{potential::OnePotential, potential_factory::create_lj, composite_potential::CompositePotential},
    utility::linspace,
};

pub struct SingleChannel {}

impl ProblemSelector for SingleChannel {
    const NAME: &'static str = "single channel";

    fn list() -> Vec<&'static str> {
        vec![
            "wave function",
            "scattering length",
            "propagation distance",
            "mass scaling",
        ]
    }

    fn methods(number: &str, _args: &mut VecDeque<String>) {
        match number {
            "0" => Self::wave_function(),
            "1" => Self::scattering_length(),
            "2" => Self::propagation_distance(),
            "3" => Self::mass_scaling(),
            _ => println!("No method found for number {}", number),
        }
    }
}

impl SingleChannel {
    fn create_collision_params() -> (Particles, CompositePotential) {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = EnergyUnit::Kelvin.to_au(1e-7);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.internals.insert_value("l", 0.0);

        let potential = create_lj(0.002, 9.0, 0.0);

        (particles, potential)
    }

    fn wave_function() {
        println!("Calculating wave function...");
        let start = Instant::now();

        let (particles, potential) = Self::create_collision_params();
        let particles = Rc::new(particles);
        let potential: Rc<dyn OnePotential + 'static> = Rc::new(potential);
        
        let mut numerov = RatioNumerov::new_single(particles, potential, 1.0);
        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, SingleDefaults::boundary()));
        let (rs, waves) = numerov.propagate_values(100.0, SingleDefaults::init_wave());
        let propagation = start.elapsed() - preparation;

        let potential_values: Vec<f64> = rs
            .iter()
            .map(|r| numerov.potential.value(r))
            .collect();

        let header = vec!["position", "wave function", "potential"];
        let data = waves
            .iter()
            .zip(potential_values.iter())
            .map(|(wave, v)| vec![*wave, *v])
            .collect();

        save_param_change("single_chan/wave_function", rs, data, header).unwrap();

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
    }

    fn scattering_length() {
        println!("Calculating scattering length...");
        let start = Instant::now();

        let (particles, potential) = Self::create_collision_params();
        let particles = Rc::new(particles);
        let potential: Rc<dyn OnePotential + 'static> = Rc::new(potential);

        let mut numerov = RatioNumerov::new_single(particles.clone(), potential.clone(), 1.0);

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, SingleDefaults::boundary()));
        numerov.propagate_to(1000.0);
        let result = numerov.result();

        let propagation = start.elapsed() - preparation;

        let mut observable_extractor = ObservableExtractor::new(particles.clone(), potential.clone(), result);

        let asymptotic = potential.asymptotic_value();
        let l = particles.internals.get_value("l") as usize;

        let s_matrix = observable_extractor.calculate_s_matrix(l, asymptotic);
        let scattering_length = s_matrix.get_scattering_length(0);

        let extraction = start.elapsed() - preparation - propagation;

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }

    fn propagation_distance() {
        println!("Calculating scattering length distance dependence...");

        let (particles, potential) = Self::create_collision_params();

        let distances = linspace(100.0, 1e4, 1000);
        let (rs, scatterings) = SingleDependencies::propagation_distance(
            distances,
            particles,
            potential,
            Boundary::new(6.5, SingleDefaults::boundary()),
        );

        let header = vec![
            "distance",
            "scattering length real",
            "scattering length imag",
        ];

        save_param_change_complex("single_chan/propagation_distance", rs, scatterings, header)
            .unwrap();
    }

    fn mass_scaling() {
        println!("Calculating scattering length vs mass scaling...");

        let (particles, potential) = Self::create_collision_params();

        let scalings = linspace(0.8, 1.2, 1000);
        fn change_function(
            scaling: &f64,
            particles: &mut Particles,
            _potential: &mut impl OnePotential,
        ) {
            particles.scale_red_mass(*scaling);
        }

        let scatterings = SingleDependencies::params_change(
            &scalings,
            change_function,
            particles,
            potential,
            Boundary::new(6.5, SingleDefaults::boundary()),
            1e4,
        );

        let header = vec![
            "mass scale factor",
            "scattering length real",
            "scattering length imag",
        ];

        save_param_change_complex("single_chan/mass_scaling", scalings, scatterings, header)
            .unwrap();
    }
}
