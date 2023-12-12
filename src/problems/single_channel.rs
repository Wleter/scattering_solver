use std::{collections::VecDeque, time::Instant};

use quantum::{
    particle_factory::create_atom,
    particles::Particles,
    problem_selector::ProblemSelector,
    saving::{save_param_change, save_param_change_complex},
    units::{energy_units::{Kelvin, Energy, CmInv}, Au}, utility::{linspace, unit_linspace},
};
use scattering_solver::{
    boundary::{Boundary, Direction},
    collision_params::CollisionParams,
    defaults::SingleDefaults,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    observables::{
        dependencies::SingleDependencies, observable_extractor::ObservableExtractor,
        s_matrix::HasSMatrix, bound_states::SingleBounds,
    },
    potentials::{potential::Potential, potential_factory::create_lj},
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
            "bound states",
        ]
    }

    fn methods(number: &str, _args: &mut VecDeque<String>) {
        match number {
            "0" => Self::wave_function(),
            "1" => Self::scattering_length(),
            "2" => Self::propagation_distance(),
            "3" => Self::mass_scaling(),
            "4" => Self::bound_states(),
            _ => println!("No method found for number {}", number),
        }
    }
}

impl SingleChannel {
    fn create_collision_params() -> CollisionParams<impl Potential<Space = f64>> {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy::new(1e-7, Kelvin);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.internals.insert_value("l", 0.0);

        let potential = create_lj(Energy::new(0.002, Au), 9.0, Energy::new(0.0, Au));

        CollisionParams::new(particles, potential)
    }

    fn wave_function() {
        println!("Calculating wave function...");
        let start = Instant::now();

        let collision_params = Self::create_collision_params();
        let mut numerov = RatioNumerov::new(&collision_params, 1.0);

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, Direction::Outwards, SingleDefaults::boundary()));
        let (rs, waves) = numerov.propagate_values(100.0, SingleDefaults::init_wave());
        let propagation = start.elapsed() - preparation;

        let potential_values: Vec<f64> = rs
            .iter()
            .map(|r| numerov.collision_params.potential.value(r))
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

        let collision_params = Self::create_collision_params();
        let mut numerov = RatioNumerov::new(&collision_params, 1.0);

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, Direction::Outwards, SingleDefaults::boundary()));
        numerov.propagate_to(1000.0);
        let result = numerov.result();

        let propagation = start.elapsed() - preparation;

        let mut observable_extractor = ObservableExtractor::new(&collision_params, result);

        let asymptotic = collision_params.potential.asymptotic_value();
        let l = collision_params.particles.internals.get_value("l") as usize;

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

        let collision_params = Self::create_collision_params();

        let distances = linspace(100.0, 1e4, 1000);
        let (rs, scatterings) = SingleDependencies::propagation_distance(
            distances,
            collision_params,
            Boundary::new(6.5, Direction::Outwards, SingleDefaults::boundary()),
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

        let collision_params = Self::create_collision_params();

        let scalings = linspace(0.8, 1.2, 1000);
        fn change_function(scaling: &f64, params: &mut CollisionParams<impl Potential>) {
            params.particles.scale_red_mass(*scaling);
        }

        let scatterings = SingleDependencies::params_change_par(
            &scalings,
            change_function,
            collision_params,
            Boundary::new(6.5, Direction::Outwards, SingleDefaults::boundary()),
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

    fn bound_states() {
        println!("Calculating bound states...");

        let collision_params = Self::create_collision_params();

        let energies = unit_linspace(Energy::new(-200.0, CmInv), Energy::new(0.0, CmInv), 5000);
        let (bound_differences, node_counts) =  SingleBounds::bound_diff_dependence(collision_params, &energies, 6.5, 70.0);
        let zipped = bound_differences
            .iter()
            .zip(node_counts.iter())
            .map(|(diff, count)| vec![*diff, *count as f64])
            .collect();
        let energies = energies.iter().map(|e| e.value).collect();

        let header = vec![
            "energy",
            "bound difference",
            "node count",
        ];
        save_param_change("single_chan/bound_diffs", energies, zipped, header).unwrap();

        let mut collision_params = Self::create_collision_params();
        let bound_energy = SingleBounds::bound_energy(&mut collision_params, -2, 6.5, 70.0, Energy::new(0.1, CmInv));
        println!("Bound energy: {:.4e} cm^-1", bound_energy.to(CmInv).value);
    }
}
