use std::{collections::VecDeque, time::Instant};

use quantum::{
    particle_factory::create_atom,
    particles::Particles,
    problem_selector::ProblemSelector,
    saving::save_data,
    units::{energy_units::{Energy, Kelvin}, Au},
    utility::linspace
};
use scattering_solver::{
    asymptotic_states::AsymptoticStates,
    boundary::{Boundary, Direction},
    collision_params::CollisionParams,
    defaults::MultiDefaults,
    numerovs::{propagator::{Numerov, Sampling}, ratio_numerov::RatioNumerov},
    observables::{
        dependencies::MultiDependencies, observable_extractor::ObservableExtractor,
        s_matrix::HasSMatrix,
    },
    potentials::{
        coupled_potential::CoupledPotential, gaussian_coupling::GaussianCoupling, multi_coupling::MultiCoupling, multi_diag_potential::MultiDiagPotential, potential::Potential, potential_factory::create_lj
    },
    types::FMatrix,
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
    fn create_collision_params() -> CollisionParams<impl Potential<Space = FMatrix<2>> + Send + Sync + Clone> {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.internals.insert_value("l", 0.0);

        let potential_lj1 = create_lj(Energy(0.002, Au), 9.0, Energy(0.0, Au));
        let potential_lj2 = create_lj(Energy(0.0021, Au), 8.9, Energy(1.0, Kelvin));

        let coupling = GaussianCoupling::new(Energy(10.0, Kelvin), 11.0, 2.0);

        let potential = MultiDiagPotential::new([potential_lj1, potential_lj2]);
        let coupling = MultiCoupling::new_neighboring(vec![coupling]);
        let coupled_potential = CoupledPotential::new(potential, coupling);

        CollisionParams::new(particles, coupled_potential)
    }

    fn wave_function() {
        println!("Calculating wave function...");
        let start = Instant::now();

        let collision_params = Self::create_collision_params();
        let mut numerov = RatioNumerov::new(&collision_params);

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, Direction::Outwards, MultiDefaults::boundary()));

        let (rs, waves) = numerov.propagate_values(100.0, MultiDefaults::init_wave(), Sampling::Uniform(1000));
        let propagation = start.elapsed() - preparation;

        let header = vec!["position", "channel 1", "channel 2"];
        let chan_1 = waves.iter().map(|wave| wave[(0, 0)]).collect();
        let chan_2 = waves.iter().map(|wave| wave[(0, 1)]).collect();
        let data = vec![rs, chan_1, chan_2];

        save_data("two_chan", "wave_function", header, data).unwrap();

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} μs", propagation.as_micros());
    }

    fn scattering_length() {
        println!("Calculating scattering length...");
        let start = Instant::now();

        let collision_params = Self::create_collision_params();
        let mut numerov = RatioNumerov::new(&collision_params);

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, Direction::Outwards, MultiDefaults::boundary()));
        numerov.propagate_to(1000.0);
        let result = numerov.result();

        let propagation = start.elapsed() - preparation;

        let mut observable_extractor = ObservableExtractor::new(&collision_params, result);
        let asymptotic = collision_params.potential.asymptotic_value();
        let asymptotic = asymptotic.diagonal().iter().map(|e| Energy(*e, Au)).collect();

        let asymptotic_states = AsymptoticStates::new(
            asymptotic, 
            FMatrix::<2>::identity(), 
            0
        );
        let l = collision_params.particles.internals.get_value("l") as usize;

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
        
        let collision_params = Self::create_collision_params();
        
        let scalings = linspace(0.8, 1.2, 1000);
        fn change_function(scaling: &f64, params: &mut CollisionParams<impl Potential>) {
            params.particles.scale_red_mass(*scaling);
        }
        
        let asymptotic = collision_params.potential.asymptotic_value();
        let asymptotic = asymptotic.diagonal().iter().map(|e| Energy(*e, Au)).collect();

        let asymptotic_states = AsymptoticStates::new(
            asymptotic, 
            FMatrix::<2>::identity(), 
            0
        );
        
        let start = Instant::now();
        let scatterings = MultiDependencies::params_change_par(
            &scalings,
            change_function,
            collision_params,
            Boundary::new(6.5, Direction::Outwards, MultiDefaults::boundary()),
            asymptotic_states,
            0,
            1e3,
        );
        let propagation = start.elapsed();
        println!("Propagation time: {:?} ms", propagation.as_millis());

        let header = vec![
            "mass scale factor",
            "scattering length real",
            "scattering length imag",
        ];
        let scat_re = scatterings.iter().map(|s| s.re).collect();
        let scat_im = scatterings.iter().map(|s| s.im).collect();
        let data = vec![scalings, scat_re, scat_im];

        save_data("two_chan", "mass_scaling", header, data).unwrap();
    }
}
