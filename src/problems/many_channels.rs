use std::{collections::VecDeque, time::Instant};

use nalgebra::{Const, Dyn};
use quantum::{
    particle_factory::create_atom, particles::Particles, problem_selector::ProblemSelector, saving::save_data, units::energy_units::{CmInv, Energy, Kelvin}, utility::{linspace, unit_linspace}
};
use scattering_solver::{
    asymptotic_states::{AsymptoticStates, DynAsymptoticStates},
    boundary::{Boundary, Direction},
    collision_params::CollisionParams,
    defaults::{DynDefaults, MultiDefaults},
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    observables::{bound_states::MultiBounds, observable_extractor::ObservableExtractor, s_matrix::HasSMatrix},
    potentials::{
        coupled_potential::CoupledPotential, gaussian_coupling::GaussianCoupling, multi_coupling::MultiCoupling, multi_diag_potential::MultiDiagPotential, potential::Potential, potential_factory::create_lj
    },
    types::{DFMatrix, FMatrix},
};

pub struct ManyChannels {}

impl ProblemSelector for ManyChannels {
    const NAME: &'static str = "large number of channels";

    fn list() -> Vec<&'static str> {
        vec![
            "scattering length static size",
            "scattering length dynamic size",
            "bound states"
        ]
    }

    fn methods(number: &str, _args: &mut VecDeque<String>) {
        match number {
            "0" => Self::scattering_length(),
            "1" => Self::scattering_length_dyn(),
            "2" => Self::bound_states(),
            _ => println!("No method found for number {}", number),
        }
    }
}

impl ManyChannels {
    fn create_collision_params<const N: usize>(
    ) -> CollisionParams<impl Potential<Space = FMatrix<N>>> {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.internals.insert_value("l", 0.0);

        let wells: [f64; N] = linspace(0.0019, 0.0022, N).try_into().unwrap();
        let potentials =
            wells.map(|well| create_lj(Energy(well, Kelvin), 9.0, Energy(well / 0.0019 - 1.0, Kelvin)));

        let couplings_strength = linspace(5.0, 15.0, N - 1);
        let couplings = couplings_strength
            .iter()
            .map(|c| GaussianCoupling::new(Energy(*c, Kelvin), 11.0, 2.0))
            .collect();

        let potential = MultiDiagPotential::new(potentials);
        let coupling = MultiCoupling::new_neighboring(Const::<N>, couplings);
        let coupled_potential = CoupledPotential::new(potential, coupling);
        CollisionParams::new(particles, coupled_potential)
    }

    fn create_dyn_collision_params(size: usize) -> CollisionParams<impl Potential<Space = DFMatrix>> {
        let particle1 = create_atom("Li6").unwrap();
        let particle2 = create_atom("Li7").unwrap();
        let energy = Energy(1e-7, Kelvin);

        let mut particles = Particles::new_pair(particle1, particle2, energy);
        particles.internals.insert_value("l", 0.0);

        let wells = linspace(0.0019, 0.0022, size);
        let potentials = wells
            .into_iter()
            .map(|well| create_lj(Energy(well, Kelvin), 9.0, Energy(well / 0.0019 - 1.0, Kelvin)))
            .collect();

        let couplings_strength = linspace(5.0, 15.0, size - 1);
        let couplings = couplings_strength
            .iter()
            .map(|c| GaussianCoupling::new(Energy(*c, Kelvin), 11.0, 2.0))
            .collect();

        let potential = MultiDiagPotential::from_vec(potentials);
        let coupling = MultiCoupling::new_neighboring(Dyn(size), couplings);
        let coupled_potential = CoupledPotential::new(potential, coupling);
        CollisionParams::new(particles, coupled_potential)
    }

    fn scattering_length() {
        println!("Calculating scattering length...");
        let start = Instant::now();

        const N: usize = 50;
        let collision_params = Self::create_collision_params::<N>();
        let mut numerov = RatioNumerov::new(&collision_params);

        let energies = linspace(0.0019, 0.0022, N)
            .iter()
            .map(|well| Energy(well / 0.0019 - 1.0, Kelvin))
            .collect();

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, Direction::Outwards, MultiDefaults::boundary()));
        numerov.propagate_to(1000.0);
        let result = numerov.result();

        let propagation = start.elapsed() - preparation;

        let mut observable_extractor = ObservableExtractor::new(&collision_params, result);

        let asymptotic_states = AsymptoticStates::new(energies, FMatrix::<N>::identity(), 0);
        let l = collision_params.particles.internals.get_value("l") as usize;

        let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic_states);
        let scattering_length = s_matrix.get_scattering_length(0);

        let extraction = start.elapsed() - preparation - propagation;

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} ms", propagation.as_millis());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }

    fn scattering_length_dyn() {
        println!("Calculating scattering length...");
        let start = Instant::now();

        const N: usize = 50;
        let collision_params = Self::create_dyn_collision_params(N);
        let mut numerov = RatioNumerov::new_dyn(&collision_params);

        let energies = linspace(0.0019, 0.0022, N)
            .iter()
            .map(|well| Energy(well / 0.0019 - 1.0, Kelvin))
            .collect();

        let preparation = start.elapsed();

        numerov.prepare(&Boundary::new(6.5, Direction::Outwards, DynDefaults::boundary(N)));
        numerov.propagate_to(1000.0);
        let result = numerov.result();

        let propagation = start.elapsed() - preparation;

        let mut observable_extractor = ObservableExtractor::new(&collision_params, result);

        let asymptotic_states = DynAsymptoticStates::new(energies, DFMatrix::identity(N, N), 0);
        let l = collision_params.particles.internals.get_value("l") as usize;

        let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic_states);
        let scattering_length = s_matrix.get_scattering_length(0);

        let extraction = start.elapsed() - preparation - propagation;

        println!("Preparation time: {:?} μs", preparation.as_micros());
        println!("Propagation time: {:?} ms", propagation.as_millis());
        println!("Extraction time: {:?} μs", extraction.as_micros());
        println!("Scattering length: {:.2e} bohr", scattering_length);
    }

    fn bound_states() {
        println!("Calculating bound states...");
        const N: usize = 50;
        let mut collision_params = Self::create_dyn_collision_params(N);

        let mut bounds = MultiBounds::new(&mut collision_params, (6.5, 1000.0));
        let energies = unit_linspace(Energy(-2.0, CmInv), Energy(0.0, CmInv), 100);

        let start = Instant::now();
        let (bound_diffs, node_counts) =  bounds.bound_diff_dependence(&energies);
        let elapsed = start.elapsed();
        println!("Elapsed time: {:?}", elapsed.as_secs_f64());

        let energies = energies.iter().map(|e| e.value()).collect();
        let node_counts = node_counts.into_iter().map(|n| n as f64).collect();
        let header = vec![
            "energy",
            "bound difference",
            "node count",
        ];
        let data = vec![energies, bound_diffs, node_counts];

        save_data("many_chan", "bound_diffs", header, data).unwrap();

        // let bound_states = vec![0, 1, 3, -1, -2, -5];
        // for n in bound_states {
        //     let bound_energy = bounds.n_bound_energy(n, Energy(0.1, CmInv));
        //     println!("n = {}, bound energy: {:.4e} cm^-1", n, bound_energy.to(CmInv).value());

        //     let (rs, wave) = bounds.bound_wave(Sampling::Variable(1000));
            
        //     let header = vec!["position", "wave function"];
        //     let data = vec![rs, wave];
        //     save_data("two_chan", &format!("bound_wave_{}", n), header, data).unwrap();
        // }
    }
}
