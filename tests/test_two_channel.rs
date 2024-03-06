use nalgebra::{Const, Dyn};
use quantum::{
    particle_factory::create_atom, particles::Particles, units::{energy_units::{Kelvin, Energy}, Au},
};
use scattering_solver::{
    asymptotic_states::{AsymptoticStates, DynAsymptoticStates},
    boundary::{Boundary, Direction},
    collision_params::CollisionParams,
    defaults::{DynDefaults, MultiDefaults},
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    observables::{observable_extractor::ObservableExtractor, s_matrix::HasSMatrix},
    potentials::{
        coupled_potential::CoupledPotential, gaussian_coupling::GaussianCoupling, multi_coupling::MultiCoupling, multi_diag_potential::MultiDiagPotential, potential::Potential, potential_factory::create_lj
    },
    types::{DFMatrix, FMatrix},
};

#[test]
fn test_two_channel() {
    let particle1 = create_atom("Li6").unwrap();
    let particle2 = create_atom("Li7").unwrap();
    let energy = Energy(1e-7, Kelvin);

    let mut particles = Particles::new_pair(particle1, particle2, energy);
    particles.internals.insert_value("l", 0.0);

    let potential_lj1 = create_lj(Energy(0.002, Au), 9.0, Energy(0.0, Au));
    let potential_lj2 = create_lj(Energy(0.0021, Au), 8.9, Energy(1.0, Kelvin));

    let coupling = GaussianCoupling::new(Energy(10.0, Kelvin), 11.0, 2.0);
    
    let potential = MultiDiagPotential::new([potential_lj1, potential_lj2]);
    let coupling = MultiCoupling::new_neighboring(Const::<2>, vec![coupling]);
    let coupled_potential = CoupledPotential::new(potential, coupling);

    let collision_params = CollisionParams::new(particles, coupled_potential);

    let mut numerov = RatioNumerov::new(&collision_params);
    numerov.prepare(&Boundary::new(6.5, Direction::Outwards, MultiDefaults::boundary()));
    numerov.propagate_to(1000.0);
    let result = numerov.result();

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
    println!("scattering length: {:.8e}", scattering_length);

    assert!(scattering_length.re > -13.137721);
    assert!(scattering_length.re < -13.137720);
    assert!(scattering_length.im > -8.7271219e-13);
    assert!(scattering_length.im < -8.7271218e-13);
}

#[test]
fn test_dyn_two_channel() {
    let particle1 = create_atom("Li6").unwrap();
    let particle2 = create_atom("Li7").unwrap();
    let energy = Energy(1e-7, Kelvin);

    let size = 2;

    let mut particles = Particles::new_pair(particle1, particle2, energy);
    particles.internals.insert_value("l", 0.0);

    let potential_lj1 = create_lj(Energy(0.002, Au), 9.0, Energy(0.0, Au));
    let potential_lj2 = create_lj(Energy(0.0021, Au), 8.9, Energy(1.0, Kelvin));

    let coupling = GaussianCoupling::new(Energy(10.0, Kelvin), 11.0, 2.0);
    
    let potential = MultiDiagPotential::from_vec(vec![potential_lj1, potential_lj2]);
    let coupling = MultiCoupling::new_neighboring(Dyn(size), vec![coupling]);
    let coupled_potential = CoupledPotential::new(potential, coupling);

    let collision_params = CollisionParams::new(particles, coupled_potential);

    let mut numerov = RatioNumerov::new_dyn(&collision_params);
    numerov.prepare(&Boundary::new(6.5, Direction::Outwards, DynDefaults::boundary(size)));
    numerov.propagate_to(1000.0);
    let result = numerov.result();

    let mut observable_extractor = ObservableExtractor::new(&collision_params, result);
    let asymptotic = collision_params.potential.asymptotic_value();
    let asymptotic = asymptotic.diagonal().iter().map(|e| Energy(*e, Au)).collect();

    let asymptotic_states = DynAsymptoticStates::new(
        asymptotic, 
        DFMatrix::identity(2, 2), 
        0
    );
    let l = collision_params.particles.internals.get_value("l") as usize;

    let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic_states);
    let scattering_length = s_matrix.get_scattering_length(0);
    println!("scattering length: {:.8e}", scattering_length);

    assert!(scattering_length.re > -13.137721);
    assert!(scattering_length.re < -13.137720);
    assert!(scattering_length.im > -8.7271219e-13);
    assert!(scattering_length.im < -8.7271218e-13);
}