use quantum::{
    particle_factory::create_atom, particles::Particles, units::{energy_units::{Kelvin, Energy}, Au},
};
use scattering_solver::{
    asymptotic_states::AsymptoticStates,
    boundary::{Boundary, Direction},
    collision_params::CollisionParams,
    defaults::MultiDefaults,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    observables::{observable_extractor::ObservableExtractor, s_matrix::HasSMatrix},
    potentials::{
        coupling_factory::couple_neighbors, gaussian_coupling::GaussianCoupling,
        potential::Potential, potential_factory::create_lj,
    },
    types::FMatrix,
};

#[test]
fn test_two_channel() {
    let particle1 = create_atom("Li6").unwrap();
    let particle2 = create_atom("Li7").unwrap();
    let energy = Energy::new(1e-7, Kelvin);

    let mut particles = Particles::new_pair(particle1, particle2, energy);
    particles.internals.insert_value("l", 0.0);

    let potential_lj1 = create_lj(Energy::new(0.002, Au), 9.0, Energy::new(0.0, Au));
    let potential_lj2 = create_lj(Energy::new(0.0021, Au), 8.9, Energy::new(1.0, Kelvin));

    let coupling = GaussianCoupling::new(Energy::new(10.0, Kelvin), 11.0, 2.0);
    let coupled_potential = couple_neighbors(vec![coupling], [potential_lj1, potential_lj2]);

    let collision_params = CollisionParams::new(particles, coupled_potential);

    let mut numerov = RatioNumerov::new(&collision_params, 1.0);
    numerov.prepare(&Boundary::new(6.5, Direction::Outwards, MultiDefaults::boundary()));
    numerov.propagate_to(1000.0);
    let result = numerov.result();

    let mut observable_extractor = ObservableExtractor::new(&collision_params, result);
    let asymptotic = collision_params.potential.asymptotic_value();
    let asymptotic = asymptotic.diagonal().iter().map(|e| Energy::new(*e, Au)).collect();

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
