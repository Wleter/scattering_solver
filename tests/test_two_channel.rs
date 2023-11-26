use quantum::{
    particle_factory::create_atom, particles::Particles, units::energy_units::EnergyUnit,
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
    let energy = EnergyUnit::Kelvin.to_au(1e-7);

    let mut particles = Particles::new_pair(particle1, particle2, energy);
    particles.internals.insert_value("l", 0.0);

    let potential_lj1 = create_lj(0.002, 9.0, 0.0);
    let potential_lj2 = create_lj(0.0021, 8.9, EnergyUnit::Kelvin.to_au(1.0));

    let coupling = GaussianCoupling::new(EnergyUnit::Kelvin.to_au(10.0), 11.0, 2.0);
    let coupled_potential = couple_neighbors(vec![coupling], [potential_lj1, potential_lj2]);

    let collision_params = CollisionParams::new(particles, coupled_potential);

    let mut numerov = RatioNumerov::new(&collision_params, 1.0);
    numerov.prepare(&Boundary::new(6.5, Direction::Outwards, MultiDefaults::boundary()));
    numerov.propagate_to(1000.0);
    let result = numerov.result();

    let mut observable_extractor = ObservableExtractor::new(&collision_params, result);
    let asymptotic = collision_params.potential.asymptotic_value();

    let asymptotic_states = AsymptoticStates {
        energies: vec![asymptotic[(0, 0)], asymptotic[(1, 1)]],
        eigenvectors: FMatrix::<2>::identity(),
        entrance_channel: 0,
    };
    let l = collision_params.particles.internals.get_value("l") as usize;

    let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic_states);
    let scattering_length = s_matrix.get_scattering_length(0);
    println!("scattering length: {:.8e}", scattering_length);

    assert!(scattering_length.re > -13.137721);
    assert!(scattering_length.re < -13.137720);
    assert!(scattering_length.im > -8.7271219e-13);
    assert!(scattering_length.im < -8.7271218e-13);
}
