use std::rc::Rc;

use quantum::{units::energy_units::EnergyUnit, particle_factory::create_atom, particles::Particles,};
use scattering_solver::{
    asymptotic_states::AsymptoticStates,
    boundary::Boundary,
    defaults::MultiDefaults,
    numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
    observables::{observable_extractor::ObservableExtractor, s_matrix::HasSMatrix},
    potentials::{
        coupling_factory::couple_neighbors, gaussian_coupling::GaussianCoupling,
        potential::{MultiPotential, OnePotential}, potential_factory::create_lj,
    }, types::FMatrix,
};

#[test]
fn test_two_channel() {
    let particle1 = create_atom("Li6").unwrap();
    let particle2 = create_atom("Li7").unwrap();
    let energy = EnergyUnit::Kelvin.to_au(1e-7);

    let mut particles = Particles::new_pair(particle1, particle2, energy);
    particles.internals.insert_value("l", 0.0);

    let potential_lj1: Box<dyn OnePotential + 'static> = Box::new(create_lj(0.002, 9.0, 0.0));
    let potential_lj2: Box<dyn OnePotential + 'static> = Box::new(create_lj(0.0021, 8.9, EnergyUnit::Kelvin.to_au(1.0)));

    let coupling = Box::new(GaussianCoupling::new(EnergyUnit::Kelvin.to_au(10.0), 11.0, 2.0));
    let potential = couple_neighbors(vec![coupling], vec![potential_lj1, potential_lj2]);
    let dim = potential.dim();

    let rc_potential: Rc<dyn MultiPotential + 'static> = Rc::new(potential);
    let rc_particles = Rc::new(particles);

    let mut numerov = RatioNumerov::new_multi(rc_particles.clone(), rc_potential.clone(), 1.0);
    numerov.prepare(&Boundary::new(6.5, MultiDefaults::boundary(dim)));
    numerov.propagate_to(1000.0);
    let result = numerov.result();

    let mut observable_extractor = ObservableExtractor::new(rc_particles.clone(), rc_potential.clone(), result);
    let asymptotic = rc_potential.asymptotic_value();

    let asymptotic_states = AsymptoticStates {
        energies: vec![asymptotic[(0, 0)], asymptotic[(1, 1)]],
        eigenvectors: FMatrix::identity(dim, dim),
        entrance_channel: 0,
    };
    let l = rc_particles.internals.get_value("l") as usize;

    let s_matrix = observable_extractor.calculate_s_matrix(l, &asymptotic_states);
    let scattering_length = s_matrix.get_scattering_length(0);

    assert!(scattering_length.re < -13.0);
    assert!(scattering_length.re > -13.2);
}