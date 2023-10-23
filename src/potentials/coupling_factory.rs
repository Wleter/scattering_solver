use super::{
    coupled_potential::CoupledPotential, multi_coupling::MultiCoupling,
    multi_diag_potential::MultiDiagPotential, potential::Potential,
};

pub fn couple_neighbors<const N: usize, P, C>(
    couplings: Vec<C>,
    potentials: [P; N],
) -> CoupledPotential<N, MultiDiagPotential<N, P>, MultiCoupling<N, C>>
where
    P: Potential<Space = f64>,
    C: Potential<Space = f64>,
{
    assert!(couplings.len() + 1 == N);

    let numbered_potentials = couplings
        .into_iter()
        .enumerate()
        .map(|(i, potential)| (potential, i, i + 1))
        .collect();

    let couplings = MultiCoupling::new(numbered_potentials, true);
    let potential = MultiDiagPotential::new(potentials);

    CoupledPotential::new(potential, couplings)
}
