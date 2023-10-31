use super::{
    coupled_potential::CoupledPotential, multi_coupling::MultiCoupling,
    multi_diag_potential::MultiDiagPotential, potential::OnePotential,
};

pub fn couple_neighbors(couplings: Vec<Box<dyn OnePotential>>, potentials: Vec<Box<dyn OnePotential>>) -> CoupledPotential
{
    assert!(couplings.len() + 1 == potentials.len());

    let numbered_potentials = couplings
        .into_iter()
        .enumerate()
        .map(|(i, potential)| (potential, i, i + 1))
        .collect();

    let couplings = MultiCoupling::new(numbered_potentials, potentials.len(), true);
    let potential = MultiDiagPotential::new(potentials);

    CoupledPotential::new(Box::new(potential), Box::new(couplings))
}
