use quantum::units::{distance_units::Distance, energy_units::Energy, Au, Unit};

use super::{composite_potential::CompositePotential, dispersion_potential::DispersionPotential};

/// Creates a Lennard-Jones potential with given parameters
pub fn create_lj<U: Unit, V: Unit>(d6: Energy<U>, r6: Distance<V>) -> CompositePotential<DispersionPotential> {
    let mut potential = CompositePotential::new();

    let d6 = d6.to_au();
    let c12 =  d6 * r6.to_au().powi(12);
    let c6 = -2.0 * d6 * r6.to_au().powi(6);

    potential.add_potential(DispersionPotential::new(c12, Au, Au, -12));
    potential.add_potential(DispersionPotential::new(c6, Au, Au, -6));

    potential
}
