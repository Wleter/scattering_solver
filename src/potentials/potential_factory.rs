use quantum::units::{Unit, energy_units::Energy, Au};

use super::{composite_potential::CompositePotential, dispersion_potential::DispersionPotential};

/// Creates a Lennard-Jones potential with given parameters
pub fn create_lj<U: Unit, V: Unit>(d6: Energy<U>, r6: f64, v0: Energy<V>) -> CompositePotential<DispersionPotential> {
    let mut potential = CompositePotential::new();

    let d6 = d6.to_au();
    let c12 = Energy(d6 * r6.powi(12), Au);
    let c6 = Energy(-2.0 * d6 * r6.powi(6), Au);
    let v0 = v0.to_au();

    potential.add_potential(DispersionPotential::new(c12, -12, v0));
    potential.add_potential(DispersionPotential::new(c6, -6, v0));

    potential
}