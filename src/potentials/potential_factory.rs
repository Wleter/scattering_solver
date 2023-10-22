use super::{composite_potential::CompositePotential, dispersion_potential::DispersionPotential};

/// Creates a Lennard-Jones potential with given parameters
pub fn create_lj(d6: f64, r6: f64, v0: f64) -> CompositePotential<DispersionPotential> {
    let mut potential = CompositePotential::new();

    potential.add_potential(DispersionPotential::new(d6 / r6.powi(12), -12, v0));
    potential.add_potential(DispersionPotential::new(-2.0 * d6 / r6.powi(6), -6, v0));

    potential
}
