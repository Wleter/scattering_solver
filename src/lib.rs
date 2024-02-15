pub mod asymptotic_states;
pub mod boundary;
pub mod collision_params;
pub mod defaults;
pub mod numerovs;
pub mod observables;
pub mod potentials;
pub mod state;
pub mod types;

extern crate nalgebra;
pub extern crate quantum;

#[cfg(test)]
mod tests {
    use nalgebra::Matrix3;
    use quantum::{
        particle_factory::create_atom, particles::Particles, units::{energy_units::{Kelvin, Energy}, Au},
    };

    use crate::{
        boundary::{Boundary, Direction},
        numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
        potentials::{
            multi_diag_potential::MultiDiagPotential,
            potential_factory::create_lj,
        },
    };

    #[test]
    fn test_potential() {
        let potential = create_lj(Energy(0.0002, Au), 8.0);
        assert_eq!(potential.value(&8.0), -0.0002);

        let multi_potential =
            MultiDiagPotential::new([potential.clone(), potential.clone(), potential.clone()]);
        let matrix = Matrix3::from_diagonal_element(potential.value(&12.0));
        assert_eq!(multi_potential.value(&12.0), matrix);
    }

    #[test]
    fn test_numerov() {
        let potential = create_lj(Energy(0.0002, Au), 8.0);

        let atom1 = create_atom("Li6").unwrap();
        let atom2 = create_atom("Li7").unwrap();
        let particles = Particles::new_pair(atom1, atom2, Energy(1e-7, Kelvin));

        let collision_params = CollisionParams::new(particles, potential);
        let mut numerov = RatioNumerov::new(&collision_params, 1.0);
        numerov.prepare(&Boundary::new(7.0, Direction::Outwards, (1.1, 1.2)));
        numerov.propagate_to(100.0);
        let result = numerov.result();

        print!("r: {:?}\n", result.r_last);
        print!("dr: {:?}\n", result.dr);
        print!("wave_ratio: {:?}\n", result.wave_ratio);
    }
}
