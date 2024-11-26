pub mod potentials;
pub mod numerovs;
pub mod boundary;
pub mod observables;
pub mod utility;
// pub mod asymptotic_states;
// pub mod collision_params;
// pub mod defaults;
// pub mod state;

// #[cfg(test)]
// mod tests {
//     use quantum::units::{energy_units::Energy, Au};

//     use crate::{
//         boundary::{Boundary, Direction},
//         collision_params::CollisionParams,
//         numerovs::{propagator::Numerov, ratio_numerov::RatioNumerov},
//         potentials::{
//             multi_diag_potential::Diagonal, potential::Potential,
//             potential_factory::create_lj,
//         },
//     };

//     #[test]
//     fn test_potential() {
//         let potential = create_lj(Energy(0.0002, Au), 8.0, Energy(0.0, Au));
//         assert_eq!(potential.value(&8.0), -0.0002);

//         let multi_potential =
//             Diagonal::new([potential.clone(), potential.clone(), potential.clone()]);
//         let matrix = Matrix3::from_diagonal_element(potential.value(&12.0));
//         assert_eq!(multi_potential.value(&12.0), matrix);
//     }

//     #[test]
//     fn test_numerov() {
//         let potential = create_lj(Energy(0.0002, Au), 8.0, Energy(0.0, Au));

//         let atom1 = create_atom("Li6").unwrap();
//         let atom2 = create_atom("Li7").unwrap();
//         let particles = Particles::new_pair(atom1, atom2, Energy(1e-7, Kelvin));

//         let collision_params = CollisionParams::new(particles, potential);
//         let mut numerov = RatioNumerov::new(&collision_params);
//         numerov.prepare(&Boundary::new(7.0, Direction::Outwards, (1.1, 1.2)));
//         numerov.propagate_to(100.0);
//         let result = numerov.result();

//         print!("r: {:?}\n", result.r_last);
//         print!("dr: {:?}\n", result.dr);
//         print!("wave_ratio: {:?}\n", result.wave_ratio);
//     }
// }
