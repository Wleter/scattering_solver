use crate::{collision_params::CollisionParams, numerovs::{ratio_numerov::RatioNumerov, propagator::Numerov}, potentials::potential::Potential, boundary::{Boundary, Direction}, defaults::SingleDefaults};


pub struct SingleBounds;

impl SingleBounds {
    // pub fn bound_energy<P>(collision_params: &mut CollisionParams<P>, n_bound: usize, r_min: f64, r_max: f64, err: f64) -> f64 
    // where
    //     P: Potential<Space = f64>    
    // {
    //     let upper_bound = 
    // }

    pub fn bound_diff_dependence<P>(mut collision_params: CollisionParams<P>, energies: &[f64], r_min: f64, r_max: f64) -> (Vec<f64>, Vec<usize>)
    where
        P: Potential<Space = f64>
    {
        let mut bound_differences = Vec::new();
        let mut node_counts = Vec::new();
        for energy in energies {
            collision_params.particles.internals.insert_value("energy", *energy);
            
            let (bound_difference, node_count) = Self::bound_diffs(&collision_params, r_min, r_max);
            bound_differences.push(bound_difference);
            node_counts.push(node_count);
        }
        
        (bound_differences, node_counts)
    }

    pub fn bound_diffs<P>(collision_params: &CollisionParams<P>, r_min: f64, r_max: f64) -> (f64, usize)
    where
        P: Potential<Space = f64>, 
    {   
        let inwards_boundary = Boundary::new(r_max, Direction::Inwards, SingleDefaults::boundary());
        let outwards_boundary = Boundary::new(r_min, Direction::Outwards, SingleDefaults::boundary());

        let mut numerov = RatioNumerov::new(&collision_params, 1.0);
        numerov.prepare(&inwards_boundary);

        while *numerov.wave_last() > 1.0 {
            numerov.single_step();
        }
        let outwards_result = numerov.result();
        
        numerov.prepare(&outwards_boundary);
        let r_match = outwards_result.r_last;
        
        let mut node_count = 0;
        while numerov.r() < r_match {
            numerov.single_step();
            if *numerov.wave_last() < 0.0 {
                node_count += 1;
            }
        }
        numerov.propagate_to(r_match);
        let inwards_result = numerov.result();

        ((1.0 / inwards_result.wave_ratio - 1.0) * (1.0 / inwards_result.dr.abs()) - (outwards_result.wave_ratio - 1.0) * (1.0 / outwards_result.dr.abs()), node_count)
    }

}