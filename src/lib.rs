pub mod collision_params;
pub mod numerovs;
pub mod potentials;
pub mod types;

extern crate nalgebra;
extern crate quantum;

#[cfg(test)]
mod tests {
    #[test]
    fn collision_params_init() {
        assert_eq!(2 + 2, 4);
    }
}
