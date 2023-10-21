pub trait MultiStepPropagator {
    fn variable_step(&mut self);

    fn step(&mut self);

    fn half_step(&mut self);

    fn double_step(&mut self);
}