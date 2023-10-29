pub struct Boundary<T> {
    pub r_start: f64,
    pub start_value: T,
    pub before_value: T,
}

impl<T> Boundary<T> {
    pub fn new(r_start: f64, values: (T, T)) -> Self {
        Self {
            r_start,
            start_value: values.0,
            before_value: values.1,
        }
    }
}
