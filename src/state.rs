use std::collections::HashMap;

pub struct State {
    dim: usize,
    values: Vec<&'static str>,
    indexes: HashMap<&'static str, usize>,
}

impl State {
    pub fn new(dim: usize, values: Vec<&'static str>) -> Self {
        let mut indexes = HashMap::new();
        for (i, value) in values.iter().enumerate() {
            indexes.insert(*value, i);
        }

        Self {
            dim,
            values,
            indexes,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get_index(&self, value: &str) -> usize {
        self.indexes[value]
    }

    pub fn get_values(&self) -> &Vec<&'static str> {
        &self.values
    }
}

pub struct CompositeState {
    states: Vec<State>,
    dim: usize,
    indexes: HashMap<&'static str, usize>,
}

impl CompositeState {
    pub fn new() -> Self {
        let indexes = HashMap::new();
        let dim = 1;

        Self {
            states: Vec::new(),
            dim,
            indexes,
        }
    }

    pub const fn dim(&self) -> usize {
        self.dim
    }

    pub fn add_state(&mut self, state: State) -> &mut Self {
        self.dim *= state.dim();
        self.states.push(state);
        self.update_indexes();

        self
    }

    fn update_indexes(&mut self) {
        self.indexes.clear();
    }
}
