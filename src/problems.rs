use std::collections::VecDeque;

use quantum::problem_selector::ProblemSelector;

use self::{single_channel::SingleChannel, two_channel::TwoChannel, many_channels::ManyChannels};

pub mod single_channel;
pub mod two_channel;
pub mod many_channels;

pub struct Problems {}

impl ProblemSelector for Problems {
    const NAME: &'static str = "test";

    fn list() -> Vec<&'static str> {
        vec!["single channel", "two channel"]
    }

    fn methods(number: &str, args: &mut VecDeque<String>) {
        match number {
            "0" => SingleChannel::select(args),
            "1" => TwoChannel::select(args),
            "2" => ManyChannels::select(args),
            _ => println!("No method found for number {}", number),
        }
    }
}
