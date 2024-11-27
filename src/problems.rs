mod single_channel;
use quantum::problems_impl;
use self::single_channel::SingleChannel;

#[cfg(feature = "faer")]
pub mod faer;
#[cfg(feature = "faer")]
use faer::FaerProblems;

#[cfg(feature = "nalgebra")]
pub mod nalgebra;
#[cfg(feature = "nalgebra")]
use nalgebra::NalgebraProblems;


pub struct Problems;

#[cfg(all(feature = "faer", feature = "nalgebra", feature = "ndarray"))]
problems_impl!(Problems, "test",
    "single channel" => SingleChannel::select,
    "faer problems" => FaerProblems::select,
    "nalgebra problems" => NalgebraProblems::select
);

// todo! maybe there is a better way to include only those features that are used
#[cfg(not(all(feature = "faer", feature = "nalgebra", feature = "ndarray")))]
problems_impl!(Problems, "test",
    "single channel" => SingleChannel::select,
);