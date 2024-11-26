use many_channels::ManyChannels;
use quantum::problems_impl;

use self::single_channel::SingleChannel;
#[cfg(feature = "faer")]
pub mod two_channel;
#[cfg(feature = "faer")]
use two_channel::TwoChannel;
#[cfg(feature = "faer")]
pub mod single_channel;
#[cfg(feature = "faer")]
pub mod many_channels;

pub struct Problems {}

#[cfg(not(feature = "faer"))]
problems_impl!(Problems, "test",
    "single channel" => SingleChannel::select
);

#[cfg(feature = "faer")]
problems_impl!(Problems, "test",
    "single channel" => SingleChannel::select,
    "two channel" => TwoChannel::select,
    "many channel" => ManyChannels::select
);

