//! Optimizers for neural network training.

mod sgd;
mod adam;

pub use sgd::SGD;
pub use adam::Adam;
