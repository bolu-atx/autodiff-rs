//! # ad_nn - Neural Network Layers for ad_tensor
//!
//! This crate provides neural network building blocks on top of the ad_tensor autodiff engine:
//!
//! - **Layers**: Linear (fully connected)
//! - **Activations**: ReLU, Sigmoid, Tanh, Softmax, Log-Softmax
//! - **Losses**: MSE, Binary Cross-Entropy, Cross-Entropy
//! - **Optimizers**: SGD (with momentum), Adam
//!
//! ## Example: Training a Simple MLP
//!
//! ```ignore
//! use ad_nn::{Linear, relu, mse_loss, SGD};
//! use ad_backend_cpu::CpuBackend;
//! use ad_tensor::prelude::*;
//!
//! // Create a simple 2-layer MLP
//! let mut layer1 = Linear::new(2, 4, true);
//! let mut layer2 = Linear::new(4, 1, true);
//!
//! let mut opt = SGD::new(0.01);
//!
//! // Training loop
//! for _ in 0..100 {
//!     let x = Tensor::var("x", CpuBackend::from_vec(vec![1.0, 2.0], Shape::new(vec![1, 2])));
//!     let target = Tensor::constant(CpuBackend::scalar(5.0));
//!
//!     // Forward pass
//!     let h = relu(&layer1.forward(&x));
//!     let pred = layer2.forward(&h);
//!
//!     // Compute loss
//!     let loss = mse_loss(&pred, &target);
//!
//!     // Backward pass
//!     let grads = loss.backward();
//!
//!     // Update parameters
//!     if let Some(g) = grads.wrt(&layer1.weight) {
//!         opt.step(&mut layer1.weight, g);
//!     }
//!     // ... update other parameters
//! }
//! ```

pub mod activations;
pub mod layers;
pub mod loss;
pub mod optim;

// Re-exports for convenience
pub use activations::{relu, sigmoid, softmax, log_softmax, tanh};
pub use layers::Linear;
pub use loss::{mse_loss, binary_cross_entropy_with_logits, soft_cross_entropy_loss, cross_entropy_loss};
pub use optim::{SGD, Adam};
