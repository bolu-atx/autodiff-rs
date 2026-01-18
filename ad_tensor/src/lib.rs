//! # ad_tensor - Tensor Autodiff with Pluggable Backends
//!
//! This crate provides a tensor-based reverse-mode automatic differentiation engine
//! with support for pluggable compute backends (CPU, Metal, CUDA).
//!
//! ## Overview
//!
//! The core abstractions are:
//! - [`Shape`] and [`Strides`] - Tensor shape and memory layout
//! - [`TensorData`] - Trait for tensor storage
//! - [`Backend`] - Trait for compute backends implementing tensor operations
//! - [`Tensor`] - Reference-counted handle to a computation graph node
//! - [`Gradients`] - Result of backward pass
//!
//! ## Example
//!
//! ```ignore
//! use ad_tensor::prelude::*;
//! use ad_backend_cpu::CpuBackend;
//!
//! type T = Tensor<CpuBackend>;
//!
//! // Create variables
//! let x = T::var("x", CpuBackend::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3])));
//! let y = T::var("y", CpuBackend::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3])));
//!
//! // Build computation: z = x * y + x.exp()
//! let z = &x * &y + x.exp();
//!
//! // Compute gradients
//! let grads = z.sum(None, false).backward();
//! let dx = grads.wrt(&x).unwrap();
//! ```

pub mod backward;
pub mod backend;
pub mod node;
pub mod shape;
pub mod tensor;

pub use backward::Gradients;
pub use backend::Backend;
pub use node::{NodeId, Tensor, TensorOp};
pub use shape::{Shape, Strides};
pub use tensor::TensorData;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::backward::Gradients;
    pub use crate::backend::Backend;
    pub use crate::node::{NodeId, Tensor, TensorOp};
    pub use crate::shape::{Shape, Strides};
    pub use crate::tensor::TensorData;
}
