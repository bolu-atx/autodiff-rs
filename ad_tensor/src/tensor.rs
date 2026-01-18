//! TensorData trait - the core abstraction for tensor storage.

use crate::shape::{Shape, Strides};

/// Core trait for tensor data storage.
/// Backends implement this to provide the actual tensor operations.
pub trait TensorData: Clone + Send + Sync + 'static {
    /// Get the shape of this tensor.
    fn shape(&self) -> &Shape;

    /// Get the strides of this tensor.
    fn strides(&self) -> &Strides;

    /// Get the total number of elements.
    fn numel(&self) -> usize {
        self.shape().numel()
    }

    /// Check if this is a scalar (0-dim tensor).
    fn is_scalar(&self) -> bool {
        self.shape().is_scalar()
    }

    /// Check if data is contiguous in memory.
    fn is_contiguous(&self) -> bool {
        self.strides() == &self.shape().contiguous_strides()
    }

    /// Get data as a contiguous f32 slice.
    fn as_slice(&self) -> &[f32];

    /// Get mutable data as a contiguous f32 slice.
    fn as_slice_mut(&mut self) -> &mut [f32];

    /// Get scalar value (panics if not scalar).
    fn scalar_value(&self) -> f32 {
        assert!(self.is_scalar(), "Expected scalar tensor");
        self.as_slice()[0]
    }
}
