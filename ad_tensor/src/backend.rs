//! Backend trait - abstraction for compute backends (CPU, Metal, CUDA).

use crate::shape::Shape;
use crate::tensor::TensorData;

/// Backend trait for tensor computation.
/// Each backend (CPU, Metal, CUDA) implements this to provide tensor operations.
pub trait Backend: Clone + Send + Sync + 'static {
    /// The tensor type for this backend.
    type Tensor: TensorData;

    // === Creation ===

    /// Create a tensor of zeros with the given shape.
    fn zeros(shape: &Shape) -> Self::Tensor;

    /// Create a tensor of ones with the given shape.
    fn ones(shape: &Shape) -> Self::Tensor;

    /// Create a tensor from a flat data vector and shape.
    fn from_vec(data: Vec<f32>, shape: Shape) -> Self::Tensor;

    /// Create a scalar (0-dim) tensor.
    fn scalar(value: f32) -> Self::Tensor;

    /// Create a tensor filled with a constant value.
    fn full(shape: &Shape, value: f32) -> Self::Tensor;

    // === Element-wise unary operations ===

    /// Negate: -x
    fn neg(x: &Self::Tensor) -> Self::Tensor;

    /// Exponential: e^x
    fn exp(x: &Self::Tensor) -> Self::Tensor;

    /// Natural logarithm: ln(x)
    fn log(x: &Self::Tensor) -> Self::Tensor;

    /// Sine: sin(x)
    fn sin(x: &Self::Tensor) -> Self::Tensor;

    /// Cosine: cos(x)
    fn cos(x: &Self::Tensor) -> Self::Tensor;

    /// ReLU: max(0, x)
    fn relu(x: &Self::Tensor) -> Self::Tensor;

    /// Sigmoid: 1 / (1 + e^(-x))
    fn sigmoid(x: &Self::Tensor) -> Self::Tensor;

    /// Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
    fn tanh(x: &Self::Tensor) -> Self::Tensor;

    /// Square root: sqrt(x)
    fn sqrt(x: &Self::Tensor) -> Self::Tensor;

    // === Element-wise binary operations ===
    // These handle broadcasting automatically.

    /// Addition: a + b
    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Subtraction: a - b
    fn sub(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Multiplication: a * b
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Division: a / b
    fn div(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Power: a^b (element-wise)
    fn pow(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Maximum: max(a, b) element-wise
    fn maximum(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Minimum: min(a, b) element-wise
    fn minimum(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    // === Comparison (returns mask tensors with 0.0 or 1.0) ===

    /// Greater than: a > b
    fn gt(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Greater than or equal: a >= b
    fn ge(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Less than: a < b
    fn lt(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Less than or equal: a <= b
    fn le(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Equal: a == b
    fn eq(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    // === Reductions ===

    /// Sum over specified axes (None = all axes -> scalar).
    fn sum(x: &Self::Tensor, axes: Option<&[usize]>, keepdims: bool) -> Self::Tensor;

    /// Mean over specified axes (None = all axes -> scalar).
    fn mean(x: &Self::Tensor, axes: Option<&[usize]>, keepdims: bool) -> Self::Tensor;

    /// Max over specified axes (None = all axes -> scalar).
    fn max(x: &Self::Tensor, axes: Option<&[usize]>, keepdims: bool) -> Self::Tensor;

    /// Min over specified axes (None = all axes -> scalar).
    fn min(x: &Self::Tensor, axes: Option<&[usize]>, keepdims: bool) -> Self::Tensor;

    // === Linear algebra ===

    /// Matrix multiplication: a @ b
    /// Supports batched matmul: (..., M, K) @ (..., K, N) -> (..., M, N)
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    // === Shape operations ===

    /// Transpose axes. None = reverse all axes.
    fn transpose(x: &Self::Tensor, axes: Option<&[usize]>) -> Self::Tensor;

    /// Reshape to new shape (must have same numel).
    fn reshape(x: &Self::Tensor, shape: &Shape) -> Self::Tensor;

    /// Broadcast to a larger shape.
    fn broadcast_to(x: &Self::Tensor, shape: &Shape) -> Self::Tensor;

    /// Sum along broadcast axes to reduce shape back.
    /// Used during backward pass for gradient reduction.
    fn sum_to(x: &Self::Tensor, shape: &Shape) -> Self::Tensor;

    /// Remove dimensions of size 1.
    fn squeeze(x: &Self::Tensor, axes: Option<&[usize]>) -> Self::Tensor;

    /// Add dimensions of size 1.
    fn unsqueeze(x: &Self::Tensor, axis: usize) -> Self::Tensor;

    // === Gradient accumulation ===

    /// Accumulate gradient: dst += src
    fn accumulate_grad(dst: &mut Self::Tensor, src: &Self::Tensor);

    /// Clone tensor data.
    fn clone_tensor(x: &Self::Tensor) -> Self::Tensor;
}
