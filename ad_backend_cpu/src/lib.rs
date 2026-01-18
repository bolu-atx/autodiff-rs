//! CPU Backend for ad_tensor with SIMD optimizations.

use ad_tensor::prelude::*;

mod simd;

/// CPU tensor storage.
#[derive(Clone, Debug)]
pub struct CpuTensor {
    data: Vec<f32>,
    shape: Shape,
    strides: Strides,
}

impl CpuTensor {
    /// Create a new CPU tensor from data and shape.
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        let strides = shape.contiguous_strides();
        assert_eq!(
            data.len(),
            shape.numel(),
            "Data length {} doesn't match shape {:?} (numel={})",
            data.len(),
            shape,
            shape.numel()
        );
        CpuTensor { data, shape, strides }
    }

    /// Get flat index from multi-dimensional indices.
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        self.strides.index(indices)
    }

    /// Iterate over all indices in the tensor.
    pub fn indices(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        TensorIndices::new(&self.shape)
    }
}

impl TensorData for CpuTensor {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &Strides {
        &self.strides
    }

    fn as_slice(&self) -> &[f32] {
        &self.data
    }

    fn as_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

/// Iterator over all multi-dimensional indices of a tensor.
struct TensorIndices<'a> {
    shape: &'a Shape,
    current: Vec<usize>,
    done: bool,
}

impl<'a> TensorIndices<'a> {
    fn new(shape: &'a Shape) -> Self {
        let ndim = shape.ndim();
        if ndim == 0 || shape.numel() == 0 {
            TensorIndices {
                shape,
                current: vec![],
                done: ndim == 0,
            }
        } else {
            TensorIndices {
                shape,
                current: vec![0; ndim],
                done: false,
            }
        }
    }
}

impl Iterator for TensorIndices<'_> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.shape.ndim() == 0 {
            self.done = true;
            return Some(vec![]);
        }

        let result = self.current.clone();

        // Increment indices (rightmost first, like odometer)
        let mut i = self.shape.ndim() - 1;
        loop {
            self.current[i] += 1;
            if self.current[i] < self.shape.dim(i) {
                break;
            }
            self.current[i] = 0;
            if i == 0 {
                self.done = true;
                break;
            }
            i -= 1;
        }

        Some(result)
    }
}

/// CPU backend marker type.
#[derive(Clone, Copy, Debug)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Tensor = CpuTensor;

    // === Creation ===

    fn zeros(shape: &Shape) -> CpuTensor {
        CpuTensor::new(vec![0.0; shape.numel()], shape.clone())
    }

    fn ones(shape: &Shape) -> CpuTensor {
        CpuTensor::new(vec![1.0; shape.numel()], shape.clone())
    }

    fn from_vec(data: Vec<f32>, shape: Shape) -> CpuTensor {
        CpuTensor::new(data, shape)
    }

    fn scalar(value: f32) -> CpuTensor {
        CpuTensor::new(vec![value], Shape::scalar())
    }

    fn full(shape: &Shape, value: f32) -> CpuTensor {
        CpuTensor::new(vec![value; shape.numel()], shape.clone())
    }

    // === Unary element-wise ===

    fn neg(x: &CpuTensor) -> CpuTensor {
        let data: Vec<f32> = x.data.iter().map(|&v| -v).collect();
        CpuTensor::new(data, x.shape.clone())
    }

    fn exp(x: &CpuTensor) -> CpuTensor {
        let data: Vec<f32> = x.data.iter().map(|&v| v.exp()).collect();
        CpuTensor::new(data, x.shape.clone())
    }

    fn log(x: &CpuTensor) -> CpuTensor {
        let data: Vec<f32> = x.data.iter().map(|&v| v.ln()).collect();
        CpuTensor::new(data, x.shape.clone())
    }

    fn sin(x: &CpuTensor) -> CpuTensor {
        let data: Vec<f32> = x.data.iter().map(|&v| v.sin()).collect();
        CpuTensor::new(data, x.shape.clone())
    }

    fn cos(x: &CpuTensor) -> CpuTensor {
        let data: Vec<f32> = x.data.iter().map(|&v| v.cos()).collect();
        CpuTensor::new(data, x.shape.clone())
    }

    fn relu(x: &CpuTensor) -> CpuTensor {
        let data: Vec<f32> = x.data.iter().map(|&v| v.max(0.0)).collect();
        CpuTensor::new(data, x.shape.clone())
    }

    fn sigmoid(x: &CpuTensor) -> CpuTensor {
        let data: Vec<f32> = x.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        CpuTensor::new(data, x.shape.clone())
    }

    fn tanh(x: &CpuTensor) -> CpuTensor {
        let data: Vec<f32> = x.data.iter().map(|&v| v.tanh()).collect();
        CpuTensor::new(data, x.shape.clone())
    }

    fn sqrt(x: &CpuTensor) -> CpuTensor {
        let data: Vec<f32> = x.data.iter().map(|&v| v.sqrt()).collect();
        CpuTensor::new(data, x.shape.clone())
    }

    // === Binary element-wise with broadcasting ===

    fn add(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| x + y)
    }

    fn sub(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| x - y)
    }

    fn mul(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| x * y)
    }

    fn div(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| x / y)
    }

    fn pow(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| x.powf(y))
    }

    fn maximum(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| x.max(y))
    }

    fn minimum(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| x.min(y))
    }

    // === Comparison ===

    fn gt(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| if x > y { 1.0 } else { 0.0 })
    }

    fn ge(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| if x >= y { 1.0 } else { 0.0 })
    }

    fn lt(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| if x < y { 1.0 } else { 0.0 })
    }

    fn le(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| if x <= y { 1.0 } else { 0.0 })
    }

    fn eq(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        binary_op_broadcast(a, b, |x, y| if (x - y).abs() < 1e-7 { 1.0 } else { 0.0 })
    }

    // === Reductions ===

    fn sum(x: &CpuTensor, axes: Option<&[usize]>, keepdims: bool) -> CpuTensor {
        reduce_op(x, axes, keepdims, 0.0, |acc, v| acc + v)
    }

    fn mean(x: &CpuTensor, axes: Option<&[usize]>, keepdims: bool) -> CpuTensor {
        let sum = Self::sum(x, axes, keepdims);
        let count = if let Some(axes) = axes {
            axes.iter().map(|&ax| x.shape.dim(ax)).product::<usize>()
        } else {
            x.shape.numel()
        };
        let count_tensor = Self::scalar(count as f32);
        Self::div(&sum, &Self::broadcast_to(&count_tensor, sum.shape()))
    }

    fn max(x: &CpuTensor, axes: Option<&[usize]>, keepdims: bool) -> CpuTensor {
        reduce_op(x, axes, keepdims, f32::NEG_INFINITY, |acc, v| acc.max(v))
    }

    fn min(x: &CpuTensor, axes: Option<&[usize]>, keepdims: bool) -> CpuTensor {
        reduce_op(x, axes, keepdims, f32::INFINITY, |acc, v| acc.min(v))
    }

    // === Linear algebra ===

    fn matmul(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        matmul_impl(a, b)
    }

    // === Shape operations ===

    fn transpose(x: &CpuTensor, axes: Option<&[usize]>) -> CpuTensor {
        let ndim = x.shape.ndim();
        if ndim == 0 {
            return x.clone();
        }

        let perm: Vec<usize> = axes.map(|a| a.to_vec()).unwrap_or_else(|| (0..ndim).rev().collect());

        let new_shape = Shape::new(perm.iter().map(|&i| x.shape.dim(i)).collect());
        let mut data = vec![0.0f32; x.shape.numel()];

        for idx in x.indices() {
            let src_flat = x.flat_index(&idx);
            let new_idx: Vec<usize> = perm.iter().map(|&i| idx[i]).collect();
            let dst_flat = new_shape.contiguous_strides().index(&new_idx);
            data[dst_flat] = x.data[src_flat];
        }

        CpuTensor::new(data, new_shape)
    }

    fn reshape(x: &CpuTensor, shape: &Shape) -> CpuTensor {
        assert_eq!(
            x.shape.numel(),
            shape.numel(),
            "Cannot reshape from {:?} to {:?}",
            x.shape,
            shape
        );
        CpuTensor::new(x.data.clone(), shape.clone())
    }

    fn broadcast_to(x: &CpuTensor, shape: &Shape) -> CpuTensor {
        if x.shape() == shape {
            return x.clone();
        }

        let ndim = shape.ndim();
        let x_ndim = x.shape.ndim();
        let offset = ndim - x_ndim;

        let mut data = vec![0.0f32; shape.numel()];

        for out_idx in TensorIndices::new(shape) {
            // Map output index to input index (accounting for broadcasting)
            let in_idx: Vec<usize> = (0..x_ndim)
                .map(|i| {
                    if x.shape.dim(i) == 1 {
                        0
                    } else {
                        out_idx[offset + i]
                    }
                })
                .collect();

            let out_flat = shape.contiguous_strides().index(&out_idx);
            let in_flat = if in_idx.is_empty() {
                0
            } else {
                x.strides.index(&in_idx)
            };
            data[out_flat] = x.data[in_flat];
        }

        CpuTensor::new(data, shape.clone())
    }

    fn sum_to(x: &CpuTensor, shape: &Shape) -> CpuTensor {
        if x.shape() == shape {
            return x.clone();
        }

        // Find axes to sum over
        let x_ndim = x.shape.ndim();
        let target_ndim = shape.ndim();
        let offset = x_ndim - target_ndim;

        let mut axes = Vec::new();

        // Leading axes that don't exist in target
        for i in 0..offset {
            axes.push(i);
        }

        // Axes that are 1 in target but larger in x
        for i in 0..target_ndim {
            if shape.dim(i) == 1 && x.shape.dim(offset + i) > 1 {
                axes.push(offset + i);
            }
        }

        if axes.is_empty() {
            return x.clone();
        }

        // Sum over these axes
        let result = Self::sum(x, Some(&axes), false);

        // Reshape to target shape
        Self::reshape(&result, shape)
    }

    fn squeeze(x: &CpuTensor, axes: Option<&[usize]>) -> CpuTensor {
        let new_dims: Vec<usize> = if let Some(axes) = axes {
            x.shape
                .dims()
                .iter()
                .enumerate()
                .filter(|(i, &d)| !axes.contains(i) || d != 1)
                .map(|(_, &d)| d)
                .collect()
        } else {
            x.shape.dims().iter().filter(|&&d| d != 1).copied().collect()
        };
        CpuTensor::new(x.data.clone(), Shape::new(new_dims))
    }

    fn unsqueeze(x: &CpuTensor, axis: usize) -> CpuTensor {
        let mut new_dims = x.shape.dims().to_vec();
        new_dims.insert(axis, 1);
        CpuTensor::new(x.data.clone(), Shape::new(new_dims))
    }

    // === Gradient accumulation ===

    fn accumulate_grad(dst: &mut CpuTensor, src: &CpuTensor) {
        assert_eq!(dst.shape, src.shape);
        for (d, s) in dst.data.iter_mut().zip(src.data.iter()) {
            *d += s;
        }
    }

    fn clone_tensor(x: &CpuTensor) -> CpuTensor {
        x.clone()
    }
}

/// Binary operation with broadcasting.
fn binary_op_broadcast<F>(a: &CpuTensor, b: &CpuTensor, op: F) -> CpuTensor
where
    F: Fn(f32, f32) -> f32,
{
    let out_shape = a
        .shape
        .broadcast_with(&b.shape)
        .expect("Shapes are not broadcast compatible");

    let a_broadcast = CpuBackend::broadcast_to(a, &out_shape);
    let b_broadcast = CpuBackend::broadcast_to(b, &out_shape);

    let data: Vec<f32> = a_broadcast
        .data
        .iter()
        .zip(b_broadcast.data.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();

    CpuTensor::new(data, out_shape)
}

/// Reduction operation over specified axes.
fn reduce_op<F>(
    x: &CpuTensor,
    axes: Option<&[usize]>,
    keepdims: bool,
    init: f32,
    op: F,
) -> CpuTensor
where
    F: Fn(f32, f32) -> f32,
{
    let ndim = x.shape.ndim();

    // Handle scalar
    if ndim == 0 {
        return x.clone();
    }

    // Determine which axes to reduce
    let reduce_axes: Vec<usize> = axes
        .map(|a| a.to_vec())
        .unwrap_or_else(|| (0..ndim).collect());

    // Compute output shape
    let out_dims: Vec<usize> = (0..ndim)
        .filter_map(|i| {
            if reduce_axes.contains(&i) {
                if keepdims {
                    Some(1)
                } else {
                    None
                }
            } else {
                Some(x.shape.dim(i))
            }
        })
        .collect();

    let out_shape = if out_dims.is_empty() {
        Shape::scalar()
    } else {
        Shape::new(out_dims)
    };

    let mut data = vec![init; out_shape.numel()];

    // Iterate over input and accumulate to output
    for in_idx in x.indices() {
        let in_flat = x.flat_index(&in_idx);

        // Compute output index (drop/keep reduced dimensions)
        let out_idx: Vec<usize> = (0..ndim)
            .filter_map(|i| {
                if reduce_axes.contains(&i) {
                    if keepdims {
                        Some(0)
                    } else {
                        None
                    }
                } else {
                    Some(in_idx[i])
                }
            })
            .collect();

        let out_flat = if out_idx.is_empty() {
            0
        } else {
            out_shape.contiguous_strides().index(&out_idx)
        };

        data[out_flat] = op(data[out_flat], x.data[in_flat]);
    }

    CpuTensor::new(data, out_shape)
}

/// Matrix multiplication implementation.
fn matmul_impl(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
    let a_ndim = a.shape.ndim();
    let b_ndim = b.shape.ndim();

    assert!(a_ndim >= 1 && b_ndim >= 1, "matmul requires at least 1D tensors");

    // Handle vector-matrix and matrix-vector cases
    if a_ndim == 1 && b_ndim == 1 {
        // Dot product
        assert_eq!(a.shape.dim(0), b.shape.dim(0));
        let dot: f32 = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).sum();
        return CpuTensor::new(vec![dot], Shape::scalar());
    }

    if a_ndim == 1 {
        // (K,) @ (K, N) -> (N,)
        let k = a.shape.dim(0);
        let n = b.shape.dim(1);
        assert_eq!(k, b.shape.dim(0));

        let mut data = vec![0.0f32; n];
        for j in 0..n {
            for i in 0..k {
                data[j] += a.data[i] * b.data[i * n + j];
            }
        }
        return CpuTensor::new(data, Shape::new(vec![n]));
    }

    if b_ndim == 1 {
        // (M, K) @ (K,) -> (M,)
        let m = a.shape.dim(0);
        let k = a.shape.dim(1);
        assert_eq!(k, b.shape.dim(0));

        let mut data = vec![0.0f32; m];
        for i in 0..m {
            for j in 0..k {
                data[i] += a.data[i * k + j] * b.data[j];
            }
        }
        return CpuTensor::new(data, Shape::new(vec![m]));
    }

    // Standard matrix multiplication: (M, K) @ (K, N) -> (M, N)
    // Also handles batched: (..., M, K) @ (..., K, N) -> (..., M, N)
    let m = a.shape.dim(a_ndim - 2);
    let k = a.shape.dim(a_ndim - 1);
    let k2 = b.shape.dim(b_ndim - 2);
    let n = b.shape.dim(b_ndim - 1);

    assert_eq!(k, k2, "Matrix dimensions don't match for matmul");

    // For now, handle only 2D case
    if a_ndim == 2 && b_ndim == 2 {
        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a.data[i * k + l] * b.data[l * n + j];
                }
                data[i * n + j] = sum;
            }
        }
        return CpuTensor::new(data, Shape::new(vec![m, n]));
    }

    // Batched matmul - broadcast batch dimensions
    let a_batch: Vec<usize> = a.shape.dims()[..a_ndim - 2].to_vec();
    let b_batch: Vec<usize> = b.shape.dims()[..b_ndim - 2].to_vec();

    let batch_shape = Shape::new(a_batch.clone())
        .broadcast_with(&Shape::new(b_batch.clone()))
        .expect("Batch dimensions not broadcastable");

    let mut out_dims = batch_shape.dims().to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::new(out_dims);

    let batch_numel = batch_shape.numel();
    let mut data = vec![0.0f32; out_shape.numel()];

    for batch_idx in 0..batch_numel {
        // Compute batch indices
        let mut remaining = batch_idx;
        let mut batch_indices = vec![0usize; batch_shape.ndim()];
        for i in (0..batch_shape.ndim()).rev() {
            batch_indices[i] = remaining % batch_shape.dim(i);
            remaining /= batch_shape.dim(i);
        }

        // Get slices for this batch
        let a_offset = compute_batch_offset(a, &batch_indices, m * k);
        let b_offset = compute_batch_offset(b, &batch_indices, k * n);
        let out_offset = batch_idx * m * n;

        // Matmul for this batch
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a.data[a_offset + i * k + l] * b.data[b_offset + l * n + j];
                }
                data[out_offset + i * n + j] = sum;
            }
        }
    }

    CpuTensor::new(data, out_shape)
}

/// Compute offset into tensor data for batched operations.
fn compute_batch_offset(t: &CpuTensor, batch_indices: &[usize], matrix_size: usize) -> usize {
    let t_batch_ndim = t.shape.ndim() - 2;
    let offset = batch_indices.len() - t_batch_ndim;

    let mut idx = 0;
    let mut stride = matrix_size;
    for i in (0..t_batch_ndim).rev() {
        let bi = batch_indices[offset + i];
        let dim = t.shape.dim(i);
        // Handle broadcasting: if dim is 1, always use index 0
        let actual_idx = if dim == 1 { 0 } else { bi };
        idx += actual_idx * stride;
        stride *= dim;
    }
    idx
}

/// Type alias for tensors using the CPU backend.
pub type CpuTensor_ = Tensor<CpuBackend>;

/// Create a variable tensor.
pub fn var(name: &str, data: Vec<f32>, shape: Shape) -> CpuTensor_ {
    Tensor::var(name, CpuBackend::from_vec(data, shape))
}

/// Create a constant tensor.
pub fn constant(data: Vec<f32>, shape: Shape) -> CpuTensor_ {
    Tensor::constant(CpuBackend::from_vec(data, shape))
}

/// Finite difference gradient check for tensor operations.
pub fn finite_diff_grad<F>(
    f: F,
    inputs: &[Vec<f32>],
    shapes: &[Shape],
    eps: f32,
) -> Vec<Vec<f32>>
where
    F: Fn(&[CpuTensor_]) -> CpuTensor_,
{
    let mut grads = Vec::new();

    for (input_idx, (input, _shape)) in inputs.iter().zip(shapes.iter()).enumerate() {
        let mut input_grads = Vec::new();

        for elem_idx in 0..input.len() {
            // f(x + eps)
            let inputs_plus: Vec<CpuTensor_> = inputs
                .iter()
                .zip(shapes.iter())
                .enumerate()
                .map(|(i, (inp, sh))| {
                    let mut data = inp.clone();
                    if i == input_idx {
                        data[elem_idx] += eps;
                    }
                    Tensor::constant(CpuBackend::from_vec(data, sh.clone()))
                })
                .collect();

            // f(x - eps)
            let inputs_minus: Vec<CpuTensor_> = inputs
                .iter()
                .zip(shapes.iter())
                .enumerate()
                .map(|(i, (inp, sh))| {
                    let mut data = inp.clone();
                    if i == input_idx {
                        data[elem_idx] -= eps;
                    }
                    Tensor::constant(CpuBackend::from_vec(data, sh.clone()))
                })
                .collect();

            let out_plus = f(&inputs_plus).sum(None, false);
            let out_minus = f(&inputs_minus).sum(None, false);

            let grad = (out_plus.item() - out_minus.item()) / (2.0 * eps);
            input_grads.push(grad);
        }

        grads.push(input_grads);
    }

    grads
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = CpuBackend::zeros(&Shape::new(vec![2, 3]));
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.as_slice(), &[0.0; 6]);

        let t2 = CpuBackend::ones(&Shape::new(vec![2, 3]));
        assert_eq!(t2.as_slice(), &[1.0; 6]);

        let s = CpuBackend::scalar(42.0);
        assert!(s.shape().is_scalar());
        assert_eq!(s.scalar_value(), 42.0);
    }

    #[test]
    fn test_unary_ops() {
        let x = CpuBackend::from_vec(vec![1.0, 2.0, -3.0], Shape::new(vec![3]));

        let neg = CpuBackend::neg(&x);
        assert_eq!(neg.as_slice(), &[-1.0, -2.0, 3.0]);

        let relu = CpuBackend::relu(&x);
        assert_eq!(relu.as_slice(), &[1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_binary_ops() {
        let a = CpuBackend::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = CpuBackend::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));

        let sum = CpuBackend::add(&a, &b);
        assert_eq!(sum.as_slice(), &[5.0, 7.0, 9.0]);

        let prod = CpuBackend::mul(&a, &b);
        assert_eq!(prod.as_slice(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_broadcasting() {
        let a = CpuBackend::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = CpuBackend::scalar(10.0);

        let result = CpuBackend::add(&a, &b);
        assert_eq!(result.as_slice(), &[11.0, 12.0, 13.0]);

        // 2D broadcasting
        let a2 = CpuBackend::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b2 = CpuBackend::from_vec(vec![10.0, 20.0], Shape::new(vec![2]));
        let result2 = CpuBackend::add(&a2, &b2);
        assert_eq!(result2.as_slice(), &[11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_sum_reduction() {
        let x = CpuBackend::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));

        // Sum all
        let total = CpuBackend::sum(&x, None, false);
        assert!(total.shape().is_scalar());
        assert_eq!(total.scalar_value(), 21.0);

        // Sum along axis 0
        let sum0 = CpuBackend::sum(&x, Some(&[0]), false);
        assert_eq!(sum0.shape().dims(), &[3]);
        assert_eq!(sum0.as_slice(), &[5.0, 7.0, 9.0]);

        // Sum along axis 1
        let sum1 = CpuBackend::sum(&x, Some(&[1]), false);
        assert_eq!(sum1.shape().dims(), &[2]);
        assert_eq!(sum1.as_slice(), &[6.0, 15.0]);
    }

    #[test]
    fn test_matmul() {
        // 2x3 @ 3x2 = 2x2
        let a = CpuBackend::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let b = CpuBackend::from_vec(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::new(vec![3, 2]),
        );

        let c = CpuBackend::matmul(&a, &b);
        assert_eq!(c.shape().dims(), &[2, 2]);
        // [1,2,3] @ [[7,8],[9,10],[11,12]] = [58, 64]
        // [4,5,6] @ [[7,8],[9,10],[11,12]] = [139, 154]
        assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_transpose() {
        let x = CpuBackend::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let t = CpuBackend::transpose(&x, None);
        assert_eq!(t.shape().dims(), &[3, 2]);
        assert_eq!(t.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_reshape() {
        let x = CpuBackend::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let r = CpuBackend::reshape(&x, &Shape::new(vec![3, 2]));
        assert_eq!(r.shape().dims(), &[3, 2]);
        assert_eq!(r.as_slice(), x.as_slice());
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let x = CpuBackend::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3, 1]));

        let squeezed = CpuBackend::squeeze(&x, None);
        assert_eq!(squeezed.shape().dims(), &[3]);

        let unsqueezed = CpuBackend::unsqueeze(&squeezed, 0);
        assert_eq!(unsqueezed.shape().dims(), &[1, 3]);
    }

    // === Autodiff tests ===

    fn assert_grad_close(name: &str, autodiff: &[f32], finite_diff: &[f32], tol: f32) {
        for (i, (ad, fd)) in autodiff.iter().zip(finite_diff.iter()).enumerate() {
            let err = (ad - fd).abs();
            assert!(
                err < tol,
                "{}: element {} mismatch: autodiff={}, finite_diff={}, err={}",
                name, i, ad, fd, err
            );
        }
    }

    #[test]
    fn test_autodiff_add() {
        let x = var("x", vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let y = var("y", vec![4.0, 5.0, 6.0], Shape::new(vec![3]));

        let z = (&x + &y).sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        let dy = grads.wrt(&y).unwrap().as_slice();

        // d(sum(x+y))/dx = 1, d(sum(x+y))/dy = 1
        assert_eq!(dx, &[1.0, 1.0, 1.0]);
        assert_eq!(dy, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_autodiff_mul() {
        let x = var("x", vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let y = var("y", vec![4.0, 5.0, 6.0], Shape::new(vec![3]));

        let z = (&x * &y).sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        let dy = grads.wrt(&y).unwrap().as_slice();

        // d(sum(x*y))/dx = y, d(sum(x*y))/dy = x
        assert_eq!(dx, &[4.0, 5.0, 6.0]);
        assert_eq!(dy, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_autodiff_exp() {
        let x_data = vec![0.0, 1.0, -1.0];
        let x = var("x", x_data.clone(), Shape::new(vec![3]));

        let z = x.exp().sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        let expected: Vec<f32> = x_data.iter().map(|v| v.exp()).collect();

        assert_grad_close("exp", dx, &expected, 1e-5);
    }

    #[test]
    fn test_autodiff_log() {
        let x_data = vec![1.0, 2.0, 3.0];
        let x = var("x", x_data.clone(), Shape::new(vec![3]));

        let z = x.log().sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        let expected: Vec<f32> = x_data.iter().map(|v| 1.0 / v).collect();

        assert_grad_close("log", dx, &expected, 1e-5);
    }

    #[test]
    fn test_autodiff_sin_cos() {
        let x_data = vec![0.0, 1.0, 2.0];
        let x = var("x", x_data.clone(), Shape::new(vec![3]));

        // sin(x)
        let z_sin = x.sin().sum(None, false);
        let grads_sin = z_sin.backward();
        let dx_sin = grads_sin.wrt(&x).unwrap().as_slice();
        let expected_sin: Vec<f32> = x_data.iter().map(|v| v.cos()).collect();
        assert_grad_close("sin", dx_sin, &expected_sin, 1e-5);

        // cos(x)
        let x2 = var("x", x_data.clone(), Shape::new(vec![3]));
        let z_cos = x2.cos().sum(None, false);
        let grads_cos = z_cos.backward();
        let dx_cos = grads_cos.wrt(&x2).unwrap().as_slice();
        let expected_cos: Vec<f32> = x_data.iter().map(|v| -v.sin()).collect();
        assert_grad_close("cos", dx_cos, &expected_cos, 1e-5);
    }

    #[test]
    fn test_autodiff_relu() {
        let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let x = var("x", x_data.clone(), Shape::new(vec![5]));

        let z = x.relu().sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        // relu gradient: 1 if x > 0, else 0
        assert_eq!(dx, &[0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_autodiff_sigmoid() {
        let x_data = vec![0.0, 1.0, -1.0];
        let x = var("x", x_data.clone(), Shape::new(vec![3]));

        let z = x.sigmoid().sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        let expected: Vec<f32> = x_data
            .iter()
            .map(|v| {
                let s = 1.0 / (1.0 + (-v).exp());
                s * (1.0 - s)
            })
            .collect();

        assert_grad_close("sigmoid", dx, &expected, 1e-5);
    }

    #[test]
    fn test_autodiff_tanh() {
        let x_data = vec![0.0, 1.0, -1.0];
        let x = var("x", x_data.clone(), Shape::new(vec![3]));

        let z = x.tanh().sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        // tanh'(x) = 1 - tanh(x)^2
        let expected: Vec<f32> = x_data.iter().map(|v| 1.0 - v.tanh().powi(2)).collect();

        assert_grad_close("tanh", dx, &expected, 1e-5);
    }

    #[test]
    fn test_autodiff_chain_rule() {
        // z = sin(x^2)
        // dz/dx = cos(x^2) * 2x
        let x_data = vec![1.0, 2.0];
        let x = var("x", x_data.clone(), Shape::new(vec![2]));

        let two = Tensor::<CpuBackend>::scalar(2.0);
        let z = x.pow(&two).sin().sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        let expected: Vec<f32> = x_data.iter().map(|v| (v * v).cos() * 2.0 * v).collect();

        assert_grad_close("chain_rule", dx, &expected, 1e-5);
    }

    #[test]
    fn test_autodiff_broadcast_grad() {
        // x is [3], y is scalar
        // z = x + y (broadcasts y to [3])
        // sum(z) has dz/dx = [1,1,1], dz/dy = 3 (sum of ones)
        let x = var("x", vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let y = Tensor::<CpuBackend>::var("y", CpuBackend::scalar(10.0));

        let z = (&x + &y).sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        let dy = grads.wrt(&y).unwrap().as_slice();

        assert_eq!(dx, &[1.0, 1.0, 1.0]);
        assert_eq!(dy, &[3.0]); // Sum of gradients from broadcast
    }

    #[test]
    fn test_autodiff_matmul() {
        // C = A @ B
        // dL/dA = dL/dC @ B^T
        // dL/dB = A^T @ dL/dC
        let a = var("A", vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b = var("B", vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));

        let c = a.matmul(&b).sum(None, false);
        let grads = c.backward();

        let da = grads.wrt(&a).unwrap();
        let db = grads.wrt(&b).unwrap();

        // Verify with finite differences
        let fd_grads = finite_diff_grad(
            |inputs| inputs[0].matmul(&inputs[1]),
            &[
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
            ],
            &[Shape::new(vec![2, 2]), Shape::new(vec![2, 2])],
            1e-4,
        );

        assert_grad_close("matmul_A", da.as_slice(), &fd_grads[0], 0.05);
        assert_grad_close("matmul_B", db.as_slice(), &fd_grads[1], 0.05);
    }

    #[test]
    fn test_autodiff_mean() {
        let x = var("x", vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));

        let z = x.mean(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        // mean gradient: 1/n for each element
        assert_eq!(dx, &[0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_autodiff_reused_variable() {
        // z = x * x = x^2
        // dz/dx = 2x
        let x = var("x", vec![2.0, 3.0], Shape::new(vec![2]));

        let z = (&x * &x).sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        assert_eq!(dx, &[4.0, 6.0]);
    }

    #[test]
    fn test_autodiff_complex_graph() {
        // z = (x * y + sin(x)) / (y + 2)
        let x = var("x", vec![1.0], Shape::new(vec![1]));
        let y = var("y", vec![2.0], Shape::new(vec![1]));
        let two = Tensor::<CpuBackend>::constant(CpuBackend::scalar(2.0));

        let numerator = &x * &y + x.sin();
        let denominator = &y + &two;
        let z = numerator / denominator;

        let loss = z.sum(None, false);
        let grads = loss.backward();

        // Verify against finite differences
        let fd_grads = finite_diff_grad(
            |inputs| {
                let two = Tensor::<CpuBackend>::constant(CpuBackend::scalar(2.0));
                let num = &inputs[0] * &inputs[1] + inputs[0].sin();
                let den = &inputs[1] + &two;
                num / den
            },
            &[vec![1.0], vec![2.0]],
            &[Shape::new(vec![1]), Shape::new(vec![1])],
            1e-4,
        );

        assert_grad_close("complex_x", grads.wrt(&x).unwrap().as_slice(), &fd_grads[0], 1e-3);
        assert_grad_close("complex_y", grads.wrt(&y).unwrap().as_slice(), &fd_grads[1], 1e-3);
    }

    #[test]
    fn test_autodiff_transpose() {
        let x = var("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));

        let z = x.transpose(None).sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap().as_slice();
        // Transpose gradient is ones (just rearranged)
        assert_eq!(dx, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_autodiff_reshape() {
        let x = var("x", vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));

        let z = x.reshape(&Shape::new(vec![4])).sum(None, false);
        let grads = z.backward();

        let dx = grads.wrt(&x).unwrap();
        assert_eq!(dx.shape().dims(), &[2, 2]);
        assert_eq!(dx.as_slice(), &[1.0, 1.0, 1.0, 1.0]);
    }
}
