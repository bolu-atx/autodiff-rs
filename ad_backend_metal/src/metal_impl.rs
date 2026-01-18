//! Metal backend implementation.

use std::sync::Arc;

use metal::{Buffer, CommandQueue, Device, Library, MTLResourceOptions};

use ad_tensor::prelude::*;

/// Metal compute shaders for tensor operations.
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Element-wise unary operations
kernel void neg(device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    output[id] = -input[id];
}

kernel void exp_f(device const float* input [[buffer(0)]],
                  device float* output [[buffer(1)]],
                  uint id [[thread_position_in_grid]]) {
    output[id] = exp(input[id]);
}

kernel void log_f(device const float* input [[buffer(0)]],
                  device float* output [[buffer(1)]],
                  uint id [[thread_position_in_grid]]) {
    output[id] = log(input[id]);
}

kernel void sin_f(device const float* input [[buffer(0)]],
                  device float* output [[buffer(1)]],
                  uint id [[thread_position_in_grid]]) {
    output[id] = sin(input[id]);
}

kernel void cos_f(device const float* input [[buffer(0)]],
                  device float* output [[buffer(1)]],
                  uint id [[thread_position_in_grid]]) {
    output[id] = cos(input[id]);
}

kernel void relu(device const float* input [[buffer(0)]],
                 device float* output [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    output[id] = max(0.0f, input[id]);
}

kernel void sigmoid(device const float* input [[buffer(0)]],
                    device float* output [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {
    output[id] = 1.0f / (1.0f + exp(-input[id]));
}

kernel void tanh_f(device const float* input [[buffer(0)]],
                   device float* output [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {
    output[id] = tanh(input[id]);
}

kernel void sqrt_f(device const float* input [[buffer(0)]],
                   device float* output [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {
    output[id] = sqrt(input[id]);
}

// Element-wise binary operations
kernel void add_f(device const float* a [[buffer(0)]],
                  device const float* b [[buffer(1)]],
                  device float* output [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    output[id] = a[id] + b[id];
}

kernel void sub_f(device const float* a [[buffer(0)]],
                  device const float* b [[buffer(1)]],
                  device float* output [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    output[id] = a[id] - b[id];
}

kernel void mul_f(device const float* a [[buffer(0)]],
                  device const float* b [[buffer(1)]],
                  device float* output [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    output[id] = a[id] * b[id];
}

kernel void div_f(device const float* a [[buffer(0)]],
                  device const float* b [[buffer(1)]],
                  device float* output [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    output[id] = a[id] / b[id];
}

kernel void pow_f(device const float* a [[buffer(0)]],
                  device const float* b [[buffer(1)]],
                  device float* output [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    output[id] = pow(a[id], b[id]);
}

kernel void maximum_f(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = max(a[id], b[id]);
}

kernel void minimum_f(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = min(a[id], b[id]);
}

// Comparison operations (output 1.0 or 0.0)
kernel void gt_f(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* output [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    output[id] = a[id] > b[id] ? 1.0f : 0.0f;
}

kernel void ge_f(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* output [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    output[id] = a[id] >= b[id] ? 1.0f : 0.0f;
}

kernel void lt_f(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* output [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    output[id] = a[id] < b[id] ? 1.0f : 0.0f;
}

kernel void le_f(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* output [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    output[id] = a[id] <= b[id] ? 1.0f : 0.0f;
}

kernel void eq_f(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* output [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    output[id] = abs(a[id] - b[id]) < 1e-7f ? 1.0f : 0.0f;
}

// Matrix multiplication: C = A @ B
// A is MxK, B is KxN, C is MxN
kernel void matmul(device const float* A [[buffer(0)]],
                   device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]],
                   constant uint& M [[buffer(3)]],
                   constant uint& K [[buffer(4)]],
                   constant uint& N [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

// Sum reduction
kernel void sum_reduce(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint id [[thread_position_in_grid]],
                       uint threads [[threads_per_grid]]) {
    float sum = 0.0f;
    for (uint i = id; i < n; i += threads) {
        sum += input[i];
    }
    output[id] = sum;
}
"#;

/// Metal device context shared between tensors.
#[derive(Clone)]
pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
}

impl MetalContext {
    /// Create a new Metal context with compiled shaders.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();

        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .expect("Failed to compile Metal shaders");

        Some(MetalContext {
            device,
            command_queue,
            library,
        })
    }

    fn create_buffer(&self, size: usize) -> Buffer {
        let size_bytes = size * std::mem::size_of::<f32>();
        self.device.new_buffer(
            size_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn create_buffer_with_data(&self, data: &[f32]) -> Buffer {
        let size_bytes = data.len() * std::mem::size_of::<f32>();
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new().expect("Failed to create Metal context - no GPU available")
    }
}

/// Metal tensor storage.
#[derive(Clone)]
pub struct MetalTensor {
    buffer: Buffer,
    shape: Shape,
    strides: Strides,
    context: Arc<MetalContext>,
}

impl MetalTensor {
    /// Create a new Metal tensor.
    pub fn new(context: Arc<MetalContext>, data: &[f32], shape: Shape) -> Self {
        let strides = shape.contiguous_strides();
        let buffer = context.create_buffer_with_data(data);
        MetalTensor {
            buffer,
            shape,
            strides,
            context,
        }
    }

    /// Get a slice of the tensor data (synchronously copies from GPU).
    pub fn to_vec(&self) -> Vec<f32> {
        let numel = self.shape.numel();
        let ptr = self.buffer.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, numel) };
        slice.to_vec()
    }
}

impl TensorData for MetalTensor {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &Strides {
        &self.strides
    }

    fn as_slice(&self) -> &[f32] {
        let numel = self.shape.numel();
        let ptr = self.buffer.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, numel) }
    }

    fn as_slice_mut(&mut self) -> &mut [f32] {
        let numel = self.shape.numel();
        let ptr = self.buffer.contents() as *mut f32;
        unsafe { std::slice::from_raw_parts_mut(ptr, numel) }
    }
}

/// Metal backend.
#[derive(Clone)]
pub struct MetalBackend {
    context: Arc<MetalContext>,
}

impl MetalBackend {
    pub fn new() -> Self {
        MetalBackend {
            context: Arc::new(MetalContext::default()),
        }
    }

    pub fn with_context(context: Arc<MetalContext>) -> Self {
        MetalBackend { context }
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

// Helper to run a unary kernel
fn run_unary_kernel(ctx: &MetalContext, input: &MetalTensor, kernel_name: &str) -> MetalTensor {
    let numel = input.shape.numel();
    let output_buffer = ctx.create_buffer(numel);

    let function = ctx
        .library
        .get_function(kernel_name, None)
        .expect("Failed to get kernel function");
    let pipeline_state = ctx
        .device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create pipeline state");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&output_buffer), 0);

    let thread_group_size = metal::MTLSize::new(256, 1, 1);
    let grid_size = metal::MTLSize::new(numel as u64, 1, 1);
    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    MetalTensor {
        buffer: output_buffer,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
        context: input.context.clone(),
    }
}

// Helper to run a binary kernel
fn run_binary_kernel(
    ctx: &MetalContext,
    a: &MetalTensor,
    b: &MetalTensor,
    output_shape: &Shape,
    kernel_name: &str,
) -> MetalTensor {
    let numel = output_shape.numel();
    let output_buffer = ctx.create_buffer(numel);

    let function = ctx
        .library
        .get_function(kernel_name, None)
        .expect("Failed to get kernel function");
    let pipeline_state = ctx
        .device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create pipeline state");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&a.buffer), 0);
    encoder.set_buffer(1, Some(&b.buffer), 0);
    encoder.set_buffer(2, Some(&output_buffer), 0);

    let thread_group_size = metal::MTLSize::new(256, 1, 1);
    let grid_size = metal::MTLSize::new(numel as u64, 1, 1);
    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    MetalTensor {
        buffer: output_buffer,
        shape: output_shape.clone(),
        strides: output_shape.contiguous_strides(),
        context: a.context.clone(),
    }
}

impl Backend for MetalBackend {
    type Tensor = MetalTensor;

    fn zeros(shape: &Shape) -> MetalTensor {
        let ctx = Arc::new(MetalContext::default());
        let data = vec![0.0f32; shape.numel()];
        MetalTensor::new(ctx, &data, shape.clone())
    }

    fn ones(shape: &Shape) -> MetalTensor {
        let ctx = Arc::new(MetalContext::default());
        let data = vec![1.0f32; shape.numel()];
        MetalTensor::new(ctx, &data, shape.clone())
    }

    fn from_vec(data: Vec<f32>, shape: Shape) -> MetalTensor {
        let ctx = Arc::new(MetalContext::default());
        MetalTensor::new(ctx, &data, shape)
    }

    fn scalar(value: f32) -> MetalTensor {
        Self::from_vec(vec![value], Shape::scalar())
    }

    fn full(shape: &Shape, value: f32) -> MetalTensor {
        let ctx = Arc::new(MetalContext::default());
        let data = vec![value; shape.numel()];
        MetalTensor::new(ctx, &data, shape.clone())
    }

    // Unary operations
    fn neg(x: &MetalTensor) -> MetalTensor {
        run_unary_kernel(&x.context, x, "neg")
    }

    fn exp(x: &MetalTensor) -> MetalTensor {
        run_unary_kernel(&x.context, x, "exp_f")
    }

    fn log(x: &MetalTensor) -> MetalTensor {
        run_unary_kernel(&x.context, x, "log_f")
    }

    fn sin(x: &MetalTensor) -> MetalTensor {
        run_unary_kernel(&x.context, x, "sin_f")
    }

    fn cos(x: &MetalTensor) -> MetalTensor {
        run_unary_kernel(&x.context, x, "cos_f")
    }

    fn relu(x: &MetalTensor) -> MetalTensor {
        run_unary_kernel(&x.context, x, "relu")
    }

    fn sigmoid(x: &MetalTensor) -> MetalTensor {
        run_unary_kernel(&x.context, x, "sigmoid")
    }

    fn tanh(x: &MetalTensor) -> MetalTensor {
        run_unary_kernel(&x.context, x, "tanh_f")
    }

    fn sqrt(x: &MetalTensor) -> MetalTensor {
        run_unary_kernel(&x.context, x, "sqrt_f")
    }

    // Binary operations (note: these assume same shapes for simplicity,
    // broadcasting needs to be done beforehand by calling broadcast_to)
    fn add(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "add_f")
    }

    fn sub(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "sub_f")
    }

    fn mul(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "mul_f")
    }

    fn div(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "div_f")
    }

    fn pow(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "pow_f")
    }

    fn maximum(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "maximum_f")
    }

    fn minimum(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "minimum_f")
    }

    // Comparison operations
    fn gt(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "gt_f")
    }

    fn ge(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "ge_f")
    }

    fn lt(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "lt_f")
    }

    fn le(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "le_f")
    }

    fn eq(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        let output_shape = a.shape.broadcast_with(&b.shape).expect("Shapes not broadcastable");
        let a_bc = Self::broadcast_to(a, &output_shape);
        let b_bc = Self::broadcast_to(b, &output_shape);
        run_binary_kernel(&a.context, &a_bc, &b_bc, &output_shape, "eq_f")
    }

    // Reductions - CPU fallback for now (GPU reductions are complex)
    fn sum(x: &MetalTensor, axes: Option<&[usize]>, keepdims: bool) -> MetalTensor {
        // CPU fallback
        let cpu_data = x.to_vec();
        let cpu_tensor = ad_backend_cpu::CpuBackend::from_vec(cpu_data, x.shape.clone());
        let result = ad_backend_cpu::CpuBackend::sum(&cpu_tensor, axes, keepdims);
        Self::from_vec(result.as_slice().to_vec(), result.shape().clone())
    }

    fn mean(x: &MetalTensor, axes: Option<&[usize]>, keepdims: bool) -> MetalTensor {
        let cpu_data = x.to_vec();
        let cpu_tensor = ad_backend_cpu::CpuBackend::from_vec(cpu_data, x.shape.clone());
        let result = ad_backend_cpu::CpuBackend::mean(&cpu_tensor, axes, keepdims);
        Self::from_vec(result.as_slice().to_vec(), result.shape().clone())
    }

    fn max(x: &MetalTensor, axes: Option<&[usize]>, keepdims: bool) -> MetalTensor {
        let cpu_data = x.to_vec();
        let cpu_tensor = ad_backend_cpu::CpuBackend::from_vec(cpu_data, x.shape.clone());
        let result = ad_backend_cpu::CpuBackend::max(&cpu_tensor, axes, keepdims);
        Self::from_vec(result.as_slice().to_vec(), result.shape().clone())
    }

    fn min(x: &MetalTensor, axes: Option<&[usize]>, keepdims: bool) -> MetalTensor {
        let cpu_data = x.to_vec();
        let cpu_tensor = ad_backend_cpu::CpuBackend::from_vec(cpu_data, x.shape.clone());
        let result = ad_backend_cpu::CpuBackend::min(&cpu_tensor, axes, keepdims);
        Self::from_vec(result.as_slice().to_vec(), result.shape().clone())
    }

    fn matmul(a: &MetalTensor, b: &MetalTensor) -> MetalTensor {
        // For now, use CPU fallback for matmul
        // TODO: Implement proper GPU matmul with the metal kernel
        let cpu_a = ad_backend_cpu::CpuBackend::from_vec(a.to_vec(), a.shape.clone());
        let cpu_b = ad_backend_cpu::CpuBackend::from_vec(b.to_vec(), b.shape.clone());
        let result = ad_backend_cpu::CpuBackend::matmul(&cpu_a, &cpu_b);
        Self::from_vec(result.as_slice().to_vec(), result.shape().clone())
    }

    fn transpose(x: &MetalTensor, axes: Option<&[usize]>) -> MetalTensor {
        let cpu_data = x.to_vec();
        let cpu_tensor = ad_backend_cpu::CpuBackend::from_vec(cpu_data, x.shape.clone());
        let result = ad_backend_cpu::CpuBackend::transpose(&cpu_tensor, axes);
        Self::from_vec(result.as_slice().to_vec(), result.shape().clone())
    }

    fn reshape(x: &MetalTensor, shape: &Shape) -> MetalTensor {
        MetalTensor {
            buffer: x.buffer.clone(),
            shape: shape.clone(),
            strides: shape.contiguous_strides(),
            context: x.context.clone(),
        }
    }

    fn broadcast_to(x: &MetalTensor, shape: &Shape) -> MetalTensor {
        if x.shape() == shape {
            return x.clone();
        }
        // CPU fallback for broadcasting
        let cpu_data = x.to_vec();
        let cpu_tensor = ad_backend_cpu::CpuBackend::from_vec(cpu_data, x.shape.clone());
        let result = ad_backend_cpu::CpuBackend::broadcast_to(&cpu_tensor, shape);
        Self::from_vec(result.as_slice().to_vec(), result.shape().clone())
    }

    fn sum_to(x: &MetalTensor, shape: &Shape) -> MetalTensor {
        if x.shape() == shape {
            return x.clone();
        }
        let cpu_data = x.to_vec();
        let cpu_tensor = ad_backend_cpu::CpuBackend::from_vec(cpu_data, x.shape.clone());
        let result = ad_backend_cpu::CpuBackend::sum_to(&cpu_tensor, shape);
        Self::from_vec(result.as_slice().to_vec(), result.shape().clone())
    }

    fn squeeze(x: &MetalTensor, axes: Option<&[usize]>) -> MetalTensor {
        let cpu_data = x.to_vec();
        let cpu_tensor = ad_backend_cpu::CpuBackend::from_vec(cpu_data, x.shape.clone());
        let result = ad_backend_cpu::CpuBackend::squeeze(&cpu_tensor, axes);
        Self::from_vec(result.as_slice().to_vec(), result.shape().clone())
    }

    fn unsqueeze(x: &MetalTensor, axis: usize) -> MetalTensor {
        let cpu_data = x.to_vec();
        let cpu_tensor = ad_backend_cpu::CpuBackend::from_vec(cpu_data, x.shape.clone());
        let result = ad_backend_cpu::CpuBackend::unsqueeze(&cpu_tensor, axis);
        Self::from_vec(result.as_slice().to_vec(), result.shape().clone())
    }

    fn accumulate_grad(dst: &mut MetalTensor, src: &MetalTensor) {
        // CPU fallback
        let dst_data: Vec<f32> = dst.to_vec().iter().zip(src.to_vec().iter()).map(|(a, b)| a + b).collect();
        let ptr = dst.buffer.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(dst_data.as_ptr(), ptr, dst_data.len());
        }
    }

    fn clone_tensor(x: &MetalTensor) -> MetalTensor {
        x.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_tensor_creation() {
        let ctx = Arc::new(MetalContext::new().expect("No Metal device"));
        let t = MetalTensor::new(ctx, &[1.0, 2.0, 3.0], Shape::new(vec![3]));
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_metal_unary_ops() {
        let x = MetalBackend::from_vec(vec![1.0, 2.0, -3.0], Shape::new(vec![3]));

        let neg = MetalBackend::neg(&x);
        assert_eq!(neg.to_vec(), vec![-1.0, -2.0, 3.0]);

        let relu = MetalBackend::relu(&x);
        assert_eq!(relu.to_vec(), vec![1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_metal_binary_ops() {
        let a = MetalBackend::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = MetalBackend::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));

        let sum = MetalBackend::add(&a, &b);
        assert_eq!(sum.to_vec(), vec![5.0, 7.0, 9.0]);

        let prod = MetalBackend::mul(&a, &b);
        assert_eq!(prod.to_vec(), vec![4.0, 10.0, 18.0]);
    }
}
