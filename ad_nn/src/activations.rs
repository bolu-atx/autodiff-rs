//! Activation functions.

use ad_backend_cpu::CpuBackend;
use ad_tensor::prelude::*;

/// ReLU activation: max(0, x)
pub fn relu(x: &Tensor<CpuBackend>) -> Tensor<CpuBackend> {
    x.relu()
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(x: &Tensor<CpuBackend>) -> Tensor<CpuBackend> {
    x.sigmoid()
}

/// Tanh activation: tanh(x)
pub fn tanh(x: &Tensor<CpuBackend>) -> Tensor<CpuBackend> {
    x.tanh()
}

/// Softmax activation: exp(x) / sum(exp(x))
///
/// Computes softmax along the last dimension (axis=-1).
/// Uses numerically stable computation: exp(x - max(x)) / sum(exp(x - max(x)))
pub fn softmax(x: &Tensor<CpuBackend>) -> Tensor<CpuBackend> {
    let ndim = x.ndim();
    let axis = if ndim > 0 { ndim - 1 } else { 0 };

    // Subtract max for numerical stability
    let x_max = x.max(Some(&[axis]), true);
    let x_shifted = x - &x_max;

    // exp(x - max)
    let exp_x = x_shifted.exp();

    // sum(exp(x - max))
    let sum_exp = exp_x.sum(Some(&[axis]), true);

    // exp(x - max) / sum(exp(x - max))
    exp_x / sum_exp
}

/// Log-softmax activation: log(softmax(x))
///
/// Computed as: x - max(x) - log(sum(exp(x - max(x))))
/// This is more numerically stable than log(softmax(x)).
pub fn log_softmax(x: &Tensor<CpuBackend>) -> Tensor<CpuBackend> {
    let ndim = x.ndim();
    let axis = if ndim > 0 { ndim - 1 } else { 0 };

    // x - max(x)
    let x_max = x.max(Some(&[axis]), true);
    let x_shifted = x - &x_max;

    // log(sum(exp(x - max)))
    let log_sum_exp = x_shifted.exp().sum(Some(&[axis]), true).log();

    // x - max - log(sum(exp(x - max)))
    x_shifted - log_sum_exp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let x = Tensor::<CpuBackend>::from_vec(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            Shape::new(vec![5]),
        );
        let y = relu(&x);
        assert_eq!(y.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let x = Tensor::<CpuBackend>::from_vec(vec![0.0], Shape::new(vec![1]));
        let y = sigmoid(&x);
        assert!((y.as_slice()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let y = softmax(&x);

        // Softmax should sum to 1
        let sum: f32 = y.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Larger values should have larger probabilities
        assert!(y.as_slice()[2] > y.as_slice()[1]);
        assert!(y.as_slice()[1] > y.as_slice()[0]);
    }

    #[test]
    fn test_log_softmax() {
        let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let y = log_softmax(&x);

        // exp(log_softmax) should sum to 1
        let exp_y: Vec<f32> = y.as_slice().iter().map(|v| v.exp()).collect();
        let sum: f32 = exp_y.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
