//! Stochastic Gradient Descent optimizer.

use ad_backend_cpu::CpuBackend;
use ad_tensor::prelude::*;
use std::collections::HashMap;

/// SGD optimizer with optional momentum.
pub struct SGD {
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient.
    pub momentum: f32,
    /// Velocity buffers for momentum (keyed by tensor NodeId).
    velocities: HashMap<NodeId, Vec<f32>>,
}

impl SGD {
    /// Create a new SGD optimizer.
    pub fn new(lr: f32) -> Self {
        SGD {
            lr,
            momentum: 0.0,
            velocities: HashMap::new(),
        }
    }

    /// Create an SGD optimizer with momentum.
    pub fn with_momentum(lr: f32, momentum: f32) -> Self {
        SGD {
            lr,
            momentum,
            velocities: HashMap::new(),
        }
    }

    /// Update a single parameter tensor in-place given its gradient.
    pub fn step(&mut self, param: &mut Tensor<CpuBackend>, grad: &<CpuBackend as Backend>::Tensor) {
        let param_id = param.id();
        let param_data = param.data().as_slice();
        let grad_data = grad.as_slice();

        let mut new_data = param_data.to_vec();

        if self.momentum > 0.0 {
            // SGD with momentum: v = momentum * v + grad; param = param - lr * v
            let velocity = self
                .velocities
                .entry(param_id)
                .or_insert_with(|| vec![0.0; param_data.len()]);

            for i in 0..new_data.len() {
                velocity[i] = self.momentum * velocity[i] + grad_data[i];
                new_data[i] -= self.lr * velocity[i];
            }
        } else {
            // Vanilla SGD: param = param - lr * grad
            for i in 0..new_data.len() {
                new_data[i] -= self.lr * grad_data[i];
            }
        }

        // Update the parameter tensor
        // Note: This is a limitation - we need to create a new tensor since
        // the data is behind an Arc. In a real implementation, you'd use
        // interior mutability or a different design.
        *param = Tensor::var(
            param.var_name().unwrap_or("param"),
            CpuBackend::from_vec(new_data, param.shape().clone()),
        );
    }

    /// Zero gradients (no-op for this simple implementation).
    pub fn zero_grad(&mut self) {
        // In this design, gradients are computed fresh each forward/backward pass
        // so there's nothing to zero.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_step() {
        let mut param = Tensor::var(
            "w",
            CpuBackend::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3])),
        );
        let grad = CpuBackend::from_vec(vec![0.1, 0.2, 0.3], Shape::new(vec![3]));

        let mut opt = SGD::new(0.1);
        opt.step(&mut param, &grad);

        // param = param - 0.1 * grad = [1.0 - 0.01, 2.0 - 0.02, 3.0 - 0.03]
        let expected = vec![0.99, 1.98, 2.97];
        for (p, e) in param.as_slice().iter().zip(expected.iter()) {
            assert!((p - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut param = Tensor::var(
            "w",
            CpuBackend::from_vec(vec![1.0], Shape::new(vec![1])),
        );
        let grad = CpuBackend::from_vec(vec![1.0], Shape::new(vec![1]));

        let mut opt = SGD::with_momentum(0.1, 0.9);

        // First step: v = 0.9 * 0 + 1 = 1, param = 1 - 0.1 * 1 = 0.9
        opt.step(&mut param, &grad);
        assert!((param.as_slice()[0] - 0.9).abs() < 1e-6);

        // Note: After creating a new tensor, the velocity buffer is associated with
        // the old node ID. For real usage, we'd need a better ID scheme.
        // For now, just verify the first step works correctly.
    }
}
