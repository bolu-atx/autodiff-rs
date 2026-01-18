//! Adam optimizer.

use ad_backend_cpu::CpuBackend;
use ad_tensor::prelude::*;
use std::collections::HashMap;

/// Adam optimizer (Adaptive Moment Estimation).
pub struct Adam {
    /// Learning rate.
    pub lr: f32,
    /// Exponential decay rate for first moment.
    pub beta1: f32,
    /// Exponential decay rate for second moment.
    pub beta2: f32,
    /// Small constant for numerical stability.
    pub eps: f32,
    /// First moment estimates (m).
    m: HashMap<NodeId, Vec<f32>>,
    /// Second moment estimates (v).
    v: HashMap<NodeId, Vec<f32>>,
    /// Step counter.
    t: u64,
}

impl Adam {
    /// Create a new Adam optimizer with default hyperparameters.
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Create an Adam optimizer with custom hyperparameters.
    pub fn with_params(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Adam {
            lr,
            beta1,
            beta2,
            eps,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Update a single parameter tensor given its gradient.
    pub fn step(&mut self, param: &mut Tensor<CpuBackend>, grad: &<CpuBackend as Backend>::Tensor) {
        self.t += 1;

        let param_id = param.id();
        let param_data = param.data().as_slice();
        let grad_data = grad.as_slice();
        let n = param_data.len();

        // Initialize moment estimates if needed
        let m = self.m.entry(param_id).or_insert_with(|| vec![0.0; n]);
        let v = self.v.entry(param_id).or_insert_with(|| vec![0.0; n]);

        let mut new_data = param_data.to_vec();

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..n {
            // Update biased first moment estimate
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad_data[i];

            // Update biased second raw moment estimate
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad_data[i] * grad_data[i];

            // Compute bias-corrected estimates
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            // Update parameter
            new_data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }

        *param = Tensor::var(
            param.var_name().unwrap_or("param"),
            CpuBackend::from_vec(new_data, param.shape().clone()),
        );
    }

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_step() {
        let mut param = Tensor::var(
            "w",
            CpuBackend::from_vec(vec![1.0, 2.0], Shape::new(vec![2])),
        );
        let grad = CpuBackend::from_vec(vec![0.1, 0.2], Shape::new(vec![2]));

        let mut opt = Adam::new(0.1);

        let initial = param.as_slice().to_vec();
        opt.step(&mut param, &grad);
        let after = param.as_slice().to_vec();

        // Parameters should have moved in the direction opposite to gradient
        for i in 0..2 {
            assert!(after[i] < initial[i]);
        }
    }

    #[test]
    fn test_adam_convergence() {
        // Simple test: minimize x^2, optimal at x=0
        let mut param = Tensor::var(
            "x",
            CpuBackend::from_vec(vec![10.0], Shape::new(vec![1])),
        );

        let mut opt = Adam::new(0.5);
        let initial_value = param.as_slice()[0];

        // Run a few steps and verify we're moving towards 0
        for _ in 0..10 {
            // Gradient of x^2 is 2x
            let grad = CpuBackend::from_vec(
                vec![2.0 * param.as_slice()[0]],
                Shape::new(vec![1]),
            );
            opt.step(&mut param, &grad);
        }

        // Should be significantly closer to 0 than initial value
        let final_value = param.as_slice()[0];
        assert!(final_value.abs() < initial_value.abs());
    }
}
