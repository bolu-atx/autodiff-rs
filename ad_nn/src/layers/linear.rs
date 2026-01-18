//! Linear (fully connected) layer.

use ad_backend_cpu::CpuBackend;
use ad_tensor::prelude::*;
use rand::Rng;

/// A linear (fully connected) layer: y = x @ W^T + b
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    pub weight: Tensor<CpuBackend>,
    /// Bias vector [out_features]
    pub bias: Option<Tensor<CpuBackend>>,
}

impl Linear {
    /// Create a new linear layer with random initialization.
    ///
    /// Uses Kaiming initialization (He et al.) for ReLU networks.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let mut rng = rand::thread_rng();

        // Kaiming initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / in_features as f32).sqrt();

        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|_| rng.gen::<f32>() * std * 2.0 - std)
            .collect();

        let weight = Tensor::var(
            "weight",
            CpuBackend::from_vec(weight_data, Shape::new(vec![out_features, in_features])),
        );

        let bias = if bias {
            let bias_data = vec![0.0f32; out_features];
            Some(Tensor::var(
                "bias",
                CpuBackend::from_vec(bias_data, Shape::new(vec![out_features])),
            ))
        } else {
            None
        };

        Linear { weight, bias }
    }

    /// Create a linear layer from existing weight and bias tensors.
    pub fn from_tensors(weight: Tensor<CpuBackend>, bias: Option<Tensor<CpuBackend>>) -> Self {
        Linear { weight, bias }
    }

    /// Forward pass: y = x @ W^T + b
    ///
    /// Input x has shape [batch, in_features] or [in_features].
    /// Output has shape [batch, out_features] or [out_features].
    pub fn forward(&self, x: &Tensor<CpuBackend>) -> Tensor<CpuBackend> {
        // x @ W^T: [batch, in_features] @ [in_features, out_features] -> [batch, out_features]
        let weight_t = self.weight.t();
        let y = x.matmul(&weight_t);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            &y + bias
        } else {
            y
        }
    }

    /// Get the number of input features.
    pub fn in_features(&self) -> usize {
        self.weight.shape().dim(1)
    }

    /// Get the number of output features.
    pub fn out_features(&self) -> usize {
        self.weight.shape().dim(0)
    }

    /// Get all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor<CpuBackend>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let layer = Linear::new(3, 2, true);

        let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let y = layer.forward(&x);

        assert_eq!(y.shape().dims(), &[1, 2]);
    }

    #[test]
    fn test_linear_backward() {
        let layer = Linear::new(3, 2, true);

        let x = Tensor::var(
            "x",
            CpuBackend::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3])),
        );
        let y = layer.forward(&x);
        let loss = y.sum(None, false);

        let grads = loss.backward();

        // Gradients should exist for weight, bias, and input
        assert!(grads.wrt(&layer.weight).is_some());
        assert!(grads.wrt(layer.bias.as_ref().unwrap()).is_some());
        assert!(grads.wrt(&x).is_some());
    }
}
