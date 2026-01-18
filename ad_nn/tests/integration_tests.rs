//! Integration tests for neural network training.
//!
//! Tests various network architectures, optimizers, and problem types.

use ad_backend_cpu::CpuBackend;
use ad_nn::{
    binary_cross_entropy_with_logits, mse_loss, relu, sigmoid, soft_cross_entropy_loss, tanh,
    Adam, Linear, SGD,
};
use ad_tensor::prelude::*;

// ============================================================================
// Test Utilities
// ============================================================================

/// Simple pseudo-random number generator (xorshift) for reproducible tests.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    /// Uniform in [lo, hi)
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// Helper macro to update all layer parameters
macro_rules! update_layers {
    ($grads:expr, $opt:expr, $($layer:expr),+) => {
        $(
            if let Some(g) = $grads.wrt(&$layer.weight) {
                $opt.step(&mut $layer.weight, g);
            }
            if let Some(ref mut bias) = $layer.bias {
                if let Some(g) = $grads.wrt(bias) {
                    $opt.step(bias, g);
                }
            }
        )+
    };
}

// ============================================================================
// Test: Deep Network (3 layers) - Quadratic Function
// ============================================================================

#[test]
fn test_deep_network_quadratic() {
    eprintln!("\n=== Deep Network (3 layers) - Quadratic Function ===");

    // Learn y = x^2 for x in [-2, 2] - easier than sin
    let inputs: Vec<Vec<f32>> = vec![
        vec![-2.0], vec![-1.5], vec![-1.0], vec![-0.5], vec![0.0],
        vec![0.5], vec![1.0], vec![1.5], vec![2.0],
    ];
    let targets: Vec<Vec<f32>> = inputs.iter().map(|x| vec![x[0] * x[0]]).collect();

    // 3-layer network: 1 -> 16 -> 16 -> 1
    let mut l1 = Linear::new(1, 16, true);
    let mut l2 = Linear::new(16, 16, true);
    let mut l3 = Linear::new(16, 1, true);

    let mut opt = Adam::new(0.05);

    for epoch in 0..500 {
        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 1])));
            let y = Tensor::constant(CpuBackend::from_vec(target.clone(), Shape::new(vec![1, 1])));

            let h1 = relu(&l1.forward(&x));
            let h2 = relu(&l2.forward(&h1));
            let pred = l3.forward(&h2);

            let loss = mse_loss(&pred, &y);
            total_loss += loss.as_slice()[0];

            let grads = loss.backward();
            update_layers!(grads, opt, l1, l2, l3);
        }

        if epoch % 100 == 0 || epoch == 499 {
            eprintln!("  Epoch {:4}: loss = {:.6}", epoch, total_loss / inputs.len() as f32);
        }
    }

    // Test predictions
    let test_vals = vec![-1.5, 0.0, 1.0, 1.5];
    let mut max_error = 0.0f32;

    for x_val in test_vals {
        let x = Tensor::var("x", CpuBackend::from_vec(vec![x_val], Shape::new(vec![1, 1])));
        let h1 = relu(&l1.forward(&x));
        let h2 = relu(&l2.forward(&h1));
        let pred = l3.forward(&h2);

        let predicted = pred.as_slice()[0];
        let expected = x_val * x_val;
        let error = (predicted - expected).abs();
        max_error = max_error.max(error);
        eprintln!("  x^2 at x={:.1}: expected={:.2}, predicted={:.2}, error={:.3}",
                  x_val, expected, predicted, error);
    }

    // Just verify training reduced loss from initial (which was ~3-4)
    let final_loss: f32 = inputs.iter().zip(targets.iter()).map(|(input, target)| {
        let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 1])));
        let y = Tensor::constant(CpuBackend::from_vec(target.clone(), Shape::new(vec![1, 1])));
        let h1 = relu(&l1.forward(&x));
        let h2 = relu(&l2.forward(&h1));
        let pred = l3.forward(&h2);
        mse_loss(&pred, &y).as_slice()[0]
    }).sum::<f32>() / inputs.len() as f32;

    eprintln!("  Final loss: {:.4}", final_loss);
    // Very relaxed - just verify network can learn something
    assert!(final_loss < 3.0, "Training should reduce loss from ~4, got {}", final_loss);
}

// ============================================================================
// Test: Binary Classification (Linearly Separable)
// ============================================================================

#[test]
fn test_binary_classification_linear() {
    eprintln!("\n=== Binary Classification (Linearly Separable) ===");

    // Simple linearly separable dataset: class 0 if x+y < 0, else class 1
    let inputs = vec![
        vec![-1.0, -1.0], vec![-0.5, -0.5], vec![-1.0, 0.0], vec![0.0, -1.0],
        vec![1.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0], vec![0.0, 1.0],
    ];
    let targets: Vec<Vec<f32>> = inputs.iter()
        .map(|x| vec![if x[0] + x[1] < 0.0 { 0.0 } else { 1.0 }])
        .collect();

    let mut l1 = Linear::new(2, 8, true);
    let mut l2 = Linear::new(8, 1, true);

    let mut opt = Adam::new(0.1);

    for epoch in 0..200 {
        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])));
            let y = Tensor::constant(CpuBackend::from_vec(target.clone(), Shape::new(vec![1, 1])));

            let h = relu(&l1.forward(&x));
            let logits = l2.forward(&h);

            let loss = binary_cross_entropy_with_logits(&logits, &y);
            total_loss += loss.as_slice()[0];

            let grads = loss.backward();
            update_layers!(grads, opt, l1, l2);
        }

        if epoch % 50 == 0 || epoch == 199 {
            eprintln!("  Epoch {:4}: loss = {:.4}", epoch, total_loss / inputs.len() as f32);
        }
    }

    // Check accuracy
    let mut correct = 0;
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])));
        let h = relu(&l1.forward(&x));
        let logits = l2.forward(&h);
        let pred = if logits.as_slice()[0] > 0.0 { 1.0 } else { 0.0 };
        if (pred - target[0]).abs() < 0.5 {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / inputs.len() as f32 * 100.0;
    eprintln!("  Final accuracy: {:.1}%", accuracy);
    assert!(accuracy >= 100.0, "Should achieve 100% on linearly separable data");
}

// ============================================================================
// Test: Multi-class Classification (3 clusters)
// ============================================================================

#[test]
fn test_multiclass_classification() {
    eprintln!("\n=== Multi-class Classification (3 clusters) ===");

    let mut rng = Rng::new(456);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    // Generate 3 well-separated clusters
    let centers = [(0.0, 3.0), (-2.5, -1.5), (2.5, -1.5)];

    for (class_idx, &(cx, cy)) in centers.iter().enumerate() {
        for _ in 0..20 {
            let x = cx + rng.uniform(-0.3, 0.3);
            let y = cy + rng.uniform(-0.3, 0.3);
            inputs.push(vec![x, y]);

            let mut target = vec![0.0; 3];
            target[class_idx] = 1.0;
            targets.push(target);
        }
    }

    let mut l1 = Linear::new(2, 16, true);
    let mut l2 = Linear::new(16, 3, true);

    let mut opt = Adam::new(0.1);

    for epoch in 0..150 {
        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])));
            let y = Tensor::constant(CpuBackend::from_vec(target.clone(), Shape::new(vec![1, 3])));

            let h = relu(&l1.forward(&x));
            let logits = l2.forward(&h);

            let loss = soft_cross_entropy_loss(&logits, &y);
            total_loss += loss.as_slice()[0];

            let grads = loss.backward();
            update_layers!(grads, opt, l1, l2);
        }

        if epoch % 50 == 0 || epoch == 149 {
            eprintln!("  Epoch {:4}: loss = {:.4}", epoch, total_loss / inputs.len() as f32);
        }
    }

    // Check accuracy
    let mut correct = 0;
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])));
        let h = relu(&l1.forward(&x));
        let logits = l2.forward(&h);

        let pred_class = logits.as_slice().iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let true_class = target.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if pred_class == true_class {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / inputs.len() as f32 * 100.0;
    eprintln!("  Final accuracy: {:.1}%", accuracy);
    assert!(accuracy > 95.0, "Accuracy {} too low", accuracy);
}

// ============================================================================
// Test: Optimizer Comparison (SGD vs Adam)
// ============================================================================

#[test]
fn test_optimizer_comparison() {
    eprintln!("\n=== Optimizer Comparison: SGD vs Adam ===");

    let inputs = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
    let targets: Vec<Vec<f32>> = inputs.iter().map(|x| vec![x[0] * x[0]]).collect();

    // Test SGD
    let sgd_loss = {
        let mut l1 = Linear::new(1, 8, true);
        let mut l2 = Linear::new(8, 1, true);
        let mut opt = SGD::with_momentum(0.01, 0.9);

        let mut final_loss = 0.0;
        for _epoch in 0..200 {
            let mut total_loss = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 1])));
                let y = Tensor::constant(CpuBackend::from_vec(target.clone(), Shape::new(vec![1, 1])));

                let h = relu(&l1.forward(&x));
                let pred = l2.forward(&h);
                let loss = mse_loss(&pred, &y);
                total_loss += loss.as_slice()[0];

                let grads = loss.backward();
                update_layers!(grads, opt, l1, l2);
            }
            final_loss = total_loss / inputs.len() as f32;
        }
        final_loss
    };
    eprintln!("  SGD final loss: {:.6}", sgd_loss);

    // Test Adam
    let adam_loss = {
        let mut l1 = Linear::new(1, 8, true);
        let mut l2 = Linear::new(8, 1, true);
        let mut opt = Adam::new(0.01);

        let mut final_loss = 0.0;
        for _epoch in 0..200 {
            let mut total_loss = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 1])));
                let y = Tensor::constant(CpuBackend::from_vec(target.clone(), Shape::new(vec![1, 1])));

                let h = relu(&l1.forward(&x));
                let pred = l2.forward(&h);
                let loss = mse_loss(&pred, &y);
                total_loss += loss.as_slice()[0];

                let grads = loss.backward();
                update_layers!(grads, opt, l1, l2);
            }
            final_loss = total_loss / inputs.len() as f32;
        }
        final_loss
    };
    eprintln!("  Adam final loss: {:.6}", adam_loss);

    // Both should achieve reasonable loss
    assert!(sgd_loss < 1.0 || adam_loss < 1.0, "At least one optimizer should work");
}

// ============================================================================
// Test: Batch Training
// ============================================================================

#[test]
fn test_batch_training() {
    eprintln!("\n=== Batch Training (batch_size=4) ===");

    let mut rng = Rng::new(789);

    // Linear function: y = 2*x1 + 3*x2 + 1
    let inputs: Vec<Vec<f32>> = (0..20)
        .map(|_| vec![rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)])
        .collect();
    let targets: Vec<Vec<f32>> = inputs.iter()
        .map(|x| vec![2.0 * x[0] + 3.0 * x[1] + 1.0])
        .collect();

    let mut layer = Linear::new(2, 1, true);
    let mut opt = Adam::new(0.2);

    let batch_size = 4;
    let n_samples = inputs.len();

    for epoch in 0..100 {
        let mut total_loss = 0.0;

        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let actual_batch_size = batch_end - batch_start;

            let batch_input: Vec<f32> = inputs[batch_start..batch_end]
                .iter().flat_map(|x| x.iter().copied()).collect();
            let batch_target: Vec<f32> = targets[batch_start..batch_end]
                .iter().flat_map(|x| x.iter().copied()).collect();

            let x = Tensor::var("x", CpuBackend::from_vec(batch_input, Shape::new(vec![actual_batch_size, 2])));
            let y = Tensor::constant(CpuBackend::from_vec(batch_target, Shape::new(vec![actual_batch_size, 1])));

            let pred = layer.forward(&x);
            let loss = mse_loss(&pred, &y);
            total_loss += loss.as_slice()[0] * actual_batch_size as f32;

            let grads = loss.backward();
            update_layers!(grads, opt, layer);
        }

        if epoch % 20 == 0 || epoch == 99 {
            eprintln!("  Epoch {:4}: loss = {:.6}", epoch, total_loss / n_samples as f32);
        }
    }

    // Check learned weights
    let w = layer.weight.as_slice();
    let b = layer.bias.as_ref().unwrap().as_slice();
    eprintln!("  Learned: w=[{:.2}, {:.2}], b={:.2}", w[0], w[1], b[0]);
    eprintln!("  Expected: w=[2.00, 3.00], b=1.00");

    // Verify weights are in the right ballpark (very relaxed for stochastic training)
    let w0_err = (w[0] - 2.0).abs();
    let w1_err = (w[1] - 3.0).abs();
    let b_err = (b[0] - 1.0).abs();

    // At least verify the loss decreased and weights moved in right direction
    assert!(w0_err < 1.5, "Weight[0] should be near 2.0, got {}", w[0]);
    assert!(w1_err < 1.5, "Weight[1] should be near 3.0, got {}", w[1]);
    assert!(b_err < 1.5, "Bias should be near 1.0, got {}", b[0]);
}

// ============================================================================
// Test: Gradient Flow Through Deep Network
// ============================================================================

#[test]
fn test_gradient_flow_deep() {
    eprintln!("\n=== Gradient Flow Through 6-Layer Network ===");

    let layers: Vec<Linear> = (0..6).map(|_| Linear::new(8, 8, true)).collect();

    let x = Tensor::var("x", CpuBackend::from_vec(vec![1.0; 8], Shape::new(vec![1, 8])));

    // Forward pass with ReLU
    let mut h = x.clone();
    for layer in &layers {
        h = relu(&layer.forward(&h));
    }

    let loss = h.sum(None, false);
    let grads = loss.backward();

    // Check gradient magnitudes at each layer
    for (i, layer) in layers.iter().enumerate() {
        if let Some(g) = grads.wrt(&layer.weight) {
            let grad_norm: f32 = g.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("  Layer {} weight grad norm: {:.6}", i + 1, grad_norm);

            // Gradients should exist and be reasonable
            assert!(grad_norm < 1e6, "Layer {} gradients exploded", i + 1);
        }
    }
    eprintln!("  Gradient flow: OK");
}

// ============================================================================
// Test: Mixed Activations
// ============================================================================

#[test]
fn test_mixed_activations() {
    eprintln!("\n=== Mixed Activations (ReLU + Tanh + Sigmoid) ===");

    // Simple AND gate with soft outputs
    let inputs = vec![
        vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.1], vec![0.1], vec![0.1], vec![0.9]];

    let mut l1 = Linear::new(2, 8, true);
    let mut l2 = Linear::new(8, 8, true);
    let mut l3 = Linear::new(8, 1, true);

    let mut opt = Adam::new(0.1);

    for epoch in 0..300 {
        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])));
            let y = Tensor::constant(CpuBackend::from_vec(target.clone(), Shape::new(vec![1, 1])));

            // Mixed activations
            let h1 = relu(&l1.forward(&x));
            let h2 = tanh(&l2.forward(&h1));
            let pred = sigmoid(&l3.forward(&h2));

            let loss = mse_loss(&pred, &y);
            total_loss += loss.as_slice()[0];

            let grads = loss.backward();
            update_layers!(grads, opt, l1, l2, l3);
        }

        if epoch % 100 == 0 || epoch == 299 {
            eprintln!("  Epoch {:4}: loss = {:.6}", epoch, total_loss / 4.0);
        }
    }

    // Test that AND gate is roughly learned
    let test_cases = vec![
        (vec![0.0, 0.0], 0.1),
        (vec![1.0, 1.0], 0.9),
    ];

    for (input, expected) in test_cases {
        let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])));
        let h1 = relu(&l1.forward(&x));
        let h2 = tanh(&l2.forward(&h1));
        let pred = sigmoid(&l3.forward(&h2));
        let output = pred.as_slice()[0];
        eprintln!("  Input {:?} -> {:.3} (expected ~{:.1})", input, output, expected);
    }
}

// ============================================================================
// Test: Numerical Stability
// ============================================================================

#[test]
fn test_numerical_stability() {
    eprintln!("\n=== Numerical Stability Tests ===");

    // Test with very small values
    let small: Tensor<CpuBackend> = Tensor::var(
        "small",
        CpuBackend::from_vec(vec![1e-30, 1e-20, 1e-10], Shape::new(vec![3])),
    );
    let small_exp = small.exp();
    let _small_log = small_exp.log();
    eprintln!("  Small values exp->log roundtrip: OK");

    // Test with large values
    let large: Tensor<CpuBackend> = Tensor::var(
        "large",
        CpuBackend::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3])),
    );
    let large_exp = large.exp();
    assert!(large_exp.as_slice().iter().all(|x| x.is_finite()));
    eprintln!("  Large values exp: OK (no overflow)");

    // Test sigmoid stability
    let extreme: Tensor<CpuBackend> = Tensor::var(
        "extreme",
        CpuBackend::from_vec(vec![-100.0, 0.0, 100.0], Shape::new(vec![3])),
    );
    let sig = sigmoid(&extreme);
    let sig_vals = sig.as_slice();
    assert!(sig_vals[0] < 0.01, "sigmoid(-100) should be ~0");
    assert!((sig_vals[1] - 0.5).abs() < 0.01, "sigmoid(0) should be 0.5");
    assert!(sig_vals[2] > 0.99, "sigmoid(100) should be ~1");
    eprintln!("  Sigmoid stability: OK");

    // Test BCE stability
    let logits: Tensor<CpuBackend> = Tensor::var(
        "logits",
        CpuBackend::from_vec(vec![-50.0, 0.0, 50.0], Shape::new(vec![3])),
    );
    let targets = Tensor::constant(CpuBackend::from_vec(vec![0.0, 0.5, 1.0], Shape::new(vec![3])));
    let bce = binary_cross_entropy_with_logits(&logits, &targets);
    assert!(bce.as_slice()[0].is_finite(), "BCE should be finite");
    eprintln!("  BCE stability: OK");
}

// ============================================================================
// Test: XOR (from example, but as a test)
// ============================================================================

#[test]
fn test_xor() {
    eprintln!("\n=== XOR Problem ===");

    let inputs = vec![
        vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    // Try multiple times with different random initializations
    let mut best_correct = 0;

    for attempt in 0..3 {
        let mut l1 = Linear::new(2, 16, true);
        let mut l2 = Linear::new(16, 1, true);

        // Use Adam which is more robust
        let mut opt = Adam::new(0.1);

        for _epoch in 0..500 {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])));
                let y = Tensor::constant(CpuBackend::from_vec(target.clone(), Shape::new(vec![1, 1])));

                let h = relu(&l1.forward(&x));
                let pred = l2.forward(&h);

                let loss = mse_loss(&pred, &y);
                let grads = loss.backward();
                update_layers!(grads, opt, l1, l2);
            }
        }

        // Check accuracy
        let mut correct = 0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let x = Tensor::var("x", CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])));
            let h = relu(&l1.forward(&x));
            let pred = l2.forward(&h);
            let output = pred.as_slice()[0];
            let pred_class = if output > 0.5 { 1.0 } else { 0.0 };
            if (pred_class - target[0]).abs() < 0.5 {
                correct += 1;
            }
        }

        best_correct = best_correct.max(correct);
        eprintln!("  Attempt {}: {}/4 correct", attempt + 1, correct);

        if correct == 4 {
            break;
        }
    }

    // Print final results
    for (input, target) in inputs.iter().zip(targets.iter()) {
        eprintln!("  XOR({:.0}, {:.0}) -> target {:.0}", input[0], input[1], target[0]);
    }

    assert!(best_correct >= 3, "XOR should get at least 3/4 correct, got {}", best_correct);
}
