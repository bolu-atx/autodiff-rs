//! Backend parity tests - verify CPU and Metal backends produce equivalent results.
//!
//! These tests ensure that all backends compute the same values (within floating-point tolerance).

use ad_backend_cpu::CpuBackend;
use ad_nn::{activations, loss};
use ad_tensor::prelude::*;

const TOLERANCE: f32 = 1e-5;

fn assert_close(cpu: f32, other: f32, name: &str) {
    let diff = (cpu - other).abs();
    assert!(
        diff < TOLERANCE,
        "{name}: CPU={cpu}, other={other}, diff={diff}"
    );
}

fn assert_tensors_close<B: Backend>(cpu: &Tensor<CpuBackend>, other: &Tensor<B>, name: &str) {
    let cpu_data = cpu.as_slice();
    let other_data = other.as_slice();
    assert_eq!(
        cpu_data.len(),
        other_data.len(),
        "{name}: shape mismatch"
    );
    for (i, (c, o)) in cpu_data.iter().zip(other_data.iter()).enumerate() {
        let diff = (c - o).abs();
        assert!(
            diff < TOLERANCE,
            "{name}[{i}]: CPU={c}, other={o}, diff={diff}"
        );
    }
}

// ============================================================================
// CPU baseline tests (always run)
// ============================================================================

#[test]
fn test_mse_loss_cpu() {
    let pred = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    let target = Tensor::<CpuBackend>::from_vec(vec![1.1, 1.9, 3.2, 3.8], Shape::new(vec![4]));

    let loss = loss::mse_loss(&pred, &target);
    // (0.1^2 + 0.1^2 + 0.2^2 + 0.2^2) / 4 = 0.025
    assert_close(loss.item(), 0.025, "mse_loss");
}

#[test]
fn test_softmax_cpu() {
    let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
    let y = activations::softmax(&x);

    // Softmax should sum to 1
    let sum: f32 = y.as_slice().iter().sum();
    assert_close(sum, 1.0, "softmax sum");
}

#[test]
fn test_bce_loss_cpu() {
    let logits = Tensor::<CpuBackend>::from_vec(vec![0.0, 2.0, -2.0], Shape::new(vec![3]));
    let targets = Tensor::<CpuBackend>::from_vec(vec![0.5, 1.0, 0.0], Shape::new(vec![3]));

    let loss = loss::binary_cross_entropy_with_logits(&logits, &targets);
    // Just verify it produces a reasonable positive value
    assert!(loss.item() > 0.0);
    assert!(loss.item() < 10.0);
}

#[test]
fn test_soft_cross_entropy_cpu() {
    let logits = Tensor::<CpuBackend>::from_vec(
        vec![2.0, 1.0, 0.1, 0.5, 1.5, 1.0],
        Shape::new(vec![2, 3]),
    );
    let targets = Tensor::<CpuBackend>::from_vec(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        Shape::new(vec![2, 3]),
    );

    let loss = loss::soft_cross_entropy_loss(&logits, &targets);
    assert!(loss.item() > 0.0);
}

// ============================================================================
// Metal parity tests (macOS only)
// ============================================================================

#[cfg(target_os = "macos")]
mod metal_parity {
    use super::*;
    use ad_backend_metal::MetalBackend;

    #[test]
    fn test_mse_loss_metal_parity() {
        let data_pred = vec![1.0, 2.0, 3.0, 4.0];
        let data_target = vec![1.1, 1.9, 3.2, 3.8];
        let shape = Shape::new(vec![4]);

        let pred_cpu = Tensor::<CpuBackend>::from_vec(data_pred.clone(), shape.clone());
        let target_cpu = Tensor::<CpuBackend>::from_vec(data_target.clone(), shape.clone());
        let loss_cpu = loss::mse_loss(&pred_cpu, &target_cpu);

        let pred_metal = Tensor::<MetalBackend>::from_vec(data_pred, shape.clone());
        let target_metal = Tensor::<MetalBackend>::from_vec(data_target, shape);
        let loss_metal = loss::mse_loss(&pred_metal, &target_metal);

        assert_close(loss_cpu.item(), loss_metal.item(), "mse_loss parity");
    }

    #[test]
    fn test_softmax_metal_parity() {
        let data = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
        let shape = Shape::new(vec![2, 3]);

        let x_cpu = Tensor::<CpuBackend>::from_vec(data.clone(), shape.clone());
        let y_cpu = activations::softmax(&x_cpu);

        let x_metal = Tensor::<MetalBackend>::from_vec(data, shape);
        let y_metal = activations::softmax(&x_metal);

        assert_tensors_close(&y_cpu, &y_metal, "softmax parity");
    }

    #[test]
    fn test_log_softmax_metal_parity() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(vec![1, 3]);

        let x_cpu = Tensor::<CpuBackend>::from_vec(data.clone(), shape.clone());
        let y_cpu = activations::log_softmax(&x_cpu);

        let x_metal = Tensor::<MetalBackend>::from_vec(data, shape);
        let y_metal = activations::log_softmax(&x_metal);

        assert_tensors_close(&y_cpu, &y_metal, "log_softmax parity");
    }

    #[test]
    fn test_relu_metal_parity() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let shape = Shape::new(vec![5]);

        let x_cpu = Tensor::<CpuBackend>::from_vec(data.clone(), shape.clone());
        let y_cpu = activations::relu(&x_cpu);

        let x_metal = Tensor::<MetalBackend>::from_vec(data, shape);
        let y_metal = activations::relu(&x_metal);

        assert_tensors_close(&y_cpu, &y_metal, "relu parity");
    }

    #[test]
    fn test_sigmoid_metal_parity() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let shape = Shape::new(vec![5]);

        let x_cpu = Tensor::<CpuBackend>::from_vec(data.clone(), shape.clone());
        let y_cpu = activations::sigmoid(&x_cpu);

        let x_metal = Tensor::<MetalBackend>::from_vec(data, shape);
        let y_metal = activations::sigmoid(&x_metal);

        assert_tensors_close(&y_cpu, &y_metal, "sigmoid parity");
    }

    #[test]
    fn test_tanh_metal_parity() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let shape = Shape::new(vec![5]);

        let x_cpu = Tensor::<CpuBackend>::from_vec(data.clone(), shape.clone());
        let y_cpu = activations::tanh(&x_cpu);

        let x_metal = Tensor::<MetalBackend>::from_vec(data, shape);
        let y_metal = activations::tanh(&x_metal);

        assert_tensors_close(&y_cpu, &y_metal, "tanh parity");
    }

    #[test]
    fn test_bce_with_logits_metal_parity() {
        let logits_data = vec![0.0, 2.0, -2.0, 1.0];
        let targets_data = vec![0.5, 1.0, 0.0, 0.5];
        let shape = Shape::new(vec![4]);

        let logits_cpu = Tensor::<CpuBackend>::from_vec(logits_data.clone(), shape.clone());
        let targets_cpu = Tensor::<CpuBackend>::from_vec(targets_data.clone(), shape.clone());
        let loss_cpu = loss::binary_cross_entropy_with_logits(&logits_cpu, &targets_cpu);

        let logits_metal = Tensor::<MetalBackend>::from_vec(logits_data, shape.clone());
        let targets_metal = Tensor::<MetalBackend>::from_vec(targets_data, shape);
        let loss_metal = loss::binary_cross_entropy_with_logits(&logits_metal, &targets_metal);

        assert_close(loss_cpu.item(), loss_metal.item(), "bce_with_logits parity");
    }

    #[test]
    fn test_soft_cross_entropy_metal_parity() {
        let logits_data = vec![2.0, 1.0, 0.1, 0.5, 1.5, 1.0];
        let targets_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let shape = Shape::new(vec![2, 3]);

        let logits_cpu = Tensor::<CpuBackend>::from_vec(logits_data.clone(), shape.clone());
        let targets_cpu = Tensor::<CpuBackend>::from_vec(targets_data.clone(), shape.clone());
        let loss_cpu = loss::soft_cross_entropy_loss(&logits_cpu, &targets_cpu);

        let logits_metal = Tensor::<MetalBackend>::from_vec(logits_data, shape.clone());
        let targets_metal = Tensor::<MetalBackend>::from_vec(targets_data, shape);
        let loss_metal = loss::soft_cross_entropy_loss(&logits_metal, &targets_metal);

        assert_close(
            loss_cpu.item(),
            loss_metal.item(),
            "soft_cross_entropy parity",
        );
    }
}
