//! Loss functions.

use ad_tensor::prelude::*;

/// Mean Squared Error loss: mean((pred - target)^2)
pub fn mse_loss<B: Backend>(pred: &Tensor<B>, target: &Tensor<B>) -> Tensor<B> {
    let diff = pred - target;
    let sq = &diff * &diff;
    sq.mean(None, false)
}

/// Binary Cross-Entropy loss with logits.
///
/// Computes: mean(max(logits, 0) - logits * targets + log(1 + exp(-|logits|)))
/// This is numerically stable.
pub fn binary_cross_entropy_with_logits<B: Backend>(
    logits: &Tensor<B>,
    targets: &Tensor<B>,
) -> Tensor<B> {
    // Numerically stable BCE:
    // max(logits, 0) - logits * targets + log(1 + exp(-|logits|))

    // max(logits, 0)
    let relu_logits = logits.relu();

    // logits * targets
    let logits_targets = logits * targets;

    // |logits|
    let abs_logits = logits.maximum(&(-logits));

    // log(1 + exp(-|logits|))
    let one = Tensor::<B>::ones(logits.shape());
    let log_term = (&one + (-&abs_logits).exp()).log();

    // Combine
    let loss = relu_logits - logits_targets + log_term;
    loss.mean(None, false)
}

/// Cross-Entropy loss for classification (with log-softmax).
///
/// Assumes:
/// - `logits` has shape [batch, num_classes]
/// - `targets` are class indices (0 to num_classes-1), shape [batch]
///
/// Returns the mean negative log probability of the correct class.
///
/// Note: This function is CPU-only because it requires direct data access via `as_slice()`.
/// For generic backend support, use [`soft_cross_entropy_loss`] with one-hot encoded targets.
pub fn cross_entropy_loss(
    logits: &Tensor<ad_backend_cpu::CpuBackend>,
    targets: &[usize],
) -> Tensor<ad_backend_cpu::CpuBackend> {
    let log_probs = crate::activations::log_softmax(logits);

    // Gather the log probabilities of the correct classes
    let batch_size = logits.shape().dim(0);
    let num_classes = logits.shape().dim(1);

    let log_probs_slice = log_probs.as_slice();
    let mut losses = Vec::with_capacity(batch_size);

    for (i, &target) in targets.iter().enumerate() {
        let idx = i * num_classes + target;
        losses.push(-log_probs_slice[idx]);
    }

    // Return mean loss
    let sum: f32 = losses.iter().sum();
    Tensor::<ad_backend_cpu::CpuBackend>::scalar(sum / batch_size as f32)
}

/// Soft Cross-Entropy loss with target probabilities.
///
/// Assumes:
/// - `logits` has shape [batch, num_classes]
/// - `targets` has shape [batch, num_classes] with probabilities (one-hot or soft labels)
///
/// Returns: -mean(sum(targets * log_softmax(logits)))
pub fn soft_cross_entropy_loss<B: Backend>(
    logits: &Tensor<B>,
    targets: &Tensor<B>,
) -> Tensor<B> {
    let log_probs = crate::activations::log_softmax(logits);

    // -targets * log_probs
    let neg_log_probs = -(targets * &log_probs);

    // Sum over classes, mean over batch
    neg_log_probs.sum(Some(&[1]), false).mean(None, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_backend_cpu::CpuBackend;

    #[test]
    fn test_mse_loss() {
        let pred = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let target = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let loss = mse_loss(&pred, &target);
        assert!(loss.item().abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let pred = Tensor::<CpuBackend>::from_vec(vec![0.0, 0.0], Shape::new(vec![2]));
        let target = Tensor::<CpuBackend>::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));

        let loss = mse_loss(&pred, &target);
        assert!((loss.item() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bce_with_logits() {
        let logits = Tensor::<CpuBackend>::from_vec(vec![0.0], Shape::new(vec![1]));
        let targets = Tensor::<CpuBackend>::from_vec(vec![0.5], Shape::new(vec![1]));

        let loss = binary_cross_entropy_with_logits(&logits, &targets);
        // At logits=0, sigmoid=0.5, BCE = -0.5*log(0.5) - 0.5*log(0.5) = log(2)
        assert!((loss.item() - 0.6931).abs() < 0.01);
    }

    #[test]
    fn test_soft_cross_entropy() {
        let logits = Tensor::<CpuBackend>::from_vec(
            vec![2.0, 1.0, 0.0],
            Shape::new(vec![1, 3]),
        );
        // One-hot target for class 0
        let targets = Tensor::<CpuBackend>::from_vec(
            vec![1.0, 0.0, 0.0],
            Shape::new(vec![1, 3]),
        );

        let loss = soft_cross_entropy_loss(&logits, &targets);
        // Loss should be -log(softmax(2.0)) for class 0
        assert!(loss.item() > 0.0);
    }
}
