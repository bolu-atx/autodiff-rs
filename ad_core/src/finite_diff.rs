//! Finite difference utilities for gradient verification.
//!
//! Provides numerical gradient computation for testing autodiff correctness.

/// Compute gradients using central finite differences.
///
/// # Arguments
/// * `f` - Function that takes a slice of variable values and returns a scalar
/// * `point` - The point at which to compute gradients
/// * `eps` - Step size for finite differences (typically 1e-7 to 1e-5)
///
/// # Returns
/// Vector of partial derivatives [df/dx_0, df/dx_1, ...] at the given point
///
/// # Example
/// ```
/// use ad_core::finite_diff_grad;
///
/// // f(x, y) = x^2 + y^2
/// // df/dx = 2x, df/dy = 2y
/// let f = |v: &[f64]| v[0] * v[0] + v[1] * v[1];
/// let grads = finite_diff_grad(f, &[3.0, 4.0], 1e-7);
///
/// assert!((grads[0] - 6.0).abs() < 1e-5); // df/dx at x=3
/// assert!((grads[1] - 8.0).abs() < 1e-5); // df/dy at y=4
/// ```
pub fn finite_diff_grad<F>(f: F, point: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = point.len();
    let mut grads = Vec::with_capacity(n);
    let mut perturbed = point.to_vec();

    for i in 0..n {
        // Central difference: (f(x + eps) - f(x - eps)) / (2 * eps)
        perturbed[i] = point[i] + eps;
        let f_plus = f(&perturbed);

        perturbed[i] = point[i] - eps;
        let f_minus = f(&perturbed);

        perturbed[i] = point[i]; // restore

        grads.push((f_plus - f_minus) / (2.0 * eps));
    }

    grads
}

/// Compute the maximum absolute difference between two gradient vectors.
///
/// Useful for comparing autodiff gradients against finite difference gradients.
pub fn max_grad_error(grad1: &[f64], grad2: &[f64]) -> f64 {
    assert_eq!(grad1.len(), grad2.len());
    grad1
        .iter()
        .zip(grad2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_diff_quadratic() {
        // f(x, y) = x^2 + 2*x*y + y^2
        // df/dx = 2x + 2y
        // df/dy = 2x + 2y
        let f = |v: &[f64]| v[0] * v[0] + 2.0 * v[0] * v[1] + v[1] * v[1];
        let grads = finite_diff_grad(f, &[1.0, 2.0], 1e-7);

        assert!((grads[0] - 6.0).abs() < 1e-5); // 2*1 + 2*2 = 6
        assert!((grads[1] - 6.0).abs() < 1e-5); // 2*1 + 2*2 = 6
    }

    #[test]
    fn test_finite_diff_transcendental() {
        // f(x) = sin(x) * exp(x)
        // df/dx = cos(x) * exp(x) + sin(x) * exp(x) = (cos(x) + sin(x)) * exp(x)
        let f = |v: &[f64]| v[0].sin() * v[0].exp();
        let grads = finite_diff_grad(f, &[1.0], 1e-7);

        let expected = (1.0_f64.cos() + 1.0_f64.sin()) * 1.0_f64.exp();
        assert!((grads[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_max_grad_error() {
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![1.1, 2.0, 2.8];

        let err = max_grad_error(&g1, &g2);
        assert!((err - 0.2).abs() < 1e-10);
    }
}
