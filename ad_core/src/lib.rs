//! # ad_core - Reverse-mode Automatic Differentiation Engine
//!
//! This crate provides a simple reverse-mode autodiff implementation for scalar-valued
//! functions f: ℝⁿ → ℝ. It builds a computation graph during the forward pass and
//! computes gradients via reverse accumulation (backpropagation).
//!
//! ## Overview
//!
//! Automatic differentiation computes exact derivatives (not numerical approximations)
//! by applying the chain rule systematically. Reverse-mode autodiff is efficient for
//! functions with many inputs and few outputs, making it ideal for gradient-based
//! optimization (e.g., training neural networks).
//!
//! ## Quick Start
//!
//! ```
//! use ad_core::{var, constant};
//!
//! // Create variables (the inputs we differentiate with respect to)
//! let x = var("x", 2.0);
//! let y = var("y", 3.0);
//!
//! // Build an expression: z = x * y + sin(x)
//! let z = &x * &y + x.sin();
//!
//! // Evaluate the expression
//! assert!((z.value() - 6.909297426825682).abs() < 1e-10);
//!
//! // Compute gradients via reverse-mode autodiff
//! let grads = z.backward();
//! let dz_dx = grads.wrt(&x).unwrap();
//! let dz_dy = grads.wrt(&y).unwrap();
//!
//! // dz/dx = y + cos(x) = 3 + cos(2) ≈ 2.5838531634528574
//! // dz/dy = x = 2
//! assert!((dz_dx - 2.5838531634528574).abs() < 1e-10);
//! assert!((dz_dy - 2.0).abs() < 1e-10);
//! ```
//!
//! ## Supported Operations
//!
//! | Category | Operations |
//! |----------|------------|
//! | Arithmetic | `+`, `-`, `*`, `/`, unary `-` |
//! | Power | [`Expr::powf`] (x^c for constant c) |
//! | Transcendental | [`Expr::exp`], [`Expr::log`], [`Expr::sin`], [`Expr::cos`] |
//!
//! ## Architecture
//!
//! - **[`Expr`]**: Reference-counted handle to a computation graph node. Cloning is O(1).
//! - **[`Gradients`]**: Result of backward pass, queryable by expression or variable name.
//! - **[`finite_diff_grad`]**: Utility for validating gradients against numerical derivatives.
//!
//! ## Example: Using with Multiple Variables
//!
//! ```
//! use ad_core::{var, constant, Gradients};
//!
//! // Compute gradients for f(x,y) = x^2 * y + y^3
//! let x = var("x", 2.0);
//! let y = var("y", 3.0);
//!
//! let f = &x.powf(2.0) * &y + y.powf(3.0);
//!
//! let grads = f.backward();
//! // df/dx = 2xy = 12
//! // df/dy = x^2 + 3y^2 = 4 + 27 = 31
//! assert!((grads.wrt(&x).unwrap() - 12.0).abs() < 1e-10);
//! assert!((grads.wrt(&y).unwrap() - 31.0).abs() < 1e-10);
//! ```

mod node;
mod ops;
mod backward;
mod finite_diff;

pub use node::{Expr, NodeId};
pub use backward::Gradients;
pub use finite_diff::{finite_diff_grad, max_grad_error};

/// Create a new variable with the given name and value.
///
/// Variables are the leaves of the computation graph that we differentiate with respect to.
/// Each call creates a new variable with a unique identity, even if the name is the same.
pub fn var(name: &str, value: f64) -> Expr {
    Expr::var(name, value)
}

/// Create a constant expression.
///
/// Constants have zero gradient - they are treated as fixed values in differentiation.
pub fn constant(value: f64) -> Expr {
    Expr::constant(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let x = var("x", 2.0);
        let y = var("y", 3.0);

        // Test addition
        let sum = &x + &y;
        assert!((sum.value() - 5.0).abs() < 1e-10);

        // Test subtraction
        let diff = &x - &y;
        assert!((diff.value() - (-1.0)).abs() < 1e-10);

        // Test multiplication
        let prod = &x * &y;
        assert!((prod.value() - 6.0).abs() < 1e-10);

        // Test division
        let quot = &x / &y;
        assert!((quot.value() - (2.0 / 3.0)).abs() < 1e-10);

        // Test negation
        let neg = -&x;
        assert!((neg.value() - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_transcendental_functions() {
        let x = var("x", 1.0);

        assert!((x.exp().value() - 1.0_f64.exp()).abs() < 1e-10);
        assert!((x.log().value() - 1.0_f64.ln()).abs() < 1e-10);
        assert!((x.sin().value() - 1.0_f64.sin()).abs() < 1e-10);
        assert!((x.cos().value() - 1.0_f64.cos()).abs() < 1e-10);
        assert!((x.powf(2.0).value() - 1.0).abs() < 1e-10);

        let y = var("y", 2.0);
        assert!((y.powf(3.0).value() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_add() {
        // z = x + y
        // dz/dx = 1, dz/dy = 1
        let x = var("x", 2.0);
        let y = var("y", 3.0);
        let z = &x + &y;

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 1.0).abs() < 1e-10);
        assert!((grads.wrt(&y).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_sub() {
        // z = x - y
        // dz/dx = 1, dz/dy = -1
        let x = var("x", 2.0);
        let y = var("y", 3.0);
        let z = &x - &y;

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 1.0).abs() < 1e-10);
        assert!((grads.wrt(&y).unwrap() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_mul() {
        // z = x * y
        // dz/dx = y, dz/dy = x
        let x = var("x", 2.0);
        let y = var("y", 3.0);
        let z = &x * &y;

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 3.0).abs() < 1e-10);
        assert!((grads.wrt(&y).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_div() {
        // z = x / y
        // dz/dx = 1/y, dz/dy = -x/y^2
        let x = var("x", 2.0);
        let y = var("y", 4.0);
        let z = &x / &y;

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 0.25).abs() < 1e-10);
        assert!((grads.wrt(&y).unwrap() - (-2.0 / 16.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_neg() {
        // z = -x
        // dz/dx = -1
        let x = var("x", 2.0);
        let z = -&x;

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_powf() {
        // z = x^3
        // dz/dx = 3*x^2
        let x = var("x", 2.0);
        let z = x.powf(3.0);

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_exp() {
        // z = exp(x)
        // dz/dx = exp(x)
        let x = var("x", 1.0);
        let z = x.exp();

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 1.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_log() {
        // z = log(x)
        // dz/dx = 1/x
        let x = var("x", 2.0);
        let z = x.log();

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_sin() {
        // z = sin(x)
        // dz/dx = cos(x)
        let x = var("x", 1.0);
        let z = x.sin();

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 1.0_f64.cos()).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_cos() {
        // z = cos(x)
        // dz/dx = -sin(x)
        let x = var("x", 1.0);
        let z = x.cos();

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - (-1.0_f64.sin())).abs() < 1e-10);
    }

    #[test]
    fn test_chain_rule() {
        // z = sin(x^2)
        // dz/dx = cos(x^2) * 2x
        let x = var("x", 2.0);
        let z = x.powf(2.0).sin();

        let grads = z.backward();
        let expected = 4.0_f64.cos() * 4.0;
        assert!((grads.wrt(&x).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_reused_variable() {
        // z = x * x (same variable used twice)
        // dz/dx = 2x
        let x = var("x", 3.0);
        let z = &x * &x;

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_diamond_graph() {
        // Diamond-shaped graph: z = (x + y) * (x - y) = x^2 - y^2
        // dz/dx = 2x, dz/dy = -2y
        let x = var("x", 3.0);
        let y = var("y", 2.0);
        let a = &x + &y;
        let b = &x - &y;
        let z = &a * &b;

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 6.0).abs() < 1e-10);
        assert!((grads.wrt(&y).unwrap() - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_complex_expression() {
        // z = (x*y + sin(x)) / (y + 2)
        let x = var("x", 1.0);
        let y = var("y", 2.0);

        let numerator = &x * &y + x.sin();
        let denominator = &y + constant(2.0);
        let z = numerator / denominator;

        // Verify value
        let expected_value = (1.0 * 2.0 + 1.0_f64.sin()) / (2.0 + 2.0);
        assert!((z.value() - expected_value).abs() < 1e-10);

        // Verify gradients against finite differences
        let grads = z.backward();
        let vars = vec![x.clone(), y.clone()];
        let f = |vals: &[f64]| {
            let x = var("x", vals[0]);
            let y = var("y", vals[1]);
            let numerator = &x * &y + x.sin();
            let denominator = &y + constant(2.0);
            (numerator / denominator).value()
        };

        let fd_grads = finite_diff_grad(f, &[1.0, 2.0], 1e-7);
        assert!((grads.wrt(&vars[0]).unwrap() - fd_grads[0]).abs() < 1e-5);
        assert!((grads.wrt(&vars[1]).unwrap() - fd_grads[1]).abs() < 1e-5);
    }

    #[test]
    fn test_constant_gradient() {
        // z = x + 5
        // dz/dx = 1, constant has no gradient
        let x = var("x", 2.0);
        let c = constant(5.0);
        let z = &x + &c;

        let grads = z.backward();
        assert!((grads.wrt(&x).unwrap() - 1.0).abs() < 1e-10);
        // Constants don't appear in all_named_grads
        assert!(grads.all_named_grads().get("5").is_none());
    }

    #[test]
    fn test_grad_by_name() {
        let x = var("x", 2.0);
        let y = var("y", 3.0);
        let z = &x * &y;

        let grads = z.backward();
        assert!((grads.by_name("x").unwrap() - 3.0).abs() < 1e-10);
        assert!((grads.by_name("y").unwrap() - 2.0).abs() < 1e-10);
        assert!(grads.by_name("nonexistent").is_none());
    }

    #[test]
    fn test_finite_diff_random_graph() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Random values
        let x_val: f64 = rng.gen_range(-2.0..2.0);
        let y_val: f64 = rng.gen_range(0.5..2.0); // positive for log

        let x = var("x", x_val);
        let y = var("y", y_val);

        // Random-ish expression: exp(x) * log(y) + sin(x*y)
        let z = x.exp() * y.log() + (&x * &y).sin();

        let grads = z.backward();

        let f = |vals: &[f64]| {
            let x = var("x", vals[0]);
            let y = var("y", vals[1]);
            (x.exp() * y.log() + (&x * &y).sin()).value()
        };

        let fd_grads = finite_diff_grad(f, &[x_val, y_val], 1e-7);

        assert!(
            (grads.wrt(&x).unwrap() - fd_grads[0]).abs() < 1e-5,
            "dz/dx mismatch: autodiff={}, fd={}",
            grads.wrt(&x).unwrap(),
            fd_grads[0]
        );
        assert!(
            (grads.wrt(&y).unwrap() - fd_grads[1]).abs() < 1e-5,
            "dz/dy mismatch: autodiff={}, fd={}",
            grads.wrt(&y).unwrap(),
            fd_grads[1]
        );
    }
}
