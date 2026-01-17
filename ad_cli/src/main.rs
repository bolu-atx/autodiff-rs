//! CLI demo for the reverse-mode autodiff library.
//!
//! Demonstrates building expressions, computing values and gradients,
//! and validating against finite differences.

use ad_core::{constant, finite_diff_grad, var};

fn main() {
    println!("=== Reverse-Mode Autodiff Demo ===\n");

    // Build the expression: z = (x*y + sin(x)) / (y + 2)
    let x_val = 1.5;
    let y_val = 2.5;

    let x = var("x", x_val);
    let y = var("y", y_val);

    let numerator = &x * &y + x.sin();
    let denominator = &y + constant(2.0);
    let z = &numerator / &denominator;

    // Evaluate
    let value = z.value();
    println!("Expression: z = (x*y + sin(x)) / (y + 2)");
    println!("At point:   x = {}, y = {}", x_val, y_val);
    println!("Value:      z = {:.10}\n", value);

    // Compute gradients via autodiff
    let grads = z.backward();
    let dz_dx = grads.wrt(&x).unwrap();
    let dz_dy = grads.wrt(&y).unwrap();

    println!("Autodiff gradients:");
    println!("  dz/dx = {:.10}", dz_dx);
    println!("  dz/dy = {:.10}\n", dz_dy);

    // Compute gradients via finite differences for validation
    let f = |vals: &[f64]| {
        let x = var("x", vals[0]);
        let y = var("y", vals[1]);
        let num = &x * &y + x.sin();
        let den = &y + constant(2.0);
        (&num / &den).value()
    };

    let fd_grads = finite_diff_grad(f, &[x_val, y_val], 1e-7);

    println!("Finite difference gradients (eps=1e-7):");
    println!("  dz/dx = {:.10}", fd_grads[0]);
    println!("  dz/dy = {:.10}\n", fd_grads[1]);

    // Compute and report errors
    let err_x = (dz_dx - fd_grads[0]).abs();
    let err_y = (dz_dy - fd_grads[1]).abs();
    let max_err = err_x.max(err_y);

    println!("Gradient errors:");
    println!("  |autodiff - fd| for x: {:.2e}", err_x);
    println!("  |autodiff - fd| for y: {:.2e}", err_y);
    println!("  Max absolute error:    {:.2e}\n", max_err);

    // Pass/fail check
    let tolerance = 1e-5;
    if max_err < tolerance {
        println!("PASS: Max error ({:.2e}) < tolerance ({:.2e})", max_err, tolerance);
    } else {
        println!("FAIL: Max error ({:.2e}) >= tolerance ({:.2e})", max_err, tolerance);
        std::process::exit(1);
    }

    // Additional examples
    println!("\n=== Additional Examples ===\n");

    // Example 1: Chain rule
    println!("1. Chain rule: z = sin(x^2)");
    let x = var("x", 2.0);
    let z = x.powf(2.0).sin();
    let grads = z.backward();
    println!("   At x = 2.0:");
    println!("   z = {:.10}", z.value());
    println!("   dz/dx = {:.10} (expected: cos(4) * 4 = {:.10})\n",
             grads.wrt(&x).unwrap(),
             4.0_f64.cos() * 4.0);

    // Example 2: Multi-variable with shared nodes
    println!("2. Diamond graph: z = (x + y) * (x - y) = x^2 - y^2");
    let x = var("x", 3.0);
    let y = var("y", 2.0);
    let a = &x + &y;
    let b = &x - &y;
    let z = &a * &b;
    let grads = z.backward();
    println!("   At x = 3.0, y = 2.0:");
    println!("   z = {:.10} (expected: 5)", z.value());
    println!("   dz/dx = {:.10} (expected: 2x = 6)", grads.wrt(&x).unwrap());
    println!("   dz/dy = {:.10} (expected: -2y = -4)\n", grads.wrt(&y).unwrap());

    // Example 3: More complex expression
    println!("3. Complex: z = exp(x) * log(y) + cos(x * y)");
    let x = var("x", 0.5);
    let y = var("y", 2.0);
    let z = x.exp() * y.log() + (&x * &y).cos();
    let grads = z.backward();
    println!("   At x = 0.5, y = 2.0:");
    println!("   z = {:.10}", z.value());
    println!("   dz/dx = {:.10}", grads.wrt(&x).unwrap());
    println!("   dz/dy = {:.10}", grads.wrt(&y).unwrap());

    // Verify with finite differences
    let f = |vals: &[f64]| {
        let x = var("x", vals[0]);
        let y = var("y", vals[1]);
        (x.exp() * y.log() + (&x * &y).cos()).value()
    };
    let fd = finite_diff_grad(f, &[0.5, 2.0], 1e-7);
    println!("   FD check: dz/dx={:.10}, dz/dy={:.10}", fd[0], fd[1]);
}
