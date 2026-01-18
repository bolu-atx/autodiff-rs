//! XOR problem training example.
//!
//! Demonstrates training a simple MLP to learn the XOR function.
//! XOR is a classic non-linearly separable problem that requires hidden layers.

use ad_backend_cpu::CpuBackend;
use ad_nn::{mse_loss, relu, Linear, SGD};
use ad_tensor::prelude::*;

fn main() {
    // XOR dataset
    let inputs = [
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = [0.0, 1.0, 1.0, 0.0];

    // Create a simple 2-layer MLP: 2 -> 16 -> 1
    let mut layer1 = Linear::new(2, 16, true);
    let mut layer2 = Linear::new(16, 1, true);

    // SGD with smaller learning rate
    let mut opt = SGD::with_momentum(0.1, 0.9);

    println!("Training XOR network...\n");

    for epoch in 0..2000 {
        let mut total_loss = 0.0;

        for (input, &target) in inputs.iter().zip(targets.iter()) {
            // Create input tensor [1, 2] (batch of 1, 2 features)
            let x = Tensor::var(
                "x",
                CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])),
            );
            let y = Tensor::constant(CpuBackend::from_vec(vec![target], Shape::new(vec![1, 1])));

            // Forward pass
            let h = relu(&layer1.forward(&x));
            let pred = layer2.forward(&h);

            // Compute loss (MSE returns a scalar tensor)
            let loss = mse_loss(&pred, &y);
            total_loss += loss.as_slice()[0];

            // Backward pass
            let grads = loss.backward();

            // Update layer1 parameters
            if let Some(g) = grads.wrt(&layer1.weight) {
                opt.step(&mut layer1.weight, g);
            }
            if let Some(bias) = &mut layer1.bias {
                if let Some(g) = grads.wrt(bias) {
                    opt.step(bias, g);
                }
            }

            // Update layer2 parameters
            if let Some(g) = grads.wrt(&layer2.weight) {
                opt.step(&mut layer2.weight, g);
            }
            if let Some(bias) = &mut layer2.bias {
                if let Some(g) = grads.wrt(bias) {
                    opt.step(bias, g);
                }
            }
        }

        if epoch % 200 == 0 || epoch == 1999 {
            println!("Epoch {:4}: avg loss = {:.6}", epoch, total_loss / 4.0);
        }
    }

    // Test the trained network
    println!("\nTesting trained network:");
    println!("========================");

    for (input, &target) in inputs.iter().zip(targets.iter()) {
        let x = Tensor::var(
            "x",
            CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])),
        );

        let h = relu(&layer1.forward(&x));
        let pred = layer2.forward(&h);
        let output = pred.as_slice()[0];

        println!(
            "Input: [{:.0}, {:.0}] -> Output: {:.4} (target: {:.0})",
            input[0], input[1], output, target
        );
    }

    // Check if we learned XOR correctly (output > 0.5 means 1, else 0)
    let mut correct = 0;
    for (input, &target) in inputs.iter().zip(targets.iter()) {
        let x = Tensor::var(
            "x",
            CpuBackend::from_vec(input.clone(), Shape::new(vec![1, 2])),
        );
        let h = relu(&layer1.forward(&x));
        let pred = layer2.forward(&h);
        let output = pred.as_slice()[0];
        let predicted_class = if output > 0.5 { 1.0 } else { 0.0 };
        if (predicted_class - target).abs() < 0.01 {
            correct += 1;
        }
    }

    println!("\nAccuracy: {}/4", correct);
    if correct == 4 {
        println!("Successfully learned XOR!");
    }
}
