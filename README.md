# autodiff-rs

A learning project implementing reverse-mode automatic differentiation in Rust with Python bindings.

[![CI](https://github.com/bolu-atx/autodiff-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bolu-atx/autodiff-rs/actions/workflows/ci.yml)

## Overview

This project demonstrates how to build a complete autodiff system from scratch:

- **Scalar autodiff** with expression nodes and cheap cloning via `Arc`
- **Tensor autodiff** with pluggable compute backends (CPU with SIMD, Metal GPU)
- **Reverse-mode differentiation** (backpropagation) using topological sorting
- **Neural network primitives** including layers, activations, losses, and optimizers
- **Python bindings** via PyO3 and maturin

## Project Structure

```
autodiff-rs/
├── ad_core/           # Scalar autodiff engine
├── ad_tensor/         # Tensor autodiff (backend-agnostic)
├── ad_backend_cpu/    # CPU backend with SIMD (AVX2/NEON)
├── ad_backend_metal/  # Metal GPU backend (Apple Silicon)
├── ad_nn/             # Neural network layers, losses, optimizers
├── ad_py/             # Python bindings (PyO3/maturin)
├── ad_cli/            # CLI demo
└── tests/             # Python tests
```

## Quick Start

### Scalar Autodiff (Rust)

```rust
use ad_core::{var, constant};

let x = var("x", 2.0);
let y = var("y", 3.0);

// z = x * y + sin(x)
let z = &x * &y + x.sin();

println!("z = {}", z.value());  // 6.909...

let grads = z.backward();
println!("dz/dx = {}", grads.wrt(&x).unwrap());  // 3.583...
println!("dz/dy = {}", grads.wrt(&y).unwrap());  // 2.0
```

### Tensor Autodiff (Rust)

```rust
use ad_tensor::prelude::*;
use ad_backend_cpu::CpuBackend;

type T = Tensor<CpuBackend>;

// Create tensors
let x = T::var("x", CpuBackend::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3])));
let y = T::var("y", CpuBackend::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3])));

// Element-wise: z = x * y + exp(x)
let z = &x * &y + x.exp();

// Compute gradients (reduce to scalar first)
let grads = z.sum(None, false).backward();
let dx = grads.wrt(&x).unwrap();
```

### Neural Network Training

```rust
use ad_tensor::prelude::*;
use ad_backend_cpu::CpuBackend;
use ad_nn::{Linear, Adam, relu, mse_loss};

// Create a 2-layer network: 2 -> 8 -> 1
let mut l1 = Linear::new(2, 8, true);  // with bias
let mut l2 = Linear::new(8, 1, true);
let mut opt = Adam::new(0.01);

// Training loop
for (input, target) in dataset {
    let x = Tensor::var("x", CpuBackend::from_vec(input, Shape::new(vec![1, 2])));
    let y = Tensor::constant(CpuBackend::from_vec(target, Shape::new(vec![1, 1])));

    // Forward pass
    let h = relu(&l1.forward(&x));
    let pred = l2.forward(&h);
    let loss = mse_loss(&pred, &y);

    // Backward pass + optimization
    let grads = loss.backward();
    opt.step(&mut l1.weight, grads.wrt(&l1.weight).unwrap());
    opt.step(&mut l2.weight, grads.wrt(&l2.weight).unwrap());
}
```

### Python

```python
from ad_py import var, constant

x = var("x", 2.0)
y = var("y", 3.0)

z = x * y + x.sin()
print(f"z = {z.value()}")  # 6.909...

grads = z.backward()
print(f"dz/dx = {grads['x']}")  # 3.583...
print(f"dz/dy = {grads['y']}")  # 2.0
```

## Supported Operations

### Scalar Operations

| Operation | Rust | Python |
|-----------|------|--------|
| Addition | `&a + &b` | `a + b` |
| Subtraction | `&a - &b` | `a - b` |
| Multiplication | `&a * &b` | `a * b` |
| Division | `&a / &b` | `a / b` |
| Negation | `-&a` | `-a` |
| Power | `a.powf(c)` | `a ** c` |
| Exponential | `a.exp()` | `a.exp()` |
| Logarithm | `a.log()` | `a.log()` |
| Sine | `a.sin()` | `a.sin()` |
| Cosine | `a.cos()` | `a.cos()` |

### Tensor Operations

| Category | Operations |
|----------|------------|
| Element-wise | `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sin`, `cos`, `sqrt` |
| Activations | `relu`, `sigmoid`, `tanh` |
| Linear Algebra | `matmul`, `transpose` |
| Reductions | `sum`, `mean`, `max` (with axes, keepdims) |
| Shape | `reshape`, `broadcast_to`, `squeeze`, `unsqueeze` |

### Neural Network Components

| Component | Available |
|-----------|-----------|
| Layers | `Linear` (dense/fully-connected) |
| Activations | `relu`, `sigmoid`, `tanh` |
| Losses | `mse_loss`, `binary_cross_entropy_with_logits`, `soft_cross_entropy_loss` |
| Optimizers | `SGD` (with momentum), `Adam` |

## Compute Backends

### CPU Backend (`ad_backend_cpu`)

- SIMD acceleration: AVX2 (x86_64), NEON (ARM)
- Runtime feature detection
- Contiguous row-major storage

### Metal Backend (`ad_backend_metal`)

- Apple GPU acceleration via Metal compute shaders
- Optimized for Apple Silicon (M1/M2/M3)

## Building

### Prerequisites

- Rust (stable, 2021 edition)
- Python 3.8+ (for bindings)
- [maturin](https://github.com/PyO3/maturin) (for Python builds)
- [uv](https://github.com/astral-sh/uv) (recommended for Python env management)

### Commands

```bash
# Run all Rust tests
cargo test --workspace

# Run neural network integration tests
cargo test -p ad_nn

# Run CLI demo
cargo run -p ad_cli

# Build Python bindings
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install maturin pytest
cd ad_py && maturin develop

# Run Python tests
pytest tests/ -v

# Generate documentation
cargo doc --workspace --open
```

## How It Works

### Forward Pass

Expressions are built by composing operations. Each operation creates a new node in the computation graph:

```rust
let x = var("x", 2.0);  // Leaf node (variable)
let y = x.sin();        // Unary op node
let z = &y * &x;        // Binary op node
```

Nodes are reference-counted (`Arc<Node>`), so cloning is O(1) and subexpressions can be shared.

### Backward Pass

Gradient computation uses reverse-mode autodiff:

1. **Topological sort**: DFS from output to find all reachable nodes
2. **Reverse traversal**: Walk from output to leaves
3. **Chain rule**: Accumulate adjoints: `child_adjoint += parent_adjoint * local_gradient`

For tensors, broadcasting is handled automatically with gradient reduction along broadcast dimensions.

## Inspiration & References

- **[micrograd](https://github.com/karpathy/micrograd)** by Andrej Karpathy - A tiny scalar-valued autograd engine in Python
- **["Automatic Differentiation in Machine Learning: A Survey"](https://arxiv.org/abs/1502.05767)** (Baydin et al., 2018)
- **["Calculus on Computational Graphs: Backpropagation"](https://colah.github.io/posts/2015-08-Backprop/)** by Chris Olah
- **[PyTorch Autograd](https://pytorch.org/docs/stable/autograd.html)** - Production-grade implementation
- **[The Simple Essence of Automatic Differentiation](http://conal.net/papers/essence-of-ad/)** (Elliott, 2018)

## License

MIT
