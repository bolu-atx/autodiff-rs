# autodiff-rs

A learning project implementing reverse-mode automatic differentiation in Rust with Python bindings.

[![CI](https://github.com/bolu-atx/autodiff-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bolu-atx/autodiff-rs/actions/workflows/ci.yml)

## Overview

This project demonstrates how to build a simple but complete autodiff system from scratch:

- **Computation graphs** with expression nodes and cheap cloning via `Arc`
- **Reverse-mode differentiation** (backpropagation) using topological sorting
- **Python bindings** via PyO3 and maturin

## Project Structure

```
autodiff-rs/
├── ad_core/     # Pure Rust autodiff engine
├── ad_cli/      # CLI demo binary
├── ad_py/       # Python bindings (PyO3/maturin)
└── tests/       # Python tests
```

## Quick Start

### Rust

```rust
use ad_core::{var, constant};

// Create variables
let x = var("x", 2.0);
let y = var("y", 3.0);

// Build expression: z = x * y + sin(x)
let z = &x * &y + x.sin();

// Evaluate
println!("z = {}", z.value());  // 6.909...

// Compute gradients via reverse-mode autodiff
let grads = z.backward();
println!("dz/dx = {}", grads.wrt(&x).unwrap());  // 3.583...
println!("dz/dy = {}", grads.wrt(&y).unwrap());  // 2.0
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

| Operation | Rust | Python |
|-----------|------|--------|
| Addition | `&a + &b` | `a + b` |
| Subtraction | `&a - &b` | `a - b` |
| Multiplication | `&a * &b` | `a * b` |
| Division | `&a / &b` | `a / b` |
| Negation | `-&a` | `-a` |
| Power | `a.powf(c)` | `a.pow(c)` or `a ** c` |
| Exponential | `a.exp()` | `a.exp()` |
| Logarithm | `a.log()` | `a.log()` |
| Sine | `a.sin()` | `a.sin()` |
| Cosine | `a.cos()` | `a.cos()` |

## Building

### Prerequisites

- Rust (stable, 2021 edition)
- Python 3.8+ (for bindings)
- [maturin](https://github.com/PyO3/maturin) (for Python builds)
- [uv](https://github.com/astral-sh/uv) (recommended for Python env management)

### Commands

```bash
# Run Rust tests
cargo test -p ad_core

# Run CLI demo
cargo run -p ad_cli

# Build Python bindings (with venv)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install maturin pytest
cd ad_py && maturin develop

# Run Python tests
pytest tests/ -v

# Generate documentation
cargo doc -p ad_core --open
```

## How It Works

### Forward Pass

Expressions are built by composing operations. Each operation creates a new node in the computation graph:

```rust
let x = var("x", 2.0);  // Leaf node (variable)
let y = x.sin();        // Unary op node: sin(x)
let z = &y * &x;        // Binary op node: sin(x) * x
```

Nodes are reference-counted (`Arc<Node>`), so cloning is O(1) and subexpressions can be shared.

### Backward Pass

Gradient computation uses reverse-mode autodiff:

1. **Topological sort**: DFS from output to find all reachable nodes
2. **Reverse traversal**: Walk from output to leaves
3. **Chain rule**: Accumulate adjoints: `child_adjoint += parent_adjoint * local_gradient`

This computes all gradients in a single backward pass, making it efficient for many-input, single-output functions.

## Inspiration & References

This project draws from several excellent resources on automatic differentiation:

- **[micrograd](https://github.com/karpathy/micrograd)** by Andrej Karpathy - A tiny scalar-valued autograd engine in Python. The simplicity and educational clarity of micrograd heavily influenced this implementation.

- **["Automatic Differentiation in Machine Learning: A Survey"](https://arxiv.org/abs/1502.05767)** (Baydin et al., 2018) - Comprehensive overview of AD techniques.

- **["Calculus on Computational Graphs: Backpropagation"](https://colah.github.io/posts/2015-08-Backprop/)** by Chris Olah - Intuitive visual explanation of backprop.

- **[PyTorch Autograd](https://pytorch.org/docs/stable/autograd.html)** - Production-grade implementation that inspired the API design.

- **[The Simple Essence of Automatic Differentiation](http://conal.net/papers/essence-of-ad/)** (Elliott, 2018) - Elegant functional perspective on AD.

## License

MIT
