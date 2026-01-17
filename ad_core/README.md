# ad_core

Pure Rust reverse-mode automatic differentiation engine for scalar-valued functions.

## Features

- **Reverse-mode autodiff** for functions f: ℝⁿ → ℝ
- **Expression graph** with cheap cloning via `Arc`
- **Supported operations**: add, sub, mul, div, neg, powf, exp, log, sin, cos
- **No dependencies** (only `rand` for tests)

## Quick Start

```rust
use ad_core::{var, constant};

// Create variables
let x = var("x", 2.0);
let y = var("y", 3.0);

// Build expression: z = x * y + sin(x)
let z = &x * &y + x.sin();

// Evaluate
println!("z = {}", z.value());  // 6.909...

// Compute gradients
let grads = z.backward();
println!("dz/dx = {}", grads.wrt(&x).unwrap());  // 3.583...
println!("dz/dy = {}", grads.wrt(&y).unwrap());  // 2.0
```

## How It Works

1. **Forward pass**: Build a computation graph by composing expressions
2. **Backward pass**: Traverse the graph in reverse topological order, accumulating adjoints via the chain rule

The implementation uses `Arc<Node>` for expression sharing, enabling diamond-shaped graphs where the same subexpression feeds into multiple parents.

## API Overview

### Creating Expressions

- `var(name, value)` - Create a variable (differentiable leaf)
- `constant(value)` - Create a constant (zero gradient)

### Operations

| Operation | Syntax |
|-----------|--------|
| Addition | `&a + &b` |
| Subtraction | `&a - &b` |
| Multiplication | `&a * &b` |
| Division | `&a / &b` |
| Negation | `-&a` |
| Power | `a.powf(c)` |
| Exponential | `a.exp()` |
| Logarithm | `a.log()` |
| Sine | `a.sin()` |
| Cosine | `a.cos()` |

### Gradient Computation

```rust
let grads = expr.backward();

// Query by expression
grads.wrt(&x)           // Option<f64>

// Query by name
grads.by_name("x")      // Option<f64>

// Get all named gradients
grads.all_named_grads() // HashMap<String, f64>
```

## Testing

```bash
cargo test -p ad_core
```

Includes gradient correctness tests for all operations and finite-difference validation.
