//! Local gradient computations for each operation.
//!
//! Each operation knows how to compute its local gradients with respect to its inputs.
//! These local gradients are used during the backward pass to accumulate adjoints.

use crate::node::{Expr, Op};

/// Compute local gradients for a node given its children's values.
///
/// Returns a vector of (child_index, local_gradient) pairs.
/// The local gradient is d(output)/d(input_i).
///
/// For nodes with multiple inputs, returns gradients for all inputs.
pub fn local_gradients(op: &Op, children: &[Expr]) -> Vec<f64> {
    match op {
        Op::Const(_) | Op::Var { .. } => {
            // Leaf nodes have no children
            vec![]
        }

        Op::Add => {
            // z = a + b
            // dz/da = 1, dz/db = 1
            vec![1.0, 1.0]
        }

        Op::Sub => {
            // z = a - b
            // dz/da = 1, dz/db = -1
            vec![1.0, -1.0]
        }

        Op::Mul => {
            // z = a * b
            // dz/da = b, dz/db = a
            let a_val = children[0].value();
            let b_val = children[1].value();
            vec![b_val, a_val]
        }

        Op::Div => {
            // z = a / b
            // dz/da = 1/b, dz/db = -a/b^2
            let a_val = children[0].value();
            let b_val = children[1].value();
            vec![1.0 / b_val, -a_val / (b_val * b_val)]
        }

        Op::Neg => {
            // z = -a
            // dz/da = -1
            vec![-1.0]
        }

        Op::Pow { exponent } => {
            // z = a^c (c is constant)
            // dz/da = c * a^(c-1)
            let a_val = children[0].value();
            vec![exponent * a_val.powf(exponent - 1.0)]
        }

        Op::Exp => {
            // z = exp(a)
            // dz/da = exp(a)
            let a_val = children[0].value();
            vec![a_val.exp()]
        }

        Op::Log => {
            // z = ln(a)
            // dz/da = 1/a
            let a_val = children[0].value();
            vec![1.0 / a_val]
        }

        Op::Sin => {
            // z = sin(a)
            // dz/da = cos(a)
            let a_val = children[0].value();
            vec![a_val.cos()]
        }

        Op::Cos => {
            // z = cos(a)
            // dz/da = -sin(a)
            let a_val = children[0].value();
            vec![-a_val.sin()]
        }
    }
}
