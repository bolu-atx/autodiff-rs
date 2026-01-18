//! Reverse-mode automatic differentiation for tensors.

use std::collections::{HashMap, HashSet};

use crate::backend::Backend;
use crate::node::{NodeId, Tensor, TensorOp};

/// Container for gradient tensors computed during backward pass.
pub struct Gradients<B: Backend> {
    /// Map from node ID to gradient tensor.
    adjoints: HashMap<NodeId, B::Tensor>,
    /// Map from variable name to (NodeId, gradient) pairs.
    name_to_grads: HashMap<String, Vec<(NodeId, B::Tensor)>>,
}

impl<B: Backend> Gradients<B> {
    /// Get gradient with respect to a tensor expression.
    pub fn wrt(&self, expr: &Tensor<B>) -> Option<&B::Tensor> {
        self.adjoints.get(&expr.id())
    }

    /// Get gradient by variable name (first match if multiple).
    pub fn by_name(&self, name: &str) -> Option<&B::Tensor> {
        self.name_to_grads
            .get(name)
            .and_then(|grads| grads.first().map(|(_, g)| g))
    }

    /// Get all named gradients.
    pub fn all_named(&self) -> &HashMap<String, Vec<(NodeId, B::Tensor)>> {
        &self.name_to_grads
    }
}

/// Compute gradients via reverse-mode autodiff.
pub fn backward<B: Backend>(output: &Tensor<B>) -> Gradients<B> {
    // Step 1: Topological sort
    let topo_order = topological_sort(output);

    // Step 2: Initialize adjoints - output gradient is 1 (or ones tensor)
    let mut adjoints: HashMap<NodeId, B::Tensor> = HashMap::new();
    adjoints.insert(output.id(), B::ones(output.shape()));

    // Step 3: Reverse traversal
    for expr in topo_order.iter().rev() {
        let Some(node_adjoint) = adjoints.get(&expr.id()) else {
            continue;
        };
        let node_adjoint = B::clone_tensor(node_adjoint);

        // Compute local gradients for each child
        let child_grads = compute_local_gradients::<B>(expr, &node_adjoint);

        // Accumulate gradients to children
        for (i, child) in expr.children().iter().enumerate() {
            if let Some(grad) = &child_grads[i] {
                adjoints
                    .entry(child.id())
                    .and_modify(|existing| B::accumulate_grad(existing, grad))
                    .or_insert_with(|| B::clone_tensor(grad));
            }
        }
    }

    // Step 4: Build name -> gradients map
    let mut name_to_grads: HashMap<String, Vec<(NodeId, B::Tensor)>> = HashMap::new();
    for expr in &topo_order {
        if let TensorOp::Var { name } = expr.op() {
            if let Some(grad) = adjoints.get(&expr.id()) {
                name_to_grads
                    .entry(name.clone())
                    .or_default()
                    .push((expr.id(), B::clone_tensor(grad)));
            }
        }
    }

    Gradients {
        adjoints,
        name_to_grads,
    }
}

/// Compute local gradients for each child of a node.
/// Returns a vector where each element is Option<gradient> for the corresponding child.
fn compute_local_gradients<B: Backend>(
    expr: &Tensor<B>,
    upstream_grad: &B::Tensor,
) -> Vec<Option<B::Tensor>> {
    let children = expr.children();

    match expr.op() {
        TensorOp::Const | TensorOp::Var { .. } => vec![],

        // === Unary element-wise ===
        TensorOp::Neg => {
            // d(-x)/dx = -1
            vec![Some(B::neg(upstream_grad))]
        }

        TensorOp::Exp => {
            // d(exp(x))/dx = exp(x)
            // grad = upstream * exp(x) = upstream * output
            vec![Some(B::mul(upstream_grad, expr.data()))]
        }

        TensorOp::Log => {
            // d(ln(x))/dx = 1/x
            vec![Some(B::div(upstream_grad, children[0].data()))]
        }

        TensorOp::Sin => {
            // d(sin(x))/dx = cos(x)
            let cos_x = B::cos(children[0].data());
            vec![Some(B::mul(upstream_grad, &cos_x))]
        }

        TensorOp::Cos => {
            // d(cos(x))/dx = -sin(x)
            let sin_x = B::sin(children[0].data());
            let neg_sin_x = B::neg(&sin_x);
            vec![Some(B::mul(upstream_grad, &neg_sin_x))]
        }

        TensorOp::Relu => {
            // d(relu(x))/dx = 1 if x > 0 else 0
            let zero = B::zeros(children[0].shape());
            let mask = B::gt(children[0].data(), &zero);
            vec![Some(B::mul(upstream_grad, &mask))]
        }

        TensorOp::Sigmoid => {
            // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
            // output = sigmoid(x), so grad = upstream * output * (1 - output)
            let one = B::ones(expr.shape());
            let one_minus_out = B::sub(&one, expr.data());
            let local = B::mul(expr.data(), &one_minus_out);
            vec![Some(B::mul(upstream_grad, &local))]
        }

        TensorOp::Tanh => {
            // d(tanh(x))/dx = 1 - tanh(x)^2
            let out_sq = B::mul(expr.data(), expr.data());
            let one = B::ones(expr.shape());
            let local = B::sub(&one, &out_sq);
            vec![Some(B::mul(upstream_grad, &local))]
        }

        TensorOp::Sqrt => {
            // d(sqrt(x))/dx = 1 / (2 * sqrt(x)) = 0.5 / sqrt(x)
            let half = B::scalar(0.5);
            let half_broadcast = B::broadcast_to(&half, expr.shape());
            let local = B::div(&half_broadcast, expr.data());
            vec![Some(B::mul(upstream_grad, &local))]
        }

        // === Binary element-wise with broadcasting ===
        TensorOp::Add => {
            // d(a+b)/da = 1, d(a+b)/db = 1
            let grad_a = B::sum_to(upstream_grad, children[0].shape());
            let grad_b = B::sum_to(upstream_grad, children[1].shape());
            vec![Some(grad_a), Some(grad_b)]
        }

        TensorOp::Sub => {
            // d(a-b)/da = 1, d(a-b)/db = -1
            let grad_a = B::sum_to(upstream_grad, children[0].shape());
            let neg_upstream = B::neg(upstream_grad);
            let grad_b = B::sum_to(&neg_upstream, children[1].shape());
            vec![Some(grad_a), Some(grad_b)]
        }

        TensorOp::Mul => {
            // d(a*b)/da = b, d(a*b)/db = a
            let b_broadcast = B::broadcast_to(children[1].data(), expr.shape());
            let a_broadcast = B::broadcast_to(children[0].data(), expr.shape());

            let local_a = B::mul(upstream_grad, &b_broadcast);
            let local_b = B::mul(upstream_grad, &a_broadcast);

            let grad_a = B::sum_to(&local_a, children[0].shape());
            let grad_b = B::sum_to(&local_b, children[1].shape());
            vec![Some(grad_a), Some(grad_b)]
        }

        TensorOp::Div => {
            // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
            let a = children[0].data();
            let b = children[1].data();

            let b_broadcast = B::broadcast_to(b, expr.shape());
            let a_broadcast = B::broadcast_to(a, expr.shape());

            // grad_a = upstream / b
            let one_over_b = B::div(&B::ones(expr.shape()), &b_broadcast);
            let local_a = B::mul(upstream_grad, &one_over_b);
            let grad_a = B::sum_to(&local_a, children[0].shape());

            // grad_b = -upstream * a / b^2
            let b_sq = B::mul(&b_broadcast, &b_broadcast);
            let neg_a_over_b_sq = B::neg(&B::div(&a_broadcast, &b_sq));
            let local_b = B::mul(upstream_grad, &neg_a_over_b_sq);
            let grad_b = B::sum_to(&local_b, children[1].shape());

            vec![Some(grad_a), Some(grad_b)]
        }

        TensorOp::Pow => {
            // d(a^b)/da = b * a^(b-1)
            // d(a^b)/db = a^b * ln(a)
            let a = children[0].data();
            let b = children[1].data();

            let a_broadcast = B::broadcast_to(a, expr.shape());
            let b_broadcast = B::broadcast_to(b, expr.shape());

            // grad_a = upstream * b * a^(b-1)
            let one = B::ones(expr.shape());
            let b_minus_1 = B::sub(&b_broadcast, &one);
            let a_pow_b_minus_1 = B::pow(&a_broadcast, &b_minus_1);
            let local_a = B::mul(&B::mul(&b_broadcast, &a_pow_b_minus_1), upstream_grad);
            let grad_a = B::sum_to(&local_a, children[0].shape());

            // grad_b = upstream * a^b * ln(a)
            let ln_a = B::log(&a_broadcast);
            let local_b = B::mul(&B::mul(expr.data(), &ln_a), upstream_grad);
            let grad_b = B::sum_to(&local_b, children[1].shape());

            vec![Some(grad_a), Some(grad_b)]
        }

        TensorOp::Maximum => {
            // d(max(a,b))/da = 1 if a >= b else 0
            // d(max(a,b))/db = 1 if b > a else 0
            let a = children[0].data();
            let b = children[1].data();

            let a_broadcast = B::broadcast_to(a, expr.shape());
            let b_broadcast = B::broadcast_to(b, expr.shape());

            let mask_a = B::ge(&a_broadcast, &b_broadcast);
            let mask_b = B::gt(&b_broadcast, &a_broadcast);

            let local_a = B::mul(upstream_grad, &mask_a);
            let local_b = B::mul(upstream_grad, &mask_b);

            let grad_a = B::sum_to(&local_a, children[0].shape());
            let grad_b = B::sum_to(&local_b, children[1].shape());

            vec![Some(grad_a), Some(grad_b)]
        }

        TensorOp::Minimum => {
            // d(min(a,b))/da = 1 if a <= b else 0
            // d(min(a,b))/db = 1 if b < a else 0
            let a = children[0].data();
            let b = children[1].data();

            let a_broadcast = B::broadcast_to(a, expr.shape());
            let b_broadcast = B::broadcast_to(b, expr.shape());

            let mask_a = B::le(&a_broadcast, &b_broadcast);
            let mask_b = B::lt(&b_broadcast, &a_broadcast);

            let local_a = B::mul(upstream_grad, &mask_a);
            let local_b = B::mul(upstream_grad, &mask_b);

            let grad_a = B::sum_to(&local_a, children[0].shape());
            let grad_b = B::sum_to(&local_b, children[1].shape());

            vec![Some(grad_a), Some(grad_b)]
        }

        // === Reductions ===
        TensorOp::Sum { axes, keepdims } => {
            // Gradient flows back by broadcasting
            let grad = if *keepdims {
                B::broadcast_to(upstream_grad, children[0].shape())
            } else {
                // Need to unsqueeze reduced axes first
                let expanded = expand_for_broadcast::<B>(upstream_grad, children[0].shape(), axes.as_deref());
                B::broadcast_to(&expanded, children[0].shape())
            };
            vec![Some(grad)]
        }

        TensorOp::Mean { axes, keepdims } => {
            // Same as sum but divide by count
            let input_shape = children[0].shape();
            let count = if let Some(axes) = axes {
                axes.iter().map(|&ax| input_shape.dim(ax)).product::<usize>()
            } else {
                input_shape.numel()
            };

            let grad = if *keepdims {
                B::broadcast_to(upstream_grad, input_shape)
            } else {
                let expanded = expand_for_broadcast::<B>(upstream_grad, input_shape, axes.as_deref());
                B::broadcast_to(&expanded, input_shape)
            };

            let count_tensor = B::scalar(count as f32);
            let count_broadcast = B::broadcast_to(&count_tensor, input_shape);
            vec![Some(B::div(&grad, &count_broadcast))]
        }

        TensorOp::Max { axes, keepdims } | TensorOp::Min { axes, keepdims } => {
            // Gradient only flows to the max/min element(s)
            let input_shape = children[0].shape();

            // Expand upstream gradient to input shape
            let expanded_grad = if *keepdims {
                B::broadcast_to(upstream_grad, input_shape)
            } else {
                let expanded = expand_for_broadcast::<B>(upstream_grad, input_shape, axes.as_deref());
                B::broadcast_to(&expanded, input_shape)
            };

            // Expand output to input shape to create mask
            let expanded_out = if *keepdims {
                B::broadcast_to(expr.data(), input_shape)
            } else {
                let expanded = expand_for_broadcast::<B>(expr.data(), input_shape, axes.as_deref());
                B::broadcast_to(&expanded, input_shape)
            };

            // Mask: 1 where input == max/min value
            let mask = B::eq(children[0].data(), &expanded_out);
            vec![Some(B::mul(&expanded_grad, &mask))]
        }

        // === Linear algebra ===
        TensorOp::MatMul => {
            // C = A @ B
            // dL/dA = dL/dC @ B^T
            // dL/dB = A^T @ dL/dC
            let a = children[0].data();
            let b = children[1].data();

            let b_t = B::transpose(b, None);
            let a_t = B::transpose(a, None);

            let grad_a = B::matmul(upstream_grad, &b_t);
            let grad_b = B::matmul(&a_t, upstream_grad);

            vec![Some(grad_a), Some(grad_b)]
        }

        // === Shape operations ===
        TensorOp::Transpose { axes } => {
            // Inverse transpose
            let inv_axes = if let Some(axes) = axes {
                let mut inv = vec![0; axes.len()];
                for (i, &ax) in axes.iter().enumerate() {
                    inv[ax] = i;
                }
                Some(inv)
            } else {
                None
            };
            vec![Some(B::transpose(upstream_grad, inv_axes.as_deref()))]
        }

        TensorOp::Reshape { original_shape } => {
            vec![Some(B::reshape(upstream_grad, original_shape))]
        }

        TensorOp::BroadcastTo { original_shape } => {
            vec![Some(B::sum_to(upstream_grad, original_shape))]
        }

        TensorOp::Squeeze { original_shape, .. } => {
            vec![Some(B::reshape(upstream_grad, original_shape))]
        }

        TensorOp::Unsqueeze { axis } => {
            vec![Some(B::squeeze(upstream_grad, Some(&[*axis])))]
        }
    }
}

/// Expand a reduced tensor back to original shape by unsqueezing reduced axes.
fn expand_for_broadcast<B: Backend>(
    tensor: &B::Tensor,
    target_shape: &crate::shape::Shape,
    axes: Option<&[usize]>,
) -> B::Tensor {
    if let Some(axes) = axes {
        let mut result = B::clone_tensor(tensor);
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort();
        for &ax in &sorted_axes {
            result = B::unsqueeze(&result, ax);
        }
        result
    } else {
        // All axes reduced - expand from scalar to target
        let mut result = B::clone_tensor(tensor);
        for ax in 0..target_shape.ndim() {
            result = B::unsqueeze(&result, ax);
        }
        result
    }
}

/// Topological sort via DFS postorder.
fn topological_sort<B: Backend>(root: &Tensor<B>) -> Vec<Tensor<B>> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();

    fn dfs<B: Backend>(
        expr: &Tensor<B>,
        visited: &mut HashSet<NodeId>,
        order: &mut Vec<Tensor<B>>,
    ) {
        if visited.contains(&expr.id()) {
            return;
        }
        visited.insert(expr.id());

        for child in expr.children() {
            dfs(child, visited, order);
        }
        order.push(expr.clone());
    }

    dfs(root, &mut visited, &mut order);
    order
}
