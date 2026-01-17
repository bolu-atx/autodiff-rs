//! Reverse-mode automatic differentiation implementation.
//!
//! The backward pass computes gradients by:
//! 1. Building a topological ordering of nodes reachable from the output
//! 2. Traversing in reverse order, accumulating adjoints from output to leaves

use std::collections::{HashMap, HashSet};

use crate::node::{Expr, NodeId, Op};
use crate::ops::local_gradients;

/// Container for gradient values computed during backward pass.
///
/// Provides methods to query gradients by variable expression or by name.
#[derive(Debug)]
pub struct Gradients {
    /// Map from node ID to its gradient (adjoint) value.
    adjoints: HashMap<NodeId, f64>,
    /// Map from variable name to list of (NodeId, gradient) for variables with that name.
    /// Multiple variables can share the same name but have different IDs.
    name_to_grads: HashMap<String, Vec<(NodeId, f64)>>,
}

impl Gradients {
    /// Get the gradient with respect to a specific expression.
    ///
    /// Returns `Some(gradient)` if the expression is a variable that was part
    /// of the computation graph, `None` otherwise.
    pub fn wrt(&self, expr: &Expr) -> Option<f64> {
        if !expr.is_var() {
            return None;
        }
        self.adjoints.get(&expr.id()).copied()
    }

    /// Get the gradient for a variable by name.
    ///
    /// If multiple variables share the same name, returns the gradient for the first one found.
    /// For more control, use `wrt()` with the specific expression.
    ///
    /// Returns `None` if no variable with this name exists in the graph.
    pub fn by_name(&self, name: &str) -> Option<f64> {
        self.name_to_grads
            .get(name)
            .and_then(|grads| grads.first().map(|(_, g)| *g))
    }

    /// Get all named gradients as a map from variable name to gradient.
    ///
    /// If multiple variables share the same name, their gradients are summed.
    pub fn all_named_grads(&self) -> HashMap<String, f64> {
        self.name_to_grads
            .iter()
            .map(|(name, grads)| {
                let total: f64 = grads.iter().map(|(_, g)| g).sum();
                (name.clone(), total)
            })
            .collect()
    }

    /// Get a vector of gradients for a list of expressions.
    ///
    /// Returns gradients in the same order as the input expressions.
    /// Non-variable expressions get gradient 0.0.
    pub fn wrt_many(&self, exprs: &[Expr]) -> Vec<f64> {
        exprs.iter().map(|e| self.wrt(e).unwrap_or(0.0)).collect()
    }
}

/// Compute gradients via reverse-mode autodiff.
///
/// Performs a backward pass from the given output expression to all leaf variables,
/// computing d(output)/d(variable) for each variable.
pub fn backward(output: &Expr) -> Gradients {
    // Step 1: Build topological order via DFS
    let topo_order = topological_sort(output);

    // Step 2: Initialize adjoints. Output has adjoint 1.0 (d(output)/d(output) = 1)
    let mut adjoints: HashMap<NodeId, f64> = HashMap::new();
    adjoints.insert(output.id(), 1.0);

    // Step 3: Traverse in reverse topological order (from output to leaves)
    for expr in topo_order.iter().rev() {
        let node_adjoint = *adjoints.get(&expr.id()).unwrap_or(&0.0);

        // Skip if this node has no contribution
        if node_adjoint == 0.0 {
            continue;
        }

        // Compute local gradients and propagate to children
        let local_grads = local_gradients(expr.op(), expr.children());

        for (i, child) in expr.children().iter().enumerate() {
            // Chain rule: child_adjoint += node_adjoint * local_gradient
            let contribution = node_adjoint * local_grads[i];
            *adjoints.entry(child.id()).or_insert(0.0) += contribution;
        }
    }

    // Step 4: Build name -> gradients map for variable lookup
    let mut name_to_grads: HashMap<String, Vec<(NodeId, f64)>> = HashMap::new();

    for expr in &topo_order {
        if let Op::Var { name, .. } = expr.op() {
            let grad = *adjoints.get(&expr.id()).unwrap_or(&0.0);
            name_to_grads
                .entry(name.clone())
                .or_default()
                .push((expr.id(), grad));
        }
    }

    Gradients {
        adjoints,
        name_to_grads,
    }
}

/// Build a topological ordering of all nodes reachable from the output.
///
/// Uses DFS postorder traversal, which naturally produces a valid topological order.
fn topological_sort(root: &Expr) -> Vec<Expr> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();

    fn dfs(expr: &Expr, visited: &mut HashSet<NodeId>, order: &mut Vec<Expr>) {
        if visited.contains(&expr.id()) {
            return;
        }
        visited.insert(expr.id());

        // Visit children first (postorder)
        for child in expr.children() {
            dfs(child, visited, order);
        }

        // Then add this node
        order.push(expr.clone());
    }

    dfs(root, &mut visited, &mut order);
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{constant, var};

    #[test]
    fn test_topological_sort_simple() {
        let x = var("x", 1.0);
        let y = var("y", 2.0);
        let z = &x + &y;

        let order = topological_sort(&z);

        // Should have 3 nodes: x, y, z
        assert_eq!(order.len(), 3);

        // z should be last (postorder)
        assert_eq!(order[2].id(), z.id());

        // x and y should come before z
        let z_idx = order.iter().position(|e| e.id() == z.id()).unwrap();
        let x_idx = order.iter().position(|e| e.id() == x.id()).unwrap();
        let y_idx = order.iter().position(|e| e.id() == y.id()).unwrap();

        assert!(x_idx < z_idx);
        assert!(y_idx < z_idx);
    }

    #[test]
    fn test_topological_sort_shared_node() {
        let x = var("x", 1.0);
        // x is used twice: z = x * x
        let z = &x * &x;

        let order = topological_sort(&z);

        // Should have 2 unique nodes: x and z
        assert_eq!(order.len(), 2);
    }

    #[test]
    fn test_backward_simple_add() {
        let x = var("x", 2.0);
        let y = var("y", 3.0);
        let z = &x + &y;

        let grads = backward(&z);

        assert_eq!(grads.wrt(&x), Some(1.0));
        assert_eq!(grads.wrt(&y), Some(1.0));
    }

    #[test]
    fn test_backward_chain() {
        // z = (x + 1)^2
        // dz/dx = 2(x + 1)
        let x = var("x", 2.0);
        let y = &x + constant(1.0);
        let z = y.powf(2.0);

        let grads = backward(&z);

        // At x=2: dz/dx = 2 * 3 = 6
        assert!((grads.wrt(&x).unwrap() - 6.0).abs() < 1e-10);
    }
}
