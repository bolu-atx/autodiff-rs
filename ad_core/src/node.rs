//! Core data structures for the computation graph.
//!
//! The computation graph is built from `Expr` nodes, which are reference-counted
//! handles to internal `Node` structures. This allows cheap cloning and sharing
//! of subexpressions.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Global counter for generating unique node IDs.
/// Uses AtomicU64 for thread-safety (needed for Python bindings).
static NODE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generates a new unique node ID.
fn next_node_id() -> u64 {
    NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u64);

/// The operation performed by a node.
#[derive(Debug, Clone)]
pub enum Op {
    /// A constant value (gradient is zero).
    Const(f64),
    /// A variable (leaf node we differentiate with respect to).
    Var { name: String, value: f64 },
    /// Addition: children[0] + children[1]
    Add,
    /// Subtraction: children[0] - children[1]
    Sub,
    /// Multiplication: children[0] * children[1]
    Mul,
    /// Division: children[0] / children[1]
    Div,
    /// Negation: -children[0]
    Neg,
    /// Power with constant exponent: children[0]^exponent
    Pow { exponent: f64 },
    /// Exponential: exp(children[0])
    Exp,
    /// Natural logarithm: ln(children[0])
    Log,
    /// Sine: sin(children[0])
    Sin,
    /// Cosine: cos(children[0])
    Cos,
}

/// Internal node structure holding the operation, children, and metadata.
#[derive(Debug)]
pub struct Node {
    /// Unique identifier for this node.
    pub id: NodeId,
    /// The operation this node performs.
    pub op: Op,
    /// Child expressions (operands).
    pub children: Vec<Expr>,
}

/// An expression in the computation graph.
///
/// `Expr` is a reference-counted handle to a `Node`. Cloning an `Expr` is cheap
/// (just incrementing a reference count) and allows the same subexpression to be
/// shared across multiple parent expressions.
///
/// Expressions are immutable once created. To evaluate at different points,
/// create new variables with different values.
#[derive(Debug, Clone)]
pub struct Expr(pub(crate) Arc<Node>);

impl Expr {
    /// Create a new variable with the given name and value.
    ///
    /// Each call creates a variable with a unique identity, even if the name matches
    /// an existing variable.
    pub fn var(name: &str, value: f64) -> Self {
        Expr(Arc::new(Node {
            id: NodeId(next_node_id()),
            op: Op::Var {
                name: name.to_string(),
                value,
            },
            children: vec![],
        }))
    }

    /// Create a constant expression.
    ///
    /// Constants are treated as fixed values during differentiation (gradient = 0).
    pub fn constant(value: f64) -> Self {
        Expr(Arc::new(Node {
            id: NodeId(next_node_id()),
            op: Op::Const(value),
            children: vec![],
        }))
    }

    /// Create a new expression node with the given operation and children.
    fn new_op(op: Op, children: Vec<Expr>) -> Self {
        Expr(Arc::new(Node {
            id: NodeId(next_node_id()),
            op,
            children,
        }))
    }

    /// Get the unique ID of this expression's node.
    pub fn id(&self) -> NodeId {
        self.0.id
    }

    /// Get a reference to the operation.
    pub fn op(&self) -> &Op {
        &self.0.op
    }

    /// Get the child expressions.
    pub fn children(&self) -> &[Expr] {
        &self.0.children
    }

    /// Check if this expression is a variable.
    pub fn is_var(&self) -> bool {
        matches!(self.0.op, Op::Var { .. })
    }

    /// Get the variable name, if this is a variable.
    pub fn var_name(&self) -> Option<&str> {
        match &self.0.op {
            Op::Var { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Evaluate this expression, returning its numerical value.
    ///
    /// This recursively evaluates all child expressions.
    pub fn value(&self) -> f64 {
        match &self.0.op {
            Op::Const(v) => *v,
            Op::Var { value, .. } => *value,
            Op::Add => self.0.children[0].value() + self.0.children[1].value(),
            Op::Sub => self.0.children[0].value() - self.0.children[1].value(),
            Op::Mul => self.0.children[0].value() * self.0.children[1].value(),
            Op::Div => self.0.children[0].value() / self.0.children[1].value(),
            Op::Neg => -self.0.children[0].value(),
            Op::Pow { exponent } => self.0.children[0].value().powf(*exponent),
            Op::Exp => self.0.children[0].value().exp(),
            Op::Log => self.0.children[0].value().ln(),
            Op::Sin => self.0.children[0].value().sin(),
            Op::Cos => self.0.children[0].value().cos(),
        }
    }

    // === Unary operations ===

    /// Compute the exponential: exp(self)
    pub fn exp(&self) -> Expr {
        Expr::new_op(Op::Exp, vec![self.clone()])
    }

    /// Compute the natural logarithm: ln(self)
    pub fn log(&self) -> Expr {
        Expr::new_op(Op::Log, vec![self.clone()])
    }

    /// Compute sine: sin(self)
    pub fn sin(&self) -> Expr {
        Expr::new_op(Op::Sin, vec![self.clone()])
    }

    /// Compute cosine: cos(self)
    pub fn cos(&self) -> Expr {
        Expr::new_op(Op::Cos, vec![self.clone()])
    }

    /// Raise to a constant power: self^exponent
    pub fn powf(&self, exponent: f64) -> Expr {
        Expr::new_op(Op::Pow { exponent }, vec![self.clone()])
    }

    /// Compute gradients via reverse-mode autodiff.
    ///
    /// Returns a `Gradients` struct that can be queried for the gradient
    /// with respect to any variable in the expression.
    pub fn backward(&self) -> crate::Gradients {
        crate::backward::backward(self)
    }
}

// === Operator overloads ===

impl std::ops::Neg for &Expr {
    type Output = Expr;

    fn neg(self) -> Expr {
        Expr::new_op(Op::Neg, vec![self.clone()])
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Expr {
        Expr::new_op(Op::Neg, vec![self])
    }
}

impl std::ops::Add for &Expr {
    type Output = Expr;

    fn add(self, rhs: &Expr) -> Expr {
        Expr::new_op(Op::Add, vec![self.clone(), rhs.clone()])
    }
}

impl std::ops::Add<Expr> for &Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Expr {
        Expr::new_op(Op::Add, vec![self.clone(), rhs])
    }
}

impl std::ops::Add<&Expr> for Expr {
    type Output = Expr;

    fn add(self, rhs: &Expr) -> Expr {
        Expr::new_op(Op::Add, vec![self, rhs.clone()])
    }
}

impl std::ops::Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Expr {
        Expr::new_op(Op::Add, vec![self, rhs])
    }
}

impl std::ops::Sub for &Expr {
    type Output = Expr;

    fn sub(self, rhs: &Expr) -> Expr {
        Expr::new_op(Op::Sub, vec![self.clone(), rhs.clone()])
    }
}

impl std::ops::Sub<Expr> for &Expr {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Expr {
        Expr::new_op(Op::Sub, vec![self.clone(), rhs])
    }
}

impl std::ops::Sub<&Expr> for Expr {
    type Output = Expr;

    fn sub(self, rhs: &Expr) -> Expr {
        Expr::new_op(Op::Sub, vec![self, rhs.clone()])
    }
}

impl std::ops::Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Expr {
        Expr::new_op(Op::Sub, vec![self, rhs])
    }
}

impl std::ops::Mul for &Expr {
    type Output = Expr;

    fn mul(self, rhs: &Expr) -> Expr {
        Expr::new_op(Op::Mul, vec![self.clone(), rhs.clone()])
    }
}

impl std::ops::Mul<Expr> for &Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        Expr::new_op(Op::Mul, vec![self.clone(), rhs])
    }
}

impl std::ops::Mul<&Expr> for Expr {
    type Output = Expr;

    fn mul(self, rhs: &Expr) -> Expr {
        Expr::new_op(Op::Mul, vec![self, rhs.clone()])
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        Expr::new_op(Op::Mul, vec![self, rhs])
    }
}

impl std::ops::Div for &Expr {
    type Output = Expr;

    fn div(self, rhs: &Expr) -> Expr {
        Expr::new_op(Op::Div, vec![self.clone(), rhs.clone()])
    }
}

impl std::ops::Div<Expr> for &Expr {
    type Output = Expr;

    fn div(self, rhs: Expr) -> Expr {
        Expr::new_op(Op::Div, vec![self.clone(), rhs])
    }
}

impl std::ops::Div<&Expr> for Expr {
    type Output = Expr;

    fn div(self, rhs: &Expr) -> Expr {
        Expr::new_op(Op::Div, vec![self, rhs.clone()])
    }
}

impl std::ops::Div for Expr {
    type Output = Expr;

    fn div(self, rhs: Expr) -> Expr {
        Expr::new_op(Op::Div, vec![self, rhs])
    }
}
