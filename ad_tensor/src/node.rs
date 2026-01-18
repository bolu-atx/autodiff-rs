//! Computation graph nodes for tensor autodiff.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::backend::Backend;
use crate::shape::Shape;
use crate::tensor::TensorData;

/// Global counter for unique node IDs.
static NODE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_node_id() -> u64 {
    NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u64);

/// Operations in the computation graph.
#[derive(Debug, Clone)]
pub enum TensorOp {
    // === Leaf nodes ===
    /// Constant tensor (gradient is zero).
    Const,
    /// Variable tensor (gradient is tracked).
    Var { name: String },

    // === Unary element-wise ===
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Relu,
    Sigmoid,
    Tanh,
    Sqrt,

    // === Binary element-wise ===
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,

    // === Reductions ===
    Sum {
        axes: Option<Vec<usize>>,
        keepdims: bool,
    },
    Mean {
        axes: Option<Vec<usize>>,
        keepdims: bool,
    },
    Max {
        axes: Option<Vec<usize>>,
        keepdims: bool,
    },
    Min {
        axes: Option<Vec<usize>>,
        keepdims: bool,
    },

    // === Linear algebra ===
    MatMul,

    // === Shape operations ===
    Transpose {
        axes: Option<Vec<usize>>,
    },
    Reshape {
        original_shape: Shape,
    },
    BroadcastTo {
        original_shape: Shape,
    },
    Squeeze {
        axes: Option<Vec<usize>>,
        original_shape: Shape,
    },
    Unsqueeze {
        axis: usize,
    },
}

/// Internal node structure.
pub struct TensorNode<B: Backend> {
    pub id: NodeId,
    pub op: TensorOp,
    pub data: B::Tensor,
    pub children: Vec<Tensor<B>>,
}

impl<B: Backend> std::fmt::Debug for TensorNode<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorNode")
            .field("id", &self.id)
            .field("op", &self.op)
            .field("shape", self.data.shape())
            .field("children", &self.children.len())
            .finish()
    }
}

/// A tensor expression in the computation graph.
/// Reference-counted for efficient sharing.
#[derive(Clone)]
pub struct Tensor<B: Backend>(pub(crate) Arc<TensorNode<B>>);

impl<B: Backend> std::fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.0.id)
            .field("op", &self.0.op)
            .field("shape", self.shape())
            .finish()
    }
}

impl<B: Backend> Tensor<B> {
    /// Create a new tensor node.
    fn new_node(op: TensorOp, data: B::Tensor, children: Vec<Tensor<B>>) -> Self {
        Tensor(Arc::new(TensorNode {
            id: NodeId(next_node_id()),
            op,
            data,
            children,
        }))
    }

    // === Constructors ===

    /// Create a variable tensor (tracked for gradients).
    pub fn var(name: &str, data: B::Tensor) -> Self {
        Self::new_node(TensorOp::Var { name: name.to_string() }, data, vec![])
    }

    /// Create a constant tensor (not tracked for gradients).
    pub fn constant(data: B::Tensor) -> Self {
        Self::new_node(TensorOp::Const, data, vec![])
    }

    /// Create a zeros tensor.
    pub fn zeros(shape: &Shape) -> Self {
        Self::constant(B::zeros(shape))
    }

    /// Create a ones tensor.
    pub fn ones(shape: &Shape) -> Self {
        Self::constant(B::ones(shape))
    }

    /// Create a scalar tensor.
    pub fn scalar(value: f32) -> Self {
        Self::constant(B::scalar(value))
    }

    /// Create a tensor from data.
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Self {
        Self::constant(B::from_vec(data, shape))
    }

    // === Accessors ===

    /// Get unique node ID.
    pub fn id(&self) -> NodeId {
        self.0.id
    }

    /// Get the operation.
    pub fn op(&self) -> &TensorOp {
        &self.0.op
    }

    /// Get the tensor data.
    pub fn data(&self) -> &B::Tensor {
        &self.0.data
    }

    /// Get child tensors.
    pub fn children(&self) -> &[Tensor<B>] {
        &self.0.children
    }

    /// Get the shape.
    pub fn shape(&self) -> &Shape {
        self.0.data.shape()
    }

    /// Get number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape().ndim()
    }

    /// Get number of elements.
    pub fn numel(&self) -> usize {
        self.shape().numel()
    }

    /// Check if this is a variable.
    pub fn is_var(&self) -> bool {
        matches!(self.0.op, TensorOp::Var { .. })
    }

    /// Get variable name if this is a variable.
    pub fn var_name(&self) -> Option<&str> {
        match &self.0.op {
            TensorOp::Var { name } => Some(name),
            _ => None,
        }
    }

    /// Get data as slice (for reading values).
    pub fn as_slice(&self) -> &[f32] {
        self.0.data.as_slice()
    }

    /// Get scalar value.
    pub fn item(&self) -> f32 {
        self.0.data.scalar_value()
    }

    // === Unary operations ===

    /// Negate: -self
    pub fn neg(&self) -> Self {
        Self::new_node(TensorOp::Neg, B::neg(&self.0.data), vec![self.clone()])
    }

    /// Exponential: exp(self)
    pub fn exp(&self) -> Self {
        Self::new_node(TensorOp::Exp, B::exp(&self.0.data), vec![self.clone()])
    }

    /// Natural log: ln(self)
    pub fn log(&self) -> Self {
        Self::new_node(TensorOp::Log, B::log(&self.0.data), vec![self.clone()])
    }

    /// Sine: sin(self)
    pub fn sin(&self) -> Self {
        Self::new_node(TensorOp::Sin, B::sin(&self.0.data), vec![self.clone()])
    }

    /// Cosine: cos(self)
    pub fn cos(&self) -> Self {
        Self::new_node(TensorOp::Cos, B::cos(&self.0.data), vec![self.clone()])
    }

    /// ReLU: max(0, self)
    pub fn relu(&self) -> Self {
        Self::new_node(TensorOp::Relu, B::relu(&self.0.data), vec![self.clone()])
    }

    /// Sigmoid: 1 / (1 + exp(-self))
    pub fn sigmoid(&self) -> Self {
        Self::new_node(TensorOp::Sigmoid, B::sigmoid(&self.0.data), vec![self.clone()])
    }

    /// Tanh: tanh(self)
    pub fn tanh(&self) -> Self {
        Self::new_node(TensorOp::Tanh, B::tanh(&self.0.data), vec![self.clone()])
    }

    /// Square root: sqrt(self)
    pub fn sqrt(&self) -> Self {
        Self::new_node(TensorOp::Sqrt, B::sqrt(&self.0.data), vec![self.clone()])
    }

    // === Binary operations ===

    /// Add: self + other
    pub fn add(&self, other: &Self) -> Self {
        Self::new_node(
            TensorOp::Add,
            B::add(&self.0.data, &other.0.data),
            vec![self.clone(), other.clone()],
        )
    }

    /// Subtract: self - other
    pub fn sub(&self, other: &Self) -> Self {
        Self::new_node(
            TensorOp::Sub,
            B::sub(&self.0.data, &other.0.data),
            vec![self.clone(), other.clone()],
        )
    }

    /// Multiply: self * other
    pub fn mul(&self, other: &Self) -> Self {
        Self::new_node(
            TensorOp::Mul,
            B::mul(&self.0.data, &other.0.data),
            vec![self.clone(), other.clone()],
        )
    }

    /// Divide: self / other
    pub fn div(&self, other: &Self) -> Self {
        Self::new_node(
            TensorOp::Div,
            B::div(&self.0.data, &other.0.data),
            vec![self.clone(), other.clone()],
        )
    }

    /// Power: self ^ other
    pub fn pow(&self, other: &Self) -> Self {
        Self::new_node(
            TensorOp::Pow,
            B::pow(&self.0.data, &other.0.data),
            vec![self.clone(), other.clone()],
        )
    }

    /// Power with constant exponent: self ^ exp
    pub fn powf(&self, exp: f32) -> Self {
        let exp_tensor = Self::scalar(exp);
        self.pow(&exp_tensor)
    }

    /// Maximum: max(self, other) element-wise
    pub fn maximum(&self, other: &Self) -> Self {
        Self::new_node(
            TensorOp::Maximum,
            B::maximum(&self.0.data, &other.0.data),
            vec![self.clone(), other.clone()],
        )
    }

    /// Minimum: min(self, other) element-wise
    pub fn minimum(&self, other: &Self) -> Self {
        Self::new_node(
            TensorOp::Minimum,
            B::minimum(&self.0.data, &other.0.data),
            vec![self.clone(), other.clone()],
        )
    }

    // === Reductions ===

    /// Sum over axes (None = all axes -> scalar).
    pub fn sum(&self, axes: Option<&[usize]>, keepdims: bool) -> Self {
        Self::new_node(
            TensorOp::Sum {
                axes: axes.map(|a| a.to_vec()),
                keepdims,
            },
            B::sum(&self.0.data, axes, keepdims),
            vec![self.clone()],
        )
    }

    /// Mean over axes (None = all axes -> scalar).
    pub fn mean(&self, axes: Option<&[usize]>, keepdims: bool) -> Self {
        Self::new_node(
            TensorOp::Mean {
                axes: axes.map(|a| a.to_vec()),
                keepdims,
            },
            B::mean(&self.0.data, axes, keepdims),
            vec![self.clone()],
        )
    }

    /// Max over axes (None = all axes -> scalar).
    pub fn max(&self, axes: Option<&[usize]>, keepdims: bool) -> Self {
        Self::new_node(
            TensorOp::Max {
                axes: axes.map(|a| a.to_vec()),
                keepdims,
            },
            B::max(&self.0.data, axes, keepdims),
            vec![self.clone()],
        )
    }

    /// Min over axes (None = all axes -> scalar).
    pub fn min(&self, axes: Option<&[usize]>, keepdims: bool) -> Self {
        Self::new_node(
            TensorOp::Min {
                axes: axes.map(|a| a.to_vec()),
                keepdims,
            },
            B::min(&self.0.data, axes, keepdims),
            vec![self.clone()],
        )
    }

    // === Linear algebra ===

    /// Matrix multiplication: self @ other
    pub fn matmul(&self, other: &Self) -> Self {
        Self::new_node(
            TensorOp::MatMul,
            B::matmul(&self.0.data, &other.0.data),
            vec![self.clone(), other.clone()],
        )
    }

    // === Shape operations ===

    /// Transpose. None = reverse all axes.
    pub fn transpose(&self, axes: Option<&[usize]>) -> Self {
        Self::new_node(
            TensorOp::Transpose {
                axes: axes.map(|a| a.to_vec()),
            },
            B::transpose(&self.0.data, axes),
            vec![self.clone()],
        )
    }

    /// Simple transpose (swap last two axes) for matrices.
    pub fn t(&self) -> Self {
        self.transpose(None)
    }

    /// Reshape to new shape.
    pub fn reshape(&self, shape: &Shape) -> Self {
        let original_shape = self.shape().clone();
        Self::new_node(
            TensorOp::Reshape { original_shape },
            B::reshape(&self.0.data, shape),
            vec![self.clone()],
        )
    }

    /// Broadcast to larger shape.
    pub fn broadcast_to(&self, shape: &Shape) -> Self {
        let original_shape = self.shape().clone();
        Self::new_node(
            TensorOp::BroadcastTo { original_shape },
            B::broadcast_to(&self.0.data, shape),
            vec![self.clone()],
        )
    }

    /// Squeeze dimensions of size 1.
    pub fn squeeze(&self, axes: Option<&[usize]>) -> Self {
        let original_shape = self.shape().clone();
        Self::new_node(
            TensorOp::Squeeze {
                axes: axes.map(|a| a.to_vec()),
                original_shape,
            },
            B::squeeze(&self.0.data, axes),
            vec![self.clone()],
        )
    }

    /// Unsqueeze: add dimension of size 1 at axis.
    pub fn unsqueeze(&self, axis: usize) -> Self {
        Self::new_node(
            TensorOp::Unsqueeze { axis },
            B::unsqueeze(&self.0.data, axis),
            vec![self.clone()],
        )
    }

    /// Compute gradients via reverse-mode autodiff.
    pub fn backward(&self) -> crate::backward::Gradients<B> {
        crate::backward::backward(self)
    }
}

// === Operator overloads ===

impl<B: Backend> std::ops::Neg for &Tensor<B> {
    type Output = Tensor<B>;
    fn neg(self) -> Tensor<B> {
        self.neg()
    }
}

impl<B: Backend> std::ops::Neg for Tensor<B> {
    type Output = Tensor<B>;
    fn neg(self) -> Tensor<B> {
        (&self).neg()
    }
}

impl<B: Backend> std::ops::Add for &Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: &Tensor<B>) -> Tensor<B> {
        self.add(rhs)
    }
}

impl<B: Backend> std::ops::Add<Tensor<B>> for &Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: Tensor<B>) -> Tensor<B> {
        self.add(&rhs)
    }
}

impl<B: Backend> std::ops::Add<&Tensor<B>> for Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: &Tensor<B>) -> Tensor<B> {
        (&self).add(rhs)
    }
}

impl<B: Backend> std::ops::Add for Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: Tensor<B>) -> Tensor<B> {
        (&self).add(&rhs)
    }
}

impl<B: Backend> std::ops::Sub for &Tensor<B> {
    type Output = Tensor<B>;
    fn sub(self, rhs: &Tensor<B>) -> Tensor<B> {
        self.sub(rhs)
    }
}

impl<B: Backend> std::ops::Sub<Tensor<B>> for &Tensor<B> {
    type Output = Tensor<B>;
    fn sub(self, rhs: Tensor<B>) -> Tensor<B> {
        self.sub(&rhs)
    }
}

impl<B: Backend> std::ops::Sub<&Tensor<B>> for Tensor<B> {
    type Output = Tensor<B>;
    fn sub(self, rhs: &Tensor<B>) -> Tensor<B> {
        (&self).sub(rhs)
    }
}

impl<B: Backend> std::ops::Sub for Tensor<B> {
    type Output = Tensor<B>;
    fn sub(self, rhs: Tensor<B>) -> Tensor<B> {
        (&self).sub(&rhs)
    }
}

impl<B: Backend> std::ops::Mul for &Tensor<B> {
    type Output = Tensor<B>;
    fn mul(self, rhs: &Tensor<B>) -> Tensor<B> {
        self.mul(rhs)
    }
}

impl<B: Backend> std::ops::Mul<Tensor<B>> for &Tensor<B> {
    type Output = Tensor<B>;
    fn mul(self, rhs: Tensor<B>) -> Tensor<B> {
        self.mul(&rhs)
    }
}

impl<B: Backend> std::ops::Mul<&Tensor<B>> for Tensor<B> {
    type Output = Tensor<B>;
    fn mul(self, rhs: &Tensor<B>) -> Tensor<B> {
        (&self).mul(rhs)
    }
}

impl<B: Backend> std::ops::Mul for Tensor<B> {
    type Output = Tensor<B>;
    fn mul(self, rhs: Tensor<B>) -> Tensor<B> {
        (&self).mul(&rhs)
    }
}

impl<B: Backend> std::ops::Div for &Tensor<B> {
    type Output = Tensor<B>;
    fn div(self, rhs: &Tensor<B>) -> Tensor<B> {
        self.div(rhs)
    }
}

impl<B: Backend> std::ops::Div<Tensor<B>> for &Tensor<B> {
    type Output = Tensor<B>;
    fn div(self, rhs: Tensor<B>) -> Tensor<B> {
        self.div(&rhs)
    }
}

impl<B: Backend> std::ops::Div<&Tensor<B>> for Tensor<B> {
    type Output = Tensor<B>;
    fn div(self, rhs: &Tensor<B>) -> Tensor<B> {
        (&self).div(rhs)
    }
}

impl<B: Backend> std::ops::Div for Tensor<B> {
    type Output = Tensor<B>;
    fn div(self, rhs: Tensor<B>) -> Tensor<B> {
        (&self).div(&rhs)
    }
}
