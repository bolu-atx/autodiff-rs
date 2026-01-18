//! Shape and stride utilities for tensors.

use std::fmt;

/// A tensor shape (dimensions).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    /// Create a new shape from dimensions.
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    /// Create a scalar shape (0-dimensional).
    pub fn scalar() -> Self {
        Shape(vec![])
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Get dimension at index.
    pub fn dim(&self, idx: usize) -> usize {
        self.0[idx]
    }

    /// Get dimensions as slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.0.iter().product::<usize>().max(1)
    }

    /// Check if this is a scalar (0-dim tensor).
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Compute row-major (C-contiguous) strides for this shape.
    pub fn contiguous_strides(&self) -> Strides {
        let ndim = self.0.len();
        if ndim == 0 {
            return Strides(vec![]);
        }

        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * self.0[i + 1];
        }
        Strides(strides)
    }

    /// Check if two shapes are broadcast-compatible.
    /// Returns the broadcast result shape if compatible.
    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        let ndim = self.ndim().max(other.ndim());
        let mut result = vec![0usize; ndim];

        for i in 0..ndim {
            let d1 = if i < ndim - self.ndim() {
                1
            } else {
                self.0[i - (ndim - self.ndim())]
            };
            let d2 = if i < ndim - other.ndim() {
                1
            } else {
                other.0[i - (ndim - other.ndim())]
            };

            if d1 == d2 {
                result[i] = d1;
            } else if d1 == 1 {
                result[i] = d2;
            } else if d2 == 1 {
                result[i] = d1;
            } else {
                return None; // Incompatible shapes
            }
        }

        Some(Shape(result))
    }

    /// Compute which axes need to be reduced when going from broadcast shape back to this shape.
    /// Returns axes that were broadcast (size 1 expanded to larger).
    pub fn reduction_axes_from(&self, broadcast_shape: &Shape) -> Vec<usize> {
        let mut axes = Vec::new();
        let offset = broadcast_shape.ndim() - self.ndim();

        // Leading dimensions that don't exist in self
        for i in 0..offset {
            axes.push(i);
        }

        // Dimensions that were size 1 in self but expanded
        for i in 0..self.ndim() {
            if self.0[i] == 1 && broadcast_shape.0[offset + i] > 1 {
                axes.push(offset + i);
            }
        }

        axes
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.0)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        if self.0.len() == 1 {
            write!(f, ",")?;
        }
        write!(f, ")")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Shape(v)
    }
}

impl From<&[usize]> for Shape {
    fn from(s: &[usize]) -> Self {
        Shape(s.to_vec())
    }
}

/// Tensor strides (step size in each dimension).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Strides(pub Vec<usize>);

impl Strides {
    pub fn new(strides: Vec<usize>) -> Self {
        Strides(strides)
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// Compute flat index from multi-dimensional indices.
    pub fn index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(self.0.len(), indices.len());
        self.0.iter().zip(indices.iter()).map(|(s, i)| s * i).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_basics() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.dim(0), 2);
        assert_eq!(s.dim(1), 3);
        assert_eq!(s.dim(2), 4);
        assert_eq!(s.numel(), 24);
        assert!(!s.is_scalar());
    }

    #[test]
    fn test_scalar_shape() {
        let s = Shape::scalar();
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
        assert!(s.is_scalar());
    }

    #[test]
    fn test_contiguous_strides() {
        let s = Shape::new(vec![2, 3, 4]);
        let strides = s.contiguous_strides();
        assert_eq!(strides.0, vec![12, 4, 1]);

        let s2 = Shape::new(vec![3, 4]);
        let strides2 = s2.contiguous_strides();
        assert_eq!(strides2.0, vec![4, 1]);
    }

    #[test]
    fn test_broadcast_same_shape() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![2, 3]);
        assert_eq!(a.broadcast_with(&b), Some(Shape::new(vec![2, 3])));
    }

    #[test]
    fn test_broadcast_scalar() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::scalar();
        assert_eq!(a.broadcast_with(&b), Some(Shape::new(vec![2, 3])));
    }

    #[test]
    fn test_broadcast_leading_dims() {
        let a = Shape::new(vec![3, 4]);
        let b = Shape::new(vec![2, 3, 4]);
        assert_eq!(a.broadcast_with(&b), Some(Shape::new(vec![2, 3, 4])));
    }

    #[test]
    fn test_broadcast_expand_ones() {
        let a = Shape::new(vec![1, 4]);
        let b = Shape::new(vec![3, 1]);
        assert_eq!(a.broadcast_with(&b), Some(Shape::new(vec![3, 4])));
    }

    #[test]
    fn test_broadcast_incompatible() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![2, 4]);
        assert_eq!(a.broadcast_with(&b), None);
    }

    #[test]
    fn test_reduction_axes() {
        let original = Shape::new(vec![1, 4]);
        let broadcast = Shape::new(vec![3, 4]);
        assert_eq!(original.reduction_axes_from(&broadcast), vec![0]);

        let original2 = Shape::new(vec![4]);
        let broadcast2 = Shape::new(vec![2, 3, 4]);
        assert_eq!(original2.reduction_axes_from(&broadcast2), vec![0, 1]);
    }

    #[test]
    fn test_stride_index() {
        let strides = Strides::new(vec![12, 4, 1]);
        assert_eq!(strides.index(&[0, 0, 0]), 0);
        assert_eq!(strides.index(&[0, 0, 1]), 1);
        assert_eq!(strides.index(&[0, 1, 0]), 4);
        assert_eq!(strides.index(&[1, 0, 0]), 12);
        assert_eq!(strides.index(&[1, 2, 3]), 12 + 8 + 3);
    }
}
