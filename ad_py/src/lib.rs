//! Python bindings for the tensor autodiff library.
//!
//! Provides a `Tensor` class with numpy interop and automatic differentiation.

use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ad_backend_cpu::CpuBackend;
use ad_tensor::prelude::*;

type CpuTensorExpr = Tensor<CpuBackend>;

/// A tensor with automatic differentiation support.
///
/// Tensors can be combined using arithmetic operators (+, -, *, /, unary -)
/// and mathematical functions (exp, log, sin, cos, relu, sigmoid, tanh).
/// Supports numpy interop for creating tensors from arrays.
///
/// Example:
///     >>> import numpy as np
///     >>> x = tensor(np.array([1.0, 2.0, 3.0]), name="x")
///     >>> y = tensor(np.array([4.0, 5.0, 6.0]), name="y")
///     >>> z = (x * y).sum()
///     >>> grads = z.backward()
///     >>> grads["x"]
///     array([4., 5., 6.], dtype=float32)
#[pyclass(name = "Tensor")]
#[derive(Clone)]
struct PyTensor {
    inner: CpuTensorExpr,
}

impl PyTensor {
    fn new(inner: CpuTensorExpr) -> Self {
        PyTensor { inner }
    }
}

#[pymethods]
impl PyTensor {
    /// Get the shape of this tensor as a tuple.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    /// Get the number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Get the total number of elements.
    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Check if this is a variable (tracked for gradients).
    fn is_var(&self) -> bool {
        self.inner.is_var()
    }

    /// Get the variable name if this is a variable.
    fn var_name(&self) -> Option<String> {
        self.inner.var_name().map(|s| s.to_string())
    }

    /// Get the tensor data as a numpy array.
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let data = self.inner.as_slice();
        Ok(PyArray1::from_slice(py, data))
    }

    /// Get scalar value (only valid for 0-dim tensors).
    fn item(&self) -> PyResult<f32> {
        if !self.inner.shape().is_scalar() && self.inner.numel() != 1 {
            return Err(PyValueError::new_err("item() only works for scalar tensors or tensors with one element"));
        }
        Ok(self.inner.as_slice()[0])
    }

    // === Unary operations ===

    /// Exponential: exp(self)
    fn exp(&self) -> PyTensor {
        PyTensor::new(self.inner.exp())
    }

    /// Natural logarithm: log(self)
    fn log(&self) -> PyTensor {
        PyTensor::new(self.inner.log())
    }

    /// Sine: sin(self)
    fn sin(&self) -> PyTensor {
        PyTensor::new(self.inner.sin())
    }

    /// Cosine: cos(self)
    fn cos(&self) -> PyTensor {
        PyTensor::new(self.inner.cos())
    }

    /// ReLU: max(0, self)
    fn relu(&self) -> PyTensor {
        PyTensor::new(self.inner.relu())
    }

    /// Sigmoid: 1 / (1 + exp(-self))
    fn sigmoid(&self) -> PyTensor {
        PyTensor::new(self.inner.sigmoid())
    }

    /// Tanh: tanh(self)
    fn tanh(&self) -> PyTensor {
        PyTensor::new(self.inner.tanh())
    }

    /// Square root: sqrt(self)
    fn sqrt(&self) -> PyTensor {
        PyTensor::new(self.inner.sqrt())
    }

    // === Reductions ===

    /// Sum over axes (None = all axes).
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn sum(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyTensor> {
        let axes = parse_axes(axis)?;
        Ok(PyTensor::new(self.inner.sum(axes.as_deref(), keepdims)))
    }

    /// Mean over axes (None = all axes).
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn mean(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyTensor> {
        let axes = parse_axes(axis)?;
        Ok(PyTensor::new(self.inner.mean(axes.as_deref(), keepdims)))
    }

    /// Max over axes (None = all axes).
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn max(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyTensor> {
        let axes = parse_axes(axis)?;
        Ok(PyTensor::new(self.inner.max(axes.as_deref(), keepdims)))
    }

    /// Min over axes (None = all axes).
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn min(&self, axis: Option<&Bound<'_, PyAny>>, keepdims: bool) -> PyResult<PyTensor> {
        let axes = parse_axes(axis)?;
        Ok(PyTensor::new(self.inner.min(axes.as_deref(), keepdims)))
    }

    // === Linear algebra ===

    /// Matrix multiplication: self @ other
    fn matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor::new(self.inner.matmul(&other.inner))
    }

    // === Shape operations ===

    /// Transpose axes (None = reverse all axes).
    #[pyo3(signature = (axes=None))]
    fn transpose(&self, axes: Option<Vec<usize>>) -> PyTensor {
        PyTensor::new(self.inner.transpose(axes.as_deref()))
    }

    /// Simple transpose (swap last two axes).
    #[getter]
    fn T(&self) -> PyTensor {
        PyTensor::new(self.inner.t())
    }

    /// Reshape to new shape.
    fn reshape(&self, shape: Vec<usize>) -> PyTensor {
        PyTensor::new(self.inner.reshape(&Shape::new(shape)))
    }

    /// Squeeze dimensions of size 1.
    #[pyo3(signature = (axis=None))]
    fn squeeze(&self, axis: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
        let axes = parse_axes(axis)?;
        Ok(PyTensor::new(self.inner.squeeze(axes.as_deref())))
    }

    /// Unsqueeze: add dimension of size 1 at axis.
    fn unsqueeze(&self, axis: usize) -> PyTensor {
        PyTensor::new(self.inner.unsqueeze(axis))
    }

    // === Autodiff ===

    /// Compute gradients via reverse-mode autodiff.
    ///
    /// Returns a dictionary mapping variable names to gradient arrays.
    fn backward(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let grads = self.inner.backward();
        let dict = PyDict::new(py);

        for (name, grad_list) in grads.all_named() {
            // Sum gradients if there are multiple variables with the same name
            if let Some((_, first_grad)) = grad_list.first() {
                let mut total = first_grad.as_slice().to_vec();
                for (_, grad) in grad_list.iter().skip(1) {
                    for (t, g) in total.iter_mut().zip(grad.as_slice()) {
                        *t += g;
                    }
                }
                let arr = PyArray1::from_vec(py, total);
                dict.set_item(name, arr)?;
            }
        }

        Ok(dict.into())
    }

    /// Compute gradients with respect to specific tensors.
    ///
    /// Args:
    ///     vars: List of Tensor objects to compute gradients for
    ///
    /// Returns:
    ///     List of gradient arrays in the same order as input tensors
    fn grad(&self, py: Python<'_>, vars: Vec<PyRef<PyTensor>>) -> PyResult<Py<PyList>> {
        let grads = self.inner.backward();
        let list = PyList::empty(py);

        for v in vars {
            if let Some(grad) = grads.wrt(&v.inner) {
                list.append(PyArray1::from_slice(py, grad.as_slice()))?;
            } else {
                // Return zeros if no gradient
                let zeros: Bound<'_, PyArray1<f32>> = PyArray1::zeros(py, [v.inner.numel()], false);
                list.append(zeros)?;
            }
        }

        Ok(list.into())
    }

    // === Operator overloads ===

    fn __add__(&self, other: TensorOrScalar) -> PyTensor {
        match other {
            TensorOrScalar::Tensor(t) => PyTensor::new(&self.inner + &t.inner),
            TensorOrScalar::Scalar(s) => {
                let scalar = CpuTensorExpr::scalar(s);
                PyTensor::new(&self.inner + &scalar)
            }
        }
    }

    fn __radd__(&self, other: TensorOrScalar) -> PyTensor {
        self.__add__(other)
    }

    fn __sub__(&self, other: TensorOrScalar) -> PyTensor {
        match other {
            TensorOrScalar::Tensor(t) => PyTensor::new(&self.inner - &t.inner),
            TensorOrScalar::Scalar(s) => {
                let scalar = CpuTensorExpr::scalar(s);
                PyTensor::new(&self.inner - &scalar)
            }
        }
    }

    fn __rsub__(&self, other: TensorOrScalar) -> PyTensor {
        match other {
            TensorOrScalar::Tensor(t) => PyTensor::new(&t.inner - &self.inner),
            TensorOrScalar::Scalar(s) => {
                let scalar = CpuTensorExpr::scalar(s);
                PyTensor::new(&scalar - &self.inner)
            }
        }
    }

    fn __mul__(&self, other: TensorOrScalar) -> PyTensor {
        match other {
            TensorOrScalar::Tensor(t) => PyTensor::new(&self.inner * &t.inner),
            TensorOrScalar::Scalar(s) => {
                let scalar = CpuTensorExpr::scalar(s);
                PyTensor::new(&self.inner * &scalar)
            }
        }
    }

    fn __rmul__(&self, other: TensorOrScalar) -> PyTensor {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: TensorOrScalar) -> PyTensor {
        match other {
            TensorOrScalar::Tensor(t) => PyTensor::new(&self.inner / &t.inner),
            TensorOrScalar::Scalar(s) => {
                let scalar = CpuTensorExpr::scalar(s);
                PyTensor::new(&self.inner / &scalar)
            }
        }
    }

    fn __rtruediv__(&self, other: TensorOrScalar) -> PyTensor {
        match other {
            TensorOrScalar::Tensor(t) => PyTensor::new(&t.inner / &self.inner),
            TensorOrScalar::Scalar(s) => {
                let scalar = CpuTensorExpr::scalar(s);
                PyTensor::new(&scalar / &self.inner)
            }
        }
    }

    fn __neg__(&self) -> PyTensor {
        PyTensor::new(-&self.inner)
    }

    fn __pow__(&self, exponent: f32, _modulo: Option<i64>) -> PyTensor {
        PyTensor::new(self.inner.powf(exponent))
    }

    fn __matmul__(&self, other: &PyTensor) -> PyTensor {
        self.matmul(other)
    }

    fn __repr__(&self) -> String {
        let shape_str: Vec<String> = self.inner.shape().dims().iter().map(|d| d.to_string()).collect();
        if let Some(name) = self.inner.var_name() {
            format!("Tensor(name='{}', shape=({}))", name, shape_str.join(", "))
        } else {
            format!("Tensor(shape=({}))", shape_str.join(", "))
        }
    }
}

/// Union type for tensor or scalar arguments.
#[derive(FromPyObject)]
enum TensorOrScalar {
    Tensor(PyTensor),
    Scalar(f32),
}

/// Parse axis argument which can be None, int, or list of ints.
fn parse_axes(axis: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<usize>>> {
    match axis {
        None => Ok(None),
        Some(obj) => {
            if let Ok(i) = obj.extract::<usize>() {
                Ok(Some(vec![i]))
            } else if let Ok(list) = obj.extract::<Vec<usize>>() {
                Ok(Some(list))
            } else {
                Err(PyTypeError::new_err("axis must be None, int, or list of ints"))
            }
        }
    }
}

/// Create a tensor from a numpy array.
///
/// Args:
///     data: numpy array (will be converted to float32)
///     name: optional variable name for gradient tracking
///
/// Returns:
///     A new Tensor
///
/// Example:
///     >>> import numpy as np
///     >>> x = tensor(np.array([1.0, 2.0, 3.0]), name="x")
#[pyfunction]
#[pyo3(signature = (data, name=None))]
fn tensor(data: PyReadonlyArrayDyn<'_, f32>, name: Option<&str>) -> PyResult<PyTensor> {
    let shape: Vec<usize> = data.shape().to_vec();
    let flat_data: Vec<f32> = data.as_slice()?.to_vec();
    let tensor_shape = Shape::new(shape);
    let cpu_tensor = CpuBackend::from_vec(flat_data, tensor_shape);

    let inner = if let Some(n) = name {
        CpuTensorExpr::var(n, cpu_tensor)
    } else {
        CpuTensorExpr::constant(cpu_tensor)
    };

    Ok(PyTensor::new(inner))
}

/// Create a scalar tensor.
///
/// Args:
///     value: scalar value
///     name: optional variable name for gradient tracking
///
/// Example:
///     >>> s = scalar(5.0, name="learning_rate")
#[pyfunction]
#[pyo3(signature = (value, name=None))]
fn scalar(value: f32, name: Option<&str>) -> PyTensor {
    let cpu_tensor = CpuBackend::scalar(value);
    let inner = if let Some(n) = name {
        CpuTensorExpr::var(n, cpu_tensor)
    } else {
        CpuTensorExpr::constant(cpu_tensor)
    };
    PyTensor::new(inner)
}

/// Create a tensor of zeros.
///
/// Args:
///     shape: tuple of dimensions
///
/// Example:
///     >>> z = zeros([2, 3])
#[pyfunction]
fn zeros(shape: Vec<usize>) -> PyTensor {
    PyTensor::new(CpuTensorExpr::zeros(&Shape::new(shape)))
}

/// Create a tensor of ones.
///
/// Args:
///     shape: tuple of dimensions
///
/// Example:
///     >>> o = ones([2, 3])
#[pyfunction]
fn ones(shape: Vec<usize>) -> PyTensor {
    PyTensor::new(CpuTensorExpr::ones(&Shape::new(shape)))
}

// === Legacy scalar API for backwards compatibility ===
// These wrap the old ad_core scalar implementation

use ad_core::{constant as core_constant, var as core_var, Expr as CoreExpr};

/// Legacy scalar expression (for backwards compatibility).
#[pyclass(name = "Expr")]
#[derive(Clone)]
struct PyExpr {
    inner: CoreExpr,
}

impl PyExpr {
    fn new(inner: CoreExpr) -> Self {
        PyExpr { inner }
    }
}

#[pymethods]
impl PyExpr {
    fn value(&self) -> f64 {
        self.inner.value()
    }

    fn is_var(&self) -> bool {
        self.inner.is_var()
    }

    fn var_name(&self) -> Option<String> {
        self.inner.var_name().map(|s| s.to_string())
    }

    fn exp(&self) -> PyExpr {
        PyExpr::new(self.inner.exp())
    }

    fn log(&self) -> PyExpr {
        PyExpr::new(self.inner.log())
    }

    fn sin(&self) -> PyExpr {
        PyExpr::new(self.inner.sin())
    }

    fn cos(&self) -> PyExpr {
        PyExpr::new(self.inner.cos())
    }

    #[pyo3(name = "pow")]
    fn powf(&self, exponent: f64) -> PyExpr {
        PyExpr::new(self.inner.powf(exponent))
    }

    fn backward(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let grads = self.inner.backward();
        let dict = PyDict::new(py);
        for (name, value) in grads.all_named_grads() {
            dict.set_item(name, value)?;
        }
        Ok(dict.into())
    }

    fn __add__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::new(&self.inner + &other.inner)
    }

    fn __radd__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::new(&other.inner + &self.inner)
    }

    fn __sub__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::new(&self.inner - &other.inner)
    }

    fn __rsub__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::new(&other.inner - &self.inner)
    }

    fn __mul__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::new(&self.inner * &other.inner)
    }

    fn __rmul__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::new(&other.inner * &self.inner)
    }

    fn __truediv__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::new(&self.inner / &other.inner)
    }

    fn __rtruediv__(&self, other: &PyExpr) -> PyExpr {
        PyExpr::new(&other.inner / &self.inner)
    }

    fn __neg__(&self) -> PyExpr {
        PyExpr::new(-&self.inner)
    }

    fn __pow__(&self, exponent: f64, _modulo: Option<i64>) -> PyExpr {
        PyExpr::new(self.inner.powf(exponent))
    }

    fn __repr__(&self) -> String {
        if let Some(name) = self.inner.var_name() {
            format!("Expr(var='{}', value={})", name, self.inner.value())
        } else {
            format!("Expr(value={})", self.inner.value())
        }
    }
}

/// Create a scalar variable (legacy API).
#[pyfunction]
fn var(name: &str, value: f64) -> PyExpr {
    PyExpr::new(core_var(name, value))
}

/// Create a scalar constant (legacy API).
#[pyfunction]
fn constant(value: f64) -> PyExpr {
    PyExpr::new(core_constant(value))
}

/// Python module for tensor automatic differentiation.
#[pymodule]
fn ad_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // New tensor API
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(scalar, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;

    // Legacy scalar API
    m.add_class::<PyExpr>()?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(constant, m)?)?;

    Ok(())
}
