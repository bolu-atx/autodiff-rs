//! Python bindings for the reverse-mode autodiff library.
//!
//! Provides a `Expr` class with operator overloads and methods for building
//! and differentiating computational graphs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ad_core::{constant as core_constant, var as core_var, Expr as CoreExpr};

/// A differentiable expression in the computation graph.
///
/// Expressions can be combined using arithmetic operators (+, -, *, /, unary -)
/// and mathematical functions (exp, log, sin, cos, pow).
///
/// Example:
///     >>> x = var("x", 2.0)
///     >>> y = var("y", 3.0)
///     >>> z = x * y + x.sin()
///     >>> z.value()
///     6.909297426825682
///     >>> grads = z.backward()
///     >>> grads["x"]
///     2.5838531634528574
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
    /// Evaluate this expression and return its numerical value.
    fn value(&self) -> f64 {
        self.inner.value()
    }

    /// Check if this expression is a variable.
    fn is_var(&self) -> bool {
        self.inner.is_var()
    }

    /// Get the variable name if this is a variable, else None.
    fn var_name(&self) -> Option<String> {
        self.inner.var_name().map(|s| s.to_string())
    }

    /// Compute exponential: exp(self)
    fn exp(&self) -> PyExpr {
        PyExpr::new(self.inner.exp())
    }

    /// Compute natural logarithm: log(self)
    fn log(&self) -> PyExpr {
        PyExpr::new(self.inner.log())
    }

    /// Compute sine: sin(self)
    fn sin(&self) -> PyExpr {
        PyExpr::new(self.inner.sin())
    }

    /// Compute cosine: cos(self)
    fn cos(&self) -> PyExpr {
        PyExpr::new(self.inner.cos())
    }

    /// Raise to a constant power: self ** exponent
    #[pyo3(name = "pow")]
    fn powf(&self, exponent: f64) -> PyExpr {
        PyExpr::new(self.inner.powf(exponent))
    }

    /// Compute gradients for all variables via reverse-mode autodiff.
    ///
    /// Returns a dictionary mapping variable names to their gradients.
    /// If multiple variables share the same name, their gradients are summed.
    ///
    /// Example:
    ///     >>> x = var("x", 2.0)
    ///     >>> y = var("y", 3.0)
    ///     >>> z = x * y
    ///     >>> z.backward()
    ///     {'x': 3.0, 'y': 2.0}
    fn backward(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let grads = self.inner.backward();
        let dict = PyDict::new(py);
        for (name, value) in grads.all_named_grads() {
            dict.set_item(name, value)?;
        }
        Ok(dict.into())
    }

    /// Compute gradients with respect to specific variables.
    ///
    /// Args:
    ///     vars: List of Expr objects (must be variables) to compute gradients for
    ///
    /// Returns:
    ///     List of gradients in the same order as the input variables
    ///
    /// Example:
    ///     >>> x = var("x", 2.0)
    ///     >>> y = var("y", 3.0)
    ///     >>> z = x * y
    ///     >>> z.grad([x, y])
    ///     [3.0, 2.0]
    fn grad(&self, vars: Vec<PyRef<PyExpr>>) -> PyResult<Vec<f64>> {
        let grads = self.inner.backward();
        let mut result = Vec::with_capacity(vars.len());
        for v in vars {
            if !v.inner.is_var() {
                return Err(PyValueError::new_err(
                    "grad() requires variable expressions",
                ));
            }
            result.push(grads.wrt(&v.inner).unwrap_or(0.0));
        }
        Ok(result)
    }

    // === Operator overloads ===

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

/// Create a new variable with the given name and value.
///
/// Variables are the leaves of the computation graph that we differentiate
/// with respect to. Each call creates a new variable with a unique identity.
///
/// Args:
///     name: Variable name (used for gradient lookup)
///     value: Initial value
///
/// Returns:
///     A new Expr representing this variable
///
/// Example:
///     >>> x = var("x", 2.0)
///     >>> x.value()
///     2.0
#[pyfunction]
fn var(name: &str, value: f64) -> PyExpr {
    PyExpr::new(core_var(name, value))
}

/// Create a constant expression.
///
/// Constants are fixed values in the computation graph. They have zero gradient.
///
/// Args:
///     value: The constant value
///
/// Returns:
///     A new Expr representing this constant
///
/// Example:
///     >>> c = constant(5.0)
///     >>> c.value()
///     5.0
#[pyfunction]
fn constant(value: f64) -> PyExpr {
    PyExpr::new(core_constant(value))
}

/// Python module for reverse-mode automatic differentiation.
#[pymodule]
fn ad_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(constant, m)?)?;
    Ok(())
}
