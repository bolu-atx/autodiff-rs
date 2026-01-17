"""
Tests for the ad_py Python bindings.

Run with: pytest tests/ -v
"""

import math
import pytest

from ad_py import var, constant


class TestBasicOperations:
    """Test basic arithmetic operations."""

    def test_var_creation(self):
        x = var("x", 2.0)
        assert x.value() == 2.0
        assert x.is_var()
        assert x.var_name() == "x"

    def test_constant_creation(self):
        c = constant(5.0)
        assert c.value() == 5.0
        assert not c.is_var()
        assert c.var_name() is None

    def test_addition(self):
        x = var("x", 2.0)
        y = var("y", 3.0)
        z = x + y
        assert z.value() == 5.0

    def test_subtraction(self):
        x = var("x", 5.0)
        y = var("y", 3.0)
        z = x - y
        assert z.value() == 2.0

    def test_multiplication(self):
        x = var("x", 2.0)
        y = var("y", 3.0)
        z = x * y
        assert z.value() == 6.0

    def test_division(self):
        x = var("x", 6.0)
        y = var("y", 2.0)
        z = x / y
        assert z.value() == 3.0

    def test_negation(self):
        x = var("x", 2.0)
        z = -x
        assert z.value() == -2.0


class TestTranscendentalFunctions:
    """Test exp, log, sin, cos, pow."""

    def test_exp(self):
        x = var("x", 1.0)
        z = x.exp()
        assert abs(z.value() - math.exp(1.0)) < 1e-10

    def test_log(self):
        x = var("x", math.e)
        z = x.log()
        assert abs(z.value() - 1.0) < 1e-10

    def test_sin(self):
        x = var("x", math.pi / 2)
        z = x.sin()
        assert abs(z.value() - 1.0) < 1e-10

    def test_cos(self):
        x = var("x", 0.0)
        z = x.cos()
        assert abs(z.value() - 1.0) < 1e-10

    def test_pow(self):
        x = var("x", 2.0)
        z = x.pow(3.0)
        assert abs(z.value() - 8.0) < 1e-10

    def test_pow_operator(self):
        x = var("x", 2.0)
        z = x ** 3.0
        assert abs(z.value() - 8.0) < 1e-10


class TestGradients:
    """Test gradient computation."""

    def test_add_gradient(self):
        # z = x + y
        # dz/dx = 1, dz/dy = 1
        x = var("x", 2.0)
        y = var("y", 3.0)
        z = x + y

        grads = z.backward()
        assert abs(grads["x"] - 1.0) < 1e-10
        assert abs(grads["y"] - 1.0) < 1e-10

    def test_sub_gradient(self):
        # z = x - y
        # dz/dx = 1, dz/dy = -1
        x = var("x", 2.0)
        y = var("y", 3.0)
        z = x - y

        grads = z.backward()
        assert abs(grads["x"] - 1.0) < 1e-10
        assert abs(grads["y"] - (-1.0)) < 1e-10

    def test_mul_gradient(self):
        # z = x * y
        # dz/dx = y, dz/dy = x
        x = var("x", 2.0)
        y = var("y", 3.0)
        z = x * y

        grads = z.backward()
        assert abs(grads["x"] - 3.0) < 1e-10
        assert abs(grads["y"] - 2.0) < 1e-10

    def test_div_gradient(self):
        # z = x / y
        # dz/dx = 1/y, dz/dy = -x/y^2
        x = var("x", 6.0)
        y = var("y", 2.0)
        z = x / y

        grads = z.backward()
        assert abs(grads["x"] - 0.5) < 1e-10  # 1/2
        assert abs(grads["y"] - (-1.5)) < 1e-10  # -6/4

    def test_neg_gradient(self):
        # z = -x
        # dz/dx = -1
        x = var("x", 2.0)
        z = -x

        grads = z.backward()
        assert abs(grads["x"] - (-1.0)) < 1e-10

    def test_pow_gradient(self):
        # z = x^3
        # dz/dx = 3*x^2
        x = var("x", 2.0)
        z = x ** 3.0

        grads = z.backward()
        assert abs(grads["x"] - 12.0) < 1e-10  # 3 * 4

    def test_exp_gradient(self):
        # z = exp(x)
        # dz/dx = exp(x)
        x = var("x", 1.0)
        z = x.exp()

        grads = z.backward()
        assert abs(grads["x"] - math.exp(1.0)) < 1e-10

    def test_log_gradient(self):
        # z = log(x)
        # dz/dx = 1/x
        x = var("x", 2.0)
        z = x.log()

        grads = z.backward()
        assert abs(grads["x"] - 0.5) < 1e-10

    def test_sin_gradient(self):
        # z = sin(x)
        # dz/dx = cos(x)
        x = var("x", 1.0)
        z = x.sin()

        grads = z.backward()
        assert abs(grads["x"] - math.cos(1.0)) < 1e-10

    def test_cos_gradient(self):
        # z = cos(x)
        # dz/dx = -sin(x)
        x = var("x", 1.0)
        z = x.cos()

        grads = z.backward()
        assert abs(grads["x"] - (-math.sin(1.0))) < 1e-10


class TestChainRule:
    """Test chain rule through composed expressions."""

    def test_sin_of_square(self):
        # z = sin(x^2)
        # dz/dx = cos(x^2) * 2x
        x = var("x", 2.0)
        z = (x ** 2.0).sin()

        grads = z.backward()
        expected = math.cos(4.0) * 4.0
        assert abs(grads["x"] - expected) < 1e-10

    def test_exp_of_product(self):
        # z = exp(x * y)
        # dz/dx = y * exp(x*y)
        # dz/dy = x * exp(x*y)
        x = var("x", 1.0)
        y = var("y", 2.0)
        z = (x * y).exp()

        grads = z.backward()
        expected_dx = 2.0 * math.exp(2.0)
        expected_dy = 1.0 * math.exp(2.0)
        assert abs(grads["x"] - expected_dx) < 1e-10
        assert abs(grads["y"] - expected_dy) < 1e-10

    def test_log_of_sum(self):
        # z = log(x + y)
        # dz/dx = 1/(x+y)
        # dz/dy = 1/(x+y)
        x = var("x", 1.0)
        y = var("y", 3.0)
        z = (x + y).log()

        grads = z.backward()
        expected = 0.25  # 1/4
        assert abs(grads["x"] - expected) < 1e-10
        assert abs(grads["y"] - expected) < 1e-10


class TestSharedNodes:
    """Test expressions where the same variable appears multiple times."""

    def test_x_squared(self):
        # z = x * x
        # dz/dx = 2x
        x = var("x", 3.0)
        z = x * x

        grads = z.backward()
        assert abs(grads["x"] - 6.0) < 1e-10

    def test_diamond_graph(self):
        # z = (x + y) * (x - y) = x^2 - y^2
        # dz/dx = 2x
        # dz/dy = -2y
        x = var("x", 3.0)
        y = var("y", 2.0)
        a = x + y
        b = x - y
        z = a * b

        grads = z.backward()
        assert abs(grads["x"] - 6.0) < 1e-10
        assert abs(grads["y"] - (-4.0)) < 1e-10


class TestGradMethod:
    """Test the grad() method that takes a list of variables."""

    def test_grad_ordered(self):
        x = var("x", 2.0)
        y = var("y", 3.0)
        z = x * y

        # Get gradients in specified order
        grads = z.grad([x, y])
        assert abs(grads[0] - 3.0) < 1e-10  # dz/dx = y
        assert abs(grads[1] - 2.0) < 1e-10  # dz/dy = x

        # Reversed order
        grads_rev = z.grad([y, x])
        assert abs(grads_rev[0] - 2.0) < 1e-10  # dz/dy = x
        assert abs(grads_rev[1] - 3.0) < 1e-10  # dz/dx = y


class TestComplexExpressions:
    """Test more complex expressions against finite differences."""

    def finite_diff(self, f, point, eps=1e-7):
        """Compute gradients using central finite differences."""
        grads = []
        for i in range(len(point)):
            perturbed_plus = point.copy()
            perturbed_plus[i] += eps
            perturbed_minus = point.copy()
            perturbed_minus[i] -= eps
            grad = (f(perturbed_plus) - f(perturbed_minus)) / (2 * eps)
            grads.append(grad)
        return grads

    def test_complex_expression(self):
        # z = (x*y + sin(x)) / (y + 2)
        x_val, y_val = 1.5, 2.5

        x = var("x", x_val)
        y = var("y", y_val)
        z = (x * y + x.sin()) / (y + constant(2.0))

        grads = z.backward()

        # Finite difference check
        def f(vals):
            x = var("x", vals[0])
            y = var("y", vals[1])
            return ((x * y + x.sin()) / (y + constant(2.0))).value()

        fd_grads = self.finite_diff(f, [x_val, y_val])

        assert abs(grads["x"] - fd_grads[0]) < 1e-5
        assert abs(grads["y"] - fd_grads[1]) < 1e-5

    def test_nested_transcendental(self):
        # z = exp(sin(x)) * cos(log(y))
        x_val, y_val = 0.5, 2.0

        x = var("x", x_val)
        y = var("y", y_val)
        z = x.sin().exp() * y.log().cos()

        grads = z.backward()

        def f(vals):
            x = var("x", vals[0])
            y = var("y", vals[1])
            return (x.sin().exp() * y.log().cos()).value()

        fd_grads = self.finite_diff(f, [x_val, y_val])

        assert abs(grads["x"] - fd_grads[0]) < 1e-5
        assert abs(grads["y"] - fd_grads[1]) < 1e-5


class TestRepr:
    """Test string representation."""

    def test_var_repr(self):
        x = var("x", 2.0)
        assert "x" in repr(x)
        assert "2" in repr(x)  # value may show as 2 or 2.0

    def test_expr_repr(self):
        x = var("x", 2.0)
        y = var("y", 3.0)
        z = x + y
        assert "5" in repr(z)  # value may show as 5 or 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
