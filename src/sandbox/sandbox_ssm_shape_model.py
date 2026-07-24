"""Sandbox: the osimfit-ssm/CLEAN_PLAN.md shape-model contract, and an SSM
shape factor threaded through a CasADi callback the same way
sandbox_jacobians.py threads a generalized coordinate q through
PositionCallback.

CLEAN_PLAN.md section 2's contract, with one simplification (see
`ShapeModel`'s docstring below): here, every shape model must supply an exact
analytic `geometry_jacobian`, not just `apply`.

    n_params, nominal, bounds  -- how many numbers in s, the s that
                                   reproduces the stock geometry, and its
                                   plausible range
    apply(s)             -> morphed geometry
    geometry_jacobian(s) -> exact d(apply)/ds
    prior(s)             -> (value, grad) -- regularizer, ||s - nominal||^2
                              family

`ShapeModel` below is that contract as an ABC. `FemurHeadShapeModel` is one
real implementation of it: the combined femur+hip mean shape from a
536-subject PCA (osimfit-ssm's Joint_Femur_Hip_SSM), reduced to its first
principal component, restricted to the femoral-head landmark patch, with
`apply` the morphed femoral-head sphere center

    p(s) = sphere_fit(mean[head] + s * sqrt(eigenvalue_1) * pc_1[head])

and an exact analytic `geometry_jacobian`. `ShapeModelCallback` threads `s`
through CasADi using that Jacobian directly -- it works unmodified for any
`ShapeModel`, not just this one, as long as that model provides
`geometry_jacobian`.

Note: the landmark morph itself is linear in s (mean + s * basis), but the
sphere-center estimator built on top of those landmarks is not -- `apply`'s
analytic Jacobian below is exact but varies with s (see check_jacobian_fd's
output at a few different s0 to see the drift). See osimfit-ssm's
CLEAN_PLAN.md for the full version: 7 modes instead of 1 and per-joint offset
frames instead of a single landmark.

`prior(s)` joins the NLP cost the same way `apply(s)` does: through a CasADi
callback (`ShapePriorCallback`) built from `prior`'s analytic gradient, not
reimplemented as a CasADi expression at the call site. That keeps the "narrow
interface is enough" claim honest for the regularizer too, not just the
geometry.
"""

import csv
from abc import ABC, abstractmethod
from pathlib import Path

import casadi as ca
import numpy as np

DATA = Path(__file__).parent / "ssm_shape_model_data"


def _as_scalar(s):
    """float(s) that also accepts length-1 arrays/CasADi DMs (NumPy 2 stopped
    allowing float() on anything with ndim > 0, even size 1)."""
    return float(np.asarray(s).reshape(-1)[0])


class ShapeModel(ABC):
    """s in, morphed geometry out -- CLEAN_PLAN.md's contract, with one
    simplification for this sandbox: `geometry_jacobian` is required, not
    optional. CLEAN_PLAN.md allows a shape model to provide only `apply` and
    let the fitter finite-difference the rest; here we assume every shape
    model provides an exact analytic Jacobian up front, so ShapeModelCallback
    never needs an FD fallback path.

    n_params/nominal/bounds are abstract properties rather than bare
    annotations, so a subclass that forgets one fails at instantiation
    instead of with an AttributeError the first time something reads it.
    """

    @property
    @abstractmethod
    def n_params(self):
        """Length of s."""

    @property
    @abstractmethod
    def nominal(self):
        """The s that reproduces the stock (unmorphed) geometry."""

    @property
    @abstractmethod
    def bounds(self):
        """(lower, upper), broadcast over n_params."""

    @abstractmethod
    def apply(self, s):
        """Morphed geometry at shape factor s, as a flat array."""

    @abstractmethod
    def geometry_jacobian(self, s):
        """Exact d(apply)/ds, shape (len(apply(s)), n_params)."""

    @abstractmethod
    def prior(self, s):
        """(value, grad) of the shape-factor regularizer at s."""


def _sphere_center(L, dL):
    """Least-squares (Kasa) sphere center through points L (m x 3), and its
    Jacobian w.r.t. a single shape factor given dL = dL/ds (m x 3).

    Solves the normal equations M c = A^T b for c = [center; d], with
    A row j = [2x_j, 2y_j, 2z_j, 1] and b_j = x_j^2 + y_j^2 + z_j^2, then
    differentiates the solution once along s.
    """
    x, y, z = L[:, 0], L[:, 1], L[:, 2]
    A = np.column_stack([2 * x, 2 * y, 2 * z, np.ones(len(L))])
    b = x * x + y * y + z * z
    Minv = np.linalg.inv(A.T @ A)
    c = Minv @ (A.T @ b)
    residual = b - A @ c

    dx, dy, dz = dL[:, 0], dL[:, 1], dL[:, 2]
    dA = np.column_stack([2 * dx, 2 * dy, 2 * dz, np.zeros(len(L))])
    db = 2 * (x * dx + y * dy + z * dz)
    dc = Minv @ (dA.T @ residual + A.T @ (db - dA @ c))
    return c[:3], dc[:3]


def load_head_indices():
    """Femoral-head vertex indices, from a ParaView point-selection CSV."""
    with open(DATA / "femur_head_points.csv") as f:
        reader = csv.reader(f)
        col = next(reader).index("vtkOriginalPointIds")
        idx = [int(float(row[col])) for row in reader]
    return np.unique(idx)


class FemurHeadShapeModel(ShapeModel):
    """One-mode linear shape model over the femoral-head landmark patch.

    The landmarks x(s) = mean + s * basis are exactly linear in s (basis =
    sqrt(eigenvalue_1) * pc_1), so s is in standard-deviation units. `apply`
    is the sphere center through those landmarks -- nonlinear in the
    landmarks, so `geometry_jacobian(s)` is exact but s-dependent, not a
    constant.
    """

    n_params = 1
    bounds = (-3.0, 3.0)  # +/- 3 SD

    @property
    def nominal(self):
        return np.zeros(self.n_params)

    def __init__(self):
        mean = np.load(DATA / "mean_shape.npy").reshape(-1, 3)
        pc1 = np.load(DATA / "pc1.npy").reshape(-1, 3)
        eig1 = float(np.load(DATA / "eigenvalue1.npy"))
        idx = load_head_indices()
        self.mean_pts = mean[idx]
        self.basis_pts = pc1[idx] * np.sqrt(eig1)
        self._cache = None  # (s, p, dp) for the most recent call

    def landmarks(self, s):
        """Morphed femoral-head point patch (m x 3) at shape factor s."""
        return self.mean_pts + self.basis_pts * _as_scalar(s)

    def _sphere(self, s):
        s = _as_scalar(s)
        if self._cache is None or self._cache[0] != s:
            p, dp = _sphere_center(self.landmarks(s), self.basis_pts)
            self._cache = (s, p, dp)
        return self._cache[1], self._cache[2]

    def apply(self, s):
        p, _ = self._sphere(s)
        return p

    def geometry_jacobian(self, s):
        _, dp = self._sphere(s)
        return dp.reshape(3, 1)

    def prior(self, s):
        """Mahalanobis prior in SD units: value and gradient."""
        s = _as_scalar(s)
        return s ** 2, np.array([2.0 * s])


class ShapeModelCallback(ca.Callback):
    """CasADi callback s -> shape_model.apply(s), for any `ShapeModel`, using
    shape_model.geometry_jacobian(s) as the exact Jacobian. Mirrors
    PositionCallback_Jac in sandbox_jacobians.py, but for a shape factor
    instead of a generalized coordinate, and generic over output size /
    n_params instead of hardcoded to one landmark.
    """

    def __init__(self, name, shape_model, output_size, opts={}):
        self.shape_model = shape_model
        self.output_size = output_size
        ca.Callback.__init__(self)
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.shape_model.n_params, 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.output_size, 1)

    def eval(self, arg):
        s = np.asarray(arg[0]).flatten()
        return [np.asarray(self.shape_model.apply(s)).reshape(-1, 1)]

    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        shape_model = self.shape_model
        output_size = self.output_size

        class JacFun(ca.Callback):
            def __init__(self, opts={}):
                ca.Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 2   # nominal in, nominal out
            def get_n_out(self): return 1

            def get_sparsity_in(self, i):
                if i == 0:
                    return ca.Sparsity.dense(shape_model.n_params, 1)
                return ca.Sparsity(output_size, 1)

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(output_size, shape_model.n_params)

            def eval(self, arg):
                J = shape_model.geometry_jacobian(np.asarray(arg[0]).flatten())
                return [np.asarray(J).reshape(output_size, shape_model.n_params)]

        self._jac_callback = JacFun()
        return self._jac_callback


class ShapePriorCallback(ca.Callback):
    """CasADi callback s -> shape_model.prior(s)[0], using
    shape_model.prior(s)[1] as the exact gradient. Same wrapping pattern as
    ShapeModelCallback, so the regularizer joins the NLP cost through the
    same kind of narrow interface as the geometry does, instead of being
    reimplemented as a CasADi expression at the call site.
    """

    def __init__(self, name, shape_model, opts={}):
        self.shape_model = shape_model
        ca.Callback.__init__(self)
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.shape_model.n_params, 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(1, 1)

    def eval(self, arg):
        s = np.asarray(arg[0]).flatten()
        value, _ = self.shape_model.prior(s)
        return [np.array([[value]])]

    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        shape_model = self.shape_model

        class JacFun(ca.Callback):
            def __init__(self, opts={}):
                ca.Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 2   # nominal in, nominal out
            def get_n_out(self): return 1

            def get_sparsity_in(self, i):
                if i == 0:
                    return ca.Sparsity.dense(shape_model.n_params, 1)
                return ca.Sparsity(1, 1)

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(1, shape_model.n_params)

            def eval(self, arg):
                s = np.asarray(arg[0]).flatten()
                _, grad = shape_model.prior(s)
                return [np.asarray(grad).reshape(1, shape_model.n_params)]

        self._jac_callback = JacFun()
        return self._jac_callback


def check_jacobian_fd(shape_model, s0=0.3, eps=1e-4):
    """Analytic geometry_jacobian vs central differences on apply -- a
    one-off correctness check, not something the fitter runs at solve time.
    Also shows that dp/ds is exact but not constant in s (call this at a few
    different s0 and compare)."""
    dp = shape_model.geometry_jacobian(s0).flatten()
    p_plus = np.asarray(shape_model.apply(s0 + eps)).flatten()
    p_minus = np.asarray(shape_model.apply(s0 - eps)).flatten()
    dp_fd = (p_plus - p_minus) / (2 * eps)
    return dp, dp_fd


def fit_shape_factor(shape_model, target, prior_weight=1e-3):
    """Recover s* by minimizing ||apply(s) - target||^2 + prior_weight *
    prior(s): the single-landmark analogue of osimfit-ssm's bilevel shape
    fit, and CLEAN_PLAN.md section 5's "recovery of a known s*" check."""
    s = ca.MX.sym("s")
    head = ShapeModelCallback("head", shape_model, output_size=3)
    prior = ShapePriorCallback("prior", shape_model)
    cost = ca.sumsqr(head(s) - target) + prior_weight * prior(s)
    # limited-memory: our callbacks only supply Jacobians, not Hessians
    # (same reason osimfit's own solvers.py sets this -- see solvers.py).
    solver = ca.nlpsol("solver", "ipopt", {"x": s, "f": cost},
                        {"print_time": False, "ipopt.print_level": 0,
                         "ipopt.hessian_approximation": "limited-memory"})
    sol = solver(x0=0.0, lbx=shape_model.bounds[0], ubx=shape_model.bounds[1])
    return float(sol["x"])


if __name__ == "__main__":
    model = FemurHeadShapeModel()

    print("analytic dp/ds at a few s0 (exact, but not constant -- see the")
    print("module docstring: apply() is linear in s, geometry_jacobian isn't):")
    for s0 in (-2.0, 0.0, 2.0):
        dp, dp_fd = check_jacobian_fd(model, s0=s0)
        print(f"  s0={s0:5.1f}  analytic={dp}  max FD diff={np.max(np.abs(dp - dp_fd)):.2e}")

    s_true = -1.4
    target = model.apply(s_true)
    s_hat = fit_shape_factor(model, target)
    print(f"\ntrue s = {s_true:.3f}, recovered s = {s_hat:.3f}")
