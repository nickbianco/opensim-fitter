"""Sandbox: plugging a statistical shape model into opensim-fitter's NLP the
same way a body scale already does -- a small parameter vector `s` in,
morphed geometry out, through a CasADi callback with an analytic Jacobian.

`ShapeModel` is the interface a shape model needs to implement:
`n_params`/`nominal`/`bounds` describe `s`, `apply(s)` morphs the geometry,
`geometry_jacobian(s)` is its exact derivative, and `prior(s)` regularizes
implausible `s`. `FemurHeadShapeModel` is one real implementation, built
from a real 536-subject femur+hip PCA: it morphs a femoral-head landmark
patch and fits a sphere through it.

`ShapeModelCallback` wraps any `ShapeModel` as a CasADi `Callback`, the same
way `PositionCallback`/`PositionCallback_Jac` in sandbox_jacobians.py wrap a
generalized coordinate `q` -- needed because `apply`/`geometry_jacobian`
call real NumPy linear algebra CasADi can't see inside of. `prior(s)`
doesn't need that: it's plain arithmetic, so CasADi differentiates it
directly when `fit_shape_factor` calls it on a symbolic `s`.
"""

import csv
from abc import ABC, abstractmethod
from pathlib import Path

import casadi as ca
import numpy as np

DATA = Path(__file__).parent / "ssm_shape_model_data"


class ShapeModel(ABC):
    """s in, morphed geometry out. Every shape model must supply an exact
    analytic `geometry_jacobian` -- no finite-difference fallback here.

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


class FemurHeadShapeModel(ShapeModel):
    """Linear shape model over the femoral-head landmark patch, apply()'d
    through a least-squares sphere fit.

    landmarks(s) = mean + basis @ s is exactly linear in s (basis columns =
    sqrt(eigenvalue_i) * pc_i for each of the n_modes requested at
    construction, so s is in standard-deviation units). `apply` is the
    sphere center through those landmarks -- nonlinear in the landmarks, so
    `geometry_jacobian(s)` is exact but s-dependent, not a constant.
    """

    bounds = (-3.0, 3.0)  # +/- 3 SD

    def __init__(self, n_modes=1):
        mean = np.load(DATA / "mean_shape.npy").reshape(-1, 3)
        idx = self._load_head_indices()
        self.mean_pts = mean[idx]
        # (n_pts, 3, n_modes). Only mode 1 ships with this sandbox, so
        # n_modes > 1 raises FileNotFoundError until more modes' data exists.
        self.basis_pts = np.stack(
            [self._load_pca_basis(mode)[idx] for mode in range(1, n_modes + 1)],
            axis=-1,
        )
        self._n_params = n_modes
        self._cache = None  # (s, p, dp) for the most recent call

    @property
    def n_params(self):
        return self._n_params

    @property
    def nominal(self):
        # s=0 reproduces the mean shape by construction, so nominal is
        # always zero here (not true for every ShapeModel, hence still
        # abstract on the base class).
        return np.zeros(self.n_params)

    @staticmethod
    def _load_head_indices():
        """Femoral-head vertex indices, from a ParaView point-selection CSV."""
        with open(DATA / "femur_head_points.csv") as f:
            reader = csv.reader(f)
            col = next(reader).index("vtkOriginalPointIds")
            idx = [int(float(row[col])) for row in reader]
        return np.unique(idx)

    @staticmethod
    def _load_pca_basis(mode):
        """One PCA mode's basis vector (sqrt(eigenvalue) * pc), from
        `pc{mode}.npy` and `eigenvalue{mode}.npy` in DATA, both flattened to
        (n_vertices, 3). This per-mode-file layout is specific to how this
        sandbox's data happens to be split up -- not a convention any other
        ShapeModel is expected to follow.
        """
        pc = np.load(DATA / f"pc{mode}.npy").reshape(-1, 3)
        eig = float(np.load(DATA / f"eigenvalue{mode}.npy"))
        return pc * np.sqrt(eig)

    def landmarks(self, s):
        """Morphed femoral-head point patch (m x 3) at shape factor s."""
        return self.mean_pts + self.basis_pts @ np.asarray(s).reshape(-1)

    @staticmethod
    def _sphere_center(L, dL):
        """Least-squares (Kasa) sphere center through points L (m x 3), plus
        its Jacobian given dL = d(landmarks)/ds (m x 3 x n_params) --
        differentiates the closed-form fit once per column of dL.
        """
        x, y, z = L[:, 0], L[:, 1], L[:, 2]
        A = np.column_stack([2 * x, 2 * y, 2 * z, np.ones(len(L))])
        b = x * x + y * y + z * z
        Minv = np.linalg.inv(A.T @ A)
        c = Minv @ (A.T @ b)
        residual = b - A @ c

        n_params = dL.shape[-1]
        dc = np.empty((4, n_params))
        for k in range(n_params):
            dx, dy, dz = dL[:, 0, k], dL[:, 1, k], dL[:, 2, k]
            dA = np.column_stack([2 * dx, 2 * dy, 2 * dz, np.zeros(len(L))])
            db = 2 * (x * dx + y * dy + z * dz)
            dc[:, k] = Minv @ (dA.T @ residual + A.T @ (db - dA @ c))
        return c[:3], dc[:3]

    def _sphere(self, s):
        s = np.asarray(s).reshape(-1)
        key = tuple(s)
        if self._cache is None or self._cache[0] != key:
            p, dp = self._sphere_center(self.landmarks(s), self.basis_pts)
            self._cache = (key, p, dp)
        return self._cache[1], self._cache[2]

    def apply(self, s):
        p, _ = self._sphere(s)
        return p

    def geometry_jacobian(self, s):
        _, dp = self._sphere(s)
        return dp

    def prior(self, s):
        """Mahalanobis prior in SD units: value and gradient. Written with
        only `.T`/`@`/`*`, so it works unmodified whether s is numeric or a
        CasADi symbol -- see fit_shape_factor, which calls this directly on
        the symbolic decision variable instead of wrapping it in a
        Callback.
        """
        return s.T @ s, 2.0 * s


class ShapeModelCallback(ca.Callback):
    """CasADi callback s -> shape_model.apply(s), using
    shape_model.geometry_jacobian(s) as the exact Jacobian. Works for any
    ShapeModel, generic over n_params and output size. Needed because
    apply/geometry_jacobian call real NumPy linear algebra CasADi can't see
    inside of -- unlike prior(s), which is plain arithmetic and gets called
    directly on the symbolic s with no Callback at all.
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


def check_jacobian_fd(shape_model, s0=0.3, eps=1e-4):
    """Analytic geometry_jacobian vs. central differences on apply -- a
    one-off correctness check for a new shape model's Jacobian."""
    dp = shape_model.geometry_jacobian(s0).flatten()
    p_plus = np.asarray(shape_model.apply(s0 + eps)).flatten()
    p_minus = np.asarray(shape_model.apply(s0 - eps)).flatten()
    dp_fd = (p_plus - p_minus) / (2 * eps)
    return dp, dp_fd


def fit_shape_factor(shape_model, target, prior_weight=1e-3):
    """Recover s* by minimizing ||apply(s) - target||^2 + prior_weight *
    prior(s) with ipopt -- the same kind of NLP osimfit's BilevelSolver runs.

    TODO: once this moves into src/osimfit, add a test asserting recovery of
    a known s* (see __main__ below for the exact check)."""
    s = ca.MX.sym("s", shape_model.n_params)
    head = ShapeModelCallback("head", shape_model, output_size=3)
    prior_value, _ = shape_model.prior(s)
    cost = ca.sumsqr(head(s) - target) + prior_weight * prior_value
    # limited-memory: the geometry callback only supplies a Jacobian, not a
    # Hessian (same reason osimfit's own solvers.py sets this).
    solver = ca.nlpsol("solver", "ipopt", {"x": s, "f": cost},
                        {"print_time": False, "ipopt.print_level": 0,
                         "ipopt.hessian_approximation": "limited-memory"})
    sol = solver(x0=np.zeros(shape_model.n_params),
                 lbx=shape_model.bounds[0], ubx=shape_model.bounds[1])
    return np.asarray(sol["x"]).flatten()


if __name__ == "__main__":
    model = FemurHeadShapeModel()

    print("analytic dp/ds at a few s0 (exact, but not constant -- apply()")
    print("is linear in s, geometry_jacobian isn't):")
    for s0 in (-2.0, 0.0, 2.0):
        dp, dp_fd = check_jacobian_fd(model, s0=s0)
        print(f"  s0={s0:5.1f}  analytic={dp}  max FD diff={np.max(np.abs(dp - dp_fd)):.2e}")

    s_true = -1.4
    target = model.apply(s_true)
    s_hat = fit_shape_factor(model, target)
    print(f"\ntrue s = {s_true:.3f}, recovered s = {s_hat[0]:.3f}")
