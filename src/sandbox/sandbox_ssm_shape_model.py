"""Sandbox: plugging a statistical shape model into opensim-fitter's bilevel
machinery -- a small, self-contained worked example, deliberately decoupled from the
real code so it can be read start to finish without jumping into osimfit.

The real integration (see SHAPE_MODEL.md) has two layers on top of what's here:

1. `osimfit.model.ShapeModel` is the same math contract as the `ShapeModel` below,
   plus the glue that makes it a bilevel `Parameter`: `expand()` (fold `s` into a
   marker/frame offset), `to_groups()`, `prior_expr()`, and per-path
   `validate()`/`apply_to_model()` via `MarkerTargetMixin`/`FrameTargetMixin`. None of
   that lives here -- this file only implements the math, not the wiring.
2. `osimfit.callbacks.ShapeModelCallback` is the class below, promoted essentially
   unchanged.

`n_params`/`nominal`/`bounds` describe the shape factor `s`, `apply(s)` morphs
geometry, `geometry_jacobian(s)` is its exact derivative, and `prior(s)` regularizes
implausible `s`. `FemurHeadLandmarkShapeModel` is a real implementation, built from a
real 536-subject femur+hip PCA: it picks one vertex on the femoral head and moves it
along the PCA modes. That's the simplest possible linear shape model -- exactly the
shape `osimfit.model.ShapeModel.expand()` requires (a constant `geometry_jacobian`),
and the shape every real PCA-based SSM has.
"""

import csv
from abc import ABC, abstractmethod
from pathlib import Path

import casadi as ca
import numpy as np

DATA = Path(__file__).parent / "ssm_shape_model_data"


class ShapeModel(ABC):
    """s in, morphed geometry out. Every shape model supplies an exact analytic
    `geometry_jacobian` -- no finite-difference fallback here.

    n_params/nominal/bounds are abstract properties rather than bare annotations, so a
    subclass that forgets one fails at instantiation instead of with an
    AttributeError the first time something reads it.
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


class FemurHeadLandmarkShapeModel(ShapeModel):
    """One landmark on the femoral head, morphed along the first `n_modes` PCA modes
    of a real 536-subject femur+hip PCA.

    landmark(s) = mean_pt + basis_pt @ s is exactly linear in s (basis columns =
    sqrt(eigenvalue_i) * pc_i at the chosen vertex, for each requested mode, so s is
    in standard-deviation units). Because apply() is linear, geometry_jacobian is just
    the constant basis_pt matrix, not a function of s at all -- see __main__ below.
    """

    def __init__(self, n_modes=1, vertex_index=None):
        mean = np.load(DATA / "mean_shape.npy").reshape(-1, 3)
        head_idx = self._load_head_indices()
        # One vertex from the femoral-head patch, not an average or a surface fit
        # through it -- the simplest possible landmark.
        vertex_index = head_idx[0] if vertex_index is None else vertex_index
        self.mean_pt = mean[vertex_index]
        self.basis_pt = np.stack(
            [self._load_pca_basis(mode)[vertex_index]
             for mode in range(1, n_modes + 1)],
            axis=-1,
        )  # (3, n_modes)
        self._n_params = n_modes

    @property
    def n_params(self):
        return self._n_params

    @property
    def nominal(self):
        # s=0 reproduces the mean shape by construction.
        return np.zeros(self.n_params)

    @property
    def bounds(self):
        return (-3.0, 3.0)  # +/- 3 SD

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
        """One PCA mode's basis vector (sqrt(eigenvalue) * pc), from `pc{mode}.npy`
        and `eigenvalue{mode}.npy` in DATA, both flattened to (n_vertices, 3)."""
        pc = np.load(DATA / f"pc{mode}.npy").reshape(-1, 3)
        eig = float(np.load(DATA / f"eigenvalue{mode}.npy"))
        return pc * np.sqrt(eig)

    def apply(self, s):
        return self.mean_pt + self.basis_pt @ np.asarray(s).reshape(-1)

    def geometry_jacobian(self, s):
        return self.basis_pt

    def prior(self, s):
        """Mahalanobis prior in SD units: value and gradient. Written with only
        `.T`/`@`/`*`, so it works unmodified whether s is numeric or a CasADi symbol
        -- see fit_shape_factor, which calls this directly on the symbolic decision
        variable instead of wrapping it in a Callback.
        """
        return s.T @ s, 2.0 * s


class ShapeModelCallback(ca.Callback):
    """CasADi callback s -> shape_model.apply(s), using
    shape_model.geometry_jacobian(s) as the exact Jacobian. Works for any
    ShapeModel, generic over n_params and output size. Needed because
    apply/geometry_jacobian call real NumPy linear algebra CasADi can't see
    inside of -- unlike prior(s), which is plain arithmetic and gets called
    directly on the symbolic s with no Callback at all.

    In the real code (osimfit.callbacks.ShapeModelCallback, promoted from here
    essentially unchanged), ShapeModel.expand() -- the bilevel-Parameter fast path --
    doesn't use this: a linear shape model's Jacobian is constant, so expand() caches
    geometry_jacobian(nominal) once as a plain matrix instead of wrapping a Callback.
    This class is still needed wherever geometry_jacobian must be re-evaluated at a
    changing s, e.g. fit_shape_factor below.
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
    dp = np.asarray(shape_model.geometry_jacobian(s0)).flatten()
    p_plus = np.asarray(shape_model.apply(s0 + eps)).flatten()
    p_minus = np.asarray(shape_model.apply(s0 - eps)).flatten()
    dp_fd = (p_plus - p_minus) / (2 * eps)
    return dp, dp_fd


def fit_shape_factor(shape_model, target, prior_weight=1e-3):
    """Recover s* by minimizing ||apply(s) - target||^2 + prior_weight *
    prior(s) with ipopt -- the same kind of NLP osimfit's BilevelSolver runs, but
    standalone: this is the use case ShapeModel.expand() doesn't cover, since expand()
    only folds a shape model into an existing bilevel solve, not a solve of its own.
    """
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
    model = FemurHeadLandmarkShapeModel()

    print("analytic dp/ds at a few s0 (exact, and constant -- apply() is linear,")
    print("so geometry_jacobian doesn't actually depend on s0):")
    for s0 in (-2.0, 0.0, 2.0):
        dp, dp_fd = check_jacobian_fd(model, s0=s0)
        print(f"  s0={s0:5.1f}  analytic={dp}  max FD diff={np.max(np.abs(dp - dp_fd)):.2e}")

    s_true = -1.4
    target = model.apply(s_true)
    s_hat = fit_shape_factor(model, target)
    print(f"\ntrue s = {s_true:.3f}, recovered s = {s_hat[0]:.3f}")
