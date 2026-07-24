# `sandbox_ssm_shape_model.py`

A runnable sandbox exploring one question: **could osimfit fit a statistical
shape model (SSM) the same way it already fits per-body scale factors?** No
new theory here — it's a small, self-contained proof that the *interface*
between a shape model and the fitter can be narrow, backed by one real
landmark from a real 536-subject bone-shape PCA (not synthetic data).

Run it with:

```
python src/sandbox/sandbox_ssm_shape_model.py
```

No OpenSim model is involved — just `numpy` and `casadi` (already
dependencies of this repo).

---

## 1. The idea in one sentence

Today, `osimfit` morphs a model via `Scaler`/`BodyScale`
(`src/osimfit/scaling.py`): a vector of per-body, per-axis scale factors goes
in, a linearly-scaled `osim.Model` comes out, and `BilevelSolver`
(`src/osimfit/solvers.py`) optimizes those factors alongside kinematics.

A statistical shape model is a *different kind* of deformation — instead of
"stretch this body along X," it's "move along the top few modes of shape
variation seen across hundreds of real bones" — but the shape from the
fitter's point of view is identical: **a small parameter vector `s` goes in,
morphed geometry comes out.** If that's true, `s` should be able to sit
alongside `body_scales` in the same NLP decision vector, with no changes to
how the solver works. This file checks that claim on a single landmark before
anyone invests in wiring up the real thing.

## 2. Walkthrough, in the order the file executes

### `ShapeModel` (the contract)

An ABC with exactly what a shape model must provide:

| Member | Meaning |
|---|---|
| `n_params` | length of `s` |
| `nominal` | the `s` that reproduces the *stock* (unmorphed) geometry |
| `bounds` | plausible range for `s` (here, ±3 standard deviations) |
| `apply(s)` | **required** — morphed geometry at `s` |
| `geometry_jacobian(s)` | **required** — exact `d(apply)/ds` |
| `prior(s)` | **required** — `(value, grad)` of a regularizer, so the fitter can penalize implausible `s` the way it already penalizes extreme body scales |

`n_params`/`nominal`/`bounds` are abstract *properties*, not bare
annotations, so a `ShapeModel` subclass that forgets one fails at
instantiation rather than with a confusing `AttributeError` the first time
something reads it.

A shape model that hasn't been ported to this repo yet has a wider version of
this contract, where `geometry_jacobian` is *optional* and the fitter falls
back to finite differences when it's missing (see osimfit-ssm's
`CLEAN_PLAN.md` if you want the full reasoning — it recommends FD as the v1
default, precisely so a shape model with no derivative code still works).
This sandbox makes the narrower assumption on purpose: every shape model here
provides an exact analytic Jacobian, so `ShapeModelCallback` has exactly one
path and there's no FD branch to reason about. If a future shape model can't
supply one, that's the point where the contract would need to widen back out
to match CLEAN_PLAN.md.

### `FemurHeadShapeModel` (one real implementation)

Real data, deliberately shrunk to the simplest non-trivial case:

- **Where the data comes from:** a PCA over the combined femur+hip mesh of
  536 subjects. The full dataset has ~100k mesh vertices and ~7 shape modes;
  this sandbox keeps only the first mode (`pc1.npy`, `eigenvalue1.npy`) and
  only the ~1500 vertices belonging to the femoral head
  (`femur_head_points.csv`). `mean_shape.npy` is still the full mesh mean —
  it's small either way (2.4 MB) — but everything downstream indexes into
  just the head patch. Total added data: ~3.5 MB, vs. ~22 MB for the full
  7-mode version.
- **The morph:** `landmarks(s) = mean_head_points + s * basis`, where
  `basis = sqrt(eigenvalue_1) * pc_1` restricted to the head patch. This part
  is exactly linear in `s`.
- **`apply(s)`:** fits a sphere through those morphed points (least-squares /
  Kasa fit, in `_sphere_center`) and returns its center. This is the kind of
  "which anatomical feature does this shape factor move" question a real
  integration cares about — e.g., hip-socket congruency.
- **`geometry_jacobian(s)`:** `_sphere_center` differentiates its own
  closed-form solution w.r.t. `s`, so this is an *exact* analytic Jacobian —
  not a finite-difference approximation.
- **Important subtlety:** the landmark morph is linear in `s`, but the sphere
  fit built on top of those landmarks is *not*. So `geometry_jacobian(s)` is
  exact but genuinely depends on `s` — it is not a constant matrix you could
  compute once and reuse. Don't assume "linear shape model" implies "constant
  Jacobian." The `__main__` block below prints `geometry_jacobian` at three
  different `s` values so this is visible rather than assumed.

### `ShapeModelCallback` (the CasADi bridge)

This is the piece that would actually let `s` join a CasADi NLP. It wraps
*any* `ShapeModel` (not just the femur-head one) as a `casadi.Callback`:

- `eval` calls `shape_model.apply(s)`.
- `get_jacobian` calls `shape_model.geometry_jacobian(s)` directly. No
  fallback branch — every `ShapeModel` is required to provide one.

`ShapePriorCallback` does the same wrapping for `prior(s)`: `eval` returns
`prior(s)`'s value, `get_jacobian` returns its analytic gradient. This
matters because `fit_shape_factor` uses the *callback*, not a hand-written
`s**2` CasADi expression, to build the regularizer term — so the "narrow
interface, wired through CasADi" claim actually gets exercised for the
prior too, not just for the geometry. A shape model with a non-quadratic
prior would still work here unmodified.

This mirrors `PositionCallback` / `PositionCallback_Jac` in
`sandbox_jacobians.py` almost exactly — that pair threads a *generalized
coordinate* `q` through CasADi with an analytic Jacobian from OpenSim's
station-Jacobian machinery. `ShapeModelCallback` threads a *shape factor* `s`
through CasADi the same way, just generic over `n_params` and output size
instead of hardcoded to one body's position.

### `check_jacobian_fd` and `fit_shape_factor`

- `check_jacobian_fd` is the sanity check you'd run on any new
  `geometry_jacobian` before trusting it: compare against central
  differences. `__main__` runs it at `s0 = -2, 0, 2` and prints the max
  difference (all around `1e-6`, i.e. FD-noise-level agreement).
- `fit_shape_factor` is the smallest possible version of a "bilevel-style"
  fit: given a target femoral-head position, recover the `s` that produces
  it, by minimizing `||apply(s) - target||^2 + prior_weight * prior(s)` with
  `ca.nlpsol(..., "ipopt", ...)` — the same solver call
  `src/osimfit/solvers.py` already uses for the real fitter. (One detail
  worth knowing if you extend this: `ipopt.hessian_approximation =
  "limited-memory"` is set because our callback supplies a Jacobian but not a
  Hessian — same reason `solvers.py` sets it.)

Running the file recovers a known `s = -1.4` to solver tolerance, confirming
the whole chain — morph, sphere fit, analytic Jacobian, CasADi callback,
ipopt solve — is self-consistent on real anatomical data.

## 3. How this maps onto what's already in `osimfit`

| Already built (this repo) | This sandbox | Relationship |
|---|---|---|
| `Scaler` / `BodyScale` ABC (`scaling.py`) | `ShapeModel` | Both are "parameter vector → morphed model." `BodyScale` is per-body-axis linear scaling; `ShapeModel` is a PCA-based deformation. Same slot, different deformation family. |
| `BilevelSolver.body_scales`, `BodyScaleGroup` (`solvers.py`) | `FemurHeadShapeModel.n_params` / `.nominal` / `.bounds` | Where `s` would eventually join the NLP decision vector, alongside (not instead of) body scales and spline control points, the same way `body_scales` is added today. |
| `Function`, `MarkerTrackingCost` (`callbacks.py`) | `ShapeModelCallback`, `ShapePriorCallback` | The general "wrap an OpenSim-dependent computation as a CasADi `Callback` with an analytic Jacobian" pattern already used for marker tracking cost. |
| `PositionCallback` / `PositionCallback_Jac` (`sandbox_jacobians.py`) | `ShapeModelCallback` | Nearly the same class shape, applied to a shape factor `s` instead of a coordinate `q`. Read that file first if this one is unclear — it's the same pattern with OpenSim's real station Jacobian instead of a closed-form sphere fit. |

**Naming trap:** `solvers.py` defines its *own* `BodyScale` — a small
`@dataclass` pairing a `BodyScaleGroup` with `Bounds`, used internally by
`BilevelSolver` — which is unrelated to `scaling.py`'s `BodyScale` ABC in the
first table row above (`solvers.py` doesn't even import it; it imports
`ManualBodyScale` instead). `BilevelSolver.body_scales` in the second row is
a list of `solvers.py`'s `BodyScale`, not `scaling.py`'s. Two different
classes, same name, in two different files — don't assume the table row
above tells you what's inside `BilevelSolver.body_scales`.

## 4. What this is *not*

This sandbox does not morph an actual `.osim` model, does not touch mesh
geometry beyond one landmark patch, and does not implement the full 7-mode
femur+hip shape model. That fuller version — whole-mesh morphing, per-joint
offset frames derived from multiple landmarks, mirroring, mass/inertia
scaling — exists in a separate repo (`osimfit-ssm`) as real, working code; it
was kept out of here on purpose so this file could stay a single,
readable script with a small (~3.5 MB) real-data footprint instead of
vendoring ~22 MB of PCA data or a second package dependency into
`opensim-fitter`. If the interface proven here holds up, that's the next
thing to port over.
