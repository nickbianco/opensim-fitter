import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .bounds import Bounds
from .data_sources import Trial
from .costs import (BilevelCost, BilevelCostRep, Cost, CostInput, CostRep,
                    TrackingCost)
from .model import (ModelCache, Parameter, BodyScale, MarkerOffset, FrameOffset,
                    EllipsoidRadii, BeamLength)
from .scaling import Axis, Scaler, ManualBodyScale


############
# SOLUTION #
############

@dataclass
class Solution:
    """
    The result of a solve, returned by every solver.

    Every solver optimizes over one or more `Trial` objects, so the kinematic result is
    always a states table per trial, keyed by trial name. Parameters coupled across
    trials (e.g., body scales, placement offsets) are shared and so live in a single
    flat list. Solvers can also provide additional solution information via `outputs`.

    Attributes
    ----------
    states_tables: dict[str, osim.TimeSeriesTable]
        The table of optimized model states for each trial, keyed by trial name.
    parameters: list[Parameter], optional
        The optimized parameters (e.g., body scales, marker and frame offsets), each
        carrying its optimal ``value``, shared across every trial. ``None`` for solvers
        that optimize no parameters. This is an independent snapshot of the solver's
        parameter configuration; the same list (with values set) can be handed back to
        ``solve()`` as an initial guess.
    outputs: dict[str, Any]
        Solver-specific results that are neither states tables nor parameters, keyed by
        output name. Per-trial results nest a dict keyed by trial name under the output
        name, e.g. ``outputs['spline_nodes']['walk_1']``.
    """
    states_tables: dict[str, osim.TimeSeriesTable] = field(default_factory=dict)
    parameters: list[Parameter] = None
    outputs: dict[str, Any] = field(default_factory=dict)

    def get_parameter(self, path: str, cls: type = Parameter) -> Parameter:
        """
        Return the optimized parameter of type `cls` whose group contains `path`.

        Parameters
        ----------
        path: str
            Absolute model path of a component in the target parameter's group.
        cls: type, optional
            Restrict the search to parameters of this `Parameter` subtype (e.g.,
            `BodyScale`, `MarkerOffset`, or `FrameOffset`). Defaults to `Parameter`
            (any type).

        Raises
        ------
        KeyError
            If not exactly one parameter of type `cls` has `path` in its group.
        """
        matches = [p for p in (self.parameters or [])
                   if isinstance(p, cls) and path in p.paths]
        if len(matches) != 1:
            raise KeyError(
                f'Expected exactly one {cls.__name__} whose group contains {path}, '
                f'but found {len(matches)}.')
        return matches[0]

    @staticmethod
    def create_states_table(model, state, coordinate_indexes, times,
                            q_opt, qdot_opt=None) -> osim.TimeSeriesTable:
        """
        Build an OpenSim StatesTrajectory and export it to a TimeSeriesTable.

        Parameters
        ----------
        model: osim.Model
        state: osim.State
            An initialized state that will be mutated in place during construction.
        coordinate_indexes: list[int]
            Indexes of the independent coordinates in the full state vector.
        times: sequence of float
        q_opt: np.ndarray, shape (num_times, num_coords)
        qdot_opt: np.ndarray, shape (num_times, num_coords), optional
        """
        statesTraj = osim.StatesTrajectory()
        for i, time in enumerate(times):
            state.setTime(time)
            q = np.zeros(state.getNQ())
            q[coordinate_indexes] = q_opt[i, :]
            state.setQ(osim.Vector.createFromMat(q))
            if qdot_opt is not None:
                # The spline differentiates the coordinates, giving qdot, while the
                # State stores generalized speeds u. Since qdot = N(q) u, convert with
                # NInv rather than assigning qdot directly; the two coincide only where
                # qdot == u, which is not the case for e.g. an EllipsoidJoint.
                qdot = np.zeros(state.getNQ())
                qdot[coordinate_indexes] = qdot_opt[i, :]
                model.realizePosition(state)
                u = osim.Vector(state.getNU(), 0.0)
                model.multiplyByNInv(
                    state, False, osim.Vector.createFromMat(qdot), u)
                state.setU(u)
            statesTraj.append(state)
        return statesTraj.exportToTable(model)


###########
# SOLVERS #
###########

class Solver(ABC):
    """
    An abstract base class for CasADi-based solvers that leverage computations from
    OpenSim models. Subclasses must implement the solve() method, which should return
    a Solution object containing the solution trajectory. This base class also
    provides common functionality for building IPOPT options and managing the OpenSim
    model and state.

    Reference data is organized into `Trial` objects, one per motion, each bundling the
    data sources collected together for that motion. A trial can be registered with
    `add_trial`. Each concrete solver determines how to optimize over the registered
    (e.g., sequentially, simultaneously, etc.).

    Parameters
    ----------
    model: str or osim.Model
        The OpenSim model to use for the optimization problem. Can be provided as a file
        path or as an already-loaded osim.Model object.
    convergence_tolerance: float, optional
        The convergence tolerance to use for the IPOPT solver. Default is 1e-4.
    """
    # Concrete subclasses override to define the set of CostInput fields supported by
    # the solver.
    SUPPORTED_INPUTS: frozenset[str] = frozenset()

    def __init__(self, model: str | osim.Model, convergence_tolerance: float=1e-4):
        super().__init__()

        # Remove muscles and create the ModelCache.
        modelProcessor = osim.ModelProcessor(model)
        modelProcessor.append(osim.ModOpRemoveMuscles())
        self.mc = ModelCache(modelProcessor.process())
        self.state = self.mc.state

        # Convenience aliases for the cached coordinate maps.
        self.coordinate_map = self.mc.coordinate_map
        self.coordinate_indexes = self.mc.coordinate_indexes

        # Optimization settings.
        self.convergence_tolerance = convergence_tolerance

        # Additional user-registered costs (e.g., regularization).
        self.costs: list[Cost] = []

        # Reference data, one Trial per motion.
        self.trials: list[Trial] = []

    def add_trial(self, trial: Trial):
        """
        Register a `Trial` containing the reference data for a single motion trial.

        Parameters
        ----------
        trial: Trial
            The trial object. Must carry at least one data source, and its name must be
            unique among the trials already registered.

        Raises
        ------
        ValueError
            If `trial` is not a `Trial`, carries no reference data, or reuses the name
            of an already-registered trial.
        """
        if not isinstance(trial, Trial):
            raise ValueError(
                f'add_trial expected a Trial, but got {type(trial).__name__}.')
        if not trial.marker_data and not trial.frame_data:
            raise ValueError(
                f"Trial '{trial.name}' has no reference data; add a data source before "
                f'registering it with a solver.')
        if any(trial.name == registered.name for registered in self.trials):
            raise ValueError(
                f"A trial named '{trial.name}' is already registered with this solver; "
                f'trial names must be unique.')

        self.trials.append(trial)

    def _assert_has_trials(self):
        """
        Verify that at least one `Trial` has been registered. Called at the start of
        solve(), since there is nothing to fit without reference data.
        """
        if not self.trials:
            raise ValueError(
                f'{type(self).__name__} has no reference data; register at least one '
                f'Trial via add_trial() before calling solve().')

    def add_cost(self, cost: Cost):
        """
        Register an additional `Cost` to include in the solver's objective. The cost is
        evaluated by calling it with a `CostInput`.

        Raises
        ------
        TypeError
            If `cost` is not a `Cost`; a `TrackingCostBase` in particular is built by a
            solver internally and cannot be registered here.
        ValueError
            If the cost depends on a `CostInput` field this solver does not provide.
        """
        if not isinstance(cost, Cost):
            raise TypeError(
                f'{type(self).__name__}.add_cost expects a Cost, but got '
                f'{type(cost).__name__}.')
        unsupported = cost.required_inputs - self.SUPPORTED_INPUTS
        if unsupported:
            raise ValueError(
                f'{type(self).__name__} does not support {type(cost).__name__}: it '
                f'requires cost input(s) {sorted(unsupported)} that this solver does '
                f'not provide (supported: {sorted(self.SUPPORTED_INPUTS)}).')
        self.costs.append(cost)

    def get_ipopt_options(self, print_level=0):
        """
        Get a dictionary of common IPOPT options for use with CasADi's nlpsolver.
        """
        ipopt_options = {}
        ipopt_options['print_level'] = print_level

        # We only support callback functions with first-order derivatives.
        ipopt_options['hessian_approximation'] = 'limited-memory'
        ipopt_options['limited_memory_max_history'] = 50
        ipopt_options['mu_strategy'] = 'adaptive'

        ipopt_options['tol'] = self.convergence_tolerance
        # ipopt_options['dual_inf_tol'] = self.convergence_tolerance
        # ipopt_options['compl_inf_tol'] = self.convergence_tolerance
        # ipopt_options['acceptable_tol'] = self.convergence_tolerance
        # ipopt_options['acceptable_dual_inf_tol'] = self.convergence_tolerance
        # ipopt_options['acceptable_compl_inf_tol'] = self.convergence_tolerance

        # Avoids crashes in CasADi for larger problems.
        ipopt_options['mumps_pivot_order'] = 0

        return ipopt_options

    def _validate_guess(self, guess: Solution):
        """
        Validate that `guess` is a `Solution` providing a usable states table for each
        registered trial. Subclasses may override to add solver-specific checks.
        """
        if not isinstance(guess, Solution):
            raise TypeError(
                f'{type(self).__name__} expected an initial guess of type Solution, '
                f'but got {type(guess).__name__}.')

        # The guess must cover exactly the registered trials. Each trial's guess is
        # looked up by name, so the order of the dict is irrelevant.
        expected = {trial.name for trial in self.trials}
        got = set(guess.states_tables)
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        if missing or extra:
            raise ValueError(
                f'Initial guess trials do not match the solver configuration. Missing '
                f'guesses for {missing}; unexpected guesses for {extra}.')

        for name, table in guess.states_tables.items():
            self._validate_guess_states_table(name, table)

    def _validate_guess_states_table(self, name: str,
                                     table: osim.TimeSeriesTable):
        """
        Validate that a trial's guess states table is non-empty and provides a column
        for every independent coordinate.
        """
        if table is None or table.getNumRows() == 0 or table.getNumColumns() == 0:
            shape = ('None' if table is None else
                     f'{table.getNumRows()} rows, {table.getNumColumns()} columns')
            raise ValueError(
                f"Initial guess states table for trial '{name}' is empty ({shape}).")

        labels = set(table.getColumnLabels())
        missing = [coord_path + '/value' for coord_path in self.coordinate_map
                   if coord_path + '/value' not in labels]
        if missing:
            raise ValueError(
                f"Initial guess states table for trial '{name}' is missing required "
                f'coordinate columns: {missing}.')

    def extract_guess_coordinates(self, table: osim.TimeSeriesTable) -> np.ndarray:
        """
        Read a guess states table into a ``(num_rows, num_coords)`` array of coordinate
        values, with columns ordered to match this solver's `coordinate_map`.
        """
        return np.column_stack([
            table.getDependentColumn(coord_path + '/value').to_numpy()
            for coord_path in self.coordinate_map])

    @staticmethod
    def compute_average_trapezoidal_error(errors, times):
        """
        Time-averaged error computed from a per-timestep symbolic error vector using the
        trapezoidal rule:

            cost = (1 / (t_{N-1} - t_0))
                   * sum_{i=0}^{N-2} 0.5 * (t_{i+1} - t_i) * (e_i + e_{i+1})

        Compared to a simple mean (``ca.sum(errors) / num_times``), this is an
        exact time average for piecewise-linear ``errors`` and handles
        non-uniform time spacing correctly. Dividing by the total duration
        keeps the cost in the same units as the per-timestep error so weights
        on companion cost terms (e.g., body-scale regularization) need not be
        retuned when switching averaging schemes.

        Parameters
        ----------
        errors: ca.MX, shape (num_times, 1)
            Symbolic per-timestep errors.
        times: array-like of float, length num_times
            Strictly increasing time vector associated with `errors`.

        Returns
        -------
        ca.MX
            Scalar time-averaged error expression.
        """
        times = np.asarray(times, dtype=float)
        dt = np.diff(times)
        weights = np.zeros(len(times))
        weights[:-1] += 0.5 * dt
        weights[1:]  += 0.5 * dt
        duration = times[-1] - times[0]
        return ca.dot(ca.DM(weights), errors) / duration

    @abstractmethod
    def solve(self, guess=None) -> Solution:
        pass


class TrackingSolver(Solver):
    """
    An abstract base class for solvers that track reference data. Data within a trial
    can be position-based (e.g., marker trajectories) or orientation-based (e.g., Theia
    frames). Concrete subclasses must implement the solve() method, which should return
    one states table per registered trial.

    Parameters
    ----------
    model: str or osim.Model
        See `Solver`.
    convergence_tolerance: float, optional
        See `Solver`.
    position_weight: float, optional
        The weight to use for position-based tracking costs. Default is 1.0.
    orientation_weight: float, optional
        The weight to use for orientation-based tracking costs. Default is 1.0.
    """
    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0):
        super().__init__(model, convergence_tolerance)

        # Cost function weights.
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

    def _validate_guess(self, guess: Solution):
        super()._validate_guess(guess)
        for trial in self.trials:
            num_rows = guess.states_tables[trial.name].getNumRows()
            if num_rows != trial.num_times:
                raise ValueError(
                    f"Initial guess states table for trial '{trial.name}' has "
                    f"{num_rows} rows but that trial's reference data has "
                    f'{trial.num_times} time samples.')


##############################
# INVERSE KINEMATICS SOLVERS #
##############################


class InverseKinematicsSolver(TrackingSolver):
    """
    Solve the inverse kinematics problem to find the set of model coordinate values that
    best track provided position (e.g., marker trajectories) and/or orientation (e.g.,
    frame orientations) data. Trials are solved sequentially.

    Parameters
    ----------
    model: str or osim.Model
        See `Solver`.
    convergence_tolerance: float, optional
        See `Solver`.
    position_weight: float, optional
        See `TrackingSolver`.
    orientation_weight: float, optional
        See `TrackingSolver`.
    """

    SUPPORTED_INPUTS = frozenset({'coordinates'})

    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0):
        super().__init__(model, convergence_tolerance, position_weight,
                         orientation_weight)

    def create_tracking_solver(self, trial: Trial, itime: int,
                               tracking_cost: TrackingCost, cost_reps: list[CostRep]):
        """
        A helper function to create a CasADi solver for the tracking problem at a
        given time step of a given trial.

        Parameters
        ----------
        trial: Trial
            The trial supplying the reference data.
        itime: int
            Index of the time sample within `trial` to track.
        tracking_cost: TrackingCost
            The tracking cost description, built once per solve.
        cost_reps: list[CostRep]
            Reps of the solver's registered costs, built once per solve.
        """
        x = ca.SX.sym('x', len(self.coordinate_indexes))
        tracking_rep = tracking_cost.create_rep('tracking_cost', self.mc, trial, itime)
        cost_input = CostInput(coordinates=x)
        f = tracking_rep(cost_input)
        for cost_rep in cost_reps:
            f += cost_rep(cost_input)
        nlp = {'x': x, 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options()
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        return tracking_rep, solver

    def solve(self, guess: Solution = None) -> Solution:
        self._assert_has_trials()

        if guess is not None:
            self._validate_guess(guess)

        # Per-coordinate bounds, plus the default initial values used to seed the first
        # time step of each trial when no guess is supplied.
        default_x0 = []
        lbx = []
        ubx = []
        for coord_path in self.coordinate_map:
            coord = osim.Coordinate.safeDownCast(self.mc.model.getComponent(coord_path))
            default_x0.append(coord.getDefaultValue())
            lbx.append(coord.getRangeMin())
            ubx.append(coord.getRangeMax())

        # Solve each trial sequentially, restarting the warm start from the default
        # coordinate values (or that trial's guess) at each trial's first time step.
        cost_reps = [cost.create_rep(self.mc) for cost in self.costs]
        tracking_cost = TrackingCost(self.position_weight, self.orientation_weight)
        states_tables: dict[str, osim.TimeSeriesTable] = {}
        for trial in self.trials:
            times = trial.times
            num_times = len(times)

            # When a guess is provided, pre-extract a (num_times, num_coords) array of
            # initial values from this trial's guess states_table so each timestep can
            # be seeded from the corresponding row.
            guess_q = None
            if guess is not None:
                guess_q = self.extract_guess_coordinates(
                    guess.states_tables[trial.name])

            # Iterate over all of the time steps in the tracking data and solve the
            # optimization problem at each time step.
            x0 = list(default_x0)
            statesTraj = osim.StatesTrajectory()
            q_traj = np.zeros((num_times, len(self.coordinate_indexes)))
            for itime, time in enumerate(times):
                print(f"Trial '{trial.name}': solving time {itime+1} of {num_times} "
                      f'(t={time:.3f} s)...')

                if guess_q is not None:
                    x0 = guess_q[itime, :].tolist()

                tracking_rep, solver = self.create_tracking_solver(
                    trial, itime, tracking_cost, cost_reps)
                sol = solver(x0=x0, lbx=lbx, ubx=ubx)

                q_traj[itime, :] = np.squeeze(sol['x'].full())

                # Write solution into the rep's state.
                tracking_rep.state.setTime(time)
                q = np.zeros(tracking_rep.state.getNQ())
                q[self.coordinate_indexes] = q_traj[itime, :]
                tracking_rep.state.setQ(osim.Vector.createFromMat(q))
                statesTraj.append(tracking_rep.state)

                if guess_q is None:
                    x0 = sol['x']

            states_tables[trial.name] = statesTraj.exportToTable(self.mc.model)

        return Solution(states_tables=states_tables)


#############################
# SPLINED KINEMATICS SOLVER #
#############################

class SplinedKinematicsSolver(TrackingSolver):
    """
    Solve for model kinematics by representing each coordinate trajectory as a B-spline
    and optimizing over the spline control points across the whole trial. Because the
    control points couple every time step, this solver can optionally also optimize
    global bilevel parameters (e.g., body scales, marker and frame offsets); register
    them with ``add_parameter``. With no parameters registered, it reduces to a
    spline-based inverse kinematics problem tracking the reference data.

    All registered trials are optimized simultaneously. Each trial
    contributes its own block of spline control points, with its own knot vector sized
    from its own duration, while the bilevel parameters are shared across every trial.
    The tracking objective uses the mean over trials of each trial's time-averaged
    error. A registered `Cost` that depends only on the shared parameters is evaluated
    once, not per trial; one that reads the coordinates (e.g. `CoordinateStiffnessCost`)
    is evaluated at every time sample and time-averaged along with the tracking error.

    Parameters
    ----------
    model: str or osim.Model
        See `Solver`.
    convergence_tolerance: float, optional
        See `Solver`.
    position_weight: float, optional
        See `TrackingSolver`.
    orientation_weight: float, optional
        See `TrackingSolver`.
    degree: int, optional
        The degree of the B-spline basis functions. Default is 3 (i.e., cubic splines).
    knot_interval: float, optional
        The interval between knots in the B-spline basis. Default is 0.05 seconds. Every
        registered trial must span at least ``degree + 1`` knot intervals.
    """
    SUPPORTED_INPUTS = frozenset({'coordinates', 'body_scales', 'marker_offsets',
                                 'frame_offsets', 'ellipsoid_radii', 'beam_lengths'})

    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0, degree=3, knot_interval=0.05):
        super().__init__(model, convergence_tolerance=convergence_tolerance,
                         position_weight=position_weight,
                         orientation_weight=orientation_weight)
        self._parameters_by_input: dict[str, list[Parameter]] = {}
        self.degree = degree
        self.knot_interval = knot_interval

    def build_knots_vector(self, times, num_knots):
        """
        Create a clamped knot vector. For n control points and degree p, there are
        n+p+1 knots. The first and last p+1 knots are clamped to the first and last time,
        respectively, and the interior knots are uniformly spaced between the first
        and last time.
        """
        knots = np.concatenate([
            np.repeat(times[0], self.degree),
            np.linspace(times[0], times[-1], num_knots - self.degree + 1),
            np.repeat(times[-1], self.degree),
        ])
        return knots

    def build_spline_basis_matrix(self, times, knots):
        """
        Build the spline basis matrix B and its derivative dB. B[i,j] = N_j(t_i),
        where N_j is the j-th B-spline basis function evaluated at time t_i.
        """

        # Build basis matrix B[i,j] = N_j(t_i) numerically.
        t = ca.MX.sym("t")
        num_knots = len(knots) - self.degree - 1

        # Scalar spline function for building B matrix.
        c_temp = ca.MX.sym("c_temp", num_knots, 1)
        spline = ca.bspline(t, c_temp, [knots], [self.degree], 1)
        spline_fn = ca.Function("spline", [t, c_temp], [spline])

        # Derivative of the spline w.r.t. time.
        spline_dt = ca.jacobian(spline, t)
        spline_fn_dt = ca.Function("spline_dt", [t, c_temp], [spline_dt])

        # Build basis matrix B[i,j] = N_j(t_i) by evaluating with unit coefficient
        # vectors.
        B = np.zeros((len(times), num_knots))
        dB = np.zeros((len(times), num_knots))
        for j in range(num_knots):
            e_j = np.zeros(num_knots)
            e_j[j] = 1.0
            B[:, j] = [float(spline_fn(ti, e_j)) for ti in times]
            dB[:, j] = [float(spline_fn_dt(ti, e_j)) for ti in times]

        return ca.DM(B), ca.DM(dB)

    def extract_coordinate_initial_guess(self, states_table, B, coord_path):
        """Extract an initial guess for the spline control points for a given coordinate
          by solving a least squares problem.
        """
        q_col = states_table.getDependentColumn(coord_path + '/value').to_numpy()
        q_guess, _, _, _ = np.linalg.lstsq(np.array(B), q_col, rcond=None)
        return q_guess.tolist()

    @property
    def parameters(self) -> list[Parameter]:
        """
        All registered parameters, flattened in `CostInput.INPUT_ORDER` (the order in
        which their variable blocks are concatenated into the optimization vector).
        Within an input, registration order is preserved.
        """
        return [p for name in CostInput.INPUT_ORDER
                for p in self._parameters_by_input.get(name, [])]

    @property
    def body_scales(self) -> list[BodyScale]:
        """
        The registered `BodyScale` parameters, in registration order.
        """
        return list(self._parameters_by_input.get(BodyScale.cost_input, []))

    @property
    def marker_offsets(self) -> list[MarkerOffset]:
        """
        The registered `MarkerOffset` parameters, in registration order.
        """
        return list(self._parameters_by_input.get(MarkerOffset.cost_input, []))

    @property
    def frame_offsets(self) -> list[FrameOffset]:
        """
        The registered `FrameOffset` parameters, in registration order.
        """
        return list(self._parameters_by_input.get(FrameOffset.cost_input, []))

    @property
    def ellipsoid_radii(self) -> list[EllipsoidRadii]:
        """
        The registered `EllipsoidRadii` parameters, in registration order.
        """
        return list(self._parameters_by_input.get(EllipsoidRadii.cost_input, []))

    @property
    def beam_lengths(self) -> list[BeamLength]:
        """
        The registered `BeamLength` parameters, in registration order.
        """
        return list(self._parameters_by_input.get(BeamLength.cost_input, []))

    def add_parameter(self, parameter: Parameter):
        """
        Register a `Parameter` to be optimized over in the bilevel optimization problem.
        The parameter is validated against the model at registration time.

        Parameters
        ----------
        parameter: Parameter
            The parameter to optimize (e.g., a `BodyScale`).

        Raises
        ------
        ValueError
            If `parameter` is not a `Parameter`, or its `cost_input` is not a recognized
            `CostInput` field.
        """
        if not isinstance(parameter, Parameter):
            raise ValueError(
                f'add_parameter expected a Parameter, but got '
                f'{type(parameter).__name__}.')
        CostInput.field_index(parameter.cost_input)
        parameter.validate(self.mc)
        self._parameters_by_input.setdefault(parameter.cost_input, []).append(parameter)
        self.mc.add_parameter_group(parameter.to_group())

    def assert_offset_groups_used(self, reps: list[BilevelCostRep]):
        """
        Verify that every registered offset group is tracked by at least one task in at
        least one of `reps`, ensuring that the offset is properly constrained in
        the bilevel optimization problem.

        Raises
        ------
        ValueError
            If any registered marker or frame offset group is tracked by no task.
        """
        def assert_used(used, offset_groups, label):
            for i, group in enumerate(offset_groups):
                if i not in used:
                    raise ValueError(
                        f'{label.capitalize()} offset group {group.component_paths} is '
                        f'not tracked by any registered {label} in any trial; its '
                        f'offset would be unconstrained.')

        used_markers = {g for rep in reps
                        for g in rep.marker_term.offset_group_indexes
                        if g is not None}
        used_frames = {g for rep in reps
                       for g in rep.frame_term.offset_group_indexes
                       if g is not None}
        assert_used(used_markers, self.mc.marker_offset_groups, 'marker')
        assert_used(used_frames, self.mc.frame_offset_groups, 'frame')

    def update_model(self, model: osim.Model, solution: Solution) -> osim.Model:
        """
        Apply the solution's optimized parameters to `model` and return it.
        """
        model.initSystem()

        # Get pre-`Model::scale()` quanities.
        translation_scales = ModelCache.get_custom_joint_translation_scales(model)
        ellipsoid_radii = ModelCache.get_ellipsoid_joint_radii(model)

        # Construct a scaler using the optimized body scales as manual scale factors.
        # This calls Model::scale() under the hood. Body scales are applied via the
        # Scaler rather than each BodyScale's apply_to_model (see
        # BodyScale.apply_to_model).
        scaler = Scaler(model)
        axes = (Axis.XAxis, Axis.YAxis, Axis.ZAxis)
        for parameter in solution.parameters:
            if not isinstance(parameter, BodyScale):
                continue
            for body_path in parameter.paths:
                body_name = osim.Body.safeDownCast(
                    model.getComponent(body_path)).getName()
                for ax_idx, axis in enumerate(axes):
                    scaler.add_body_scale(ManualBodyScale(
                        body_name, axis, float(parameter.value[ax_idx])))
        model = scaler.scale()

        # Apply pre-`Model::scale()` quanities. Restoring the ellipsoid radii undoes
        # `EllipsoidJoint::extendScale`, which multiplies them by the parent frame's
        # body scale factors: this solver treats a joint's mobilizer geometry as
        # independent of the body scales, so an `EllipsoidRadii` parameter is the only
        # thing that may change it.
        ModelCache.apply_custom_joint_translation_scales(model, translation_scales)
        ModelCache.apply_ellipsoid_joint_radii(model, ellipsoid_radii)

        # Apply the remaining optimized parameters (e.g., marker and frame offsets) to
        # the scaled model.
        for parameter in solution.parameters:
            if not isinstance(parameter, BodyScale):
                parameter.apply_to_model(model)

        # Finalize the system and return.
        model.finalizeConnections()
        model.initSystem()
        return model

    def _validate_guess(self, guess: Solution):
        super()._validate_guess(guess)
        if not self.parameters:
            return

        # Check that the solver parameters and guess parameters are the same size.
        expected = self.parameters
        got = guess.parameters or []
        if len(got) != len(expected):
            raise ValueError(
                f'Initial guess has {len(got)} parameter(s) but the solver is '
                f'configured with {len(expected)}.')

        # Enforce that the guess lists its parameters grouped in CostInput.INPUT_ORDER.
        positions = [CostInput.field_index(g.cost_input) for g in got]
        if positions != sorted(positions):
            raise ValueError(
                f'Initial guess parameters must be ordered by CostInput.INPUT_ORDER '
                f'{CostInput.INPUT_ORDER}, but got {[g.cost_input for g in got]}.')

        for e, g in zip(expected, got):
            if type(g) is not type(e) or g.paths != e.paths:
                raise ValueError(
                    f'Initial guess parameters do not match the solver configuration. '
                    f'Expected {type(e).__name__} on {e.paths}, got '
                    f'{type(g).__name__} on {g.paths}.')
            if g.value is None or np.asarray(g.value).shape != (e.num_variables,):
                shape = None if g.value is None else np.asarray(g.value).shape
                raise ValueError(
                    f'Initial guess value for {type(e).__name__} on {e.paths} must '
                    f'have shape ({e.num_variables},), got {shape}.')


    def solve(self, guess: Solution = None) -> Solution:
        self._assert_has_trials()

        if guess is not None:
            self._validate_guess(guess)

        num_coords = len(self.coordinate_indexes)

        # Build a spline basis per trial.
        trial_times: list[list[float]] = []
        trial_num_knots: list[int] = []
        trial_B: list[ca.DM] = []
        trial_dB: list[ca.DM] = []
        for trial in self.trials:
            times = trial.times
            duration = times[-1] - times[0]

            num_knots = int(round(duration / self.knot_interval))
            if num_knots < self.degree + 1:
                minimum_duration = (self.degree + 1) * self.knot_interval
                raise ValueError(
                    f"Trial '{trial.name}' spans {duration:.4g} s, which yields only "
                    f"{num_knots} B-spline control point(s) at a knot interval of "
                    f"{self.knot_interval:.4g} s; a degree-{self.degree} spline "
                    f"requires at least {self.degree + 1}. Either provide a trial "
                    f"spanning at least {minimum_duration:.4g} s, or reduce "
                    f"knot_interval or degree.")

            knots = self.build_knots_vector(times, num_knots)
            B, dB = self.build_spline_basis_matrix(times, knots)
            trial_times.append(times)
            trial_num_knots.append(num_knots)
            trial_B.append(B)
            trial_dB.append(dB)

        # Extract parameter dimensions.
        num_params = len(self.parameters)

        # Apply the parameters from the initial guess to the solver's list of registered
        # parameters.
        if num_params > 0 and guess is not None and guess.parameters is not None:
            for sp, gp in zip(self.parameters, guess.parameters):
                sp.value = np.asarray(gp.value, dtype=float)

        # Define the optimization variables: one block of control points per trial,
        # followed by the parameter blocks shared across all trials.
        # One symbolic block per parameter type, sized from the groups registered on
        # the ModelCache and ordered by CostInput.INPUT_ORDER, matching the layout of
        # the optimization vector assembled below.
        coeffs = [ca.MX.sym(f'coeffs_{itrial}', num_knots, num_coords)
                  for itrial, num_knots in enumerate(trial_num_knots)]
        blocks = {name: ca.MX.sym(name, self.mc.parameter_groups[name].num_variables)
                  for name in CostInput.PARAMETER_INPUTS}
        x0 = []
        lbx = []
        ubx = []
        for itrial, trial in enumerate(self.trials):
            num_knots = trial_num_knots[itrial]
            guess_table = (None if guess is None else guess.states_tables[trial.name])
            for coord_path in self.coordinate_map:
                coord = osim.Coordinate.safeDownCast(
                    self.mc.model.getComponent(coord_path))
                x0 += ([coord.getDefaultValue()] * num_knots if guess_table is None
                       else self.extract_coordinate_initial_guess(
                           guess_table, trial_B[itrial], coord_path))
                lbx += [coord.getRangeMin()] * num_knots
                ubx += [coord.getRangeMax()] * num_knots

        # Append each parameter's initial guess and bounds, in type order, matching the
        # [coeffs_0, ..., coeffs_J, s, mo, fo] layout of the optimization vector below.
        for p in self.parameters:
            p.append_guess_and_bounds(x0, lbx, ubx)

        # Accumulate the tracking cost for each trial. Reps are held in a list for the
        # lifetime of the solve so CasADi's references to them stay valid.
        #
        # Registered costs are split by whether they read the coordinates. A
        # coordinate-dependent cost has a different value at every time sample, so it
        # is accumulated with the tracking error and receives the same time averaging;
        # a cost over the shared parameters alone is evaluated once, after the loop.
        coordinate_cost_reps = [
            cost.create_rep(self.mc) for cost in self.costs
            if 'coordinates' in cost.required_inputs]
        parameter_cost_reps = [
            cost.create_rep(self.mc) for cost in self.costs
            if 'coordinates' not in cost.required_inputs]

        f = 0
        tracking_reps = []
        cost_type = BilevelCost if num_params > 0 else TrackingCost
        tracking_cost = cost_type(self.position_weight, self.orientation_weight)
        for itrial, trial in enumerate(self.trials):
            times = trial_times[itrial]
            num_times = len(times)

            # Map this trial's control points to its full predicted trajectory via its
            # spline basis matrix.
            q = trial_B[itrial] @ coeffs[itrial]

            # Compute the tracking cost at each time step via a callback rep.
            errors = ca.MX(num_times, 1)
            for itime in range(num_times):
                tracking_rep = tracking_cost.create_rep(
                    f'tracking_cost_trial_{itrial}_time_{itime}', self.mc, trial,
                    itime)
                tracking_reps.append(tracking_rep)
                cost_input = CostInput(coordinates=q[itime, :].T, **blocks)
                error = tracking_rep(cost_input)
                for cost_rep in coordinate_cost_reps:
                    error += cost_rep(cost_input)
                errors[itime] = error

            f += self.compute_average_trapezoidal_error(errors, times)

        # Average across trials.
        f /= len(self.trials)

        # Every offset group must be tracked in at least one trial.
        if num_params > 0:
            self.assert_offset_groups_used(tracking_reps)

        # Add the cost terms on the parameters shared across all trials.
        parameter_input = CostInput(**blocks)
        for cost_rep in parameter_cost_reps:
            f += cost_rep(parameter_input)

        # Solve.
        x = ca.vertcat(*[ca.vec(c) for c in coeffs],
                       *[blocks[name] for name in CostInput.PARAMETER_INPUTS])
        nlp = {'x': x, 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options(print_level=5)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        # Reconstruct each trial's optimal trajectory by evaluating its spline at that
        # trial's input data time points.
        states_tables: dict[str, osim.TimeSeriesTable] = {}
        spline_nodes: dict[str, np.ndarray] = {}
        i = 0
        for itrial, trial in enumerate(self.trials):
            num_coeff_vars = trial_num_knots[itrial] * num_coords
            coeffs_opt = ca.reshape(sol['x'][i : i + num_coeff_vars],
                                    trial_num_knots[itrial], num_coords)
            q_opt = np.array(trial_B[itrial] @ coeffs_opt)
            qdot_opt = np.array(trial_dB[itrial] @ coeffs_opt)
            states_tables[trial.name] = Solution.create_states_table(
                self.mc.model, self.state, self.coordinate_indexes,
                trial_times[itrial], q_opt, qdot_opt)
            spline_nodes[trial.name] = np.array(coeffs_opt)
            i += num_coeff_vars

        # Slice each parameter's optimized value from the flat solution vector, when any
        # parameters were registered. The parameter blocks follow every trial's control
        # points, so `i` is already at the start of the first parameter block.
        solution_parameters = None
        if num_params > 0:
            x_flat = np.array(sol['x']).flatten()
            for p in self.parameters:
                p.value = x_flat[i : i + p.num_variables].reshape(-1)
                i += p.num_variables
            solution_parameters = [p.with_value(p.value) for p in self.parameters]

        return Solution(
            states_tables=states_tables,
            parameters=solution_parameters,
            outputs={'spline_nodes': spline_nodes},
        )


#################
# MARKER PLACER #
#################

class MarkerPlacer(Solver):
    """
    A solver for placing unfixed, e.g. "tracking", markers on the model.

    The solver minimizes the squared distance between the model's marker positions and
    the reference marker positions provided by each trial's `MarkerSource`. Markers
    whose '<fixed>' property is set to ``True`` will be used to pose the model, as in a
    typical inverse kinematics problem. Markers whose '<fixed>' property is set to
    ``False`` will have the position offsets optimized to place them as close as
    possible to the reference positions.

    The solver operates on the first time point of each registered trial. If multiple
    trials are registered, the solver operates on all time points simultaneously.

    Parameters
    ----------
    model: str or osim.Model
        See `Solver`.
    offset_bounds: Bounds, optional
        The bounds on the marker position offsets to optimize. Default is
        [-0.5, 0.5] meters in each direction.
    convergence_tolerance: float, optional
        See `Solver`.
    """
    SUPPORTED_INPUTS = frozenset({'coordinates', 'marker_offsets'})

    def __init__(self, model: osim.Model, offset_bounds: Bounds = Bounds(-0.5, 0.5),
                 convergence_tolerance=1e-4):
        super().__init__(model, convergence_tolerance)
        self.offset_bounds = offset_bounds

    def _assert_no_costs(self):
        """
        Verify that no additional `Cost` is registered. This solver's objective is the
        marker placement error alone, so a registered cost has nothing to contribute to.

        Raises
        ------
        ValueError
            If any cost was registered via `add_cost`.
        """
        if self.costs:
            registered = sorted(type(cost).__name__ for cost in self.costs)
            raise ValueError(
                f'{type(self).__name__} does not accept additional costs; its '
                f'objective is the marker placement error alone, but {registered} '
                f'{"was" if len(registered) == 1 else "were"} registered.')

    def _validate_trials(self):
        """
        Verify that every registered trial carries marker data this solver can place,
        and nothing it cannot.

        Raises
        ------
        ValueError
            If a trial carries no marker data, carries frame data, or labels a marker
            that is not a marker in the model.
        """
        markerset = self.mc.model.getMarkerSet()
        model_markers = {markerset.get(i).getAbsolutePathString()
                         for i in range(markerset.getSize())}
        for trial in self.trials:
            if not trial.marker_data:
                raise ValueError(
                    f"Trial '{trial.name}' carries no marker data; MarkerPlacer places "
                    f'markers and so requires a MarkerSource in every trial.')
            if trial.frame_data:
                raise ValueError(
                    f"Trial '{trial.name}' carries frame data, which MarkerPlacer does "
                    f'not support; it places markers only.')
            for data in trial.marker_data:
                missing = sorted(set(data.labels) - model_markers)
                if missing:
                    raise ValueError(
                        f"Trial '{trial.name}' has reference data labels that are not "
                        f'markers in {self.mc.model.getName()}: {missing}.')

    def solve(self, guess: Solution = None) -> Solution:
        self._assert_has_trials()
        self._validate_trials()
        self._assert_no_costs()

        if guess is not None:
            self._validate_guess(guess)

        num_coords = len(self.coordinate_indexes)

        # Define the marker offset parameters. These are shared across every trial.
        marker_offsets: list[MarkerOffset] = []
        initial_offset = np.zeros(3)
        for tracking_marker in self.mc.get_tracking_marker_paths():
            marker_offset = MarkerOffset(tracking_marker, self.offset_bounds,
                                         initial_offset)
            marker_offset.validate(self.mc)
            marker_offsets.append(marker_offset)
        self.mc.marker_offset_groups = [mo.to_group() for mo in marker_offsets]

        # Define the optimization variables: one pose per trial, followed by the shared
        # offsets.
        poses = [ca.MX.sym(f'pose_{itrial}', num_coords)
                 for itrial in range(len(self.trials))]
        blocks = {name: ca.MX.sym(name, self.mc.parameter_groups[name].num_variables)
                  for name in CostInput.PARAMETER_INPUTS}

        # Define each pose's initial guess and bounds, seeding from the first row of
        # that trial's guess states table when one is supplied.
        x0 = []
        lbx = []
        ubx = []
        for trial in self.trials:
            guess_q = (None if guess is None else self.extract_guess_coordinates(
                guess.states_tables[trial.name])[0, :])
            for icoord, coord_path in enumerate(self.coordinate_map):
                coord = osim.Coordinate.safeDownCast(
                    self.mc.model.getComponent(coord_path))
                x0.append(coord.getDefaultValue() if guess_q is None
                          else float(guess_q[icoord]))
                lbx.append(coord.getRangeMin())
                ubx.append(coord.getRangeMax())
        for marker_offset in marker_offsets:
            marker_offset.append_guess_and_bounds(x0, lbx, ubx)

        # Accumulate the placement error at each trial's first time point. Reps are held
        # in a list for the lifetime of the solve so CasADi's references to them stay
        # valid. _validate_trials rejects frame data, so a BilevelCost's frame terms are
        # always empty here and each rep tracks markers only.
        errors = ca.MX(len(self.trials), 1)
        placement_reps = []
        placement_cost = BilevelCost(position_weight=1.0)
        for itrial, trial in enumerate(self.trials):
            placement_rep = placement_cost.create_rep(
                f'marker_placer_cost_trial_{itrial}', self.mc, trial, 0)
            placement_reps.append(placement_rep)
            errors[itrial] = placement_rep(
                CostInput(coordinates=poses[itrial], **blocks))

        # Average over trials so the objective's magnitude is independent of the number
        # of trials.
        f = ca.sum1(errors) / len(self.trials)

        # Solve.
        nlp = {'x': ca.vertcat(
            *poses, *[blocks[name] for name in CostInput.PARAMETER_INPUTS]), 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options(print_level=5)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        # Slice each trial's optimized pose from the flat solution vector, storing it as
        # a one-row states table stamped with that trial's first time.
        x_flat = np.array(sol['x']).flatten()
        states_tables: dict[str, osim.TimeSeriesTable] = {}
        i = 0
        for trial in self.trials:
            pose = x_flat[i : i + num_coords].reshape(1, num_coords)
            states_tables[trial.name] = Solution.create_states_table(
                self.mc.model, self.state, self.coordinate_indexes,
                [trial.times[0]], pose)
            i += num_coords

        # The shared offset blocks follow every trial's pose, so `i` is already at the
        # start of the first offset block.
        for marker_offset in marker_offsets:
            marker_offset.value = x_flat[
                i : i + marker_offset.num_variables].reshape(-1)
            i += marker_offset.num_variables

        return Solution(states_tables=states_tables, parameters=marker_offsets)

    def update_model(self, model: osim.Model,
                     solution: Solution) -> osim.Model:
        """
        Apply the solution's optimized marker placement offsets to `model` in place and
        return it. Each offset is baked into its marker's ``location`` property as an
        additive translation expressed in the marker's base frame.
        """
        model.initSystem()
        for parameter in solution.parameters or []:
            parameter.apply_to_model(model)
        model.finalizeConnections()
        model.initSystem()
        return model
