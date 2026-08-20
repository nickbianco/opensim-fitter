import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .bounds import Bounds
from .data_sources import DataSource, MarkerSource, TheiaFrameSource
from .costs import (Cost, CostInput, TrackingCostFunction, BilevelCostFunction)
from .model import ModelCache, Parameter, BodyScale, MarkerOffset, FrameOffset
from .scaling import Axis, Scaler, ManualBodyScale


################
# DATA STRUCTS #
################

@dataclass
class TheiaFrameData:
    labels: list[str]
    positions: osim.TimeSeriesTableVec3
    orientations: osim.TimeSeriesTableQuaternion


@dataclass
class MarkerData:
    labels: list[str]
    positions: osim.TimeSeriesTableVec3


############
# SOLUTION #
############

@dataclass
class Solution:
    """
    Base class for solver solutions.
    """

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
                qdot = np.zeros(state.getNQ())
                qdot[coordinate_indexes] = qdot_opt[i, :]
                state.setU(osim.Vector.createFromMat(qdot))
            statesTraj.append(state)
        return statesTraj.exportToTable(model)


@dataclass
class InverseKinematicsSolution(Solution):
    """
    Solution for the `InverseKinematicsSolver`. Contains the optimized model states as
    an OpenSim TimeSeriesTable.

    Attributes
    ----------
    states_table: osim.TimeSeriesTable
        The table of optimized model states.
    """
    states_table: osim.TimeSeriesTable


@dataclass
class SplinedKinematicsSolution(Solution):
    """
    Solution for the `SplinedKinematicsSolver`. Contains the optimized model states, the
    optimal B-spline control points (nodes) for each coordinate, and, when the solver
    was configured with bilevel parameters, the optimized parameters.

    Attributes
    ----------
    states_table: osim.TimeSeriesTable
        The table of optimized model states.
    spline_nodes: np.ndarray, shape (num_knots, num_coords), optional
        The optimal B-spline control points for each coordinate.
    parameters: list[Parameter], optional
        The optimized parameters (e.g., body scales, marker and frame offsets), each
        carrying its optimal ``value``, or None if the solver optimized kinematics only.
        This is an independent snapshot of the solver's parameter configuration; the same
        list (with values set) can be handed back to ``solve()`` as an initial guess.
    """
    states_table: osim.TimeSeriesTable
    spline_nodes: np.ndarray = None
    parameters: list[Parameter] = None

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


@dataclass
class MarkerPlacerSolution(Solution):
    """
    Solution for the `MarkerPlacer` solver.

    Attributes
    ----------
    pose: np.ndarray, shape (num_independent_coords,)
        Optimized independent-coordinate values for the placement pose, in the order of
        the solver's ``coordinate_indexes``.
    marker_offsets: list[MarkerOffset]
        The optimized marker placement offsets, each carrying its optimal ``value`` (an
        XYZ translation expressed in the marker's base frame).
    """
    pose: np.ndarray = None
    marker_offsets: list[MarkerOffset] = None


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

    Parameters
    ----------
    model: str or osim.Model
        The OpenSim model to use for the optimization problem. Can be provided as a file
        path or as an already-loaded osim.Model object.
    convergence_tolerance: float, optional
        The convergence tolerance to use for the IPOPT solver. Default is 1e-4.
    """
    # Concrete subclasses set this to the exact Solution subclass they accept as
    # an initial guess (and return from solve()).
    _guess_type: type = Solution

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

        # Additional user-registered costs (e.g., regularization) appended via
        # add_cost()
        self.costs: list[Cost] = []

    def add_cost(self, cost: Cost):
        """
        Register an additional `Cost` to include in the solver's objective. The cost is
        evaluated by calling it with a `CostInput`.

        Raises
        ------
        ValueError
            If the cost depends on a `CostInput` field this solver does not provide.
        """
        unsupported = cost.required_inputs - self.SUPPORTED_INPUTS
        if unsupported:
            raise ValueError(
                f'{type(self).__name__} does not support {type(cost).__name__}: it '
                f'requires cost input(s) {sorted(unsupported)} that this solver does '
                f'not provide (supported: {sorted(self.SUPPORTED_INPUTS)}).')
        self.costs.append(cost)

    def _initialize_costs(self):
        """
        Give every registered cost a chance to precompute model-derived data from this
        solver's `ModelCache`. Called at the start of solve(), after all parameters are
        registered, so costs need not be constructed with the model or parameters.
        """
        for cost in self.costs:
            cost.initialize(self.mc)

    def get_ipopt_options(self, print_level=0):
        """
        Get a dictionary of common IPOPT options for use with CasADi's nlpsolver.
        """
        ipopt_options = {}
        ipopt_options['hessian_approximation'] = 'limited-memory'
        ipopt_options['tol'] = self.convergence_tolerance
        ipopt_options['dual_inf_tol'] = self.convergence_tolerance
        ipopt_options['compl_inf_tol'] = self.convergence_tolerance
        ipopt_options['acceptable_tol'] = self.convergence_tolerance
        ipopt_options['acceptable_dual_inf_tol'] = self.convergence_tolerance
        ipopt_options['acceptable_compl_inf_tol'] = self.convergence_tolerance
        ipopt_options['print_level'] = print_level

        return ipopt_options

    def _validate_guess(self, guess: Solution):
        """
        Validate that `guess` matches the solver's expected guess type and contains
        usable data. Subclasses may override to add solver-specific checks; in that
        case they should call `super()._validate_guess(guess)` first.
        """
        if type(guess) is not self._guess_type:
            raise TypeError(
                f'{type(self).__name__} expected an initial guess of type '
                f'{self._guess_type.__name__}, but got {type(guess).__name__}.')

        table = guess.states_table
        if table.getNumRows() == 0 or table.getNumColumns() == 0:
            raise ValueError(
                'Initial guess states_table is empty '
                f'({table.getNumRows()} rows, {table.getNumColumns()} columns).')

        labels = set(table.getColumnLabels())
        missing = [coord_path + '/value' for coord_path in self.coordinate_map
                   if coord_path + '/value' not in labels]
        if missing:
            raise ValueError(
                f'Initial guess states_table is missing required coordinate columns: '
                f'{missing}.')

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
    An abstract base class for solvers that track reference data. Reference data can be
    position-based (e.g., marker trajectories) or orientation-based (e.g., Theia frames)
    and should be provided as DataSource objects via the helper methods. Concrete
    subclasses must implement the solve() method, which should return a Solution object.

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

        # Data sources.
        self.theia_frame_data: list[TheiaFrameData] = []
        self.marker_data: list[MarkerData] = []

    def add_theia_frame_reference_data(self, theia_frame_source: TheiaFrameSource):
        """
        Add a TheiaFrameSource as reference data for this solver.
        """
        positions = theia_frame_source.get_positions_table()
        orientations = theia_frame_source.get_orientations_table()
        DataSource.assert_position_orientation_consistent(positions, orientations)
        labels = positions.getColumnLabels()

        self.theia_frame_data.append(TheiaFrameData(labels, positions, orientations))

    def add_marker_reference_data(self, marker_source: MarkerSource):
        """
        Add a MarkerSource as reference data for this solver.
        """
        positions = marker_source.get_positions_table()
        labels = positions.getColumnLabels()

        self.marker_data.append(MarkerData(labels, positions))

    def get_times_from_reference_data(self):
        """
        Extract the time vector from the reference data, asserting that all data
        sources share the same time vector.
        """
        tables = [data.positions for data in self.theia_frame_data]
        tables += [data.positions for data in self.marker_data]
        return DataSource.assert_tables_share_times(tables)

    def create_tracking_callback(self, name: str, itime: int,
                                 position_weight: float,
                                 orientation_weight: float) -> TrackingCostFunction:
        """
        Create a CasADi callback function for computing the tracking cost at a given
        time step, which can be used in the formulation of an optimization problem.
        """
        callback = TrackingCostFunction(name, self.mc)

        for data in self.theia_frame_data:
            for iframe, frame_path in enumerate(data.labels):
                callback.add_frame_tracking_cost_term(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=position_weight,
                    orientation_weight=orientation_weight)

        for data in self.marker_data:
            for iframe, marker_path in enumerate(data.labels):
                callback.add_marker_tracking_cost_term(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    weight=position_weight)

        return callback

    def _validate_guess(self, guess: Solution):
        super()._validate_guess(guess)
        num_times = len(self.get_times_from_reference_data())
        num_rows = guess.states_table.getNumRows()
        if num_rows != num_times:
            raise ValueError(
                f'Initial guess states_table has {num_rows} rows but the reference '
                f'data has {num_times} time samples.')


##############################
# INVERSE KINEMATICS SOLVERS #
##############################


class InverseKinematicsSolver(TrackingSolver):
    """
    Solve the inverse kinematics problem to find the set of model coordinate values that
    best track provided position (e.g., marker trajectories) and/or orientation (e.g.,
    frame orientations) data.

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

    _guess_type = InverseKinematicsSolution
    SUPPORTED_INPUTS = frozenset({'coordinates'})

    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0):
        super().__init__(model, convergence_tolerance, position_weight,
                         orientation_weight)

    def create_tracking_solver(self, itime, position_weight, orientation_weight):
        """
        A helper function to create a CasADi solver for the tracking problem at a
        given time step.
        """
        x = ca.SX.sym('x', len(self.coordinate_indexes))
        callback = self.create_tracking_callback('tracking_cost', itime,
                                                 position_weight=position_weight,
                                                 orientation_weight=orientation_weight)
        cost_input = CostInput(coordinates=x)
        f = callback(cost_input)
        for cost in self.costs:
            f += cost(cost_input)
        nlp = {'x': x, 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options()
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        return callback, solver

    def solve(self, guess: InverseKinematicsSolution = None
              ) -> InverseKinematicsSolution:

        self._initialize_costs()
        times = self.get_times_from_reference_data()
        num_times = len(times)

        if guess is not None:
            self._validate_guess(guess)

        # Per-coordinate bounds, plus an initial x0 to use for the first time step
        # when no guess is supplied. The loop below carries x0 forward from the
        # previous step's solution (or pulls from the guess if provided).
        x0 = []
        lbx = []
        ubx = []
        for coord_path in self.coordinate_map:
            coord = osim.Coordinate.safeDownCast(self.mc.model.getComponent(coord_path))
            x0.append(coord.getDefaultValue())
            lbx.append(coord.getRangeMin())
            ubx.append(coord.getRangeMax())

        # When a guess is provided, pre-extract a (num_times, num_coords) array of
        # initial values from the guess states_table so each timestep can be seeded
        # from the corresponding row.
        guess_q = None
        if guess is not None:
            guess_q = np.column_stack([
                guess.states_table.getDependentColumn(
                    coord_path + '/value').to_numpy()
                for coord_path in self.coordinate_map])

        # Iterate over all of the time steps in the tracking data and solve the
        # optimization problem at each time step.
        statesTraj = osim.StatesTrajectory()
        q_traj = np.zeros((num_times, len(self.coordinate_indexes)))
        for itime, time in enumerate(times):
            print(f'Solving time {itime+1} of {num_times} (t={time:.3f} s)...')

            if guess_q is not None:
                x0 = guess_q[itime, :].tolist()

            callback, solver = self.create_tracking_solver(itime,
                    position_weight=self.position_weight,
                    orientation_weight=self.orientation_weight)
            sol = solver(x0=x0, lbx=lbx, ubx=ubx)

            q_traj[itime, :] = np.squeeze(sol['x'].full())

            # Write solution into callback.state — avoids calling initSystem() again,
            # which would invalidate the state handle held by the callback.
            # StatesTrajectory.append() copies the state by value, so reuse is safe.
            callback.state.setTime(time)
            q = np.zeros(callback.state.getNQ())
            q[self.coordinate_indexes] = q_traj[itime, :]
            callback.state.setQ(osim.Vector.createFromMat(q))
            statesTraj.append(callback.state)

            if guess_q is None:
                x0 = sol['x']

        return InverseKinematicsSolution(
            states_table=statesTraj.exportToTable(self.mc.model),
        )


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
        The interval between knots in the B-spline basis. Default is 0.05 seconds.
    """
    _guess_type = SplinedKinematicsSolution
    SUPPORTED_INPUTS = frozenset({'body_scales', 'marker_offsets', 'frame_offsets'})

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

    def create_bilevel_callback(self, name: str, itime: int,
                                position_weight: float,
                                orientation_weight: float) -> BilevelCostFunction:

        # Construct the bilevel callback function.
        callback = BilevelCostFunction(name, self.mc)

        # Map each offset target path to the index of its offset group.
        marker_index_of = {path: i for i, grp in enumerate(self.mc.marker_offset_groups)
                                   for path in grp.component_paths}
        frame_index_of = {path: i for i, grp in enumerate(self.mc.frame_offset_groups)
                                  for path in grp.component_paths}

        # Add the tracking cost terms for each frame.
        for data in self.theia_frame_data:
            for iframe, frame_path in enumerate(data.labels):
                callback.add_frame_bilevel_cost_term(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=position_weight,
                    orientation_weight=orientation_weight,
                    offset_group_index=frame_index_of.get(frame_path))

        # Add the tracking cost terms for each marker.
        for data in self.marker_data:
            for iframe, marker_path in enumerate(data.labels):
                callback.add_marker_bilevel_cost_term(marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    weight=position_weight,
                    offset_group_index=marker_index_of.get(marker_path))

        # Every offset group must be used by at least one registered task.
        def assert_offset_groups_used(offset_group_indexes, offset_groups, label):
            used = {g for g in offset_group_indexes if g is not None}
            for i, group in enumerate(offset_groups):
                if i not in used:
                    raise ValueError(
                        f'{label.capitalize()} offset group {group.component_paths} is '
                        f'not tracked by any registered {label}; its offset would be '
                        f'unconstrained.')
        assert_offset_groups_used(callback.marker_term.offset_group_indexes,
                                  self.mc.marker_offset_groups, 'marker')
        assert_offset_groups_used(callback.frame_term.offset_group_indexes,
                                  self.mc.frame_offset_groups, 'frame')

        return callback

    def update_model(self, model: osim.Model,
                     solution: SplinedKinematicsSolution) -> osim.Model:
        """
        Apply the solution's optimized parameters to `model` and return it.
        """
        model.initSystem()

        # Get pre-`Model::scale()` quanities.
        translation_scales = ModelCache.get_custom_joint_translation_scales(model)

        # Construct a scaler using the optimized body scales as manual scale factors.
        # This calls Model::scale() under the hood. Body scales are baked via the Scaler
        # rather than each BodyScale's apply_to_model (see BodyScale.apply_to_model).
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

        # Apply pre-`Model::scale()` quanities.
        ModelCache.apply_custom_joint_translation_scales(model, translation_scales)

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


    def solve(self, guess: SplinedKinematicsSolution = None
              ) -> SplinedKinematicsSolution:

        self._initialize_costs()
        times = self.get_times_from_reference_data()
        num_times = len(times)

        if guess is not None:
            self._validate_guess(guess)

        # Define the knot vector.
        num_knots = int((times[-1] - times[0]) / self.knot_interval)
        knots = self.build_knots_vector(times, num_knots)

        # Pre-compute the spline basis matrix and its derivative.
        B, dB = self.build_spline_basis_matrix(times, knots)

        # Extract parameter dimensions.
        num_params = len(self.parameters)
        num_scales = sum(p.num_variables for p in self.parameters
                         if isinstance(p, BodyScale))
        num_markers = sum(p.num_variables for p in self.parameters
                          if isinstance(p, MarkerOffset))
        num_frames = sum(p.num_variables for p in self.parameters
                         if isinstance(p, FrameOffset))

        # Apply the parameters from the initial guess to the solver's list of registered
        # parameters.
        if num_params > 0 and guess is not None and guess.parameters is not None:
            for sp, gp in zip(self.parameters, guess.parameters):
                sp.value = np.asarray(gp.value, dtype=float)

        # Define the optimization variables.
        coeffs = ca.MX.sym('coeffs', num_knots, len(self.coordinate_indexes))
        s = ca.MX.sym('body_scales', num_scales)
        mo = ca.MX.sym('marker_offsets', num_markers)
        fo = ca.MX.sym('frame_offsets', num_frames)
        x0 = []
        lbx = []
        ubx = []
        for coord_path in self.coordinate_map:
            coord = osim.Coordinate.safeDownCast(self.mc.model.getComponent(coord_path))
            x0 += ([coord.getDefaultValue()] * num_knots if guess is None
                   else self.extract_coordinate_initial_guess(
                       guess.states_table, B, coord_path))
            lbx += [coord.getRangeMin()] * num_knots
            ubx += [coord.getRangeMax()] * num_knots

        # Append each parameter's initial guess and bounds, in type order, matching the
        # [coeffs, s, mo, fo] layout of the optimization vector below.
        for p in self.parameters:
            p.append_guess_and_bounds(x0, lbx, ubx)

        # Map the control points to the full predicted trajectory via the spline basis
        # matrix.
        q = B @ coeffs

        # Compute the tracking cost at each time step via a callback.
        errors = ca.MX(num_times, 1)
        callbacks = []
        for itime in range(num_times):
            if num_params > 0:
                callback = self.create_bilevel_callback(
                    f'tracking_cost_time_{itime}', itime,
                    position_weight=self.position_weight,
                    orientation_weight=self.orientation_weight)
            else:
                callback = self.create_tracking_callback(
                    f'tracking_cost_time_{itime}', itime,
                    position_weight=self.position_weight,
                    orientation_weight=self.orientation_weight)
            callbacks.append(callback)
            cost_input = CostInput(coordinates=q[itime, :].T, body_scales=s,
                                   marker_offsets=mo, frame_offsets=fo)
            errors[itime] = callback(cost_input)

        # Compute the total cost.
        f = self.compute_average_trapezoidal_error(errors, times)
        trial_input = CostInput(body_scales=s, marker_offsets=mo, frame_offsets=fo)
        for cost in self.costs:
            f += cost(trial_input)

        # Solve.
        nlp = {'x': ca.vertcat(ca.vec(coeffs), s, mo, fo), 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options(print_level=5)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        # Reconstruct the optimal trajectory by evaluating the spline at the input data
        # time points.
        num_coeff_vars = num_knots * len(self.coordinate_indexes)
        coeffs_opt = ca.reshape(sol['x'][:num_coeff_vars], num_knots,
                                len(self.coordinate_indexes))
        q_opt = np.array(B @ coeffs_opt)
        qdot_opt = np.array(dB @ coeffs_opt)

        # Slice each parameter's optimized value from the flat solution vector, when any
        # parameters were registered.
        solution_parameters = None
        if num_params > 0:
            x_flat = np.array(sol['x']).flatten()
            i = num_coeff_vars
            for p in self.parameters:
                p.value = x_flat[i : i + p.num_variables].reshape(-1)
                i += p.num_variables
            solution_parameters = [p.with_value(p.value) for p in self.parameters]

        return SplinedKinematicsSolution(
            states_table=Solution.create_states_table(
                self.mc.model, self.state, self.coordinate_indexes,
                times, q_opt, qdot_opt),
            spline_nodes=np.array(coeffs_opt),
            parameters=solution_parameters,
        )


#################
# MARKER PLACER #
#################

class MarkerPlacer(Solver):
    """
    A solver for placing unfixed, e.g. "tracking", markers on the model.

    The solver uses minimize the squared distance between the model's marker
    positions and the reference marker positions provided by a `MarkerSource`.
    Markers whose '<fixed>' property is set to ``True`` will be use to pose the
    model, as in a typical inverse kinematics problem. Markers whose '<fixed>'
    property is set to ``False`` will have the position offsets optimized to place
    them as close as possible to the reference positions.

    Parameters
    ----------
    model: str or osim.Model
        See `Solver`.
    marker_source: MarkerSource
        A MarkerSource as reference data for the marker placement optimization
        problem. The column labels must match the absolute paths of the markers in
        the model.
    marker_index: int, optional
        The row index of the marker positions table from the marker source to use
        as the reference positions for the optimization.
    offset_bounds: Bounds, optional
        The bounds on the marker position offsets to optimize. Default is
        [-0.5, 0.5] meters in each direction.
    convergence_tolerance: float, optional
        See `Solver`.
    """
    _guess_type = MarkerPlacerSolution
    SUPPORTED_INPUTS = frozenset({'coordinates', 'marker_offsets'})

    def __init__(self, model: osim.Model, marker_source: MarkerSource,
                 marker_index: int = 0, offset_bounds: Bounds = Bounds(-0.5, 0.5),
                 convergence_tolerance=1e-4):
        super().__init__(model, convergence_tolerance)
        self.marker_source = marker_source
        self.marker_index = marker_index
        self.offset_bounds = offset_bounds

    def solve(self, guess: MarkerPlacerSolution = None) -> MarkerPlacerSolution:

        self._initialize_costs()
        # Validate the marker source and extract the marker paths to track.
        self.marker_source.validate_marker_paths(self.mc.model)
        positions = self.marker_source.get_positions_table()
        marker_paths = positions.getColumnLabels()

        # Validate the guess.
        if guess is not None:
            self._validate_guess(guess)

        # Define the marker offset parameters.
        marker_offsets: list[MarkerOffset] = []
        initial_offset = np.zeros(3)
        for tracking_marker in self.mc.get_tracking_marker_paths():
            marker_offset = MarkerOffset(tracking_marker, self.offset_bounds,
                                         initial_offset)
            marker_offset.validate(self.mc)
            marker_offsets.append(marker_offset)
        self.mc.marker_offset_groups = [mo.to_group() for mo in marker_offsets]

        # Define variables.
        num_markers = sum(mo.num_variables for mo in marker_offsets)
        q = ca.MX.sym('q', len(self.coordinate_indexes))
        s = ca.MX.sym('body_scales', 0)
        mo = ca.MX.sym('marker_offsets', num_markers)
        fo = ca.MX.sym('frame_offsets', 0)

        # Define bounds.
        x0 = []
        lbx = []
        ubx = []
        for coord_path in self.coordinate_map:
            coord = osim.Coordinate.safeDownCast(self.mc.model.getComponent(coord_path))
            x0.append(coord.getDefaultValue())
            lbx.append(coord.getRangeMin())
            ubx.append(coord.getRangeMax())
        for marker_offset in marker_offsets:
            marker_offset.append_guess_and_bounds(x0, lbx, ubx)

        # Define the cost bilevel function. It reads its parameter groups from the
        # ModelCache.
        callback = BilevelCostFunction('marker_placer_cost', self.mc)
        marker_index_of = {path: i
                           for i, grp in enumerate(self.mc.marker_offset_groups)
                           for path in grp.component_paths}
        for imarker, marker_path in enumerate(marker_paths):
            callback.add_marker_bilevel_cost_term(marker_path,
                positions.getRowAtIndex(self.marker_index).getElt(0, imarker),
                weight=1.0, offset_group_index=marker_index_of.get(marker_path))
        cost_input = CostInput(coordinates=q, body_scales=s, marker_offsets=mo,
                               frame_offsets=fo)
        f = callback(cost_input)
        for cost in self.costs:
            f += cost(cost_input)

        # Solve.
        nlp = {'x': ca.vertcat(q, s, mo, fo), 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options(print_level=5)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        # Slice the optimized pose and marker offsets from the flat solution vector.
        x_flat = np.array(sol['x']).flatten()
        num_coords = len(self.coordinate_indexes)
        pose = x_flat[:num_coords]
        i = num_coords
        for marker_offset in marker_offsets:
            marker_offset.value = x_flat[
                i : i + marker_offset.num_variables].reshape(-1)
            i += marker_offset.num_variables

        return MarkerPlacerSolution(pose=pose, marker_offsets=marker_offsets)

    def update_model(self, model: osim.Model,
                     solution: MarkerPlacerSolution) -> osim.Model:
        """
        Apply the solution's optimized marker placement offsets to `model` in place and
        return it. Each offset is baked into its marker's ``location`` property as an
        additive translation expressed in the marker's base frame.
        """
        model.initSystem()
        for marker_offset in solution.marker_offsets:
            marker_offset.apply_to_model(model)
        model.finalizeConnections()
        model.initSystem()
        return model
