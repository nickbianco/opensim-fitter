import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from .data_sources import Trial
from .model import ModelCache, StationCache
from .scaling import AnthropometricMeasurement
from .anthropometrics import build_ansur_distribution


##################
# COST INTERFACE #
##################

@dataclass
class CostInput:
    """
    Bundles the optimization variables passed to a cost evaluation. The canonical
    ordering of cost inputs is defined via the `INPUT_ORDER` attribute. Unused inputs
    default to empty symbolic arrays so they remain valid callback arguments.

    Attributes
    ----------
    INPUT_ORDER: tuple[str, ...]
        The canonical order of the optimization-variable inputs.
    coordinates: ca.MX, optional
        Coordinate values (e.g., joint angles).
    body_scales: ca.MX, optional
        Flattened per-group XYZ body-scale factors.
    marker_offsets: ca.MX, optional
        Flattened per-group XYZ marker offsets.
    frame_offsets: ca.MX, optional
        Flattened per-group XYZ frame offsets.
    ellipsoid_radii: ca.MX, optional
        Flattened per-group XYZ factors on EllipsoidJoint radii.
    beam_lengths: ca.MX, optional
        Per-group factors on CantileverFreeBeamJoint beam lengths, one per group.

    Notes
    -----
    `INPUT_ORDER` is also the order in which a bilevel cost composes parameter blocks
    onto a tracking term's stations, so body scales (which set a station absolutely)
    must precede offsets (which add to it). See `ParameterGroups`.
    """
    INPUT_ORDER: ClassVar[tuple[str, ...]] = (
        'coordinates', 'body_scales', 'marker_offsets', 'frame_offsets',
        'ellipsoid_radii', 'beam_lengths')
    TRIPLET_INPUTS: ClassVar[tuple[str, ...]] = (
        'body_scales', 'marker_offsets', 'frame_offsets', 'ellipsoid_radii')
    PARAMETER_INPUTS: ClassVar[tuple[str, ...]] = INPUT_ORDER[1:]

    coordinates: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    body_scales: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    marker_offsets: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    frame_offsets: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    ellipsoid_radii: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    beam_lengths: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))

    @classmethod
    def field_index(cls, name: str) -> int:
        """
        Return the canonical position of `name` in `INPUT_ORDER`, validating that it is
        a recognized input field. Solvers use this both to reject parameters that
        declare an unknown input and to order parameter blocks by `INPUT_ORDER`.

        Parameters
        ----------
        name: str
            The `CostInput` field name to look up.

        Returns
        -------
        int
            The index of `name` in `INPUT_ORDER`.

        Raises
        ------
        ValueError
            If `name` is not a recognized `CostInput` field.
        """
        if name not in cls.INPUT_ORDER:
            raise ValueError(
                f'{name!r} is not a recognized CostInput field {cls.INPUT_ORDER}.')
        return cls.INPUT_ORDER.index(name)

    def as_triplets(self, name: str) -> ca.MX:
        """
        Return a per-group XYZ input reshaped to an ``(n, 3)`` matrix whose row ``i`` is
        the ``(x, y, z)`` triplet of group ``i``. The flat storage is laid out
        triplet-contiguous (``[s0x, s0y, s0z, s1x, ...]``), so the view is the
        column-major ``(3, n)`` reshape transposed.

        Parameters
        ----------
        name: str
            The field to view. Must be one of `TRIPLET_INPUTS` (`body_scales`,
            `marker_offsets`, or `frame_offsets`).

        Returns
        -------
        ca.MX
            The field as an ``(n, 3)`` matrix.

        Raises
        ------
        ValueError
            If `name` is not a triplet input, or its length is not a multiple of 3.
        """
        if name not in self.TRIPLET_INPUTS:
            raise ValueError(
                f'{name!r} is not a triplet input; expected one of '
                f'{self.TRIPLET_INPUTS}.')
        value = getattr(self, name)
        num_entries = value.numel()
        if num_entries % 3 != 0:
            raise ValueError(
                f'{name!r} has length {num_entries}, which is not a multiple of 3.')
        return ca.reshape(value, 3, num_entries // 3).T


class CostRep(ABC):
    """
    A cost's per-solve representation that a solver evaluates.

    A `CostRep` is constructed via a cost's `create_rep` from the solver's `ModelCache`
    and lives only for the duration of one solve.
    """

    @abstractmethod
    def __call__(self, input: CostInput) -> ca.MX:
        pass


class CostBase(ABC):
    """
    A model-independent description of a cost term: the weights, targets, and reference
    data the user (or a solver) supplies, with no OpenSim state of its own. A cost is
    inert and reusable. All model-bound work happens in the `CostRep` it creates.

    Attributes
    ----------
    required_inputs: frozenset[str]
        The `CostInput` field names this cost reads and therefore requires the solver to
        provide (e.g., ``{'body_scales'}``). A solver validates that it provides every
        required input before accepting the cost; see `Solver.add_cost`.
    """
    required_inputs: frozenset[str] = frozenset()


class Cost(CostBase):
    """
    A cost whose rep is built from the `ModelCache` alone, and so the kind of cost a
    user can register on a solver via `Solver.add_cost`.
    """

    @abstractmethod
    def create_rep(self, mc: ModelCache) -> CostRep:
        """
        Build one of this cost's representations against `mc`. Solvers call this once
        for every rep the problem needs, after all parameter groups are registered, and
        hold the returned reps for the lifetime of the solve.

        Parameters
        ----------
        mc: ModelCache
            The solver's `ModelCache`, from which the rep initializes any model-derived
            quantities.
        """


class TrackingCostBase(CostBase):
    """
    A cost evaluated at a single time sample of a single trial, whose rep therefore
    needs that trial and sample index at construction.
    """

    @abstractmethod
    def create_rep(self, name: str, mc: ModelCache, trial: Trial,
                   itime: int) -> CostRep:
        """
        Build this cost's representation of one time sample of one trial. Solvers call
        this once for every sample the problem tracks, after all parameter groups are
        registered, and hold the returned reps for the lifetime of the solve.

        Parameters
        ----------
        name: str
            The name of the rep's callback function.
        mc: ModelCache
            The solver's `ModelCache`, from which the rep initializes any model-derived
            quantities.
        trial: Trial
            The trial supplying the reference data.
        itime: int
            Index of the time sample within `trial` that the rep tracks.
        """


class SymbolicCost(Cost):
    """
    A `Cost` whose rep is a plain CasADi expression, requiring no OpenSim evaluation.
    It is differentiated symbolically by CasADi and incurs no callback overhead.
    """

    @abstractmethod
    def evaluate(self, input: CostInput) -> ca.MX:
        """
        Return this cost's CasADi expression for `input`.
        """

    def create_rep(self, mc: ModelCache) -> 'SymbolicCostRep':
        return SymbolicCostRep(self)


class SymbolicCostRep(CostRep):
    """
    The rep of a `SymbolicCost`. It holds no model-derived state, since the cost is a
    pure expression over the optimization variables, and simply defers to
    `SymbolicCost.evaluate`.

    Parameters
    ----------
    cost: SymbolicCost
        The cost this rep represents.
    """

    def __init__(self, cost: SymbolicCost):
        self.cost = cost

    def __call__(self, input: CostInput) -> ca.MX:
        return self.cost.evaluate(input)


class Function(ca.Callback, ABC):
    """
    A base class for CasADi callback functions that evaluate the function and its
    Jacobian using OpenSim. To implement a new callback, extend this class and implement
    the abstract methods to define the number of inputs and outputs and provide the
    function evaluation and its Jacobian.

    Parameters
    ----------
    name: str
        The name of the callback function.
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function
        and its Jacobian and caching model information.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its analytic
        Jacobian (`get_jacobian`). Default is ``False``.
    """
    def __init__(self, name: str, mc: ModelCache, enable_fd: bool = False):
        ca.Callback.__init__(self)
        self.mc = mc
        self.state = self.mc.state
        self.enable_fd = enable_fd
        self.construct(name, {'enable_fd': True} if enable_fd else {})

    def get_n_in(self): return self._get_num_inputs()
    def get_n_out(self): return self._get_num_outputs()

    def get_input_size(self, i):
        return self._get_input_size(i)

    def get_output_size(self, i):
        return self._get_output_size(i)

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.get_input_size(i), 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.get_output_size(i), 1)

    def eval(self, arg):
        return self._eval(arg)

    def has_jacobian(self): return not self.enable_fd

    def get_jacobian(self, name, inames, onames, opts):
        class JacobianFunction(ca.Callback):
            def __init__(self, callback, opts={}):
                ca.Callback.__init__(self)
                self.callback = callback
                self.construct(name, opts)

            def get_n_in(self):
                return self.callback.get_n_in() + self.callback.get_n_out()
            def get_n_out(self):
                return self.callback.get_n_in()

            def get_sparsity_in(self,i):
                if i < self.callback.get_n_in():
                    return ca.Sparsity.dense(self.callback.get_input_size(i), 1)
                elif i < self.callback.get_n_in() + self.callback.get_n_out():
                    iout = i - self.callback.get_n_in()
                    return ca.Sparsity.dense(self.callback.get_output_size(iout), 1)
                else:
                    return ca.Sparsity.dense(0, 0)

            def get_sparsity_out(self,i):
                iin = i % self.callback.get_n_in()
                iout = i // self.callback.get_n_in()
                return ca.Sparsity.dense(self.callback.get_output_size(iout),
                                         self.callback.get_input_size(iin))

            def eval(self, arg):
                return self.callback._jac_eval(arg)

        self.jacobian_callback = JacobianFunction(self)
        return self.jacobian_callback

    @abstractmethod
    def _get_num_inputs(self):
        pass

    @abstractmethod
    def _get_num_outputs(self):
        pass

    @abstractmethod
    def _get_input_size(self, i):
        pass

    @abstractmethod
    def _get_output_size(self, i):
        pass

    @abstractmethod
    def _eval(self, arg):
        pass

    @abstractmethod
    def _jac_eval(self, arg):
        pass


class CallbackCostRep(CostRep, Function):
    """
    A `CostRep` backed by a CasADi callback function that evaluates the cost and its
    Jacobian through OpenSim. Constructed with a fully-populated `ModelCache`, so the
    input sizes it declares to CasADi match the solver's registered parameter groups.
    """

    def __call__(self, input: CostInput) -> ca.MX:
        return ca.Function.__call__(
            self, *(getattr(input, name) for name in CostInput.INPUT_ORDER))

    def _get_num_inputs(self):
        return len(CostInput.INPUT_ORDER)

    def _get_num_outputs(self):
        return 1

    def _get_input_size(self, i):
        order = CostInput.INPUT_ORDER
        if not 0 <= i < len(order):
            raise IndexError(f'Invalid input index {i} for {type(self).__name__}.')
        name = order[i]
        if name == 'coordinates':
            return len(self.mc.coordinate_indexes)
        return self.mc.parameter_groups[name].num_variables

    def _empty_parameter_jacobians(self):
        """
        A zero Jacobian block for every parameter input, sized from the solver's
        registered parameter groups. Used by reps whose cost does not depend on any
        parameter.
        """
        return [np.zeros((1, self._get_input_size(i)))
                for i in range(1, len(CostInput.INPUT_ORDER))]

    def _get_output_size(self, i):
        if i == 0:
            return 1
        raise IndexError(f'Invalid output index {i} for {type(self).__name__}.')


class BodyScaleRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on body-scale factors that encourages each toward `target`:

        cost = weight * sum_i (s_i - target)^2

    Keeping the scales near ``target`` (typically 1.0, i.e., identity scaling) means the
    optimizer only deviates from the nominal scaling when doing so substantially
    improves the primary tracking cost.

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    target: float, optional
        Per-component target value. Default is 1.0.
    """
    required_inputs = frozenset({'body_scales'})

    def __init__(self, weight: float, target: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.target = target

    def evaluate(self, input: CostInput) -> ca.MX:
        return self.weight * ca.sum((input.body_scales - self.target)**2)


class BodyScaleIsotropyCost(SymbolicCost):
    """
    A quadratic penalty encouraging each body-scale group to scale isotropically, i.e.
    equally along X, Y, and Z:

        cost = weight * sum_g sum_axis (s_{g,axis} - mean_axis(s_g))^2

    where ``mean_axis(s_g)`` is the average of group ``g``'s three scale factors. It
    penalizes a group being stretched more along one axis than another without
    constraining its overall size.

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    """
    required_inputs = frozenset({'body_scales'})

    def __init__(self, weight: float):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight

    def evaluate(self, input: CostInput) -> ca.MX:
        scales = input.as_triplets('body_scales')
        axis_means = ca.sum2(scales) / 3
        deviations = scales - ca.repmat(axis_means, 1, 3)
        return self.weight * ca.sumsqr(deviations)


class OffsetRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on marker and frame XYZ offsets, penalizing offsets away from
    zero:

        cost = weight * sum_i offset_i^2

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    """
    required_inputs = frozenset({'marker_offsets', 'frame_offsets'})

    def __init__(self, weight: float):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight

    def evaluate(self, input: CostInput) -> ca.MX:
        offsets = ca.vertcat(input.marker_offsets, input.frame_offsets)
        return self.weight * ca.sum(offsets**2)


class MobilizerDimensionRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on the mobilizer dimension factors -- ellipsoid radii and beam
    lengths -- that encourages each toward `target`:

        cost = weight * sum_i (f_i - target)^2

    Both parameter types are dimensionless factors on the model's baseline geometry, so
    a target of 1.0 keeps a joint's dimensions at their nominal values and the optimizer
    only deviates when doing so substantially improves the primary tracking cost.
    Without such a penalty these factors are prone to absorbing marker error and
    running to their bounds, since a joint's internal geometry can often mimic the
    effect of a pose change.

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    target: float, optional
        Per-factor target value. Default is 1.0.
    """
    required_inputs = frozenset({'ellipsoid_radii', 'beam_lengths'})

    def __init__(self, weight: float, target: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.target = target

    def evaluate(self, input: CostInput) -> ca.MX:
        dimensions = ca.vertcat(input.ellipsoid_radii, input.beam_lengths)
        return self.weight * ca.sum((dimensions - self.target)**2)


class CoordinateStiffnessCost(Cost):
    """
    A quadratic penalty that acts like a spring on selected coordinates, holding each
    near a target value:

        cost = weight * sum_i k_i * (q_i - target_i)^2

    Reference data does not constrain every coordinate equally well. Scapula and spine
    coordinates in particular are only weakly determined by surface markers, so they
    are free to drift toward whatever value marginally improves the tracking error,
    often into a non-physiological posture. A stiffness holds such a coordinate near a
    neutral value unless the data gives a clear reason to move it, without constraining
    it outright the way a narrowed range would.

    The penalty is a function of the coordinates, so it is evaluated at every time
    sample and time-averaged alongside the tracking error rather than once per solve.
    Each stiffness therefore trades off directly against the time-averaged tracking
    error.

    Note that a coordinate's units set the scale of its stiffness: `k_i` multiplies
    squared radians for a rotational coordinate and squared meters for a translational
    one, so stiffnesses are not comparable across the two.

    Parameters
    ----------
    stiffnesses: dict[str, float]
        Mapping from absolute coordinate path to that coordinate's non-negative
        stiffness. Only the coordinates named here are penalized.
    targets: dict[str, float], optional
        Mapping from absolute coordinate path to the value that coordinate is pulled
        toward. Any coordinate absent from this mapping is pulled toward its default
        value in the model, which is the neutral posture the solvers also seed from.
        Defaults to ``None`` (every target taken from the model).
    weight: float, optional
        Non-negative scalar applied to the whole sum. Default is 1.0.

    Raises
    ------
    ValueError
        If `weight` or any stiffness is negative, or `stiffnesses` is empty.
        `CoordinateStiffnessCostRep` additionally validates the coordinate paths
        against the model.
    """
    required_inputs = frozenset({'coordinates'})

    def __init__(self, stiffnesses: dict[str, float],
                 targets: dict[str, float] = None, weight: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        if not stiffnesses:
            raise ValueError(
                'CoordinateStiffnessCost requires at least one coordinate stiffness.')
        for path, stiffness in stiffnesses.items():
            if stiffness < 0:
                raise ValueError(
                    f'Expected the stiffness for {path} to be non-negative, but got '
                    f'{stiffness}.')
        self.weight = weight
        self.stiffnesses = dict(stiffnesses)
        self.targets = dict(targets) if targets else {}

    def create_rep(self, mc: ModelCache) -> 'CoordinateStiffnessCostRep':
        return CoordinateStiffnessCostRep(self, mc)


class CoordinateStiffnessCostRep(CostRep):
    """
    The rep of a `CoordinateStiffnessCost`. It resolves each coordinate path to its
    position in the `CostInput.coordinates` vector and fills in any target left
    unspecified from the model, then evaluates a plain CasADi expression, so the cost is
    differentiated symbolically with no callback overhead.

    Parameters
    ----------
    cost: CoordinateStiffnessCost
        The cost this rep represents.
    mc: ModelCache
        The solver's `ModelCache`, supplying the coordinate ordering and the default
        coordinate values.

    Raises
    ------
    ValueError
        If a coordinate path is not an independent coordinate of the model. Dependent
        (e.g. constrained) coordinates are absent from `ModelCache.coordinate_map` and
        so cannot be penalized directly.
    """

    def __init__(self, cost: CoordinateStiffnessCost, mc: ModelCache):
        self.cost = cost
        # Element j of the coordinates vector is the j-th entry of coordinate_map, so a
        # coordinate's position in that ordering is its index into the input.
        order = list(mc.coordinate_map)
        self.indexes: list[int] = []
        self.stiffnesses: list[float] = []
        self.targets: list[float] = []
        for path, stiffness in cost.stiffnesses.items():
            if path not in mc.coordinate_map:
                known = 'is a dependent coordinate' if mc.model.hasComponent(path) \
                    else 'is not a coordinate in the model'
                raise ValueError(
                    f'Cannot apply a coordinate stiffness to {path}: it {known}. '
                    f'Expected one of the model\'s independent coordinates.')
            self.indexes.append(order.index(path))
            self.stiffnesses.append(float(stiffness))
            if path in cost.targets:
                self.targets.append(float(cost.targets[path]))
            else:
                coordinate = osim.Coordinate.safeDownCast(mc.model.getComponent(path))
                self.targets.append(float(coordinate.getDefaultValue()))

    def __call__(self, input: CostInput) -> ca.MX:
        penalty = 0
        for index, stiffness, target in zip(
                self.indexes, self.stiffnesses, self.targets):
            penalty += stiffness * (input.coordinates[index] - target)**2
        return self.cost.weight * penalty


###########
# HELPERS #
###########

def _calc_quaternion(state, frame):
    rotation = frame.getRotationInGround(state)
    quaternion = rotation.convertRotationToQuaternion()
    return np.array([quaternion.get(i) for i in range(4)])

def _project_to_coordinates(mc: ModelCache, state: osim.State,
                            f_u: osim.Vector) -> np.ndarray:
    """
    Convert a mobility-space gradient into a row of the error Jacobian with respect to
    the independent generalized coordinates.

    Simbody's station and frame Jacobian-transpose operators return a generalized force
    in u-space, of length ``state.getNU()``, while `ModelCache.coordinate_map` indexes
    q. The two spaces coincide only where qdot == u; in general, since qdot = N u, a
    coordinate increment maps to a mobility increment through NInv, so the q-space
    gradient is ``~NInv * f_u``.

    Parameters
    ----------
    mc: ModelCache
        The cache supplying the model and the independent coordinate q-indexes.
    state: osim.State
        A State realized through Position.
    f_u: osim.Vector
        The mobility-space gradient, length ``state.getNU()``.

    Returns
    -------
    np.ndarray
        The gradient with respect to the independent coordinates, shape
        ``(1, len(mc.coordinate_indexes))``.
    """
    f_q = osim.Vector(state.getNQ(), 0.0)
    mc.model.multiplyByNInv(state, True, f_u, f_q)
    return np.expand_dims(f_q.to_numpy()[mc.coordinate_indexes], axis=0)


def _calc_quaternion_jacobian(eps):
    # Simbody -> /SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L712
    e = 0.5 * eps
    return np.array([
        [-e[1], -e[2], -e[3]],
        [ e[0],  e[3], -e[2]],
        [-e[3],  e[0],  e[1]],
        [ e[2], -e[1],  e[0]],
    ])

##################
# ERROR GRADIENT #
##################

@dataclass
class ErrorGradient:
    """
    The intermediate gradient quantities a bilevel tracking term computes once per
    evaluation, from which every registered parameter block assembles its own Jacobian.

    This is the interface between costs and parameters: a term knows how its error
    depends on body positions and station placements, a `ParameterGroups` knows how its
    variables move those, and neither needs to know about the other. See
    `ParameterGroups.calc_jacobian_block`.

    Attributes
    ----------
    state: osim.State
        The term's working State, realized to Position with this evaluation's parameter
        values already applied.
    Jq: np.ndarray
        The error Jacobian with respect to the independent generalized coordinates,
        shape ``(1, len(ModelCache.coordinate_indexes))``.
    dp_GB: osim.VectorVec3
        The error gradient with respect to each mobilized body's origin position in
        Ground, length `ModelCache.num_mobod`. Parameters that shift a body rigidly
        (body scales, mobilizer parameters) differentiate through this.
    doffset: np.ndarray
        The error gradient with respect to a shift of each task's station within its
        base frame, shape ``(num_tasks, 3)``. Parameters that move a station within its
        base frame (offsets, and the station-location part of a body scale)
        differentiate through this.
    tasks: Tasks
        The term's task table, supplying `station_caches`, `base_stations`,
        `offset_group_indexes`, `num_tasks`, and `offset_input`.
    """
    state: osim.State
    Jq: np.ndarray
    dp_GB: osim.VectorVec3
    doffset: np.ndarray
    tasks: 'Tasks'


#########
# TASKS #
#########

class Tasks(ABC):
    """
    A base class for task-specific storage and registration.

    Attributes
    ----------
    offset_input: str
        The `CostInput` field whose offset groups apply to these tasks. An
        `OffsetGroups` block contributes to a term only when this matches its
        `input_name`, which is what keeps marker offsets out of the frame-offset block
        and vice versa.
    """
    offset_input: str = None

    @abstractmethod
    def initialize_tasks(self, state: osim.State, **kwargs) -> float:
        pass


class MarkerTasks(Tasks):
    """
    Marker-specific task storage and registration.
    """
    offset_input = 'marker_offsets'

    def initialize_tasks(self):
        self.markers = []
        self.station_caches: list[StationCache] = []
        self.mobod_indexes = osim.SimTKArrayInt()
        self.stations = osim.SimTKArrayVec3()
        self.num_tasks: int = 0
        self.positions = []
        self.weights = []
        self.base_frames = []
        self.base_stations = []
        self.offset_group_indexes: list[int] = []

    def add_marker(self, marker_path: str, position: osim.Vec3, weight: float = 1.0,
                   offset_group_index: int | None = None):
        """
        Register a marker to track.

        Parameters
        ----------
        marker_path: str
            The OpenSim Model path to the tracking marker.
        position: osim.Vec3
            The reference position data tracked by the model marker.
        weight: float, optional
            The cost weight for the position error. Default: 1.0.
        offset_group_index: int | None, optional
            The index of the offset group whose XYZ offset applies to this marker, or
            ``None`` if this marker's placement is not offset. Default: ``None``.
        """
        if not self.mc.model.hasComponent(marker_path):
            raise ValueError(f'Model does not have a component at path {marker_path}.')
        if weight < 0:
            raise ValueError(f'Expected weight to be non-negative, but got {weight}.')

        self.mc.model.realizePosition(self.mc.state)
        marker = osim.Marker.safeDownCast(self.mc.model.getComponent(marker_path))
        cache = StationCache.from_station(self.mc, marker)
        self.station_caches.append(cache)
        self.markers.append(marker)
        self.mobod_indexes.push_back(cache.base_frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(*[float(v) for v in cache.base_station]))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(position.to_numpy())
        self.weights.append(weight)
        self.base_frames.append(cache.base_frame)
        self.base_stations.append(cache.base_station)
        self.offset_group_indexes.append(offset_group_index)


class FrameTasks(Tasks):
    """
    Frame-specific task storage and registration.
    """
    offset_input = 'frame_offsets'

    def initialize_tasks(self):
        self.frames = []
        self.station_caches: list[StationCache] = []
        self.mobod_indexes = osim.SimTKArrayInt()
        self.stations = osim.SimTKArrayVec3()
        self.num_tasks: int = 0
        self.positions = []
        self.orientations = []
        self.position_weights = []
        self.orientation_weights = []
        self.base_frames = []
        self.base_stations = []
        self.offset_group_indexes: list[int] = []

    def add_frame(self, frame_path: str, position: osim.Vec3,
                  orientation: osim.Quaternion, position_weight: float = 1.0,
                  orientation_weight: float = 1.0,
                  offset_group_index: int | None = None):
        """
        Register a frame to track.

        Parameters
        ----------
        frame_path: str
            The OpenSim Model path to the tracking frame.
        position: osim.Vec3
            The reference position data tracked by the model frame.
        orientation: osim.Quaternion
            The reference orientation, expressed as a quaternion, tracked by the model
            frame.
        position_weight: float, optional
            The cost weight for the position error. Default: 1.0.
        orientation_weight: float, optional
            The cost weight for the orientation error. Default: 1.0.
        offset_group_index: int | None, optional
            The index of the offset group whose XYZ offset applies to this frame, or
            ``None`` if this frame's placement is not offset. Default: ``None``.
        """
        if not self.mc.model.hasComponent(frame_path):
            raise ValueError(f'Model does not have a component at path {frame_path}.')
        if position_weight < 0:
            raise ValueError(f'Expected position_weight to be non-negative, but got '
                             f'{position_weight}.')
        if orientation_weight < 0:
            raise ValueError(f'Expected orientation_weight to be non-negative, but got '
                             f'{orientation_weight}.')

        frame = osim.PhysicalFrame.safeDownCast(self.mc.model.getComponent(frame_path))
        cache = StationCache.from_frame(self.mc, frame)
        self.station_caches.append(cache)
        self.frames.append(frame)
        self.mobod_indexes.push_back(cache.base_frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(*[float(v) for v in cache.base_station]))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(position.to_numpy())
        self.orientations.append(np.array([orientation.get(i) for i in range(4)]))
        self.position_weights.append(position_weight)
        self.orientation_weights.append(orientation_weight)
        self.base_frames.append(cache.base_frame)
        self.base_stations.append(cache.base_station)
        self.offset_group_indexes.append(offset_group_index)


##############
# COST TERMS #
##############

class TrackingTerm(ABC):
    """
    A base class for tracking cost terms that compute a scalar error and its
    derivatives with respect to a solver's optimization variables.

    Every term implements `calc_error`. For derivatives, a term over the generalized
    coordinates alone implements `calc_jacobian`, returning the coordinate Jacobian; a
    term that also supports bilevel parameters implements `calc_error_gradient`
    instead, returning the intermediate gradients each parameter block differentiates
    through. See `BilevelTerm`.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def calc_error(self, state: osim.State, **kwargs) -> float:
        pass


class FrameTrackingTerm(FrameTasks, TrackingTerm):
    """
    A tracking cost term that computes the aggregate error between model frames'
    positions and orientations and corresponding reference data as a function of the
    model's generalized coordinates. Individual frames are registered via add_frame().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
        error = 0.0
        for i, frame in enumerate(self.frames):
            p_model = frame.getPositionInGround(state).to_numpy()
            position_error = self.position_weights[i] * np.square(
                np.linalg.norm(p_model - self.positions[i]))

            eps = _calc_quaternion(state, frame)
            orientation_error = self.orientation_weights[i] * (
                1.0 - np.square(np.dot(eps, self.orientations[i])))

            error += position_error + orientation_error
        return error

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        if self.num_tasks == 0:
            return [np.zeros((1, len(self.mc.coordinate_indexes)))]

        # Loop over all frames and compute the "spatial error" (i.e., the combined
        # position and orientation error) for each.
        spatialError = osim.VectorOfSpatialVec(self.num_tasks, osim.SpatialVec(0))
        for i, frame in enumerate(self.frames):
            wp = self.position_weights[i]
            wo = self.orientation_weights[i]

            # Position error.
            p_model = frame.getPositionInGround(state)
            p_error = osim.Vec3(
                2.0 * wp * (p_model[0] - self.positions[i][0]),
                2.0 * wp * (p_model[1] - self.positions[i][1]),
                2.0 * wp * (p_model[2] - self.positions[i][2]))

            # Orientation error.
            eps = _calc_quaternion(state, frame)
            jac_eps = _calc_quaternion_jacobian(eps)
            omega = jac_eps.T @ self.orientations[i]
            scale = wo * -2.0 * np.dot(eps, self.orientations[i])
            w_error = osim.Vec3(scale * omega[0], scale * omega[1], scale * omega[2])

            # Combine the position and orientation into a SpatialVec to pass to the
            # frame Jacobian operator below.
            spatialError.set(i, osim.SpatialVec(w_error, p_error))

        # Calculate the frame (position and orientation) error Jacobian.
        vec = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, vec)

        return [_project_to_coordinates(self.mc, state, vec)]


class MarkerTrackingTerm(MarkerTasks, TrackingTerm):
    """
    A tracking cost term that computes the aggregate error between model markers'
    positions and corresponding reference positions as a function of the model's
    generalized coordinates. Individual markers are registered via add_marker().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
        error = 0.0
        for marker, position, weight in zip(
                self.markers, self.positions, self.weights):
            p_model = marker.getLocationInGround(state).to_numpy()
            error += weight * np.square(np.linalg.norm(p_model - position))
        return error

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        if self.num_tasks == 0:
            return [np.zeros((1, len(self.mc.coordinate_indexes)))]

        # Inialize the array used to calculate the position error Jacobian via the
        # grouped Simbody operator.
        f_GP = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        for i, (marker, position, weight) in enumerate(
                zip(self.markers, self.positions, self.weights)):
            p_model = marker.getLocationInGround(state)
            f_GP.set(i, osim.Vec3(
                2.0 * weight * (p_model[0] - position[0]),
                2.0 * weight * (p_model[1] - position[1]),
                2.0 * weight * (p_model[2] - position[2])))

        # Calculate the position error Jacobian.
        vec = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, f_GP, vec)

        return [_project_to_coordinates(self.mc, state, vec)]


class BilevelTerm(TrackingTerm):
    """
    An intermediate base class for tracking cost terms whose error depends on bilevel
    parameters as well as on the generalized coordinates.

    Rather than assembling a Jacobian block per parameter type, a bilevel term computes
    the two intermediate gradients that every parameter block differentiates through
    and returns them in an `ErrorGradient`. Subclasses (via `MarkerTasks`/`FrameTasks`)
    supply the per-task `station_caches`, `stations`, and `offset_group_indexes` that
    the gradient carries.
    """
    def __init__(self):
        super().__init__()

    def calc_zero_gradient(self, state: osim.State) -> 'ErrorGradient':
        """
        The `ErrorGradient` of a term with no registered tasks: every block is zero.
        """
        return ErrorGradient(
            state=state,
            Jq=np.zeros((1, len(self.mc.coordinate_indexes))),
            dp_GB=osim.VectorVec3(self.mc.num_mobod, osim.Vec3(0)),
            doffset=np.zeros((0, 3)),
            tasks=self)

    def calc_station_gradient(self, state: osim.State,
                              dp_GS: osim.VectorVec3) -> np.ndarray:
        """
        Return the per-task sensitivity of the error to a shift of each task's station
        within its base frame, shape ``(num_tasks, 3)``, given the per-task error
        gradient in Ground.
        """
        doffset = np.zeros((self.num_tasks, 3))
        for i, base_frame in enumerate(self.base_frames):
            rotation = base_frame.getRotationInGround(state)
            R_GB = np.array([[rotation.get(r, c) for c in range(3)] for r in range(3)])
            doffset[i] = dp_GS.get(i).to_numpy() @ R_GB
        return doffset

    def scatter_to_bodies(self, dp_GS: osim.VectorVec3) -> osim.VectorVec3:
        """
        Scatter the per-task error gradient into a per-mobilized-body gradient with
        respect to body origin positions in Ground.

        Every parameter that shifts a body rigidly differentiates through this vector.
        Because such a shift translates a station on that body identically and applies
        no rotation, ``dp_GS_i / dp_GB[k_i] = I``, so the scatter is just

            dp_GB[k] += dp_GS[i]   # for each task i on body k
        """
        dp_GB = osim.VectorVec3(self.mc.num_mobod, osim.Vec3(0))
        for i in range(self.num_tasks):
            k = int(self.mobod_indexes.getElt(i))
            total = dp_GB.get(k).to_numpy() + dp_GS.get(i).to_numpy()
            dp_GB.set(k, osim.Vec3(
                float(total[0]), float(total[1]), float(total[2])))
        return dp_GB

    @abstractmethod
    def calc_error_gradient(self, state: osim.State) -> 'ErrorGradient':
        """
        Return this term's `ErrorGradient` at `state`, which must already be realized
        to Position with the current parameter values applied.
        """


class MarkerBilevelTerm(MarkerTasks, BilevelTerm):
    """
    A tracking cost term that computes the aggregate error between model markers' scaled
    positions and corresponding reference positions as a function of the model's
    generalized coordinates and body scales. Individual markers are registered via
    add_marker().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
        error = 0.0
        for i, (frame, position, weight) in enumerate(
                zip(self.base_frames, self.positions, self.weights)):
            p_model = frame.findStationLocationInGround(
                state, self.stations.getElt(i)).to_numpy()
            error += weight * np.square(np.linalg.norm(p_model - position))
        return error

    def calc_error_gradient(self, state) -> ErrorGradient:
        if self.num_tasks == 0:
            return self.calc_zero_gradient(state)

        # Calculate the per-marker error gradient in Ground. This is a force-like term
        # that will be multiplied with (the transpose of) each position Jacobian.
        dp_GS = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        for i, (frame, position, weight) in enumerate(
                zip(self.base_frames, self.positions, self.weights)):
            p_GS = frame.findStationLocationInGround(state, self.stations.getElt(i))
            dp_GS.set(i, osim.Vec3(2.0 * weight * (p_GS[0] - position[0]),
                                   2.0 * weight * (p_GS[1] - position[1]),
                                   2.0 * weight * (p_GS[2] - position[2])))

        # Calculate the Jacobian of the position error with respect to the coordinates.
        vec = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, dp_GS, vec)
        Jq = _project_to_coordinates(self.mc, state, vec)

        return ErrorGradient(
            state=state,
            Jq=Jq,
            dp_GB=self.scatter_to_bodies(dp_GS),
            doffset=self.calc_station_gradient(state, dp_GS),
            tasks=self)


class FrameBilevelTerm(FrameTasks, BilevelTerm):
    """
    A tracking cost term that computes the aggregate error between model frames' scaled
    positions and corresponding reference positions as a function of the model's
    generalized coordinates and body scales. Individual frames are registered via
    add_frame().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
        error = 0.0
        for i, (frame, base_frame) in enumerate(zip(self.frames, self.base_frames)):
            p_model = base_frame.findStationLocationInGround(
                state, self.stations.getElt(i)).to_numpy()
            position_error = self.position_weights[i] * np.square(
                np.linalg.norm(p_model - self.positions[i]))

            eps = _calc_quaternion(state, frame)
            orientation_error = self.orientation_weights[i] * (
                1.0 - np.square(np.dot(eps, self.orientations[i])))

            error += position_error + orientation_error
        return error

    def calc_error_gradient(self, state) -> ErrorGradient:
        if self.num_tasks == 0:
            return self.calc_zero_gradient(state)

        # Loop over all frames and compute the "spatial error" (i.e., the combined
        # position and orientation error) for each. Store the position-error gradient
        # along the way, since the parameter blocks differentiate through it.
        spatialError = osim.VectorOfSpatialVec(self.num_tasks, osim.SpatialVec(0))
        dp_GF = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        for i, (frame, base_frame) in enumerate(zip(self.frames, self.base_frames)):
            wp = self.position_weights[i]
            wo = self.orientation_weights[i]
            position = self.positions[i]

            # The frame's ground position is computed from its (possibly offset) cached
            # station so that the gradient is consistent with any applied offsets.
            p_GF = base_frame.findStationLocationInGround(
                state, self.stations.getElt(i))
            dp_GF.set(i, osim.Vec3(2.0 * wp * (p_GF[0] - position[0]),
                                   2.0 * wp * (p_GF[1] - position[1]),
                                   2.0 * wp * (p_GF[2] - position[2])))

            # Calculate the per-frame orientation error in Ground. No bilevel parameter
            # rotates a frame, so the orientation error enters the coordinate Jacobian
            # only and is absent from the gradients the parameters consume.
            eps = _calc_quaternion(state, frame)
            jac_eps = _calc_quaternion_jacobian(eps)
            omega = jac_eps.T @ self.orientations[i]
            scale = wo * -2.0 * np.dot(eps, self.orientations[i])
            dw_GF = osim.Vec3(scale * omega[0], scale * omega[1], scale * omega[2])

            # Combine the position and orientation into a SpatialVec to pass to the
            # frame Jacobian operator below.
            spatialError.set(i, osim.SpatialVec(dw_GF, dp_GF.get(i)))

        # Calculate the frame (position and orientation) error Jacobian.
        vec = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, vec)
        Jq = _project_to_coordinates(self.mc, state, vec)

        return ErrorGradient(
            state=state,
            Jq=Jq,
            dp_GB=self.scatter_to_bodies(dp_GF),
            doffset=self.calc_station_gradient(state, dp_GF),
            tasks=self)


##################
# COST FUNCTIONS #
##################

class TrackingCost(TrackingCostBase):
    """
    The weighted, squared error between the model's markers and frames and a trial's
    reference data, as a function of the model's generalized coordinates.

    Parameters
    ----------
    position_weight: float, optional
        Weight applied to marker and frame-origin position errors. Default is 1.0.
    orientation_weight: float, optional
        Weight applied to frame orientation errors. Default is 1.0.
    """
    required_inputs = frozenset({'coordinates'})

    def __init__(self, position_weight: float = 1.0,
                 orientation_weight: float = 1.0):
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

    def create_rep(self, name: str, mc: ModelCache, trial: Trial,
                   itime: int) -> 'TrackingCostRep':
        rep = TrackingCostRep(name, mc)

        for data in trial.frame_data:
            for iframe, frame_path in enumerate(data.labels):
                rep.add_frame_tracking_cost_term(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=self.position_weight,
                    orientation_weight=self.orientation_weight)

        for data in trial.marker_data:
            for imarker, marker_path in enumerate(data.labels):
                rep.add_marker_tracking_cost_term(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, imarker),
                    weight=self.position_weight)

        return rep


class TrackingCostRep(CallbackCostRep):
    """
    The rep of a `TrackingCost`: a callback that evaluates the sum of tracking cost
    terms over a set of model frames and markers with respect to the model's
    generalized coordinates.

    Parameters
    ----------
    name: str
        The name of the callback function.
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its analytic
        Jacobian. Default is ``False``.
    """
    def __init__(self, name: str, mc: ModelCache, enable_fd: bool = False):
        Function.__init__(self, name, mc, enable_fd=enable_fd)
        self.marker_term = MarkerTrackingTerm(mc)
        self.frame_term = FrameTrackingTerm(mc)

    def apply_state(self, arg):
        """
        Apply the input coordinates to the model state and realize the system to the
        position stage.
        """
        q = np.zeros(self.state.getNQ())
        q[self.mc.coordinate_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.mc.model.realizePosition(self.state)

    def add_marker_tracking_cost_term(self, marker_path: str, position: osim.Vec3,
                                      weight: float = 1.0):
        self.marker_term.add_marker(marker_path, position, weight=weight)

    def add_frame_tracking_cost_term(self, frame_path: str,
                                     position: osim.Vec3,
                                     orientation: osim.Quaternion,
                                     position_weight: float = 1.0,
                                     orientation_weight: float = 1.0):
        self.frame_term.add_frame(frame_path, position, orientation,
                                  position_weight=position_weight,
                                  orientation_weight=orientation_weight)

    def _eval(self, arg):
        self.apply_state(arg)
        error = (self.marker_term.calc_error(self.state) +
                 self.frame_term.calc_error(self.state))
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        J = (self.marker_term.calc_jacobian(self.state)[0] +
             self.frame_term.calc_jacobian(self.state)[0])
        return [J] + self._empty_parameter_jacobians()


class BilevelCost(TrackingCostBase):
    """
    The tracking cost of `TrackingCost`, as a function of the model's generalized
    coordinates, its body scales, and its per-marker/frame XYZ placement offsets.

    Parameters
    ----------
    position_weight: float, optional
        Weight applied to marker and frame-origin position errors. Default is 1.0.
    orientation_weight: float, optional
        Weight applied to frame orientation errors. Default is 1.0.
    """
    required_inputs = frozenset(
        {'coordinates', 'body_scales', 'marker_offsets', 'frame_offsets'})

    def __init__(self, position_weight: float = 1.0,
                 orientation_weight: float = 1.0):
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

    def create_rep(self, name: str, mc: ModelCache, trial: Trial,
                   itime: int) -> 'BilevelCostRep':
        rep = BilevelCostRep(name, mc)
        # Map each offset target path to the index of the offset group that applies to
        # it; paths absent from a mapping are not offset.
        marker_index_of = {path: i for i, grp in enumerate(mc.marker_offset_groups)
                           for path in grp.component_paths}
        frame_index_of = {path: i for i, grp in enumerate(mc.frame_offset_groups)
                          for path in grp.component_paths}

        for data in trial.frame_data:
            for iframe, frame_path in enumerate(data.labels):
                rep.add_frame_bilevel_cost_term(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=self.position_weight,
                    orientation_weight=self.orientation_weight,
                    offset_group_index=frame_index_of.get(frame_path))

        for data in trial.marker_data:
            for imarker, marker_path in enumerate(data.labels):
                rep.add_marker_bilevel_cost_term(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, imarker),
                    weight=self.position_weight,
                    offset_group_index=marker_index_of.get(marker_path))

        return rep


class BilevelCostRep(CallbackCostRep):
    """
    The rep of a `BilevelCost`: a callback that evaluates the sum of tracking cost
    terms over a set of model markers and frames with respect to the model's generalized
    coordinates, a set of body scales, and a set of per-marker/frame XYZ placement
    offsets.

    Parameters
    ----------
    name: str
        The name of the callback function.
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information. Contains parameter information
        (e.g., body scale groups) for relevant optimization parameters.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its analytic
        Jacobian. Default is ``False``.
    """

    def __init__(self, name: str, mc: ModelCache, enable_fd: bool = False):
        Function.__init__(self, name, mc, enable_fd=enable_fd)
        self.marker_term = MarkerBilevelTerm(mc)
        self.frame_term = FrameBilevelTerm(mc)
        self.mc.cache_body_scale_group_joints()
        # The parameter values last composed onto this rep's station tables, keyed by
        # `CostInput` field name; ``None`` until the first composition. See
        # `apply_state`.
        self._composed_parameters: dict[str, np.ndarray] = None

    @property
    def terms(self) -> tuple[BilevelTerm, ...]:
        """
        This rep's tracking terms, whose errors and gradients sum to the rep's own.
        """
        return (self.marker_term, self.frame_term)

    def apply_state(self, arg):
        """
        Apply the input coordinates and every parameter block to the model State and to
        each term's station table, then realize to Position.

        Blocks are composed in `CostInput.INPUT_ORDER`, which body-scale and offset
        composition relies on: a body scale sets each station absolutely from its
        base-frame location, and an offset then adds to that result.

        Neither the State write nor the station composition depends on the
        coordinates, and the parameter blocks are shared across every trial and time
        sample, so both are skipped when this rep's inputs still carry the parameter
        values already in place. That matters because a bilevel solve holds one rep per
        trial time sample and evaluates all of them at the same parameter values, twice
        per iteration (once for the objective and once for its gradient). The State
        write is memoized on the `ModelCache`, since every rep shares one State; the
        station composition is memoized here, since the station table is this rep's
        own. Composition is all-or-nothing: an offset adds to whatever the body scales
        left behind, so re-running one block alone would compound onto its own result.
        """
        values = {}
        for i, name in enumerate(CostInput.INPUT_ORDER):
            if name == 'coordinates':
                continue
            values[name] = np.atleast_1d(np.squeeze(arg[i].full())).astype(float)

        self.mc.apply_parameters_to_state(self.state, values)

        if self._composed_parameters is None or any(
                not np.array_equal(self._composed_parameters[name], value)
                for name, value in values.items()):
            for name, value in values.items():
                groups = self.mc.parameter_groups[name]
                for term in self.terms:
                    groups.apply_to_tasks(term, value)
            self._composed_parameters = {
                name: value.copy() for name, value in values.items()}

        q = np.zeros(self.state.getNQ())
        q[self.mc.coordinate_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.mc.model.realizePosition(self.state)

    def add_marker_bilevel_cost_term(self, marker_path: str, position: osim.Vec3,
                                     weight: float = 1.0,
                                     offset_group_index: int | None = None):
        self.marker_term.add_marker(marker_path, position, weight=weight,
                                    offset_group_index=offset_group_index)

    def add_frame_bilevel_cost_term(self, frame_path: str, position: osim.Vec3,
                                    orientation: osim.Quaternion,
                                    position_weight: float = 1.0,
                                    orientation_weight: float = 1.0,
                                    offset_group_index: int | None = None):
        self.frame_term.add_frame(frame_path, position, orientation, position_weight,
                                  orientation_weight,
                                  offset_group_index=offset_group_index)

    def _eval(self, arg):
        self.apply_state(arg)
        error = 0
        error += self.marker_term.calc_error(self.state)
        error += self.frame_term.calc_error(self.state)
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)

        # Each term computes its intermediate gradients once; each parameter block then
        # projects every term's gradient onto itself and the contributions are summed.
        gradients = [term.calc_error_gradient(self.state) for term in self.terms]

        Jq = sum(gradient.Jq for gradient in gradients)
        blocks = []
        for name in CostInput.PARAMETER_INPUTS:
            groups = self.mc.parameter_groups[name]
            block = np.zeros((1, groups.num_variables))
            for gradient in gradients:
                block += groups.calc_jacobian_block(gradient)
            blocks.append(block)

        return [Jq] + blocks


class AnthropometricRegularizationCost(Cost):
    """
    A regularization penalty on body-scale factors, ``s``, that maximizes the
    log-likelihood that a set of anthropometric measurements, ``m(s)``, fall within a
    distribution fit to the ANSUR II dataset. Since it is a multivariate normal
    distribution, we use the Mahalanobis distance to define the cost:

        cost = weight * 0.5 (m(s) - μ)^T Σ^-1 (m(s) - μ)

    which equates to minimizing the negative log-likelihood of the probability density
    function.

    Users must define the set of measurements, ``m(s)``, via the parameter
    `measurements`, a list of `AnthropometricMeasurement`. Each
    `AnthropometricMeasurement` is named after a measurement in the ANSUR II dataset
    and defines the two `Station`s on the `Model` (and optionally an axis) from which
    to compute the simulated measurement. The `sex`
    parameter can be used to specify that the distribution should be fit to either
    male or female participants only; using the default value (`None`) fits across all
    participants from the ANSUR report.

    All quantities are in meters (ANSUR II millimeters are converted on load).

    Parameters
    ----------
    measurements: list[AnthropometricMeasurement]
        The measurements (each a station pair and optional axis) to compute from the
        model. Each measurement's `name` must match a measurement from the ANSUR II
        dataset.
    sex: str, optional
        Subject sex ('male' or 'female') selecting the ANSUR II subset. Defaults to None
        (the combined male-and-female dataset).
    weight: float, optional
        Non-negative scalar applied to the penalty. Default is 1.0.

    Raises
    ------
    ValueError
        If `weight` is negative or a measurement name is not present in the ANSUR II
        dataset. `AnthropometricRegularizationCostRep` additionally validates that the
        referenced components are stations.
    """
    required_inputs = frozenset({'body_scales'})

    def __init__(self, measurements: list[AnthropometricMeasurement],
                 sex: str = None, weight: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.measurements = measurements

        # Fit the ANSUR II distribution over the requested measurements, in meters.
        measurement_names = [m.name for m in self.measurements]
        distribution = build_ansur_distribution(measurement_names, sex)
        self.mean = np.asarray(distribution.get_mean(), dtype=float).reshape(-1)
        self.precision = np.linalg.inv(
            np.asarray(distribution.get_covariance(), dtype=float))

    def create_rep(self, mc: ModelCache) -> 'AnthropometricRegularizationCostRep':
        return AnthropometricRegularizationCostRep(self, mc)


class AnthropometricRegularizationCostRep(CallbackCostRep):
    """
    The rep of an `AnthropometricRegularizationCost`. It caches the model's default
    pose and a `StationCache` pair per measurement, then evaluates the Mahalanobis
    penalty and its gradient through OpenSim.

    Parameters
    ----------
    cost: AnthropometricRegularizationCost
        The cost this rep represents.
    mc: ModelCache
        The solver's `ModelCache`, whose registered body scale groups set this
        callback's input size.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback rather than using its
        analytic Jacobian. Default is ``False``.

    Raises
    ------
    ValueError
        If a measurement references a component that is not an `osim.Station`.
    """

    def __init__(self, cost: AnthropometricRegularizationCost, mc: ModelCache,
                 enable_fd: bool = False):
        self.cost = cost
        mc.model.realizePosition(mc.state)
        self.default_q = mc.state.getQ().to_numpy().copy()
        mc.cache_body_scale_group_joints()
        self.station_caches = []
        for measurement in cost.measurements:
            sc1 = StationCache.from_station(
                mc, mc.model.getComponent(measurement.station1_path))
            sc2 = StationCache.from_station(
                mc, mc.model.getComponent(measurement.station2_path))
            axis = measurement.axis.value if measurement.axis is not None else None
            self.station_caches.append((sc1, sc2, axis))
        Function.__init__(self, 'anthropometric_regularization_cost', mc,
                          enable_fd=enable_fd)

    def __call__(self, input: CostInput) -> ca.MX:
        return ca.Function.__call__(self, input.body_scales)

    def _get_num_inputs(self):
        return 1

    def _get_input_size(self, i):
        if i == 0:
            return 3 * len(self.mc.body_scale_groups)
        raise IndexError(f'Invalid input index {i} for {type(self).__name__}.')

    def _apply_body_scales(self, body_scales: np.ndarray) -> None:
        # Routed through the ModelCache so this rep shares the memoized State write
        # with the tracking reps, which apply the same body scales to the same State.
        self.mc.apply_parameters_to_state(
            self.state, {'body_scales': body_scales})
        self.state.setQ(osim.Vector.createFromMat(self.default_q))
        self.mc.model.realizePosition(self.state)

    def _eval(self, arg):
        body_scales = np.atleast_1d(np.squeeze(arg[0].full())).astype(float)
        self._apply_body_scales(body_scales)

        measurements = np.empty(len(self.station_caches))
        for i, (sc1, sc2, axis) in enumerate(self.station_caches):
            pos1 = sc1.calc_position(self.state, body_scales).to_numpy()
            pos2 = sc2.calc_position(self.state, body_scales).to_numpy()
            displacement = pos2 - pos1
            measurements[i] = (np.linalg.norm(displacement) if axis is None
                               else abs(displacement[axis]))

        residual = measurements - self.cost.mean
        return [float(self.cost.weight * 0.5 * residual
                      @ self.cost.precision @ residual)]

    def _jac_eval(self, arg):
        body_scales = np.atleast_1d(np.squeeze(arg[0].full())).astype(float)
        self._apply_body_scales(body_scales)

        num_scales = 3 * len(self.mc.body_scale_groups)
        m = np.empty(len(self.station_caches))
        jacobian = np.zeros((len(self.station_caches), num_scales))
        for i, (sc1, sc2, axis) in enumerate(self.station_caches):
            pos1 = sc1.calc_position(self.state, body_scales).to_numpy()
            pos2 = sc2.calc_position(self.state, body_scales).to_numpy()
            displacement = pos2 - pos1

            jac1 = sc1.calc_position_jacobian_wrt_body_scales(self.state)
            jac2 = sc2.calc_position_jacobian_wrt_body_scales(self.state)
            displacement_jacobian = jac2 - jac1

            if axis is None:
                norm = np.linalg.norm(displacement)
                m[i] = norm
                if norm > 0.0:
                    jacobian[i, :] = (displacement / norm) @ displacement_jacobian
            else:
                value = displacement[axis]
                m[i] = abs(value)
                jacobian[i, :] = np.sign(value) * displacement_jacobian[axis, :]
        residual = m - self.cost.mean
        gradient = self.cost.weight * (self.cost.precision @ residual) @ jacobian
        return [gradient.reshape(1, num_scales)]
