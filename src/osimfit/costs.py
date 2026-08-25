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
    """
    INPUT_ORDER: ClassVar[tuple[str, ...]] = (
        'coordinates', 'body_scales', 'marker_offsets', 'frame_offsets')
    TRIPLET_INPUTS: ClassVar[tuple[str, ...]] = (
        'body_scales', 'marker_offsets', 'frame_offsets')

    coordinates: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    body_scales: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    marker_offsets: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    frame_offsets: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))

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
        sizes = {
            'coordinates': len(self.mc.coordinate_indexes),
            'body_scales': 3 * len(self.mc.body_scale_groups),
            'marker_offsets': 3 * len(self.mc.marker_offset_groups),
            'frame_offsets': 3 * len(self.mc.frame_offset_groups),
        }
        order = CostInput.INPUT_ORDER
        if not 0 <= i < len(order):
            raise IndexError(f'Invalid input index {i} for {type(self).__name__}.')
        return sizes[order[i]]

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


###########
# HELPERS #
###########

def _calc_quaternion(state, frame):
    rotation = frame.getRotationInGround(state)
    quaternion = rotation.convertRotationToQuaternion()
    return np.array([quaternion.get(i) for i in range(4)])

def _calc_quaternion_jacobian(eps):
    # Simbody -> /SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L712
    e = 0.5 * eps
    return np.array([
        [-e[1], -e[2], -e[3]],
        [ e[0],  e[3], -e[2]],
        [-e[3],  e[0],  e[1]],
        [ e[2], -e[1],  e[0]],
    ])

#########
# TASKS #
#########

class Tasks(ABC):
    """
    A base class for task-specific storage and registration.
    """
    @abstractmethod
    def initialize_tasks(self, state: osim.State, **kwargs) -> float:
        pass


class MarkerTasks(Tasks):
    """
    Marker-specific task storage and registration.
    """
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
    Jacobian with respect to the model's generalized coordinates, body scales,
    and other optimization variables. To implement a new tracking cost term, extend this
    class and implement the abstract methods (calc_error, calc_jacobian) to compute the
    error and its Jacobian.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def calc_error(self, state: osim.State, **kwargs) -> float:
        pass

    @abstractmethod
    def calc_jacobian(self, state: osim.State, **kwargs) -> list[np.ndarray]:
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
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, vec)
        J = vec.to_numpy()

        return [np.expand_dims(J[self.mc.coordinate_indexes], axis=0)]


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
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, f_GP, vec)

        return [np.expand_dims(vec.to_numpy()[self.mc.coordinate_indexes], axis=0)]


class BilevelTerm(TrackingTerm):
    """
    An intermediate base class that provides functionality common to bilevel cost terms.

    Applying body scales and placement offsets to the cached station locations is shared
    across marker and frame terms; subclasses (via `MarkerTasks`/`FrameTasks`) supply the
    per-task `station_caches`, `stations`, and `offset_group_indexes`.
    """
    def __init__(self):
        super().__init__()

    def apply_scales(self, body_scales: np.ndarray) -> None:
        for itask, cache in enumerate(self.station_caches):
            s = cache.calc_scaled_base_station(body_scales)
            self.stations.updElt(itask).set(0, float(s[0]))
            self.stations.updElt(itask).set(1, float(s[1]))
            self.stations.updElt(itask).set(2, float(s[2]))

    def apply_offsets(self, offsets: np.ndarray) -> None:
        for i, g in enumerate(self.offset_group_indexes):
            if g is None:
                continue
            o = np.asarray(offsets[3*g : 3*g+3], dtype=float)
            s = self.stations.getElt(i).to_numpy() + o
            self.stations.updElt(i).set(0, float(s[0]))
            self.stations.updElt(i).set(1, float(s[1]))
            self.stations.updElt(i).set(2, float(s[2]))

    def apply_state(self, body_scales: np.ndarray, offsets: np.ndarray) -> None:
        self.apply_scales(body_scales)
        self.apply_offsets(offsets)


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

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.coordinate_indexes)))
        Js = np.zeros((1, 3 * len(self.mc.body_scale_groups)))
        Jo = np.zeros((1, 3 * len(self.mc.marker_offset_groups)))
        if self.num_tasks == 0:
            return [Jq, Js, Jo]

        # Calculate the per-marker error gradient in Ground. This is a force-like term
        # will be multiplied with (the transpose of) each position Jacobian below. Also,
        # precompute the sensitivity of each marker's ground position to a shift from
        # an offset variable.
        dp_GS = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        doffset = np.zeros((self.num_tasks, 3))
        for i, (frame, position, weight) in enumerate(
                zip(self.base_frames, self.positions, self.weights)):
            p_GS = frame.findStationLocationInGround(state, self.stations.getElt(i))
            dp_GS.set(i, osim.Vec3(2.0 * weight * (p_GS[0] - position[0]),
                                   2.0 * weight * (p_GS[1] - position[1]),
                                   2.0 * weight * (p_GS[2] - position[2])))
            rotation = frame.getRotationInGround(state)
            R_GB = np.array([[rotation.get(r, c) for c in range(3)] for r in range(3)])
            doffset[i] = dp_GS.get(i).to_numpy() @ R_GB

        # Calculate the Jacobian of the position error with respect to the coordinates.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, dp_GS, vec)
        Jq[0, :] = vec.to_numpy()[self.mc.coordinate_indexes]

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GS_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GS.get(i)   # for each marker i on body k
        #
        dp_GB = osim.VectorVec3(self.mc.num_mobod, osim.Vec3(0))
        for i in range(self.num_tasks):
            k = int(self.mobod_indexes.getElt(i))
            cur = dp_GB.get(k).to_numpy() + dp_GS.get(i).to_numpy()
            dp_GB.set(k, osim.Vec3(float(cur[0]), float(cur[1]), float(cur[2])))

        # Calculate the position-error Jacobian with respect to body scales.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB)

        # Assemble the marker offset Jacobian based on the offset sensitivities. Also,
        # include the contributions from the marker offsets to the Jacobian with respect
        # to body scales.
        for i in range(self.num_tasks):
            g_off = self.offset_group_indexes[i]
            g_scale = self.station_caches[i].body_scale_group_index
            if g_off is None and g_scale is None:
                continue
            if g_off is not None:
                Jo[0, 3*g_off:3*g_off+3] += doffset[i]
            if g_scale is not None:
                Js[0, 3*g_scale:3*g_scale+3] += self.base_stations[i] * doffset[i]

        return [Jq, Js, Jo]


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

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.coordinate_indexes)))
        Js = np.zeros((1, 3 * len(self.mc.body_scale_groups)))
        Jo = np.zeros((1, 3 * len(self.mc.frame_offset_groups)))
        if self.num_tasks == 0:
            return [Jq, Js, Jo]

        # Loop over all frames and compute the "spatial error" (i.e., the combined
        # position and orientation error) for each.
        spatialError = osim.VectorOfSpatialVec(self.num_tasks, osim.SpatialVec(0))
        # Store the position-error gradient along the way. We need it for the body scale
        # and offset Jacobian calculations. Also, precompute the sensitivity of each
        # frame's ground position to a shift from an offset variable.
        dp_GF = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        doffset = np.zeros((self.num_tasks, 3))
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

            # Calculate the per-frame orientation error in Ground.
            eps = _calc_quaternion(state, frame)
            jac_eps = _calc_quaternion_jacobian(eps)
            omega = jac_eps.T @ self.orientations[i]
            scale = wo * -2.0 * np.dot(eps, self.orientations[i])
            dw_GF = osim.Vec3(scale * omega[0], scale * omega[1], scale * omega[2])

            # Combine the position and orientation into a SpatialVec to pass to the
            # frame Jacobian operator below.
            spatialError.set(i, osim.SpatialVec(dw_GF, dp_GF.get(i)))

            # Precompute the position sensitivity to a base-frame station shift.
            rotation = base_frame.getRotationInGround(state)
            R_GB = np.array([[rotation.get(r, c) for c in range(3)] for r in range(3)])
            doffset[i] = dp_GF.get(i).to_numpy() @ R_GB

        # Calculate the frame (position and orientation) error Jacobian.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, vec)
        Jq[0, :] = vec.to_numpy()[self.mc.coordinate_indexes]

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GF_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GF.get(i)   # for each frame i on body k
        #
        dp_GB = osim.VectorVec3(self.mc.num_mobod, osim.Vec3(0))
        for i in range(self.num_tasks):
            k = int(self.mobod_indexes.getElt(i))
            cur = dp_GB.get(k).to_numpy() + dp_GF.get(i).to_numpy()
            dp_GB.set(k, osim.Vec3(float(cur[0]), float(cur[1]), float(cur[2])))

        # Calculate the position-error Jacobian with respect to body scales. This does
        # not include the contributions from frame offsets, we will include that below.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB)

        # Assemble the frame offset Jacobian based on the offset sensitivities. Also,
        # include the contributions from the frame offsets to the Jacobian with respect
        # to body scales.
        for i in range(self.num_tasks):
            g_off = self.offset_group_indexes[i]
            g_scale = self.station_caches[i].body_scale_group_index
            if g_off is None and g_scale is None:
                continue
            if g_off is not None:
                Jo[0, 3*g_off:3*g_off+3] += doffset[i]
            if g_scale is not None:
                Js[0, 3*g_scale:3*g_scale+3] += self.base_stations[i] * doffset[i]

        return [Jq, Js, Jo]


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
        empty = np.zeros((1, 0))
        return [J, empty, empty, empty]


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

    def apply_state(self, arg):
        """
        Apply input coordinates, body-scale variables, and offset variables to the
        model State, then realize to Position.
        """
        body_scales = np.squeeze(arg[1].full())
        body_scales = np.atleast_1d(body_scales).astype(float)
        self.mc.set_scaled_mobilizer_frame_positions(self.state, body_scales)

        marker_offsets = np.atleast_1d(np.squeeze(arg[2].full())).astype(float)
        self.marker_term.apply_state(body_scales, marker_offsets)

        frame_offsets = np.atleast_1d(np.squeeze(arg[3].full())).astype(float)
        self.frame_term.apply_state(body_scales, frame_offsets)

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
        Jq_m, Js_m, Jmo = self.marker_term.calc_jacobian(self.state)
        Jq_f, Js_f, Jfo = self.frame_term.calc_jacobian(self.state)
        return [Jq_m + Jq_f, Js_m + Js_f, Jmo, Jfo]


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
        self.mc.set_scaled_mobilizer_frame_positions(self.state, body_scales)
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
