import copy
import numpy as np
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .bounds import Bounds


##########
# GROUPS #
##########

@dataclass
class BodyScaleGroup:
    """
    A group of mobilized bodies sharing one set of XYZ body scales. The group
    defines the list of OpenSim body paths and corresponding mobilized body indexes for
    each set of body scales.

    Attributes
    ----------
    body_paths: list[str]
        Absolute model paths to the bodies in this group.
    mobod_indexes: list[int]
        `MobilizedBodyIndex` values for the bodies in this group, paired with
        body_paths.
    inboard_joints: list[osim.Joint]
        A list of `Joint`s whose inboard frames correspond to the `MobilizedBodyIndex`
        values in `mobod_indexes`.
    outboard_joints: list[osim.Joint]
        A list of `Joint`s whose outboard frames correspond to the `MobilizedBodyIndex`
        values in `mobod_indexes`.
    """
    body_paths: list[str]
    mobod_indexes: list[int]
    inboard_joints: list[osim.Joint] = field(default_factory=list, compare=False)
    outboard_joints: list[osim.Joint] = field(default_factory=list, compare=False)


@dataclass
class OffsetGroup:
    """
    A group of markers or frames sharing one set of XYZ offsets. The offset is an
    additive translation, expressed in each component's base frame, applied to the
    component's placement (a marker's location or a frame's translation).

    Attributes
    ----------
    component_paths: list[str]
        Absolute model paths to the markers or frames in this group.
    """
    component_paths: list[str]
    mobod_indexes: list[int]


@dataclass
class MarkerOffsetGroup(OffsetGroup):
    """An `OffsetGroup` whose components are markers (offsets a marker's location)."""


@dataclass
class FrameOffsetGroup(OffsetGroup):
    """An `OffsetGroup` whose components are frames (offsets a frame's translation)."""


###############
# MODEL CACHE #
###############

class ModelCache:
    """
    A thin wrapper around `osim.Model` that pre-computes and caches lookups
    used repeatedly by solvers and callback functions. It also provides useful methods
    for complicated calculations used by solvers (e.g., converting gradients with
    respect to body scales).

    Parameters
    ----------
    model: str or osim.Model
        The OpenSim model to use for the optimization problem.

    Attributes
    ----------
    model: osim.Model
        The wrapped OpenSim model.
    state: osim.State
        The model's working state (snapshot at construction time).
    matter: osim.SimbodyMatterSubsystem
        The cached matter subsystem reference.
    num_mobod: int
        Total Simbody mobod count, including Ground at index 0.
    coordinate_map: dict[str, int]
        Mapping from absolute coordinate path to its q-index in the State,
        restricted to independent coordinates (e.g., coupled coordinates are
        excluded).
    coordinate_indexes: list[int]
        The q-indexes of the independent coordinates, in registration order.
    body_scale_groups: list[BodyScaleGroup]
        The list of BodyScaleGroups associated with this model.
    marker_offset_groups: list[MarkerOffsetGroup]
        The list of MarkerOffsetGroups associated with this model.
    frame_offset_groups: list[FrameOffsetGroup]
        The list of FrameOffsetGroups associated with this model.
    parent_of: dict[int, int]
        Per-mobod parent in the multibody tree. ``parent_of[k]`` is the
        ``MobilizedBodyIndex`` of body ``k``'s parent (Ground has no entry).
    children_of: dict[int, list[int]]
        Inverse of ``parent_of``: ``children_of[k]`` is the list of mobod
        indexes whose parent is ``k``. Every mobod (including Ground at 0)
        has an entry, possibly empty.
    """
    def __init__(self, model: str | osim.Model):
        modelProcessor = osim.ModelProcessor(model)
        self.model = modelProcessor.process()
        self.state = self.model.initSystem()
        self.matter = self.model.getMatterSubsystem()
        self.num_mobod = self.model.getNumBodies() + 1
        self.coordinate_map = self._get_coordinate_index_map(self.model,
                                                    skip_dependent_coordinates=True)
        self.coordinate_indexes = list(self.coordinate_map.values())
        self.body_scale_groups: list[BodyScaleGroup] = []
        self.marker_offset_groups: list[MarkerOffsetGroup] = []
        self.frame_offset_groups: list[FrameOffsetGroup] = []

        # For now, disallow models with joints where qdot != u.
        assert(self.state.getNQ() == self.state.getNU())

        # Mobilized body parents.
        self.parent_of: dict[int, int] = {}
        for i in range(self.model.getNumJoints()):
            joint = self.model.getJointSet().get(i)
            cix = int(joint.getChildFrame().getMobilizedBodyIndex())
            pix = int(joint.getParentFrame().getMobilizedBodyIndex())
            self.parent_of[cix] = pix

        # Mobilized body children.
        self.children_of: dict[int, list[int]] = {
            k: [] for k in range(self.num_mobod)}
        for j, kp in self.parent_of.items():
            self.children_of[kp].append(j)

        # Cache baseline (unscaled) inboard (X_PF) and outboard (X_BM) mobilizer
        # frames for every mobilized body, indexed by MobilizedBodyIndex.
        self.baseline_p_PF: dict[int, np.ndarray] = {}
        self.baseline_R_PF: dict[int, osim.Rotation] = {}
        self.baseline_p_BM: dict[int, np.ndarray] = {}
        self.baseline_R_BM: dict[int, osim.Rotation] = {}
        for i in range(self.model.getNumJoints()):
            # TODO: this logic breaks for joints that contain multiple mobilized bodies
            # (e.g., ScapulothoracicJoint).
            joint = self.model.getJointSet().get(i)
            mbx = int(joint.getChildFrame().getMobilizedBodyIndex())
            X_PF = joint.getInboardFrame(self.state)
            self.baseline_p_PF[mbx] = X_PF.p().to_numpy()
            self.baseline_R_PF[mbx] = osim.Rotation(X_PF.R())
            X_BM = joint.getOutboardFrame(self.state)
            self.baseline_p_BM[mbx] = X_BM.p().to_numpy()
            self.baseline_R_BM[mbx] = osim.Rotation(X_BM.R())

    def add_parameter_group(self, group) -> None:
        """
        Append a parameter group to the appropriate cached list, dispatched by type.
        Solvers call this from `add_parameter()` so the group descriptors needed by the
        cost callbacks live on the ModelCache rather than being rebuilt on each use.

        Parameters
        ----------
        group: BodyScaleGroup, MarkerOffsetGroup, or FrameOffsetGroup
            The parameter group to register.

        Raises
        ------
        ValueError
            If `group` is not a recognized parameter group type.
        """
        if isinstance(group, BodyScaleGroup):
            self.body_scale_groups.append(group)
        elif isinstance(group, MarkerOffsetGroup):
            self.marker_offset_groups.append(group)
        elif isinstance(group, FrameOffsetGroup):
            self.frame_offset_groups.append(group)
        else:
            raise ValueError(
                f'Unsupported parameter group type {type(group).__name__}.')

    @staticmethod
    def _get_coordinate_index_map(model: osim.Model,
                                  skip_dependent_coordinates: bool=True) -> dict:
        """
        Get a mapping between coordinate paths and their indexes in the state vector.

        Parameters
        ----------
        model: osim.Model
            The OpenSim model from which to create the coordinate index map.
        skip_dependent_coordinates: bool, optional
            Whether to skip dependent (e.g., constrained) coordinates in the model.
        """
        state = model.getWorkingState()
        state_paths = osim.createStateVariableNamesInSystemOrder(model)
        coordinate_map: dict[str, int] = {}
        for i, state_path in enumerate(state_paths):
            if 'value' in state_path:
                coord_path = state_path.replace('/value', '')
                coordinate = osim.Coordinate.safeDownCast(model.getComponent(coord_path))
                if skip_dependent_coordinates:
                    if not coordinate.isDependent(state):
                        coordinate_map[coord_path] = i
                else:
                    coordinate_map[coord_path] = i

        return coordinate_map

    def get_joint_for_mobilized_body_index(self, mobod_index: int) -> osim.Joint:
        """
        Return a `Joint` whose child body is associated with provided `MobilizedBody`
        index.

        Parameters
        ----------
        mobod_index: int
            The index to a `MobilizedBody`.

        Raises
        ------
        ValueError
            If no `Joint` is found matching provided `MobilizedBody` index.
        """
        jointset = self.model.getJointSet()
        for i in range(jointset.getSize()):
            joint = jointset.get(i)
            if mobod_index == int(joint.getChildFrame().getMobilizedBodyIndex()):
                return joint

        raise ValueError(
                f"Could not find a Joint in model '{self.model.getName()}' with "
                f"MobilizedBodyIndex {mobod_index}")

    def set_scaled_mobilizer_frame_positions(self, state: osim.State,
                                             body_scales: np.ndarray) -> None:
        """
        Set the inboard (X_PF) and outboard (X_BM) mobilizer frame positions given body
        body scales. Invalidates Stage::Instance and higher.

        For each group, the outboard frame (X_BM) of every group body's joint and
        the inboard frame (X_PF) of every joint driving a group body's child are
        scaled by the group's XYZ body scale. Each scaled frame translation is
        computed elementwise from the cached baseline (relative to the body's base
        frame), so repeated calls are absolute rather than compounding.

        Parameters
        ----------
        state: osim.State
            The State to update.
        body_scales: np.ndarray, shape (3 * len(body_scale_groups),)
            Flat XYZ body-scale variables, one Vec3 per BodyScaleGroup.
        """
        for i, group in enumerate(self.body_scale_groups):
            s = np.asarray(body_scales[3*i : 3*i+3], dtype=float)

            # Outboard frame (X_BM) attached to each group body.
            for joint in group.outboard_joints:
                k = int(joint.getChildFrame().getMobilizedBodyIndex())
                p_BM = self.baseline_p_BM[k] * s
                X_BM = osim.Transform(self.baseline_R_BM[k], osim.Vec3(
                    float(p_BM[0]), float(p_BM[1]), float(p_BM[2])))
                joint.setOutboardFrame(state, X_BM)

            # Inboard frame (X_PF) of every joint driving a group body's child.
            for joint in group.inboard_joints:
                c = int(joint.getChildFrame().getMobilizedBodyIndex())
                p_PF = self.baseline_p_PF[c] * s
                X_PF = osim.Transform(self.baseline_R_PF[c], osim.Vec3(
                    float(p_PF[0]), float(p_PF[1]), float(p_PF[2])))
                joint.setInboardFrame(state, X_PF)

    @staticmethod
    def get_custom_joint_translation_scales(model: osim.Model) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping joint paths to per-axis translation scales, each
        currently applied to a CustomJoint as a length-3 array.

        Parameters
        ----------
        model: osim.Model
            The model to read from.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping joint paths to current [sx, sy, sz] translation scales.
        """
        scales: dict[str, np.ndarray] = {}
        jointset = model.getJointSet()
        for ijoint in range(jointset.getSize()):
            joint = jointset.get(ijoint)
            joint_path = joint.getAbsolutePathString()
            cj = osim.CustomJoint.safeDownCast(model.getComponent(joint_path))
            if cj is None:
                continue

            st = cj.getSpatialTransform()
            scales[joint_path] = np.ones(3)
            for i in range(3):
                axis = st.getTransformAxis(3 + i)
                if not axis.hasFunction():
                    continue
                mf = osim.MultiplierFunction.safeDownCast(axis.getFunction())
                if mf is not None:
                    scales[joint_path][i] = mf.getScale()

        return scales

    @staticmethod
    def apply_custom_joint_translation_scales(model: osim.Model, scales: dict) -> None:
        """
        For each `(joint_path, Vec3)` entry in `scales`, scale the
        translation TransformAxis functions of that CustomJoint by delegating
        to OpenSim's `SpatialTransform::scale`.

        Parameters
        ----------
        model: osim.Model
            The model to mutate.
        scales: dict[str, np.ndarray | osim.Vec3]
            Mapping from CustomJoint absolute path to a length-3 Vec3-like
            translation-scale value.
        """
        for joint_path, tscale in scales.items():
            cj = osim.CustomJoint.safeDownCast(model.getComponent(joint_path))
            if cj is None:
                raise ValueError(f'Component at {joint_path} is not a CustomJoint.')
            st = cj.upd_SpatialTransform()

            # Undo any scaling left on the translation functions by a prior
            # Model::scale().
            for j in range(3, 6):
                axis = st.updTransformAxis(j)
                if not axis.hasFunction():
                    continue
                mf = osim.MultiplierFunction.safeDownCast(axis.updFunction())
                if mf is not None:
                    mf.setScale(1.0)

            # Apply the desired translation scale.
            tscale_np = np.asarray(tscale, dtype=float)
            st.scale(osim.Vec3(float(tscale_np[0]), float(tscale_np[1]),
                               float(tscale_np[2])))

    def calc_position_jacobian_wrt_body_scales(self, state: osim.State,
                                               dp_GB: osim.VectorVec3) -> np.ndarray:
        """
        Return the position-error Jacobian with respect to body scales given a
        `State` object with scaled inboard and outboard applied and a vector `dp_GB`
        representing the position-error gradient with respect to body origin positions.

        Parameters
        ----------
        state: osim.State
            The `State` from which to compute the Jacobian. Scaled inboard and outboard
            frame positions should already be applied.
        dp_GB: osim.VectorVec3
            The gradient of the position-error with respect to body origin positions.
            Length is equal to the number of mobilized bodies in the system (including
            ground).
        body_scale_groups: list[BodyScaleGroup]
            A list of `BodyScaleGroup`, one for each body scale. The cached references
            to `Joint`s should be populated to provide to access inboard and outboard
            frame indexes.
        """
        dp_BM = osim.VectorVec3(self.num_mobod, osim.Vec3(0))
        self.model.multiplyByPositionJacobianWrtOutboardFramePositionsTranspose(
            state, dp_GB, dp_BM)
        dp_PF = osim.VectorVec3(self.num_mobod, osim.Vec3(0))
        self.model.multiplyByPositionJacobianWrtInboardFramePositionsTranspose(
            state, dp_GB, dp_PF)

        ds_body = np.zeros((self.num_mobod, 3))
        for cx in range(1, self.num_mobod):
            px = self.parent_of[cx]
            ds_body[px] += self.baseline_p_PF[cx] * dp_PF[cx].to_numpy()
            ds_body[cx] += self.baseline_p_BM[cx] * dp_BM[cx].to_numpy()

        Js = np.zeros((1, 3 * len(self.body_scale_groups)))
        for i, group in enumerate(self.body_scale_groups):
            col = np.zeros(3)
            for k in group.mobod_indexes:
                col += ds_body[k,:]
            Js[0, 3*i:3*(i+1)] = col

        return Js

    def get_tracking_marker_paths(self):
        """
        Get a list of all markers in the model whose '<fixed>' property is ``False``.
        """
        tracking_markers: list[str] = []
        for i in range(self.model.getMarkerSet().getSize()):
            marker = self.model.getMarkerSet().get(i)
            if not marker.get_fixed():
                tracking_markers.append(marker.getAbsolutePathString())

        return tracking_markers

##############
# PARAMETERS #
##############

class Parameter(ABC):
    """
    Base class for an optimized parameter. The parameter can be assigned to a single
    component, or a group of model components of the same type. Each parameter will
    create single block of optimization variables in a bilevel problem. Subclasses
    must supply the per-type behavior a solver needs by implementing the abstract
    methods `validate`, `to_group`, `append_guess_and_bounds`, and `apply_to_model`.

    Attributes
    ----------
    value: np.ndarray or None
        The optimized (or initial-guess) value for this parameter, or ``None`` when
        unset. Populated by solvers and carried on solution objects.
    group_type: type
        The math-layer descriptor type (e.g., `BodyScaleGroup`) for this parameter, as
        consumed by the cost callback.
    cost_input: str
        The name of the `CostInput` field this parameter's variable block feeds (e.g.,
        ``'body_scales'``). Solvers use it to order parameter blocks by
        `CostInput.INPUT_ORDER`.
    """
    value: np.ndarray = None
    group_type: type = None
    cost_input: str = None

    @abstractmethod
    def validate(self, mc: ModelCache) -> None:
        """
        Validate this parameter against the model and cache any derived data. Raise a
        ValueError if the configuration is invalid.
        """

    @abstractmethod
    def to_group(self):
        """
        Return the math-layer descriptor (e.g., `BodyScaleGroup`) for this parameter, as
        consumed by the cost callback.
        """

    @abstractmethod
    def append_guess_and_bounds(self, x0: list, lbx: list, ubx: list) -> None:
        """
        Append this parameter's initial guess and per-variable bounds, in place, to the
        solver's `x0`, `lbx`, and `ubx` arrays.
        """

    @abstractmethod
    def apply_to_model(self, model: osim.Model) -> None:
        """
        Apply this parameter's `value` to the `model`.
        """

    @property
    @abstractmethod
    def num_variables(self) -> int:
        """
        The number of optimization variables in this parameter's block.
        """

    def with_value(self, value: np.ndarray) -> "Parameter":
        """
        Return a copy of this parameter carrying `value`, leaving the original
        unchanged. Raise a ValueError if `value` does not have `num_variables` elements.
        """
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size != self.num_variables:
            raise ValueError(
                f'{type(self).__name__} expected a value with {self.num_variables} '
                f'element(s), but got {value.size}.')
        new = copy.copy(self)
        new.value = value
        return new


class Vec3Parameter(Parameter):
    """
    A parameter representing a Vec3 quantity in an OpenSim model.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the component(s) sharing this parameter's Vec3 value.
    bounds: Bounds
        Bounds applied to each element of the Vec3.
    value: np.ndarray
        Initial value for the Vec3.
    """
    def __init__(self, paths: str | list[str], bounds: Bounds, value: np.ndarray):
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ValueError(
                'paths must be a non-empty string or list of strings.')
        self.paths = list(paths)
        self.bounds = bounds
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size != self.num_variables:
            raise ValueError(
                f'{type(self).__name__} expected a value with {self.num_variables} '
                f'element(s), but got {value.size}.')
        self.value = value

    @property
    def num_variables(self) -> int:
        return 3

    def append_guess_and_bounds(self, x0: list, lbx: list, ubx: list) -> None:
        x0 += self.value.tolist()
        lbx += [self.bounds.lower_bound] * 3
        ubx += [self.bounds.upper_bound] * 3


class BodyScale(Vec3Parameter):
    """
    An optimized Vec3 of body scales shared across one or more bodies. Pass a single
    body path to scale one body, or a list of body paths to share one set of body scales
    across a group of bodies (e.g., for left-right symmetric scaling).

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the body or bodies whose body scale is optimized.
    bounds: Bounds
        Bounds applied to each Vec3 scale factor.
    value: np.ndarray
        Initial [sx, sy, sz] scale.
    """
    group_type = BodyScaleGroup
    cost_input = 'body_scales'

    def __init__(self, paths: str | list[str], bounds: Bounds, value: np.ndarray):
        super().__init__(paths, bounds, value)
        self.mobod_indexes: list[int] = None

    def validate(self, mc: ModelCache) -> None:
        self.mobod_indexes = []
        for path in self.paths:
            body = osim.Body.safeDownCast(mc.model.getComponent(path))
            if body is None:
                raise ValueError(f'Component at path {path} is not a Body.')
            self.mobod_indexes.append(int(body.getMobilizedBodyIndex()))

    def to_group(self) -> BodyScaleGroup:
        return BodyScaleGroup(list(self.paths), list(self.mobod_indexes))

    def apply_to_model(self, model: osim.Model) -> None:
        raise NotImplementedError(
            'BodyScale.apply_to_model is not implemented.')


class MarkerOffset(Vec3Parameter):
    """
    An optimized Vec3 offset applied to one or more markers' placement, expressed in
    each marker's base frame. Pass a single marker path to offset one marker, or a list
    to share one set of offsets across a group of markers.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the marker(s) whose placement offset is optimized.
    bounds: Bounds
        Bounds applied to each Vec3 offset component.
    value: np.ndarray, optional
        Initial [ox, oy, oz] offset. Defaults to ``None`` (unset).
    """
    group_type = MarkerOffsetGroup
    cost_input = 'marker_offsets'

    def __init__(self, paths: str | list[str], bounds: Bounds, value: np.ndarray):
        super().__init__(paths, bounds, value)
        self.mobod_indexes: list[int] = None

    def apply_to_model(self, model: osim.Model) -> None:
        for path in self.paths:
            marker = osim.Marker.safeDownCast(model.getComponent(path))
            loc = marker.get_location()
            marker.set_location(osim.Vec3(
                loc[0] + float(self.value[0]), loc[1] + float(self.value[1]),
                loc[2] + float(self.value[2])))

    def validate(self, mc: ModelCache) -> None:
        self.mobod_indexes = []
        for path in self.paths:
            marker = osim.Marker.safeDownCast(mc.model.getComponent(path))
            if marker is None:
                raise ValueError(f'Component at path {path} is not a Marker.')
            parent_frame = marker.getParentFrame()
            base_frame = osim.PhysicalFrame.safeDownCast(
                marker.getParentFrame().findBaseFrame())
            if (parent_frame.getAbsolutePathString() !=
                    base_frame.getAbsolutePathString()):
                raise ValueError(
                    f'Cannot optimize a marker offset for {path}: its parent '
                    f'frame ({parent_frame.getAbsolutePathString()}) is not its base '
                    f'frame ({base_frame.getAbsolutePathString()}). Offsets are only '
                    f'supported for markers attached directly to a body.')
            self.mobod_indexes.append(base_frame.getMobilizedBodyIndex())

    def to_group(self) -> MarkerOffsetGroup:
        return MarkerOffsetGroup(list(self.paths), list(self.mobod_indexes))


class FrameOffset(Vec3Parameter):
    """
    An optimized Vec3 offset applied to one or more `PhysicalOffsetFrame` translations,
    expressed in each frame's base frame. Pass a single frame path to offset one frame,
    or a list to share one set of offsets across a group of frames.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the frame(s) whose placement offset is optimized.
    bounds: Bounds
        Bounds applied to each Vec3 offset component.
    value: np.ndarray, optional
        Initial [ox, oy, oz] offset. Defaults to ``None`` (unset).
    """
    group_type = FrameOffsetGroup
    cost_input = 'frame_offsets'

    def __init__(self, paths: str | list[str], bounds: Bounds, value: np.ndarray):
        super().__init__(paths, bounds, value)
        self.mobod_indexes: list[int] = None

    def apply_to_model(self, model: osim.Model) -> None:
        for path in self.paths:
            frame = osim.PhysicalOffsetFrame.safeDownCast(model.getComponent(path))
            t = frame.get_translation()
            frame.set_translation(osim.Vec3(
                t[0] + float(self.value[0]), t[1] + float(self.value[1]),
                t[2] + float(self.value[2])))

    def validate(self, mc: ModelCache) -> None:
        self.mobod_indexes = []
        for path in self.paths:
            frame = osim.PhysicalOffsetFrame.safeDownCast(mc.model.getComponent(path))
            if frame is None:
                raise ValueError(
                    f'Component at path {path} is not a PhysicalOffsetFrame.')
            parent_frame = frame.getParentFrame()
            base_frame = osim.PhysicalFrame.safeDownCast(
                frame.getParentFrame().findBaseFrame())
            if (parent_frame.getAbsolutePathString() !=
                    base_frame.getAbsolutePathString()):
                raise ValueError(
                    f'Cannot optimize a frame offset for {path}: its parent '
                    f'frame ({parent_frame.getAbsolutePathString()}) is not its base '
                    f'frame ({base_frame.getAbsolutePathString()}). Offsets are only '
                    f'supported for markers attached directly to a body.')
            self.mobod_indexes.append(base_frame.getMobilizedBodyIndex())

    def to_group(self) -> FrameOffsetGroup:
        return FrameOffsetGroup(list(self.paths), list(self.mobod_indexes))
