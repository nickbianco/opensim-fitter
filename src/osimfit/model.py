import copy
import numpy as np
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass

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

    Notes
    -----
    A group holds only model-independent descriptors, so the same group may be
    registered on more than one `ModelCache`. The `Joint`s whose mobilizer frames
    scale with the group are model-specific and are therefore cached on the
    `ModelCache`; see `ModelCache.cache_body_scale_group_joints`.
    """
    body_paths: list[str]
    mobod_indexes: list[int]


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


@dataclass
class MobilizerParameterGroup:
    """
    A group of joints sharing one factor on a mobilizer's own translation (as opposed
    to the placement of its inboard or outboard frame, which is what a
    `BodyScaleGroup` scales).

    Attributes
    ----------
    joint_paths: list[str]
        Absolute model paths to the joints in this group.
    mobod_indexes: list[int]
        `MobilizedBodyIndex` values of the joints' child bodies, paired with
        joint_paths.
    """
    joint_paths: list[str]
    mobod_indexes: list[int]


@dataclass
class EllipsoidRadiiGroup(MobilizerParameterGroup):
    """A `MobilizerParameterGroup` whose joints are `osim.EllipsoidJoint`s."""


@dataclass
class BeamLengthGroup(MobilizerParameterGroup):
    """A `MobilizerParameterGroup` whose joints are `osim.CantileverFreeBeamJoint`s."""


###########
# HELPERS #
###########

def _set_vec3(vec3: osim.Vec3, value: np.ndarray) -> None:
    """
    Overwrite an `osim.Vec3` in place from a length-3 array.
    """
    for i in range(3):
        vec3.set(i, float(value[i]))


def _to_numpy(mat: osim.Mat33) -> np.ndarray:
    """
    Convert an `osim.Mat33` to a (3, 3) array. `osim.Mat33` exposes no `to_numpy`.
    """
    return np.array([[mat.get(r, c) for c in range(3)] for r in range(3)])


####################
# PARAMETER GROUPS #
####################

class ParameterGroups(ABC):
    """
    The model-bound collection of every registered parameter group of one type, owning
    one contiguous block of a bilevel problem's optimization variables.

    A `ParameterGroups` is the single place that knows how one kind of parameter
    couples to a cost: how many variables its block holds, how a value of that block is
    written into the model (`apply_to_state`) or into a tracking term's station table
    (`apply_to_tasks`), and how a term's `ErrorGradient` is projected onto that block
    (`calc_jacobian_block`). Adding a new parameter type therefore means writing one
    `ParameterGroups` subclass and registering it in `PARAMETER_GROUPS_TYPES`, rather
    than editing every cost and solver that handles parameters.

    Blocks are composed in `CostInput.INPUT_ORDER`. That order is significant for
    `apply_to_tasks`: body scales set each station absolutely from its base-frame
    location, and offsets then add to the result, so body scales must precede offsets.

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` these groups are registered on.

    Attributes
    ----------
    input_name: str
        The `CostInput` field this block feeds.
    group_type: type
        The group descriptor type (e.g., `BodyScaleGroup`) this collection holds.
    num_variables_per_group: int
        The number of optimization variables each registered group contributes.
    """
    input_name: str = None
    group_type: type = None
    num_variables_per_group: int = 3

    def __init__(self, mc: 'ModelCache'):
        self.mc = mc
        self.groups: list = []

    @property
    def num_variables(self) -> int:
        """
        The total number of optimization variables in this block.
        """
        return self.num_variables_per_group * len(self.groups)

    def group_slice(self, index: int) -> slice:
        """
        The slice of this block's variables belonging to group `index`.
        """
        n = self.num_variables_per_group
        return slice(n * index, n * (index + 1))

    def add(self, group) -> None:
        """
        Register a group descriptor, caching any model-derived data it needs.
        """
        self.groups.append(group)

    def replace(self, groups) -> None:
        """
        Replace every registered group, re-caching model-derived data.
        """
        self.groups = []
        for group in groups:
            self.add(group)

    def apply_to_state(self, state: osim.State, values: np.ndarray) -> None:
        """
        Write `values` into `state`. Invalidates Stage::Instance and higher. The
        default does nothing, for parameters that do not touch the State.
        """

    def apply_to_tasks(self, tasks, values: np.ndarray) -> None:
        """
        Compose `values` into a tracking term's per-task station table. The default
        does nothing, for parameters that do not move a station within its base frame.
        """

    @abstractmethod
    def calc_jacobian_block(self, gradient) -> np.ndarray:
        """
        Return this block's row of the error Jacobian, shape ``(1, num_variables)``,
        given one tracking term's `ErrorGradient`. A cost sums the blocks returned for
        each of its terms.
        """


class BodyScaleGroups(ParameterGroups):
    """
    The registered `BodyScaleGroup`s. Body scales stretch each group body's inboard and
    outboard mobilizer frame translations, and scale the base-frame location of any
    station attached to a group body.
    """
    input_name = 'body_scales'
    group_type = BodyScaleGroup

    def apply_to_state(self, state: osim.State, values: np.ndarray) -> None:
        self.mc.set_scaled_mobilizer_frame_positions(state, values)

    def apply_to_tasks(self, tasks, values: np.ndarray) -> None:
        # Set each station absolutely from its cached base-frame location, so repeated
        # calls do not compound and so a station on an unscaled body is reset to base.
        for itask, cache in enumerate(tasks.station_caches):
            _set_vec3(tasks.stations.updElt(itask),
                      cache.calc_scaled_base_station(values))

    def calc_jacobian_block(self, gradient) -> np.ndarray:
        Js = np.zeros((1, self.num_variables))
        if gradient.tasks.num_tasks == 0:
            return Js

        # The contribution routed through each mobilizer frame translation.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(
            gradient.state, gradient.dp_GB)

        # Plus the contribution from scaling each station's base-frame location.
        for i, cache in enumerate(gradient.tasks.station_caches):
            g = cache.body_scale_group_index
            if g is not None:
                Js[0, self.group_slice(g)] += (
                    gradient.tasks.base_stations[i] * gradient.doffset[i])

        return Js


class OffsetGroups(ParameterGroups):
    """
    The registered `OffsetGroup`s of one kind. An offset is an additive translation of a
    component's placement, expressed in the component's base frame.

    A term's tasks carry offsets for exactly one kind of component, declared by
    `Tasks.offset_input`, so a term whose tasks are markers contributes nothing to the
    frame-offset block and vice versa.
    """
    def apply_to_tasks(self, tasks, values: np.ndarray) -> None:
        if tasks.offset_input != self.input_name:
            return
        for i, g in enumerate(tasks.offset_group_indexes):
            if g is None:
                continue
            station = tasks.stations.getElt(i).to_numpy() + values[self.group_slice(g)]
            _set_vec3(tasks.stations.updElt(i), station)

    def calc_jacobian_block(self, gradient) -> np.ndarray:
        Jo = np.zeros((1, self.num_variables))
        if gradient.tasks.offset_input != self.input_name:
            return Jo
        for i, g in enumerate(gradient.tasks.offset_group_indexes):
            if g is not None:
                Jo[0, self.group_slice(g)] += gradient.doffset[i]
        return Jo


class MarkerOffsetGroups(OffsetGroups):
    """The registered `MarkerOffsetGroup`s."""
    input_name = 'marker_offsets'
    group_type = MarkerOffsetGroup


class FrameOffsetGroups(OffsetGroups):
    """The registered `FrameOffsetGroup`s."""
    input_name = 'frame_offsets'
    group_type = FrameOffsetGroup


class MobilizerParameterGroups(ParameterGroups):
    """
    The registered groups of one mobilizer parameter type (e.g., ellipsoid radii).

    Each group's variables are dimensionless factors on the model's baseline value for
    that parameter, cached per joint when the group is registered so that repeated
    evaluation is absolute rather than compounding.

    Because such a parameter shifts only the mobilizer's own translation, its Jacobian
    is the joint's local Ground-frame column composed with the matter subsystem's
    subtree-sum operator: ``dE/dfactor = baseline * (~J_local @ subtreeSum(dp_GB))``.
    The parameter leaves the mobilizer's rotation untouched, so it does not enter a
    frame orientation error.

    Attributes
    ----------
    joint_type: type
        The `osim.Joint` subclass the group's paths must resolve to.
    """
    joint_type: type = None

    def __init__(self, mc: 'ModelCache'):
        super().__init__(mc)
        self.joints: list[list] = []
        self.baselines: list[list[np.ndarray]] = []

    def add(self, group) -> None:
        super().add(group)
        joints = []
        baselines = []
        for path in group.joint_paths:
            joint = self.joint_type.safeDownCast(self.mc.model.getComponent(path))
            if joint is None:
                raise ValueError(
                    f'Component at path {path} is not an {self.joint_type.__name__}.')
            joints.append(joint)
            baselines.append(self.read_baseline(joint))
        self.joints.append(joints)
        self.baselines.append(baselines)

    def replace(self, groups) -> None:
        self.joints = []
        self.baselines = []
        super().replace(groups)

    @abstractmethod
    def read_baseline(self, joint) -> np.ndarray:
        """
        Return the joint's baseline parameter value, shape
        ``(num_variables_per_group,)``, read from its property.
        """

    @abstractmethod
    def set_value(self, joint, state: osim.State, value: np.ndarray) -> None:
        """
        Write an absolute parameter `value` for `joint` into `state`.
        """

    @abstractmethod
    def calc_local_jacobian(self, joint, state: osim.State) -> np.ndarray:
        """
        Return the joint's local Ground-frame position Jacobian of its child body
        origin with respect to this parameter, shape
        ``(3, num_variables_per_group)``.
        """

    def apply_to_state(self, state: osim.State, values: np.ndarray) -> None:
        for i, (joints, baselines) in enumerate(zip(self.joints, self.baselines)):
            factor = values[self.group_slice(i)]
            for joint, baseline in zip(joints, baselines):
                self.set_value(joint, state, baseline * factor)

    def calc_jacobian_block(self, gradient) -> np.ndarray:
        J = np.zeros((1, self.num_variables))
        if gradient.tasks.num_tasks == 0:
            return J

        for i, (group, joints, baselines) in enumerate(
                zip(self.groups, self.joints, self.baselines)):
            for mobod_index, joint, baseline in zip(
                    group.mobod_indexes, joints, baselines):
                # Sum the per-body gradient over the mobilizer's outboard subtree,
                # which the mobilizer translation shifts rigidly.
                subtree_sum = (
                    self.mc.model
                    .multiplyByPositionJacobianWrtMobilizerTranslationTranspose(
                        gradient.state, int(mobod_index),
                        gradient.dp_GB).to_numpy())
                local = self.calc_local_jacobian(joint, gradient.state)
                J[0, self.group_slice(i)] += baseline * (local.T @ subtree_sum)

        return J


class EllipsoidRadiiGroups(MobilizerParameterGroups):
    """The registered `EllipsoidRadiiGroup`s."""
    input_name = 'ellipsoid_radii'
    group_type = EllipsoidRadiiGroup
    joint_type = osim.EllipsoidJoint
    num_variables_per_group = 3

    def read_baseline(self, joint) -> np.ndarray:
        return joint.get_radii_x_y_z().to_numpy()

    def set_value(self, joint, state: osim.State, value: np.ndarray) -> None:
        joint.setRadii(state, osim.Vec3(
            float(value[0]), float(value[1]), float(value[2])))

    def calc_local_jacobian(self, joint, state: osim.State) -> np.ndarray:
        return _to_numpy(joint.calcPositionJacobianWrtRadii(state))


class BeamLengthGroups(MobilizerParameterGroups):
    """The registered `BeamLengthGroup`s."""
    input_name = 'beam_lengths'
    group_type = BeamLengthGroup
    joint_type = osim.CantileverFreeBeamJoint
    num_variables_per_group = 1

    def read_baseline(self, joint) -> np.ndarray:
        return np.array([joint.get_beam_length()], dtype=float)

    def set_value(self, joint, state: osim.State, value: np.ndarray) -> None:
        joint.setLength(state, float(value[0]))

    def calc_local_jacobian(self, joint, state: osim.State) -> np.ndarray:
        return joint.calcPositionJacobianWrtLength(state).to_numpy().reshape(3, 1)


# Every parameter type a bilevel problem supports, in `CostInput.INPUT_ORDER`. A
# `ModelCache` instantiates one collection per entry.
PARAMETER_GROUPS_TYPES: tuple[type, ...] = (
    BodyScaleGroups, MarkerOffsetGroups, FrameOffsetGroups,
    EllipsoidRadiiGroups, BeamLengthGroups)


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
    num_mobod: int
        Total Simbody mobod count, including Ground at index 0.
    coordinate_map: dict[str, int]
        Mapping from absolute coordinate path to its q-index in the State,
        restricted to independent coordinates (e.g., coupled coordinates are
        excluded).
    coordinate_indexes: list[int]
        The q-indexes of the independent coordinates, in registration order.
    parameter_groups: dict[str, ParameterGroups]
        The registered parameter groups, keyed by `CostInput` field name, one entry
        per type in `PARAMETER_GROUPS_TYPES`. Each entry owns one contiguous block of
        a bilevel problem's optimization variables; see `ParameterGroups`.
    body_scale_groups: list[BodyScaleGroup]
        The list of BodyScaleGroups associated with this model. A view of
        ``parameter_groups['body_scales'].groups``.
    marker_offset_groups: list[MarkerOffsetGroup]
        The list of MarkerOffsetGroups associated with this model. A view of
        ``parameter_groups['marker_offsets'].groups``.
    frame_offset_groups: list[FrameOffsetGroup]
        The list of FrameOffsetGroups associated with this model. A view of
        ``parameter_groups['frame_offsets'].groups``.
    ellipsoid_radii_groups: list[EllipsoidRadiiGroup]
        The list of EllipsoidRadiiGroups associated with this model. A view of
        ``parameter_groups['ellipsoid_radii'].groups``.
    beam_length_groups: list[BeamLengthGroup]
        The list of BeamLengthGroups associated with this model. A view of
        ``parameter_groups['beam_lengths'].groups``.
    parent_of: dict[int, int]
        Per-mobod parent in the multibody tree. ``parent_of[k]`` is the
        ``MobilizedBodyIndex`` of body ``k``'s parent (Ground has no entry).
    children_of: dict[int, list[int]]
        Inverse of ``parent_of``: ``children_of[k]`` is the list of mobod
        indexes whose parent is ``k``. Every mobod (including Ground at 0)
        has an entry, possibly empty.
    body_scale_group_inboard_joints: list[list[osim.Joint]]
        Per-`BodyScaleGroup` `Joint`s whose inboard (X_PF) mobilizer frames scale
        with that group, parallel to `body_scale_groups`. Populated by
        `cache_body_scale_group_joints`.
    body_scale_group_outboard_joints: list[list[osim.Joint]]
        Per-`BodyScaleGroup` `Joint`s whose outboard (X_BM) mobilizer frames scale
        with that group, parallel to `body_scale_groups`. Populated by
        `cache_body_scale_group_joints`.
    """
    def __init__(self, model: str | osim.Model):
        modelProcessor = osim.ModelProcessor(model)
        self.model = modelProcessor.process()
        self.state = self.model.initSystem()
        self.num_mobod = self.model.getNumBodies() + 1
        self.coordinate_map = self._get_coordinate_index_map(self.model,
                                                    skip_dependent_coordinates=True)
        self.coordinate_indexes = list(self.coordinate_map.values())
        self.parameter_groups: dict[str, ParameterGroups] = {
            cls.input_name: cls(self) for cls in PARAMETER_GROUPS_TYPES}
        self.body_scale_group_inboard_joints: list[list[osim.Joint]] = []
        self.body_scale_group_outboard_joints: list[list[osim.Joint]] = []

        # A quaternion-capable mobilizer (e.g. the MobilizedBody::Ellipsoid behind an
        # EllipsoidJoint, or Free and Ball) always allocates `getMaxNQ()` slots in the
        # State even when the Euler-angle modeling option is engaged and only
        # `getNQInUse()` of them are meaningful. State::getNQ() therefore counts the
        # unused slots, which is why it can exceed getNU(); `coordinate_map` is built
        # from each Coordinate's true q index so those gaps are skipped rather than
        # written through.

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
        group: BodyScaleGroup, MarkerOffsetGroup, FrameOffsetGroup, \
                EllipsoidRadiiGroup, or BeamLengthGroup
            The parameter group to register.

        Raises
        ------
        ValueError
            If `group` is not a recognized parameter group type.
        """
        for groups in self.parameter_groups.values():
            if isinstance(group, groups.group_type):
                groups.add(group)
                return

        raise ValueError(
            f'Unsupported parameter group type {type(group).__name__}.')

    @property
    def body_scale_groups(self) -> list[BodyScaleGroup]:
        return self.parameter_groups['body_scales'].groups

    @body_scale_groups.setter
    def body_scale_groups(self, groups) -> None:
        self.parameter_groups['body_scales'].replace(groups)

    @property
    def marker_offset_groups(self) -> list[MarkerOffsetGroup]:
        return self.parameter_groups['marker_offsets'].groups

    @marker_offset_groups.setter
    def marker_offset_groups(self, groups) -> None:
        self.parameter_groups['marker_offsets'].replace(groups)

    @property
    def frame_offset_groups(self) -> list[FrameOffsetGroup]:
        return self.parameter_groups['frame_offsets'].groups

    @frame_offset_groups.setter
    def frame_offset_groups(self, groups) -> None:
        self.parameter_groups['frame_offsets'].replace(groups)

    @property
    def ellipsoid_radii_groups(self) -> list[EllipsoidRadiiGroup]:
        return self.parameter_groups['ellipsoid_radii'].groups

    @ellipsoid_radii_groups.setter
    def ellipsoid_radii_groups(self, groups) -> None:
        self.parameter_groups['ellipsoid_radii'].replace(groups)

    @property
    def beam_length_groups(self) -> list[BeamLengthGroup]:
        return self.parameter_groups['beam_lengths'].groups

    @beam_length_groups.setter
    def beam_length_groups(self, groups) -> None:
        self.parameter_groups['beam_lengths'].replace(groups)

    @staticmethod
    def _get_coordinate_index_map(model: osim.Model,
                                  skip_dependent_coordinates: bool=True) -> dict:
        """
        Get a mapping between coordinate paths and their q-indexes in the state vector.

        Each index comes from the Coordinate's own mobilizer, not from its position in
        the state-variable ordering. The two differ whenever a mobilizer allocates more
        q than it uses: a quaternion-capable mobilizer always reserves `getMaxNQ()`
        slots, so a State can carry unused q slots that the state-variable ordering
        does not enumerate. Using the enumeration position would shift every coordinate
        after such a mobilizer into the wrong slot.

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
        for state_path in state_paths:
            if 'value' in state_path:
                coord_path = state_path.replace('/value', '')
                coordinate = osim.Coordinate.safeDownCast(model.getComponent(coord_path))
                if skip_dependent_coordinates and coordinate.isDependent(state):
                    continue
                coordinate_map[coord_path] = model.getCoordinateQIndex(
                    state, coordinate)

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

    def cache_body_scale_group_joints(self) -> None:
        """
        Populate `body_scale_group_outboard_joints` and
        `body_scale_group_inboard_joints` with the `Joint`s whose mobilizer frames
        scale with each registered `BodyScaleGroup`: the outboard (X_BM) frame of each
        group body's joint, and the inboard (X_PF) frame of every joint driving a group
        body's child.

        The `Joint`s belong to this `ModelCache`'s model, so they are cached here rather
        than on the groups, which may be shared across `ModelCache`s.
        """
        self.body_scale_group_outboard_joints = [
            [self.get_joint_for_mobilized_body_index(int(k))
             for k in group.mobod_indexes]
            for group in self.body_scale_groups]
        self.body_scale_group_inboard_joints = [
            [self.get_joint_for_mobilized_body_index(c)
             for k in group.mobod_indexes
             for c in self.children_of[int(k)]]
            for group in self.body_scale_groups]

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
        num_groups = len(self.body_scale_groups)
        if (len(self.body_scale_group_outboard_joints) != num_groups
                or len(self.body_scale_group_inboard_joints) != num_groups):
            raise RuntimeError(
                'cache_body_scale_group_joints() must be called after the last body '
                'scale group is registered and before scaled mobilizer frames are set.')

        for i in range(num_groups):
            s = np.asarray(body_scales[3*i : 3*i+3], dtype=float)

            # Outboard frame (X_BM) attached to each group body.
            for joint in self.body_scale_group_outboard_joints[i]:
                k = int(joint.getChildFrame().getMobilizedBodyIndex())
                p_BM = self.baseline_p_BM[k] * s
                X_BM = osim.Transform(self.baseline_R_BM[k], osim.Vec3(
                    float(p_BM[0]), float(p_BM[1]), float(p_BM[2])))
                joint.setOutboardFrame(state, X_BM)

            # Inboard frame (X_PF) of every joint driving a group body's child.
            for joint in self.body_scale_group_inboard_joints[i]:
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

    @staticmethod
    def get_ellipsoid_joint_radii(model: osim.Model) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping `osim.EllipsoidJoint` paths to their current
        'radii_x_y_z' property values.

        Paired with `apply_ellipsoid_joint_radii` to hold ellipsoid radii fixed across
        a `Model::scale()` call: `EllipsoidJoint::extendScale` multiplies the radii by
        the parent frame's body scale factors, which this fitter treats as an
        independent parameter rather than a consequence of body scaling.

        Parameters
        ----------
        model: osim.Model
            The model to read from.
        """
        radii: dict[str, np.ndarray] = {}
        jointset = model.getJointSet()
        for ijoint in range(jointset.getSize()):
            joint = osim.EllipsoidJoint.safeDownCast(jointset.get(ijoint))
            if joint is None:
                continue
            radii[joint.getAbsolutePathString()] = joint.get_radii_x_y_z().to_numpy()

        return radii

    @staticmethod
    def apply_ellipsoid_joint_radii(model: osim.Model, radii: dict) -> None:
        """
        For each `(joint_path, Vec3)` entry in `radii`, overwrite that
        `osim.EllipsoidJoint`'s 'radii_x_y_z' property.

        Parameters
        ----------
        model: osim.Model
            The model to mutate.
        radii: dict[str, np.ndarray | osim.Vec3]
            Mapping from EllipsoidJoint absolute path to a length-3 Vec3-like radii
            value, as returned by `get_ellipsoid_joint_radii`.
        """
        for joint_path, value in radii.items():
            joint = osim.EllipsoidJoint.safeDownCast(model.getComponent(joint_path))
            if joint is None:
                raise ValueError(
                    f'Component at {joint_path} is not an EllipsoidJoint.')
            value = np.asarray(value, dtype=float)
            joint.set_radii_x_y_z(osim.Vec3(
                float(value[0]), float(value[1]), float(value[2])))

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

    def find_body_scale_group_index(self, mobod_index: int):
        """
        Find the `BodyScaleGroup` index associated with a `MobilizedBody` in the model.

        Parameters
        ----------
        mobod_index: int
            The index to a `MobilizedBody` in the model.

        Raises
        ------
        Exception
            If multiple `BodyScaleGroup` indexes are found for the provided
            `osim.MobilizedBodyIndex`.
        """
        scale_groups = list()
        for g, group in enumerate(self.body_scale_groups):
            if mobod_index in [int(k) for k in group.mobod_indexes]:
                scale_groups.append(g)

        if len(scale_groups) > 1:
            raise Exception(f'Multiple scale groups found for body at index '
                            f'{mobod_index}')

        return scale_groups[0] if len(scale_groups) > 0 else None

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


class StationCache:
    """
    A thin wrapper around a point fixed on a body (a `osim.Station` or a
    `osim.PhysicalFrame`'s origin) that pre-computes values used repeatedly by solvers
    and callback functions.

    Construct via `from_station` or `from_frame`.

    Attributes
    ----------
    base_frame: osim.PhysicalFrame
        The base `osim.PhysicalFrame` of the frame to which the point is attached.
    mobod_index: int
        The index to the `osim.MobilizedBody` associated with the base frame.
    base_station: np.ndarray
        The location of the point in the base frame, shape (3,).
    body_scale_group_index: None | int
        The index to the `BodyScaleGroup` associated with this point's
        `osim.MobilizedBodyIndex`, if it exists.
    """
    def __init__(self, *args, **kwargs):
        raise TypeError(
            'Construct a StationCache via StationCache.from_station(...) or '
            'StationCache.from_frame(...).')

    @classmethod
    def _create(cls, mc: ModelCache, base_frame: osim.PhysicalFrame,
                base_station: np.ndarray):
        """
        Populate a cache from an already-resolved base frame and base-frame point.
        """
        cache = cls.__new__(cls)
        cache.mc = mc
        cache.base_frame = base_frame
        cache.mobod_index = int(base_frame.getMobilizedBodyIndex())
        cache.base_station = base_station
        cache.body_scale_group_index = mc.find_body_scale_group_index(cache.mobod_index)
        return cache

    @classmethod
    def from_station(cls, mc: ModelCache, station: osim.Station):
        """
        Build a `StationCache` for an `osim.Station`. `base_station` is the station's
        location in its base frame.

        Parameters
        ----------
        mc: ModelCache
            A previously-constructed `ModelCache`.
        station: osim.Station
            The station (or a subclass, e.g. an `osim.Marker`) to wrap.

        Raises
        ------
        ValueError
            If `station` is not an `osim.Station`.
        """
        downcast = osim.Station.safeDownCast(station)
        if downcast is None:
            raise ValueError(f'Expected an osim.Station, but got {station}.')
        base_frame = osim.PhysicalFrame.safeDownCast(
            downcast.getParentFrame().findBaseFrame())
        base_station = downcast.findLocationInFrame(mc.state, base_frame).to_numpy()
        return cls._create(mc, base_frame, base_station)

    @classmethod
    def from_frame(cls, mc: ModelCache, frame: osim.PhysicalFrame):
        """
        Build a `StationCache` for a `osim.PhysicalFrame`'s origin. `base_station` is
        the frame origin's location in its base frame.

        Parameters
        ----------
        mc: ModelCache
            A previously-constructed `ModelCache`.
        frame: osim.PhysicalFrame
            The frame whose origin to wrap.

        Raises
        ------
        ValueError
            If `frame` is not an `osim.PhysicalFrame`.
        """
        downcast = osim.PhysicalFrame.safeDownCast(frame)
        if downcast is None:
            raise ValueError(f'Expected an osim.PhysicalFrame, but got {frame}.')
        base_frame = osim.PhysicalFrame.safeDownCast(downcast.findBaseFrame())
        base_station = downcast.findTransformInBaseFrame().p().to_numpy()
        return cls._create(mc, base_frame, base_station)

    def calc_scaled_base_station(self, body_scales: np.ndarray) -> np.ndarray:
        offset = self.base_station.copy()
        if self.body_scale_group_index is not None:
            g = self.body_scale_group_index
            offset = offset * np.asarray(body_scales[3*g : 3*g+3], dtype=float)
        return offset

    def calc_position(self, state: osim.State, body_scales: np.ndarray) -> osim.Vec3:
        offset = self.calc_scaled_base_station(body_scales)
        vec = osim.Vec3(float(offset[0]), float(offset[1]), float(offset[2]))
        return self.base_frame.findStationLocationInGround(state, vec)

    def calc_position_jacobian_wrt_body_scales(self, state: osim.State) -> np.ndarray:
        rotation = self.base_frame.getRotationInGround(state)
        R_GB = np.array([[rotation.get(r, c) for c in range(3)] for r in range(3)])
        jacobian = np.zeros((3, 3 * len(self.mc.body_scale_groups)))
        for axis in range(3):
            dp_GB = osim.VectorVec3(self.mc.num_mobod, osim.Vec3(0))
            unit = [0.0, 0.0, 0.0]
            unit[axis] = 1.0
            dp_GB.set(self.mobod_index, osim.Vec3(unit[0], unit[1], unit[2]))
            row = self.mc.calc_position_jacobian_wrt_body_scales(
                state, dp_GB)[0, :].copy()

            # Add the contribution from scaling the station's base-frame location.
            if self.body_scale_group_index is not None:
                doffset = np.asarray(unit) @ R_GB
                g = self.body_scale_group_index
                row[3*g:3*g+3] += self.base_station * doffset

            jacobian[axis, :] = row

        return jacobian


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


class BlockParameter(Parameter):
    """
    A parameter whose optimization variables form one fixed-size block, shared across
    one or more model components of the same type. Subclasses fix the block size by
    implementing `num_variables`.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the component(s) sharing this parameter's value.
    bounds: Bounds
        Bounds applied to each element of the block.
    value: np.ndarray
        Initial value for the block.
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

    def append_guess_and_bounds(self, x0: list, lbx: list, ubx: list) -> None:
        x0 += self.value.tolist()
        lbx += [self.bounds.lower_bound] * self.num_variables
        ubx += [self.bounds.upper_bound] * self.num_variables


class Vec3Parameter(BlockParameter):
    """
    A parameter representing a Vec3 quantity in an OpenSim model.
    """
    @property
    def num_variables(self) -> int:
        return 3


class ScalarParameter(BlockParameter):
    """
    A parameter representing a scalar quantity in an OpenSim model.
    """
    @property
    def num_variables(self) -> int:
        return 1


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


class MobilizerParameter(Parameter):
    """
    A parameter that scales a mobilizer's own translation, as a dimensionless factor on
    the joint's baseline property value. Pass a single joint path to scale one joint, or
    a list to share one factor across a group of joints (e.g., for left-right symmetry).

    Unlike a `BodyScale`, which stretches the placement of a joint's inboard and
    outboard frames, this scales the mobilizer's internal geometry, and so is
    independent of any `BodyScale` registered on a neighboring body. `apply_to_model`
    is multiplicative on the model's current property value, which lets a solver undo
    the body scaling that `Model::scale()` may have applied to it; see
    `ModelCache.get_ellipsoid_joint_radii`.

    Attributes
    ----------
    joint_type: type
        The `osim.Joint` subclass this parameter's paths must resolve to.
    """
    joint_type: type = None

    def validate(self, mc: ModelCache) -> None:
        if self.bounds.lower_bound <= 0.0:
            raise ValueError(
                f'{type(self).__name__} on {self.paths} requires a strictly positive '
                f'lower bound, since the underlying mobilizer rejects a non-positive '
                f'value, but got {self.bounds.lower_bound}.')
        self.mobod_indexes = []
        for path in self.paths:
            joint = self.joint_type.safeDownCast(mc.model.getComponent(path))
            if joint is None:
                raise ValueError(
                    f'Component at path {path} is not an '
                    f'{self.joint_type.__name__}.')
            self.mobod_indexes.append(
                int(joint.getChildFrame().getMobilizedBodyIndex()))

    def to_group(self):
        return self.group_type(list(self.paths), list(self.mobod_indexes))


class EllipsoidRadii(MobilizerParameter, Vec3Parameter):
    """
    An optimized Vec3 of factors on one or more `osim.EllipsoidJoint`s' radii. A factor
    of 1.0 leaves a joint's 'radii_x_y_z' property unchanged.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the EllipsoidJoint(s) whose radii are optimized.
    bounds: Bounds
        Bounds applied to each factor. The lower bound must be strictly positive.
    value: np.ndarray, optional
        Initial [fx, fy, fz] factors. Defaults to no scaling.
    """
    group_type = EllipsoidRadiiGroup
    cost_input = 'ellipsoid_radii'
    joint_type = osim.EllipsoidJoint

    def __init__(self, paths: str | list[str], bounds: Bounds,
                 value: np.ndarray = (1.0, 1.0, 1.0)):
        super().__init__(paths, bounds, value)
        self.mobod_indexes: list[int] = None

    def apply_to_model(self, model: osim.Model) -> None:
        for path in self.paths:
            joint = osim.EllipsoidJoint.safeDownCast(model.getComponent(path))
            radii = joint.get_radii_x_y_z()
            joint.set_radii_x_y_z(osim.Vec3(
                radii[0] * float(self.value[0]),
                radii[1] * float(self.value[1]),
                radii[2] * float(self.value[2])))


class BeamLength(MobilizerParameter, ScalarParameter):
    """
    An optimized factor on one or more `osim.CantileverFreeBeamJoint`s' beam lengths. A
    factor of 1.0 leaves a joint's 'beam_length' property unchanged.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the CantileverFreeBeamJoint(s) whose beam length is
        optimized.
    bounds: Bounds
        Bounds applied to the factor. The lower bound must be strictly positive.
    value: np.ndarray, optional
        Initial factor. Defaults to no scaling.
    """
    group_type = BeamLengthGroup
    cost_input = 'beam_lengths'
    joint_type = osim.CantileverFreeBeamJoint

    def __init__(self, paths: str | list[str], bounds: Bounds,
                 value: np.ndarray = (1.0,)):
        super().__init__(paths, bounds, value)
        self.mobod_indexes: list[int] = None

    def apply_to_model(self, model: osim.Model) -> None:
        for path in self.paths:
            joint = osim.CantileverFreeBeamJoint.safeDownCast(
                model.getComponent(path))
            joint.set_beam_length(
                joint.get_beam_length() * float(self.value[0]))
