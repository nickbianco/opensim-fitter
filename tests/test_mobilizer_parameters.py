"""
Unit tests for the mobilizer `Parameter`s: `EllipsoidRadii` and `BeamLength`.

MobilizedBody::Ellipsoid reserves 4 q slots even in Euler-angle mode, where only 3 are
in use, so a State containing one carries an unused q slot and getNQ() exceeds getNU().
`ModelCache` handles that by indexing coordinates through their own mobilizer; see
`test_ellipsoid_coordinate_map_skips_unused_q_slot`.
"""

import pytest
import numpy as np
import casadi as ca
import opensim as osim

from osimfit.model import (ModelCache, BodyScale, BodyScaleGroup, EllipsoidRadii,
                           EllipsoidRadiiGroup, BeamLength, BeamLengthGroup,
                           MarkerOffsetGroup)
from osimfit.bounds import Bounds
from osimfit.costs import CostInput
from osimfit.solvers import SplinedKinematicsSolver
from tests.test_costs import build_bilevel_rep

# Baseline mobilizer geometry of the test model.
RADII = (0.05, 0.03, 0.04)
BEAM_LENGTH = 0.35


def create_beam_model(second_beam: bool = False):
    """
    Create a model whose chain exercises `BeamLength`, using only mobilizers with
    qdot == u so that `ModelCache` accepts it:

        ground --Pin-- torso --Gimbal-- humerus --CantileverFreeBeam-- forearm

    Markers sit on both the humerus and the forearm, so a beam-length factor reaches
    one marker through the beam mobilizer's own subtree and leaves the other alone.
    Set `second_beam` to add a mirrored 'elbow_l'/'forearm_l' pair, for grouping tests.
    """
    model = osim.Model()
    model.setName('beam')

    torso = osim.Body('torso', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    humerus = osim.Body('humerus', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    forearm = osim.Body('forearm', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    for body in (torso, humerus, forearm):
        model.addBody(body)

    model.addJoint(osim.PinJoint('ground_torso', model.getGround(), torso))
    model.addJoint(osim.GimbalJoint(
        'shoulder_r', torso, osim.Vec3(0.1, 0.2, 0.3), osim.Vec3(0.2, -0.1, 0.3),
        humerus, osim.Vec3(0.01, 0.02, 0.03), osim.Vec3(0.1, 0.2, -0.1)))
    model.addJoint(osim.CantileverFreeBeamJoint(
        'elbow_r', humerus, osim.Vec3(0.05, -0.1, 0.02), osim.Vec3(0.1, 0.1, 0.1),
        forearm, osim.Vec3(0.0, 0.01, 0.0), osim.Vec3(-0.2, 0.1, 0.05),
        BEAM_LENGTH))

    model.addMarker(osim.Marker('humerus_marker', humerus, osim.Vec3(0.02, 0.03, 0.01)))
    model.addMarker(osim.Marker('forearm_marker', forearm, osim.Vec3(0.1, 0.2, 0.3)))

    if second_beam:
        forearm_l = osim.Body('forearm_l', 1.0, osim.Vec3(0), osim.Inertia(0.1))
        model.addBody(forearm_l)
        model.addJoint(osim.CantileverFreeBeamJoint(
            'elbow_l', humerus, osim.Vec3(0.05, -0.1, -0.02), osim.Vec3(0.1, -0.1, 0.1),
            forearm_l, osim.Vec3(0.0, 0.01, 0.0), osim.Vec3(0.2, 0.1, -0.05),
            BEAM_LENGTH))
        model.addMarker(osim.Marker(
            'forearm_l_marker', forearm_l, osim.Vec3(0.1, 0.2, -0.3)))

    model.finalizeConnections()
    return model


def create_beam_recovery_model():
    """
    A minimal chain in which the beam length is strongly identifiable:

        ground --Pin-- torso --CantileverFreeBeam-- forearm

    Three markers spread across the forearm pin down the beam's endpoint pose, and one
    marker on the torso pins down the upstream pin rotation. There is no
    three-rotation joint upstream that could mimic a change in beam length.
    """
    model = osim.Model()
    model.setName('beam_recovery')

    torso = osim.Body('torso', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    forearm = osim.Body('forearm', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    for body in (torso, forearm):
        model.addBody(body)

    model.addJoint(osim.PinJoint('ground_torso', model.getGround(), torso))
    model.addJoint(osim.CantileverFreeBeamJoint(
        'elbow_r', torso, osim.Vec3(0.05, -0.1, 0.02), osim.Vec3(0.1, 0.1, 0.1),
        forearm, osim.Vec3(0.0, 0.01, 0.0), osim.Vec3(-0.2, 0.1, 0.05),
        BEAM_LENGTH))

    model.addMarker(osim.Marker('torso_marker', torso, osim.Vec3(0.1, 0.05, 0.0)))
    model.addMarker(osim.Marker('forearm_0', forearm, osim.Vec3(0.15, 0.0, 0.0)))
    model.addMarker(osim.Marker('forearm_1', forearm, osim.Vec3(0.0, 0.15, 0.0)))
    model.addMarker(osim.Marker('forearm_2', forearm, osim.Vec3(0.0, 0.0, 0.15)))

    model.finalizeConnections()
    return model


def create_ellipsoid_model(second_ellipsoid: bool = False):
    """
    Create a model whose chain exercises `EllipsoidRadii` alongside `BeamLength`:

        ground --Pin-- torso --Ellipsoid-- humerus --CantileverFreeBeam-- forearm

    The ellipsoid radii reach the humerus marker directly and the forearm marker
    through the beam mobilizer's subtree. Set `second_ellipsoid` to add a mirrored
    'shoulder_l'/'humerus_l' pair, for grouping tests.
    """
    model = osim.Model()
    model.setName('ellipsoid')

    torso = osim.Body('torso', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    humerus = osim.Body('humerus', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    forearm = osim.Body('forearm', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    for body in (torso, humerus, forearm):
        model.addBody(body)

    model.addJoint(osim.PinJoint('ground_torso', model.getGround(), torso))
    model.addJoint(osim.EllipsoidJoint(
        'shoulder_r', torso, osim.Vec3(0.1, 0.2, 0.3), osim.Vec3(0.2, -0.1, 0.3),
        humerus, osim.Vec3(0.01, 0.02, 0.03), osim.Vec3(0.1, 0.2, -0.1),
        osim.Vec3(*RADII)))
    model.addJoint(osim.CantileverFreeBeamJoint(
        'elbow_r', humerus, osim.Vec3(0.05, -0.1, 0.02), osim.Vec3(0.1, 0.1, 0.1),
        forearm, osim.Vec3(0.0, 0.01, 0.0), osim.Vec3(-0.2, 0.1, 0.05),
        BEAM_LENGTH))

    model.addMarker(osim.Marker('humerus_marker', humerus, osim.Vec3(0.02, 0.03, 0.01)))
    model.addMarker(osim.Marker('forearm_marker', forearm, osim.Vec3(0.1, 0.2, 0.3)))

    if second_ellipsoid:
        humerus_l = osim.Body('humerus_l', 1.0, osim.Vec3(0), osim.Inertia(0.1))
        model.addBody(humerus_l)
        model.addJoint(osim.EllipsoidJoint(
            'shoulder_l', torso, osim.Vec3(0.1, 0.2, -0.3), osim.Vec3(-0.2, 0.1, 0.3),
            humerus_l, osim.Vec3(0.01, 0.02, -0.03), osim.Vec3(0.1, -0.2, -0.1),
            osim.Vec3(*RADII)))
        model.addMarker(osim.Marker(
            'humerus_l_marker', humerus_l, osim.Vec3(0.02, 0.03, -0.01)))

    model.finalizeConnections()
    return model


def create_ellipsoid_recovery_model():
    """
    A minimal chain in which the ellipsoid radii are strongly identifiable:

        ground --Pin-- torso --Ellipsoid-- humerus

    Three markers spread across the humerus pin down its pose, and one on the torso
    pins down the upstream pin rotation.
    """
    model = osim.Model()
    model.setName('ellipsoid_recovery')

    torso = osim.Body('torso', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    humerus = osim.Body('humerus', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    for body in (torso, humerus):
        model.addBody(body)

    model.addJoint(osim.PinJoint('ground_torso', model.getGround(), torso))
    model.addJoint(osim.EllipsoidJoint(
        'shoulder_r', torso, osim.Vec3(0.1, 0.2, 0.3), osim.Vec3(0.2, -0.1, 0.3),
        humerus, osim.Vec3(0.01, 0.02, 0.03), osim.Vec3(0.1, 0.2, -0.1),
        osim.Vec3(*RADII)))

    model.addMarker(osim.Marker('torso_marker', torso, osim.Vec3(0.1, 0.05, 0.0)))
    model.addMarker(osim.Marker('humerus_0', humerus, osim.Vec3(0.15, 0.0, 0.0)))
    model.addMarker(osim.Marker('humerus_1', humerus, osim.Vec3(0.0, 0.15, 0.0)))
    model.addMarker(osim.Marker('humerus_2', humerus, osim.Vec3(0.0, 0.0, 0.15)))

    model.finalizeConnections()
    return model


def joint_mobod_index(model, joint_path):
    joint = model.getComponent(joint_path)
    return int(joint.getChildFrame().getMobilizedBodyIndex())


def add_marker_terms(cost):
    """
    Register the model's two markers on `cost` with reference positions offset from the
    model's default pose, so the error and its gradient are both non-zero.
    """
    cost.add_marker_bilevel_cost_term(
        '/markerset/humerus_marker', osim.Vec3(0.3, 0.1, -0.2), weight=2.0)
    cost.add_marker_bilevel_cost_term(
        '/markerset/forearm_marker', osim.Vec3(-0.1, 0.4, 0.25), weight=1.5)


##############
# VALIDATION #
##############

def test_ellipsoid_radii_rejects_non_ellipsoid_joint():
    model = create_beam_model()
    mc = ModelCache(model)
    parameter = EllipsoidRadii('/jointset/elbow_r', Bounds(0.5, 2.0))
    with pytest.raises(ValueError, match='not an EllipsoidJoint'):
        parameter.validate(mc)


def test_beam_length_rejects_non_beam_joint():
    model = create_beam_model()
    mc = ModelCache(model)
    parameter = BeamLength('/jointset/shoulder_r', Bounds(0.5, 2.0))
    with pytest.raises(ValueError, match='not an CantileverFreeBeamJoint'):
        parameter.validate(mc)


@pytest.mark.parametrize('lower_bound', [-1.0, 0.0])
def test_mobilizer_parameters_require_positive_lower_bound(lower_bound):
    mc = ModelCache(create_beam_model())
    for parameter in (EllipsoidRadii('/jointset/elbow_r', Bounds(lower_bound, 2.0)),
                      BeamLength('/jointset/elbow_r', Bounds(lower_bound, 2.0))):
        with pytest.raises(ValueError, match='strictly positive lower bound'):
            parameter.validate(mc)


def test_mobilizer_parameter_block_sizes():
    mc = ModelCache(create_beam_model())
    radii = EllipsoidRadii('/jointset/elbow_r', Bounds(0.5, 2.0))
    length = BeamLength('/jointset/elbow_r', Bounds(0.5, 2.0))
    assert radii.num_variables == 3
    assert length.num_variables == 1

    length.validate(mc)
    mc.add_parameter_group(length.to_group())
    assert mc.parameter_groups['beam_lengths'].num_variables == 1
    assert mc.parameter_groups['ellipsoid_radii'].num_variables == 0


def test_beam_length_groups_cache_baselines():
    mc = ModelCache(create_beam_model())
    length = BeamLength('/jointset/elbow_r', Bounds(0.5, 2.0))
    length.validate(mc)
    mc.add_parameter_group(length.to_group())
    np.testing.assert_allclose(
        mc.parameter_groups['beam_lengths'].baselines[0][0], [BEAM_LENGTH])


def test_ellipsoid_coordinate_map_skips_unused_q_slot():
    """
    MobilizedBody::Ellipsoid reserves 4 q slots but uses only 3 in Euler-angle mode, so
    the State carries an unused slot between the ellipsoid's coordinates and the
    beam's. The coordinate map must index through each Coordinate's own mobilizer and
    skip that gap, rather than use the state-variable enumeration position.
    """
    model = create_ellipsoid_model()
    state = model.initSystem()

    # One reserved-but-unused q slot, so allocated NQ exceeds NU.
    assert state.getNQ() == state.getNU() + 1

    mc = ModelCache(model)
    indexes = mc.coordinate_indexes

    # Indexes are strictly increasing, unique, and in range, and they are not simply
    # the enumeration positions.
    assert indexes == sorted(set(indexes))
    assert max(indexes) < state.getNQ()
    assert indexes != list(range(len(indexes)))

    # Writing through the map round-trips each Coordinate's value.
    values = 0.1 * (1 + np.arange(len(indexes)))
    q = np.zeros(state.getNQ())
    q[indexes] = values
    state.setQ(osim.Vector.createFromMat(q))
    model.realizePosition(state)
    for coord_path, value in zip(mc.coordinate_map, values):
        assert model.getComponent(coord_path).getValue(state) == \
            pytest.approx(value)


#############
# JACOBIANS #
#############

def build_jacobian_pair(model, **groups):
    """
    Build an analytic and a finite-differenced `BilevelCostRep` over the same parameter
    groups and marker terms, on independent ModelCaches.
    """
    reps = []
    for enable_fd in (False, True):
        rep = build_bilevel_rep(
            f'cost_fd_{enable_fd}', ModelCache(model), enable_fd=enable_fd, **groups)
        add_marker_terms(rep)
        reps.append(rep)
    return reps


def assert_jacobians_agree(model, values, **groups):
    """
    Assert that a `BilevelCostRep`'s analytic Jacobian matches CasADi's
    finite-difference Jacobian at `values`, a dict of per-input numeric blocks
    (excluding coordinates, which are seeded from a fixed non-trivial pose).
    """
    cost_jac, cost_fd = build_jacobian_pair(model, **groups)

    num_coords = len(cost_jac.mc.coordinate_indexes)
    symbols = {'coordinates': ca.SX.sym('q', num_coords)}
    numbers = [np.linspace(0.05, 0.25, num_coords)]
    for name in CostInput.PARAMETER_INPUTS:
        block = np.atleast_1d(np.asarray(values.get(name, []), dtype=float))
        symbols[name] = ca.SX.sym(name, block.size)
        numbers.append(block)

    x = ca.vertcat(*[symbols[name] for name in CostInput.INPUT_ORDER])
    val = np.concatenate(numbers)

    jacobians = []
    for cost in (cost_jac, cost_fd):
        jacobians.append(ca.Function(
            'J', [x], [ca.jacobian(cost(CostInput(**symbols)), x)])(val).full())

    # The Jacobian must be non-trivial for the comparison to mean anything.
    assert np.linalg.norm(jacobians[0]) > 1e-6
    np.testing.assert_allclose(jacobians[0], jacobians[1], atol=1e-6)


def test_beam_length_jacobian_matches_finite_differences():
    model = create_beam_model()
    model.initSystem()
    assert_jacobians_agree(
        model,
        {'beam_lengths': [1.15]},
        beam_length_groups=[BeamLengthGroup(
            ['/jointset/elbow_r'],
            [joint_mobod_index(model, '/jointset/elbow_r')])])


def test_ellipsoid_radii_jacobian_matches_finite_differences():
    model = create_ellipsoid_model()
    model.initSystem()
    assert_jacobians_agree(
        model,
        {'ellipsoid_radii': [1.1, 0.9, 1.05]},
        ellipsoid_radii_groups=[EllipsoidRadiiGroup(
            ['/jointset/shoulder_r'],
            [joint_mobod_index(model, '/jointset/shoulder_r')])])


def test_grouped_ellipsoid_radii_jacobian_matches_finite_differences():
    """
    Two EllipsoidJoints sharing one factor block contribute additively to it.
    """
    model = create_ellipsoid_model(second_ellipsoid=True)
    model.initSystem()
    paths = ['/jointset/shoulder_r', '/jointset/shoulder_l']
    cost_jac, cost_fd = build_jacobian_pair(
        model,
        ellipsoid_radii_groups=[EllipsoidRadiiGroup(
            paths, [joint_mobod_index(model, p) for p in paths])])
    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/humerus_l_marker', osim.Vec3(0.2, -0.3, 0.1), weight=1.0)

    num_coords = len(cost_jac.mc.coordinate_indexes)
    q = ca.SX.sym('q', num_coords)
    er = ca.SX.sym('er', 3)
    x = ca.vertcat(q, er)
    empty = ca.DM.zeros(0, 1)

    def jac(cost):
        expr = cost(CostInput(q, empty, empty, empty, er, empty))
        return ca.Function('J', [x], [ca.jacobian(expr, x)])(
            np.concatenate([np.linspace(0.05, 0.25, num_coords),
                            [1.1, 0.9, 1.05]])).full()

    assert np.linalg.norm(jac(cost_jac)) > 1e-6
    np.testing.assert_allclose(jac(cost_jac), jac(cost_fd), atol=1e-6)


def test_ellipsoid_radii_and_beam_length_jacobian_matches_finite_differences():
    """
    Both mobilizer parameters at once, in series: the ellipsoid's subtree contains the
    beam mobilizer, so the two blocks overlap in the bodies they move.
    """
    model = create_ellipsoid_model()
    model.initSystem()
    assert_jacobians_agree(
        model,
        {'ellipsoid_radii': [1.1, 0.9, 1.05], 'beam_lengths': [1.15]},
        ellipsoid_radii_groups=[EllipsoidRadiiGroup(
            ['/jointset/shoulder_r'],
            [joint_mobod_index(model, '/jointset/shoulder_r')])],
        beam_length_groups=[BeamLengthGroup(
            ['/jointset/elbow_r'],
            [joint_mobod_index(model, '/jointset/elbow_r')])])


def test_grouped_beam_length_jacobian_matches_finite_differences():
    """
    Two CantileverFreeBeamJoints sharing one factor contribute additively to it.
    """
    model = create_beam_model(second_beam=True)
    model.initSystem()
    paths = ['/jointset/elbow_r', '/jointset/elbow_l']
    cost_jac, cost_fd = build_jacobian_pair(
        model,
        beam_length_groups=[BeamLengthGroup(
            paths, [joint_mobod_index(model, p) for p in paths])])
    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/forearm_l_marker', osim.Vec3(0.2, -0.3, 0.1), weight=1.0)

    num_coords = len(cost_jac.mc.coordinate_indexes)
    q = ca.SX.sym('q', num_coords)
    bl = ca.SX.sym('bl', 1)
    x = ca.vertcat(q, bl)
    empty = ca.DM.zeros(0, 1)

    def jac(cost):
        expr = cost(CostInput(q, empty, empty, empty, empty, bl))
        return ca.Function('J', [x], [ca.jacobian(expr, x)])(
            np.concatenate([np.linspace(0.05, 0.25, num_coords), [1.15]])).full()

    assert np.linalg.norm(jac(cost_jac)) > 1e-6
    np.testing.assert_allclose(jac(cost_jac), jac(cost_fd), atol=1e-6)


def test_combined_parameter_jacobian_matches_finite_differences():
    """
    Body scales, marker offsets, ellipsoid radii, and a beam length together: each
    block's contribution must be correct in the presence of every other.
    """
    model = create_ellipsoid_model()
    model.initSystem()
    bodyset = model.getBodySet()
    body_scale_groups = [
        BodyScaleGroup([bodyset.get(i).getAbsolutePathString()],
                       [int(bodyset.get(i).getMobilizedBodyIndex())])
        for i in range(bodyset.getSize())]

    assert_jacobians_agree(
        model,
        {'body_scales': np.tile([1.1, 0.95, 1.0], bodyset.getSize()),
         'marker_offsets': [0.01, -0.02, 0.03],
         'ellipsoid_radii': [1.1, 0.9, 1.05],
         'beam_lengths': [1.15]},
        body_scale_groups=body_scale_groups,
        marker_offset_groups=[MarkerOffsetGroup(
            ['/markerset/forearm_marker'],
            [joint_mobod_index(model, '/jointset/elbow_r')])],
        ellipsoid_radii_groups=[EllipsoidRadiiGroup(
            ['/jointset/shoulder_r'],
            [joint_mobod_index(model, '/jointset/shoulder_r')])],
        beam_length_groups=[BeamLengthGroup(
            ['/jointset/elbow_r'],
            [joint_mobod_index(model, '/jointset/elbow_r')])])


def test_identity_factor_reproduces_baseline_error():
    """
    A factor of 1.0 must leave the model's mobilizer geometry untouched, so the cost at
    an identity factor equals the cost with no mobilizer parameter registered at all.
    """
    model = create_beam_model()
    model.initSystem()
    empty = ca.DM.zeros(0, 1)

    plain = build_bilevel_rep('plain', ModelCache(model))
    add_marker_terms(plain)
    num_coords = len(plain.mc.coordinate_indexes)
    q = ca.DM(np.linspace(0.05, 0.25, num_coords))
    baseline = float(plain(CostInput(q, empty, empty, empty, empty, empty)))

    parameterized = build_bilevel_rep(
        'parameterized', ModelCache(model),
        beam_length_groups=[BeamLengthGroup(
            ['/jointset/elbow_r'],
            [joint_mobod_index(model, '/jointset/elbow_r')])])
    add_marker_terms(parameterized)
    at_identity = float(parameterized(
        CostInput(q, empty, empty, empty, empty, ca.DM.ones(1))))

    assert at_identity == pytest.approx(baseline, abs=1e-12)


def test_ellipsoid_radii_jacobian_at_opensim_level():
    """
    Verify the composition an `EllipsoidRadiiGroups` block performs --
    ``~J_local @ subtreeSum(dp_GB)`` -- directly against central finite differences,
    without going through `ModelCache`. This covers the EllipsoidJoint Jacobian path
    while a model containing one cannot be wrapped in a ModelCache.
    """
    model = create_ellipsoid_model()
    state = model.initSystem()
    joint = model.getComponent('/jointset/shoulder_r')
    humerus_mobod = joint_mobod_index(model, '/jointset/shoulder_r')

    rng = np.random.default_rng(0)
    q = rng.uniform(-0.3, 0.3, state.getNQ())
    radii = np.array(RADII)
    station = np.array([0.1, 0.2, 0.3])
    target = np.array([1.0, 2.0, 3.0])
    marker = model.getComponent('/markerset/forearm_marker')

    def apply(radii_value):
        joint.setRadii(state, osim.Vec3(*[float(v) for v in radii_value]))
        state.setQ(osim.Vector.createFromMat(q))
        model.realizePosition(state)

    def error():
        p = marker.getLocationInGround(state).to_numpy()
        return 0.5 * np.sum((p - target) ** 2)

    apply(radii)
    p_GS = marker.getLocationInGround(state).to_numpy()
    num_mobod = model.getNumBodies() + 1
    dp_GB = osim.VectorVec3(num_mobod, osim.Vec3(0))
    dp_GB.set(int(marker.getParentFrame().getMobilizedBodyIndex()),
              osim.Vec3(*[float(v) for v in (p_GS - target)]))

    subtree_sum = model.multiplyByPositionJacobianWrtMobilizerTranslationTranspose(
        state, humerus_mobod, dp_GB).to_numpy()
    local = joint.calcPositionJacobianWrtRadii(state)
    local = np.array([[local.get(r, c) for c in range(3)] for r in range(3)])
    analytic = local.T @ subtree_sum

    h = 1e-7
    finite_difference = np.zeros(3)
    for i in range(3):
        plus = radii.copy(); plus[i] += h
        apply(plus); e_plus = error()
        minus = radii.copy(); minus[i] -= h
        apply(minus); e_minus = error()
        finite_difference[i] = (e_plus - e_minus) / (2.0 * h)

    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-5)


##################
# MODEL APPLYING #
##################

def test_apply_to_model_is_multiplicative():
    model = create_ellipsoid_model()
    model.initSystem()

    radii = EllipsoidRadii('/jointset/shoulder_r', Bounds(0.5, 2.0),
                           np.array([1.5, 2.0, 0.5]))
    radii.apply_to_model(model)
    np.testing.assert_allclose(
        model.getComponent('/jointset/shoulder_r').get_radii_x_y_z().to_numpy(),
        np.array(RADII) * np.array([1.5, 2.0, 0.5]))

    length = BeamLength('/jointset/elbow_r', Bounds(0.5, 2.0), np.array([2.0]))
    length.apply_to_model(model)
    assert model.getComponent('/jointset/elbow_r').get_beam_length() == \
        pytest.approx(BEAM_LENGTH * 2.0)


def test_get_and_apply_ellipsoid_joint_radii_round_trip():
    model = create_ellipsoid_model()
    model.initSystem()

    saved = ModelCache.get_ellipsoid_joint_radii(model)
    assert set(saved) == {'/jointset/shoulder_r'}

    model.getComponent('/jointset/shoulder_r').set_radii_x_y_z(
        osim.Vec3(9.0, 9.0, 9.0))
    ModelCache.apply_ellipsoid_joint_radii(model, saved)
    np.testing.assert_allclose(
        model.getComponent('/jointset/shoulder_r').get_radii_x_y_z().to_numpy(), RADII)


def test_model_scale_would_scale_radii_without_the_restore():
    """
    Document the behavior `SplinedKinematicsSolver.update_model` compensates for:
    `EllipsoidJoint::extendScale` multiplies the radii by the parent frame's body
    scale factors, and restoring the saved radii undoes exactly that.
    """
    model = create_ellipsoid_model()
    state = model.initSystem()
    saved = ModelCache.get_ellipsoid_joint_radii(model)

    scaleset = osim.ScaleSet()
    scale = osim.Scale()
    scale.setSegmentName('torso')
    scale.setScaleFactors(osim.Vec3(2.0, 2.0, 2.0))
    scaleset.cloneAndAppend(scale)
    scaleset.get(0).setName('torso')
    model.scale(state, scaleset, True)

    scaled = model.getComponent(
        '/jointset/shoulder_r').get_radii_x_y_z().to_numpy()
    np.testing.assert_allclose(scaled, np.array(RADII) * 2.0)

    ModelCache.apply_ellipsoid_joint_radii(model, saved)
    np.testing.assert_allclose(
        model.getComponent('/jointset/shoulder_r').get_radii_x_y_z().to_numpy(), RADII)


def test_update_model_leaves_beam_length_independent_of_body_scales():
    """
    `Model::scale()` does not touch 'beam_length', so a `BeamLength` factor applies
    cleanly on top of baseline regardless of a body scale on a neighboring body.
    """
    from osimfit.solvers import Solution

    solver = SplinedKinematicsSolver(create_beam_model())
    body_scale = BodyScale('/bodyset/humerus', Bounds(0.5, 2.0),
                           np.array([2.0, 2.0, 2.0]))
    length = BeamLength('/jointset/elbow_r', Bounds(0.5, 2.0), np.array([1.5]))
    solver.add_parameter(body_scale)
    solver.add_parameter(length)

    updated = solver.update_model(
        create_beam_model(), Solution(parameters=[body_scale, length]))

    assert updated.getComponent('/jointset/elbow_r').get_beam_length() == \
        pytest.approx(BEAM_LENGTH * 1.5)


##############
# END-TO-END #
##############

def create_prescribed_markers(model: osim.Model, trc_path: str,
                              num_times: int = 40, freq: float = 0.5) -> None:
    """
    Write marker positions for a smooth, prescribed coordinate trajectory to a TRC
    file. A prescribed trajectory is used rather than a forward simulation so the
    resulting fit is well conditioned and cheap to solve.
    """
    state = model.initSystem()
    coordinates = [model.getCoordinateSet().get(i)
                   for i in range(model.getCoordinateSet().getSize())]
    markers = [model.getMarkerSet().get(i)
               for i in range(model.getMarkerSet().getSize())]

    times = np.linspace(0.0, 1.0, num_times)
    table = osim.TimeSeriesTableVec3()
    for time in times:
        for icoord, coordinate in enumerate(coordinates):
            coordinate.setValue(
                state,
                0.3 * np.sin(2.0 * np.pi * freq * time + 0.7 * icoord), False)
        model.assemble(state)
        model.realizePosition(state)
        row = osim.RowVectorVec3(len(markers))
        for imarker, marker in enumerate(markers):
            location = marker.getLocationInGround(state)
            for i in range(3):
                row.updElt(0, imarker).set(i, location[i])
        table.appendRow(time, row)

    table.setColumnLabels([m.getAbsolutePathString() for m in markers])
    table.addTableMetaDataString('DataRate', str(num_times))
    table.addTableMetaDataString('Units', 'm')
    osim.TRCFileAdapter().write(table, trc_path)


def test_solver_recovers_beam_length(tmp_path):
    """
    Synthesize marker data from a model whose beam is 1.3x its nominal length, then
    solve against the nominal model while optimizing the beam-length factor. The
    recovered factor must match the truth.
    """
    from osimfit.data_sources import MarkerSource, Trial

    true_factor = 1.3

    truth = create_beam_recovery_model()
    truth.getComponent('/jointset/elbow_r').set_beam_length(
        BEAM_LENGTH * true_factor)
    truth.finalizeConnections()

    trc_path = str(tmp_path / 'markers.trc')
    create_prescribed_markers(truth, trc_path)

    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}

    model = create_beam_recovery_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(
        model, convergence_tolerance=1e-6, knot_interval=0.1, position_weight=5.0)
    solver.add_trial(Trial('beam', [MarkerSource('markers', trc_path,
                                                 label_map=label_map)]))
    solver.add_parameter(BeamLength('/jointset/elbow_r', Bounds(0.5, 2.0)))

    solution = solver.solve()

    lengths = [p for p in solution.parameters if isinstance(p, BeamLength)]
    assert len(lengths) == 1
    assert lengths[0].paths == ['/jointset/elbow_r']
    np.testing.assert_allclose(lengths[0].value, [true_factor], atol=0.01)

    # The optimized factor must bake into the updated model's property.
    updated = solver.update_model(create_beam_recovery_model(), solution)
    assert updated.getComponent('/jointset/elbow_r').get_beam_length() == \
        pytest.approx(BEAM_LENGTH * lengths[0].value[0])


def test_solver_recovers_ellipsoid_radii(tmp_path):
    """
    Synthesize marker data from a model whose ellipsoid radii are scaled by known
    factors, then solve against the nominal model while optimizing those factors.
    """
    from osimfit.data_sources import MarkerSource, Trial

    true_factors = np.array([1.4, 0.7, 1.2])

    truth = create_ellipsoid_recovery_model()
    truth.getComponent('/jointset/shoulder_r').set_radii_x_y_z(
        osim.Vec3(*[float(v) for v in np.array(RADII) * true_factors]))
    truth.finalizeConnections()

    trc_path = str(tmp_path / 'markers.trc')
    create_prescribed_markers(truth, trc_path)

    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}

    model = create_ellipsoid_recovery_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(
        model, convergence_tolerance=1e-6, knot_interval=0.1, position_weight=5.0)
    solver.add_trial(Trial('ellipsoid', [MarkerSource('markers', trc_path,
                                                      label_map=label_map)]))
    solver.add_parameter(EllipsoidRadii('/jointset/shoulder_r', Bounds(0.2, 3.0)))

    solution = solver.solve()

    radii = [p for p in solution.parameters if isinstance(p, EllipsoidRadii)]
    assert len(radii) == 1
    np.testing.assert_allclose(radii[0].value, true_factors, atol=0.02)

    # The optimized factors must bake into the updated model's property.
    updated = solver.update_model(create_ellipsoid_recovery_model(), solution)
    np.testing.assert_allclose(
        updated.getComponent('/jointset/shoulder_r').get_radii_x_y_z().to_numpy(),
        np.array(RADII) * radii[0].value)


def test_update_model_leaves_radii_independent_of_body_scales():
    """
    `EllipsoidJoint::extendScale` multiplies the radii by the parent frame's body scale
    factors during `Model::scale()`. This fitter treats mobilizer geometry as
    independent of body scaling, so `update_model` must undo that: an `EllipsoidRadii`
    factor applies on top of the baseline radii, not on top of the scaled radii.
    """
    from osimfit.solvers import Solution

    solver = SplinedKinematicsSolver(create_ellipsoid_recovery_model())
    body_scale = BodyScale('/bodyset/torso', Bounds(0.5, 2.0),
                           np.array([2.0, 2.0, 2.0]))
    radii = EllipsoidRadii('/jointset/shoulder_r', Bounds(0.2, 3.0),
                           np.array([1.5, 1.0, 1.0]))
    solver.add_parameter(body_scale)
    solver.add_parameter(radii)

    updated = solver.update_model(
        create_ellipsoid_recovery_model(),
        Solution(parameters=[body_scale, radii]))

    np.testing.assert_allclose(
        updated.getComponent('/jointset/shoulder_r').get_radii_x_y_z().to_numpy(),
        np.array(RADII) * np.array([1.5, 1.0, 1.0]))


def test_update_model_without_radii_parameter_preserves_radii():
    """
    A body scale on the ellipsoid's parent with no `EllipsoidRadii` registered must
    still leave the radii at baseline.
    """
    from osimfit.solvers import Solution

    solver = SplinedKinematicsSolver(create_ellipsoid_recovery_model())
    body_scale = BodyScale('/bodyset/torso', Bounds(0.5, 2.0),
                           np.array([2.0, 2.0, 2.0]))
    solver.add_parameter(body_scale)

    updated = solver.update_model(
        create_ellipsoid_recovery_model(), Solution(parameters=[body_scale]))

    np.testing.assert_allclose(
        updated.getComponent('/jointset/shoulder_r').get_radii_x_y_z().to_numpy(),
        RADII)


#########################
# REGULARIZATION COST   #
#########################

def test_mobilizer_dimension_regularization_penalizes_deviation():
    from osimfit.costs import MobilizerDimensionRegularizationCost

    cost = MobilizerDimensionRegularizationCost(weight=2.0)
    rep = cost.create_rep(ModelCache(create_beam_model()))

    at_target = rep(CostInput(ellipsoid_radii=ca.DM.ones(3),
                              beam_lengths=ca.DM.ones(1)))
    assert float(at_target) == pytest.approx(0.0)

    # weight * sum of squared deviations from 1.0.
    deviated = rep(CostInput(ellipsoid_radii=ca.DM([1.1, 0.9, 1.0]),
                             beam_lengths=ca.DM([1.2])))
    assert float(deviated) == pytest.approx(
        2.0 * ((0.1)**2 + (0.1)**2 + 0.0 + (0.2)**2))


def test_mobilizer_dimension_regularization_rejects_negative_weight():
    from osimfit.costs import MobilizerDimensionRegularizationCost
    with pytest.raises(ValueError, match='non-negative'):
        MobilizerDimensionRegularizationCost(weight=-1.0)


def test_splined_solver_accepts_mobilizer_dimension_regularization():
    from osimfit.costs import MobilizerDimensionRegularizationCost
    solver = SplinedKinematicsSolver(create_beam_model())
    solver.add_cost(MobilizerDimensionRegularizationCost(1e-2))
    assert len(solver.costs) == 1
