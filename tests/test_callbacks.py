"""
Unit tests for TrackingCostFunction and BilevelCostFunction.
"""

import pytest
import numpy as np
import casadi as ca
import opensim as osim
from pathlib import Path
from osimfit.callbacks import (BilevelCostFunction, BodyScaleGroup,
                               MarkerOffsetGroup, FrameOffsetGroup,
                               TrackingCostFunction)
from osimfit.model import ModelCache

# Define the test model path.
MODEL_FPATH = str(Path(__file__).parent / 'subject_scale_walk.osim')


###########
# HELPERS #
###########

def create_sliding_mass_model(child_x_offset: float = 0.0):
    """
    Create a model with ne body sliding along the X-direction in ground with two markers
    in the body frame: 'm0' at the origin, 'm1' at (0.5, 0, 0). Use `child_x_offset` to
    add offset in the X-direction for the child body's joint frame.
    """
    model = osim.Model()
    model.setName('sliding_mass')
    ground = model.getGround()
    body = osim.Body('body', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body)
    joint = osim.SliderJoint(
        'slider',
        ground, osim.Vec3(0), osim.Vec3(0),
        body, osim.Vec3(child_x_offset, 0, 0), osim.Vec3(0),
    )
    model.addJoint(joint)
    model.addMarker(osim.Marker('m0', body, osim.Vec3(0)))
    model.addMarker(osim.Marker('m1', body, osim.Vec3(0.5, 0, 0)))
    model.finalizeConnections()
    return model


def create_n_sliding_body_model(n: int, child_x_offset: float = 0.0):
    """
    Create a model with `n` independent bodies, each on its own slider joint along the
    X-direction in ground, each with one marker at body-frame (0.5, 0, 0). Mobilized
    body indexes are 1..n in body-addition order. Use `child_x_offset` to
    add offset in the X-direction for each child body's joint frame.
    """
    model = osim.Model()
    model.setName(f'{n}_sliding_mass')
    ground = model.getGround()
    for i in range(n):
        body = osim.Body(f'body_{i}', 1.0, osim.Vec3(0), osim.Inertia(1))
        model.addBody(body)
        joint = osim.SliderJoint(
            f'slider_{i}',
            ground, osim.Vec3(0), osim.Vec3(0),
            body, osim.Vec3(child_x_offset, 0, 0), osim.Vec3(0),
        )
        model.addJoint(joint)
        model.addMarker(osim.Marker(f'm{i}', body, osim.Vec3(0.5, 0, 0)))
    model.finalizeConnections()
    return model


def getP_BM(model: osim.Model, joint_index: int, state: osim.State):
    """
    Return the position of the child frame from the Joint at index `joint_index`.
    """
    return model.getJointSet().get(joint_index).getOutboardFrame(state).p().to_numpy()


##########################
# TRACKING COST FUNCTION #
##########################

def test_tracking_cost_function_constructs_marker_and_frame_subcosts():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
    assert cost.marker_cost is not None
    assert cost.frame_cost is not None


def test_tracking_cost_function_add_marker_registers_in_marker_cost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
    cost.add_marker_tracking_cost('/markerset/m0', osim.Vec3(0))
    assert len(cost.marker_cost.markers) == 1
    assert cost.marker_cost.mobod_indexes.size() == 1
    assert len(cost.frame_cost.frames) == 0


def test_tracking_cost_function_add_frame_registers_in_frame_cost():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
    cost.add_frame_tracking_cost(
        '/bodyset/pelvis', osim.Vec3(0), osim.Quaternion())
    assert len(cost.frame_cost.frames) == 1
    assert cost.frame_cost.mobod_indexes.size() == 1
    assert len(cost.marker_cost.markers) == 0


def test_empty_tracking_cost_function():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
    x = ca.DM.zeros(len(cost.mc.q_indexes))
    assert float(cost(x)) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_at_reference_yields_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
    # At q=0, m0 sits at the world origin.
    cost.add_marker_tracking_cost('/markerset/m0', osim.Vec3(0))
    x = ca.DM.zeros(len(cost.mc.q_indexes))
    assert float(cost(x)) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_off_reference_yields_squared_error():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
    # m0 at world (0.1, 0, 0) when q=0.1; reference at the origin.
    cost.add_marker_tracking_cost(
        '/markerset/m0', osim.Vec3(0.0, 0, 0), weight=1.0)
    x = ca.DM([0.1])
    assert float(cost(x)) == pytest.approx(0.01, abs=1e-9)


def test_tracking_cost_function_jacobian_sliding_mass():
    model = create_sliding_mass_model()
    model.initSystem()
    cost_jac = TrackingCostFunction('cost_jac', ModelCache(model))
    cost_fd = TrackingCostFunction('cost_fd', ModelCache(model),
                                   opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_tracking_cost(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_tracking_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)

    x = ca.SX.sym('x', len(cost_jac.mc.q_indexes))
    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(x), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(x), x)])

    assert np.allclose(J_jac(0.1).full(), J_fd(0.1).full(), atol=1e-6)


def test_tracking_cost_function_jacobian_full_body():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost_jac = TrackingCostFunction('cost_jac', ModelCache(model))
    cost_fd = TrackingCostFunction('cost_fd', ModelCache(model),
                                   opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_tracking_cost(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_tracking_cost(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)

    x = ca.SX.sym('x', len(cost_jac.mc.q_indexes))
    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(x), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(x), x)])

    assert np.allclose(J_jac(0.1).full(), J_fd(0.1).full(), atol=1e-6)


#########################
# BILEVEL COST FUNCTION #
#########################

def test_bilevel_cost_function_constructs_marker_subcost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    assert cost.marker_cost is not None
    assert cost.body_scale_groups == [BodyScaleGroup(['/bodyset/body'], [1])]


def test_bilevel_cost_function_add_marker_registers_in_marker_cost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_marker_bilevel_cost('/markerset/m0', osim.Vec3(0))
    assert cost.marker_cost.mobod_indexes.size() == 1
    assert len(cost.frame_cost.frames) == 0


def test_bilevel_cost_function_add_frame_registers_in_frame_cost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion())
    assert cost.frame_cost.mobod_indexes.size() == 1
    assert len(cost.marker_cost.markers) == 0


def test_bilevel_apply_scales_shifts_child_frame_translation():
    """
    Applying body scales through the cost should multiply each component of the model's
    child frame translation elementswise.
    """
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])

    cost.mc.set_scaled_mobilizer_frame_positions(
        cost.state, cost.body_scale_groups, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(getP_BM(model, 0, cost.state),
                               np.array([0.4 * 2.0, 0.0, 0.0]))


def test_bilevel_apply_scales_shared_group_broadcasts_across_members():
    """
    A scale group must apply the same set of scale factors to every member body's
    child frame translations.
    """
    model = create_n_sliding_body_model(2, child_x_offset=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(
            ['/bodyset/body_0', '/bodyset/body_1'], [1, 2])],
        marker_offset_groups=[], frame_offset_groups=[])

    cost.mc.set_scaled_mobilizer_frame_positions(
        cost.state, cost.body_scale_groups, np.array([2.0, 3.0, 4.0]))
    for k in (0, 1):
        np.testing.assert_allclose(getP_BM(model, k, cost.state),
                                   np.array([0.4 * 2.0, 0.0, 0.0]))


def test_bilevel_apply_scales_mixed_groups_apply_independent_vectors():
    """
    Separate scale groups must apply scale factors to owned bodies independently.
    """
    model = create_n_sliding_body_model(3, child_x_offset=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[
            BodyScaleGroup(['/bodyset/body_0', '/bodyset/body_1'], [1, 2]),
            BodyScaleGroup(['/bodyset/body_2'], [3]),
        ],
        marker_offset_groups=[], frame_offset_groups=[])

    cost.mc.set_scaled_mobilizer_frame_positions(
        cost.state, cost.body_scale_groups, np.array([2.0, 3.0, 4.0, 5.0, 5.0, 5.0]))
    for k in (0, 1):
        np.testing.assert_allclose(getP_BM(model, k, cost.state),
                                   np.array([0.4 * 2.0, 0.0, 0.0]))
    np.testing.assert_allclose(getP_BM(model, 2, cost.state),
                               np.array([0.4 * 5.0, 0.0, 0.0]))


def test_bilevel_cost_function_empty_eval_is_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s = ca.DM.ones(3)
    assert float(cost(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_marker_world_position():
    """
    Scaling a body changes both the segment length (via offset frame scaling) and the
    positions of markers on the body.
    """
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_marker_bilevel_cost('/markerset/m1', osim.Vec3(0.5, 0, 0))

    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])
    # At s_unit: m1 world = (-0.4 + 0.5) = 0.1. Error = (0.1 - 0.5)^2 = 0.16.
    assert float(cost(q, s_unit, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.16, abs=1e-9)
    # At s_scaled X=2: m1 world = (-0.8 + 1.0) = 0.2. Error = (0.2 - 0.5)^2 = 0.09.
    assert float(cost(q, s_scaled, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.09, abs=1e-9)


def test_bilevel_cost_function_frame_at_reference_yields_zero():
    """
    A frame tracked at its own world position and orientation yields zero error.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost('/bodyset/body', osim.Vec3(0), osim.Quaternion())
    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s = ca.DM([1.0, 1.0, 1.0])
    assert float(cost(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_frame_world_position():
    """
    Scaling a body with a non-zero child frame offset shifts the child frame in ground.
    The child frame offset should contribute to the squared error in the cost.
    """
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion(), position_weight=2.0)

    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])
    # At s_unit: origin = -0.4. Error = 2 * (-0.4)^2 = 0.32.
    assert float(cost(q, s_unit, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.32, abs=1e-9)
    # At s_scaled X=2: origin = -0.8. Error = 2 * (-0.8)^2 = 1.28.
    assert float(cost(q, s_scaled, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(1.28, abs=1e-9)


def test_bilevel_cost_function_jacobians_sliding_mass():
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost_jac = BilevelCostFunction(
        'cost_jac', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[],
        opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost(
            '/bodyset/body', osim.Vec3(0.5, 0, 0), osim.Quaternion(),
            position_weight=1.5, orientation_weight=1.0)

    q = ca.SX.sym('q', len(cost_jac.mc.q_indexes))
    s = ca.SX.sym('s', 3)
    x = ca.vertcat(q, s)

    J_jac = ca.Function(
        'J_jac', [x],
        [ca.jacobian(cost_jac(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)), x)])
    J_fd = ca.Function(
        'J_fd', [x],
        [ca.jacobian(cost_fd(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)), x)])

    val = np.concatenate([
        np.full(len(cost_jac.mc.q_indexes), 0.1),
        np.array([1.1, 1.0, 1.0]),
    ])
    assert np.allclose(J_jac(val).full(), J_fd(val).full(), atol=1e-6)


def test_bilevel_cost_function_jacobians_full_body():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    bodyset = model.getBodySet()
    body_scale_groups = []
    for i in range(bodyset.getSize()):
        body = bodyset.get(i)
        body_scale_groups.append(BodyScaleGroup(
            body_paths=[body.getAbsolutePathString()],
            mobod_indexes=[int(body.getMobilizedBodyIndex())]))

    cost_jac = BilevelCostFunction(
        'cost_jac', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=[], frame_offset_groups=[], opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost(
            '/bodyset/pelvis', osim.Vec3(0.3, 0.1, -0.2),
            osim.Quaternion(0.9, 0.1, 0.2, 0.3),
            position_weight=2.0, orientation_weight=1.5)

    q = ca.SX.sym('q', len(cost_jac.mc.q_indexes))
    s = ca.SX.sym('s', 3*bodyset.getSize())
    x = ca.vertcat(q, s)

    J_jac = ca.Function(
        'J_jac', [x],
        [ca.jacobian(cost_jac(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)), x)])
    J_fd = ca.Function(
        'J_fd', [x],
        [ca.jacobian(cost_fd(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)), x)])

    val = np.concatenate([
        np.full(len(cost_jac.mc.q_indexes), 0.1),
        np.tile([1.1, 1.0, 1.0], bodyset.getSize()),
    ])
    assert np.allclose(J_jac(val).full(), J_fd(val).full(), atol=1e-6)


def test_bilevel_cost_function_grouped_jacobian_sums_solo_and_matches_fd():
    """
    For a 2-body model with one marker per body, the shared-group Jacobian
    column for the shared scalar must (a) equal the sum of the solo Jacobian
    columns when both solo scales are set to the same value (chain rule), and
    (b) agree with the finite-difference Jacobian of the shared callback.
    """
    model = create_n_sliding_body_model(2, child_x_offset=0.4)
    model.initSystem()

    solo_groups = [
        BodyScaleGroup(['/bodyset/body_0'], [1]),
        BodyScaleGroup(['/bodyset/body_1'], [2]),
    ]
    shared_groups = [
        BodyScaleGroup(['/bodyset/body_0', '/bodyset/body_1'], [1, 2]),
    ]
    cost_solo = BilevelCostFunction(
        'cost_solo', ModelCache(model), body_scale_groups=solo_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_shared = BilevelCostFunction(
        'cost_shared', ModelCache(model), body_scale_groups=shared_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model), body_scale_groups=shared_groups,
        marker_offset_groups=[], frame_offset_groups=[], opts={'enable_fd': True})

    for cost in (cost_solo, cost_shared, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/m0', osim.Vec3(0.4, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost(
            '/bodyset/body_0', osim.Vec3(0.2, 0, 0), osim.Quaternion(),
            position_weight=1.0)
        cost.add_frame_bilevel_cost(
            '/bodyset/body_1', osim.Vec3(0.5, 0, 0), osim.Quaternion(),
            position_weight=1.2)

    nq = len(cost_shared.mc.q_indexes)
    q = ca.SX.sym('q', nq)
    offset = ca.DM.zeros(0, 1)

    # (b) Shared analytic ≈ FD on the shared callback.
    s_shared = ca.SX.sym('s_shared', 3)
    x_shared = ca.vertcat(q, s_shared)
    J_shared_fn = ca.Function(
        'J_shared', [x_shared],
        [ca.jacobian(cost_shared(q, s_shared, offset, offset), x_shared)])
    J_fd_fn = ca.Function(
        'J_fd', [x_shared],
        [ca.jacobian(cost_fd(q, s_shared, offset, offset), x_shared)])
    val_shared = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0]),
    ])
    J_shared = J_shared_fn(val_shared).full()
    J_fd = J_fd_fn(val_shared).full()
    assert np.allclose(J_shared, J_fd, atol=1e-6)

    # (a) Shared body-scale column equals the sum of solo body-scale
    # columns evaluated at the same s applied to both bodies.
    s_solo = ca.SX.sym('s_solo', 6)
    x_solo = ca.vertcat(q, s_solo)
    J_solo_fn = ca.Function(
        'J_solo', [x_solo],
        [ca.jacobian(cost_solo(q, s_solo, offset, offset), x_solo)])
    val_solo = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0, 1.1, 1.0, 1.0]),
    ])
    J_solo = J_solo_fn(val_solo).full()
    solo_sum_cols = J_solo[:, nq:nq+3] + J_solo[:, nq+3:nq+6]
    np.testing.assert_allclose(J_shared[:, nq:nq+3], solo_sum_cols,
                               atol=1e-9)


def test_bilevel_apply_state_shifts_station():
    """
    Use apply_state() to set each offset task's cached station to baseline + offset
    at identity body scale, leaving non-offset tasks untouched.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[MarkerOffsetGroup(['/markerset/m1'], [2])],
        frame_offset_groups=[])
    cost.add_marker_bilevel_cost('/markerset/m1', osim.Vec3(0.5, 0, 0),
                                 offset_group_index=0)
    cost.add_marker_bilevel_cost('/markerset/m0', osim.Vec3(0, 0, 0))
    mc = cost.marker_cost
    baseline_m1 = mc.base_stations[0].copy()
    baseline_m0 = mc.base_stations[1].copy()

    body_scale = np.ones(3)
    offset = np.array([0.1, -0.2, 0.3])
    mc.apply_state(body_scale, offset)
    np.testing.assert_allclose(mc.stations.getElt(0).to_numpy(),
                               baseline_m1 + offset)
    np.testing.assert_allclose(mc.stations.getElt(1).to_numpy(), baseline_m0)

    # apply_state() is idempotent.
    mc.apply_state(body_scale, offset)
    np.testing.assert_allclose(mc.stations.getElt(0).to_numpy(),
                               baseline_m1 + offset)


def test_bilevel_offset_changes_marker_error():
    """
    Applying an offset to a marker shifts its position and ground and shoudl yield a
    change in the tracking error.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[MarkerOffsetGroup(['/markerset/m1'], [2])],
        frame_offset_groups=[])
    cost.add_marker_bilevel_cost('/markerset/m1', osim.Vec3(0.5, 0, 0),
                                 offset_group_index=0)
    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s = ca.DM.ones(3)
    # No offset: m1 world = 0.5, reference = 0.5, error = 0.
    assert float(cost(q, s, ca.DM.zeros(3), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.0, abs=1e-12)
    # Offset X by 0.2: m1 world = 0.7, error = (0.7 - 0.5)^2 = 0.04.
    assert float(cost(q, s, ca.DM([0.2, 0, 0]), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.04, abs=1e-9)


def test_bilevel_offset_frame_orientation_invariant():
    """
    Applying a translation offset to frame's position should not affect its orientation,
    so an orientation-only frame cost (position_weight = 0) is invariant to the offset.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[],
        frame_offset_groups=[FrameOffsetGroup(['/bodyset/body'], [1])])
    cost.add_frame_bilevel_cost(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion(0.9, 0.1, 0.2, 0.3),
        position_weight=0.0, orientation_weight=1.0, offset_group_index=0)
    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s = ca.DM.ones(3)
    e0 = float(cost(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(3)))
    e1 = float(cost(q, s, ca.DM.zeros(0, 1), ca.DM([0.2, -0.1, 0.3])))
    assert e0 > 0.0
    assert e0 == pytest.approx(e1, abs=1e-12)


def test_bilevel_cost_function_offset_jacobians_full_body():
    """
    On the full-body model, the analytic bilevel Jacobian over [q, s, o], including the
    marker and frame offset columns and the offset-induced coupling into the q-columns,
    must match the finite-difference Jacobian.
    """
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    pelvis = osim.Body.safeDownCast(model.getComponent('/bodyset/pelvis'))
    pelvis_mbx = int(pelvis.getMobilizedBodyIndex())
    torso = osim.Body.safeDownCast(model.getComponent('/bodyset/torso'))
    torso_mbx = int(torso.getMobilizedBodyIndex())
    body_scale_groups = [BodyScaleGroup(['/bodyset/pelvis'], [pelvis_mbx])]

    marker_offset_groups = [MarkerOffsetGroup(['/markerset/R.Shoulder'], [torso_mbx])]
    frame_offset_groups = [FrameOffsetGroup(['/bodyset/pelvis'], [pelvis_mbx])]
    cost_jac = BilevelCostFunction(
        'cost_jac', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=marker_offset_groups,
        frame_offset_groups=frame_offset_groups)
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=marker_offset_groups,
        frame_offset_groups=frame_offset_groups, opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0,
            offset_group_index=0)
        cost.add_frame_bilevel_cost(
            '/bodyset/pelvis', osim.Vec3(0.3, 0.1, -0.2),
            osim.Quaternion(0.9, 0.1, 0.2, 0.3),
            position_weight=2.0, orientation_weight=1.5,
            offset_group_index=0)

    nq = len(cost_jac.mc.q_indexes)
    q = ca.SX.sym('q', nq)
    s = ca.SX.sym('s', 3)
    mo = ca.SX.sym('mo', 3)
    fo = ca.SX.sym('fo', 3)
    x = ca.vertcat(q, s, mo, fo)

    J_jac = ca.Function('J_jac', [x],
                        [ca.jacobian(cost_jac(q, s, mo, fo), x)])
    J_fd = ca.Function('J_fd', [x],
                       [ca.jacobian(cost_fd(q, s, mo, fo), x)])

    val = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0]),
        np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.0]),
    ])
    A = J_jac(val).full()
    F = J_fd(val).full()
    assert np.allclose(A, F, atol=1e-6)
    # The offset columns should be non-zero.
    assert np.any(np.abs(A[0, nq+3:nq+9]) > 1e-8)
