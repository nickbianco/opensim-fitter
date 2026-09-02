"""
Unit tests for `Cost`s.
"""

import pytest
import numpy as np
import casadi as ca
import opensim as osim

from pathlib import Path
from osimfit.model import (ModelCache, BodyScale, BodyScaleGroup, MarkerOffsetGroup,
                           FrameOffsetGroup, StationCache)
from osimfit.bounds import Bounds
from osimfit.solvers import InverseKinematicsSolver, SplinedKinematicsSolver
from osimfit.costs import (AnthropometricRegularizationCostRep, CostInput,
                           CostRep, SymbolicCost, SymbolicCostRep,
                           BodyScaleRegularizationCost, BodyScaleIsotropyCost,
                           OffsetRegularizationCost, BilevelCostRep,
                           TrackingCost, TrackingCostRep,
                           AnthropometricRegularizationCost)
from osimfit.scaling import Axis, AnthropometricMeasurement
from tests.test_double_pendulum import create_double_pendulum

# Define the test model path.
MODEL_FPATH = str(Path(__file__).parent / 'subject_scale_walk.osim')


##############
# VALIDATION #
##############

class CoordinatePenalty(SymbolicCost):
    """A minimal cost that depends only on the coordinates."""
    required_inputs = frozenset({'coordinates'})

    def evaluate(self, input: CostInput) -> ca.MX:
        return ca.sumsqr(input.coordinates)


@pytest.fixture
def double_pendulum_model():
    m = create_double_pendulum(1.0, 1.0)
    m.initSystem()
    return m

def test_inverse_kinematics_accepts_coordinate_cost(double_pendulum_model):
    solver = InverseKinematicsSolver(double_pendulum_model)
    solver.add_cost(CoordinatePenalty())
    assert len(solver.costs) == 1


def test_inverse_kinematics_rejects_parameter_cost(double_pendulum_model):
    solver = InverseKinematicsSolver(double_pendulum_model)
    with pytest.raises(ValueError, match='body_scales'):
        solver.add_cost(BodyScaleRegularizationCost(1.0))
    assert solver.costs == []


def test_splined_accepts_parameter_costs(double_pendulum_model):
    solver = SplinedKinematicsSolver(double_pendulum_model)
    solver.add_cost(BodyScaleRegularizationCost(1.0))
    solver.add_cost(OffsetRegularizationCost(1.0))
    assert len(solver.costs) == 2


def test_splined_accepts_coordinate_cost(double_pendulum_model):
    """
    A coordinate-dependent cost is evaluated at every time sample of every trial, so
    the splined solver supports it alongside its parameter costs.
    """
    solver = SplinedKinematicsSolver(double_pendulum_model)
    solver.add_cost(CoordinatePenalty())
    solver.add_cost(BodyScaleRegularizationCost(1.0))
    assert len(solver.costs) == 2


def test_registered_cost_reps_size_themselves_from_the_solvers_parameters():
    """
    A registered cost's rep is built from the solver's own ModelCache, so its declared
    CasADi input size follows the parameters registered on that solver: an
    AnthropometricRegularizationCostRep declares 3 * len(mc.body_scale_groups), which is
    only final once add_parameter() has been called for every body scale.
    """
    solver = SplinedKinematicsSolver(create_two_link_model())
    cost = AnthropometricRegularizationCost(
        [AnthropometricMeasurement('stature', '/S0', '/S1', Axis.YAxis)],
        sex='female')
    solver.add_cost(cost)
    for body_path in ('/bodyset/b0', '/bodyset/b1'):
        solver.add_parameter(BodyScale(body_path, Bounds(0.5, 2.0), np.ones(3)))

    rep = cost.create_rep(solver.mc)

    assert isinstance(rep, AnthropometricRegularizationCostRep)
    assert rep.mc is solver.mc
    num_scales = 3 * len(solver.mc.body_scale_groups)
    assert num_scales == 6
    assert rep.size1_in(0) == num_scales

    # Every solve builds a fresh rep rather than reusing the previous solve's.
    assert cost.create_rep(solver.mc) is not rep


def test_add_cost_rejects_a_tracking_cost(double_pendulum_model):
    """
    The registered-cost loop builds reps from the ModelCache alone, so a
    TrackingCostBase (which a solver is meant to construct itself, per trial and time
    sample) is not a Cost and cannot be registered.
    """
    solver = InverseKinematicsSolver(double_pendulum_model)
    with pytest.raises(TypeError, match='expects a Cost'):
        solver.add_cost(TrackingCost())
    assert solver.costs == []


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

def test_tracking_cost_function_constructs_marker_and_frame_terms():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    assert cost.marker_term is not None
    assert cost.frame_term is not None


def test_tracking_cost_function_add_marker_registers_in_marker_term():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    cost.add_marker_tracking_cost_term('/markerset/m0', osim.Vec3(0))
    assert len(cost.marker_term.markers) == 1
    assert cost.marker_term.mobod_indexes.size() == 1
    assert len(cost.frame_term.frames) == 0


def test_tracking_cost_function_add_frame_registers_in_frame_term():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    cost.add_frame_tracking_cost_term(
        '/bodyset/pelvis', osim.Vec3(0), osim.Quaternion())
    assert len(cost.frame_term.frames) == 1
    assert cost.frame_term.mobod_indexes.size() == 1
    assert len(cost.marker_term.markers) == 0


def test_empty_tracking_cost_function():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    x = ca.DM.zeros(len(cost.mc.coordinate_indexes))
    assert float(cost(CostInput(coordinates=x))) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_at_reference_yields_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    # At q=0, m0 sits at the world origin.
    cost.add_marker_tracking_cost_term('/markerset/m0', osim.Vec3(0))
    x = ca.DM.zeros(len(cost.mc.coordinate_indexes))
    assert float(cost(CostInput(coordinates=x))) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_off_reference_yields_squared_error():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    # m0 at world (0.1, 0, 0) when q=0.1; reference at the origin.
    cost.add_marker_tracking_cost_term(
        '/markerset/m0', osim.Vec3(0.0, 0, 0), weight=1.0)
    x = ca.DM([0.1])
    assert float(cost(CostInput(coordinates=x))) == pytest.approx(0.01, abs=1e-9)


def test_tracking_cost_function_jacobian_sliding_mass():
    model = create_sliding_mass_model()
    model.initSystem()
    cost_jac = TrackingCostRep('cost_jac', ModelCache(model))
    cost_fd = TrackingCostRep('cost_fd', ModelCache(model),
                                   enable_fd=True)

    for cost in (cost_jac, cost_fd):
        cost.add_marker_tracking_cost_term(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_tracking_cost_term(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)

    x = ca.SX.sym('x', len(cost_jac.mc.coordinate_indexes))
    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(CostInput(x)), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(CostInput(x)), x)])

    assert np.allclose(J_jac(0.1).full(), J_fd(0.1).full(), atol=1e-6)


def test_tracking_cost_function_jacobian_full_body():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost_jac = TrackingCostRep('cost_jac', ModelCache(model))
    cost_fd = TrackingCostRep('cost_fd', ModelCache(model),
                                   enable_fd=True)

    for cost in (cost_jac, cost_fd):
        cost.add_marker_tracking_cost_term(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_tracking_cost_term(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)

    x = ca.SX.sym('x', len(cost_jac.mc.coordinate_indexes))
    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(CostInput(x)), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(CostInput(x)), x)])

    assert np.allclose(J_jac(0.1).full(), J_fd(0.1).full(), atol=1e-6)


#########################
# BILEVEL COST FUNCTION #
#########################

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


def build_bilevel_rep(name, mc, body_scale_groups=[], marker_offset_groups=[],
                       frame_offset_groups=[], ellipsoid_radii_groups=[],
                       beam_length_groups=[], enable_fd=False):
    """
    Register the given parameter groups on `mc` and build a BilevelCostRep, which
    reads its groups from the ModelCache. Mirrors what BilevelCost.create_rep does,
    without needing a Trial to supply reference data.
    """
    mc.body_scale_groups = list(body_scale_groups)
    mc.marker_offset_groups = list(marker_offset_groups)
    mc.frame_offset_groups = list(frame_offset_groups)
    mc.ellipsoid_radii_groups = list(ellipsoid_radii_groups)
    mc.beam_length_groups = list(beam_length_groups)
    return BilevelCostRep(name, mc, enable_fd=enable_fd)


def test_bilevel_cost_function_constructs_marker_term():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    assert cost.marker_term is not None
    assert cost.mc.body_scale_groups == [BodyScaleGroup(['/bodyset/body'], [1])]


def test_bilevel_cost_function_add_marker_registers_in_marker_term():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_marker_bilevel_cost_term('/markerset/m0', osim.Vec3(0))
    assert cost.marker_term.mobod_indexes.size() == 1
    assert len(cost.frame_term.frames) == 0


def test_bilevel_cost_function_add_frame_registers_in_frame_term():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost_term(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion())
    assert cost.frame_term.mobod_indexes.size() == 1
    assert len(cost.marker_term.markers) == 0


def test_bilevel_apply_scales_shifts_child_frame_translation():
    """
    Applying body scales through the cost should multiply each component of the model's
    child frame translation elementswise.
    """
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])

    cost.mc.set_scaled_mobilizer_frame_positions(
        cost.state, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(getP_BM(model, 0, cost.state),
                               np.array([0.4 * 2.0, 0.0, 0.0]))


def test_bilevel_apply_scales_shared_group_broadcasts_across_members():
    """
    A scale group must apply the same set of scale factors to every member body's
    child frame translations.
    """
    model = create_n_sliding_body_model(2, child_x_offset=0.4)
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(
            ['/bodyset/body_0', '/bodyset/body_1'], [1, 2])],
        marker_offset_groups=[], frame_offset_groups=[])

    cost.mc.set_scaled_mobilizer_frame_positions(
        cost.state, np.array([2.0, 3.0, 4.0]))
    for k in (0, 1):
        np.testing.assert_allclose(getP_BM(model, k, cost.state),
                                   np.array([0.4 * 2.0, 0.0, 0.0]))


def test_bilevel_apply_scales_mixed_groups_apply_independent_vectors():
    """
    Separate scale groups must apply scale factors to owned bodies independently.
    """
    model = create_n_sliding_body_model(3, child_x_offset=0.4)
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[
            BodyScaleGroup(['/bodyset/body_0', '/bodyset/body_1'], [1, 2]),
            BodyScaleGroup(['/bodyset/body_2'], [3]),
        ],
        marker_offset_groups=[], frame_offset_groups=[])

    cost.mc.set_scaled_mobilizer_frame_positions(
        cost.state, np.array([2.0, 3.0, 4.0, 5.0, 5.0, 5.0]))
    for k in (0, 1):
        np.testing.assert_allclose(getP_BM(model, k, cost.state),
                                   np.array([0.4 * 2.0, 0.0, 0.0]))
    np.testing.assert_allclose(getP_BM(model, 2, cost.state),
                               np.array([0.4 * 5.0, 0.0, 0.0]))


def test_bilevel_cost_function_empty_eval_is_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    q = ca.DM.zeros(len(cost.mc.coordinate_indexes))
    s = ca.DM.ones(3)
    assert float(cost(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_marker_world_position():
    """
    Scaling a body changes both the segment length (via offset frame scaling) and the
    positions of markers on the body.
    """
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_marker_bilevel_cost_term('/markerset/m1', osim.Vec3(0.5, 0, 0))

    q = ca.DM.zeros(len(cost.mc.coordinate_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])
    # At s_unit: m1 world = (-0.4 + 0.5) = 0.1. Error = (0.1 - 0.5)^2 = 0.16.
    assert float(cost(CostInput(q, s_unit, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.16, abs=1e-9)
    # At s_scaled X=2: m1 world = (-0.8 + 1.0) = 0.2. Error = (0.2 - 0.5)^2 = 0.09.
    assert float(cost(CostInput(q, s_scaled, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.09, abs=1e-9)


def test_bilevel_cost_function_frame_at_reference_yields_zero():
    """
    A frame tracked at its own world position and orientation yields zero error.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost_term('/bodyset/body', osim.Vec3(0), osim.Quaternion())
    q = ca.DM.zeros(len(cost.mc.coordinate_indexes))
    s = ca.DM([1.0, 1.0, 1.0])
    assert float(cost(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_frame_world_position():
    """
    Scaling a body with a non-zero child frame offset shifts the child frame in ground.
    The child frame offset should contribute to the squared error in the cost.
    """
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost_term(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion(), position_weight=2.0)

    q = ca.DM.zeros(len(cost.mc.coordinate_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])
    # At s_unit: origin = -0.4. Error = 2 * (-0.4)^2 = 0.32.
    assert float(cost(CostInput(q, s_unit, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.32, abs=1e-9)
    # At s_scaled X=2: origin = -0.8. Error = 2 * (-0.8)^2 = 1.28.
    assert float(cost(CostInput(q, s_scaled, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(1.28, abs=1e-9)


def test_bilevel_cost_function_jacobians_sliding_mass():
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost_jac = build_bilevel_rep(
        'cost_jac', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = build_bilevel_rep(
        'cost_fd', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[],
        enable_fd=True)

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost_term(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost_term(
            '/bodyset/body', osim.Vec3(0.5, 0, 0), osim.Quaternion(),
            position_weight=1.5, orientation_weight=1.0)

    q = ca.SX.sym('q', len(cost_jac.mc.coordinate_indexes))
    s = ca.SX.sym('s', 3)
    x = ca.vertcat(q, s)

    J_jac = ca.Function(
        'J_jac', [x],
        [ca.jacobian(cost_jac(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))), x)])
    J_fd = ca.Function(
        'J_fd', [x],
        [ca.jacobian(cost_fd(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))), x)])

    val = np.concatenate([
        np.full(len(cost_jac.mc.coordinate_indexes), 0.1),
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

    cost_jac = build_bilevel_rep(
        'cost_jac', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = build_bilevel_rep(
        'cost_fd', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=[], frame_offset_groups=[], enable_fd=True)

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost_term(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost_term(
            '/bodyset/pelvis', osim.Vec3(0.3, 0.1, -0.2),
            osim.Quaternion(0.9, 0.1, 0.2, 0.3),
            position_weight=2.0, orientation_weight=1.5)

    q = ca.SX.sym('q', len(cost_jac.mc.coordinate_indexes))
    s = ca.SX.sym('s', 3*bodyset.getSize())
    x = ca.vertcat(q, s)

    J_jac = ca.Function(
        'J_jac', [x],
        [ca.jacobian(cost_jac(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))), x)])
    J_fd = ca.Function(
        'J_fd', [x],
        [ca.jacobian(cost_fd(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))), x)])

    val = np.concatenate([
        np.full(len(cost_jac.mc.coordinate_indexes), 0.1),
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
    cost_solo = build_bilevel_rep(
        'cost_solo', ModelCache(model), body_scale_groups=solo_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_shared = build_bilevel_rep(
        'cost_shared', ModelCache(model), body_scale_groups=shared_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = build_bilevel_rep(
        'cost_fd', ModelCache(model), body_scale_groups=shared_groups,
        marker_offset_groups=[], frame_offset_groups=[], enable_fd=True)

    for cost in (cost_solo, cost_shared, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/m0', osim.Vec3(0.4, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost_term(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost_term(
            '/bodyset/body_0', osim.Vec3(0.2, 0, 0), osim.Quaternion(),
            position_weight=1.0)
        cost.add_frame_bilevel_cost_term(
            '/bodyset/body_1', osim.Vec3(0.5, 0, 0), osim.Quaternion(),
            position_weight=1.2)

    nq = len(cost_shared.mc.coordinate_indexes)
    q = ca.SX.sym('q', nq)
    offset = ca.DM.zeros(0, 1)

    # (b) Shared analytic ≈ FD on the shared callback.
    s_shared = ca.SX.sym('s_shared', 3)
    x_shared = ca.vertcat(q, s_shared)
    J_shared_fn = ca.Function(
        'J_shared', [x_shared],
        [ca.jacobian(cost_shared(CostInput(q, s_shared, offset, offset)), x_shared)])
    J_fd_fn = ca.Function(
        'J_fd', [x_shared],
        [ca.jacobian(cost_fd(CostInput(q, s_shared, offset, offset)), x_shared)])
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
        [ca.jacobian(cost_solo(CostInput(q, s_solo, offset, offset)), x_solo)])
    val_solo = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0, 1.1, 1.0, 1.0]),
    ])
    J_solo = J_solo_fn(val_solo).full()
    solo_sum_cols = J_solo[:, nq:nq+3] + J_solo[:, nq+3:nq+6]
    np.testing.assert_allclose(J_shared[:, nq:nq+3], solo_sum_cols,
                               atol=1e-9)


def test_parameter_groups_shift_station():
    """
    Composing the body-scale and marker-offset blocks onto a term's stations, in
    CostInput.INPUT_ORDER, sets each offset task's cached station to baseline + offset
    at identity body scale, leaving non-offset tasks untouched.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[MarkerOffsetGroup(['/markerset/m1'], [2])],
        frame_offset_groups=[])
    cost.add_marker_bilevel_cost_term('/markerset/m1', osim.Vec3(0.5, 0, 0),
                                 offset_group_index=0)
    cost.add_marker_bilevel_cost_term('/markerset/m0', osim.Vec3(0, 0, 0))
    term = cost.marker_term
    baseline_m1 = term.base_stations[0].copy()
    baseline_m0 = term.base_stations[1].copy()

    body_scale = np.ones(3)
    offset = np.array([0.1, -0.2, 0.3])

    def compose():
        cost.mc.parameter_groups['body_scales'].apply_to_tasks(term, body_scale)
        cost.mc.parameter_groups['marker_offsets'].apply_to_tasks(term, offset)

    compose()
    np.testing.assert_allclose(term.stations.getElt(0).to_numpy(),
                               baseline_m1 + offset)
    np.testing.assert_allclose(term.stations.getElt(1).to_numpy(), baseline_m0)

    # Composition is idempotent: the body-scale block sets each station absolutely
    # from its base-frame location before the offset block adds to it.
    compose()
    np.testing.assert_allclose(term.stations.getElt(0).to_numpy(),
                               baseline_m1 + offset)


def test_bilevel_offset_changes_marker_error():
    """
    Applying an offset to a marker shifts its position and ground and shoudl yield a
    change in the tracking error.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[MarkerOffsetGroup(['/markerset/m1'], [2])],
        frame_offset_groups=[])
    cost.add_marker_bilevel_cost_term('/markerset/m1', osim.Vec3(0.5, 0, 0),
                                 offset_group_index=0)
    q = ca.DM.zeros(len(cost.mc.coordinate_indexes))
    s = ca.DM.ones(3)
    # No offset: m1 world = 0.5, reference = 0.5, error = 0.
    assert float(cost(CostInput(q, s, ca.DM.zeros(3), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.0, abs=1e-12)
    # Offset X by 0.2: m1 world = 0.7, error = (0.7 - 0.5)^2 = 0.04.
    assert float(cost(CostInput(q, s, ca.DM([0.2, 0, 0]), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.04, abs=1e-9)


def test_bilevel_offset_frame_orientation_invariant():
    """
    Applying a translation offset to frame's position should not affect its orientation,
    so an orientation-only frame cost (position_weight = 0) is invariant to the offset.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[],
        frame_offset_groups=[FrameOffsetGroup(['/bodyset/body'], [1])])
    cost.add_frame_bilevel_cost_term(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion(0.9, 0.1, 0.2, 0.3),
        position_weight=0.0, orientation_weight=1.0, offset_group_index=0)
    q = ca.DM.zeros(len(cost.mc.coordinate_indexes))
    s = ca.DM.ones(3)
    e0 = float(cost(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(3))))
    e1 = float(cost(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM([0.2, -0.1, 0.3]))))
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
    cost_jac = build_bilevel_rep(
        'cost_jac', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=marker_offset_groups,
        frame_offset_groups=frame_offset_groups)
    cost_fd = build_bilevel_rep(
        'cost_fd', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=marker_offset_groups,
        frame_offset_groups=frame_offset_groups, enable_fd=True)

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0,
            offset_group_index=0)
        cost.add_frame_bilevel_cost_term(
            '/bodyset/pelvis', osim.Vec3(0.3, 0.1, -0.2),
            osim.Quaternion(0.9, 0.1, 0.2, 0.3),
            position_weight=2.0, orientation_weight=1.5,
            offset_group_index=0)

    nq = len(cost_jac.mc.coordinate_indexes)
    q = ca.SX.sym('q', nq)
    s = ca.SX.sym('s', 3)
    mo = ca.SX.sym('mo', 3)
    fo = ca.SX.sym('fo', 3)
    x = ca.vertcat(q, s, mo, fo)

    J_jac = ca.Function('J_jac', [x],
                        [ca.jacobian(cost_jac(CostInput(q, s, mo, fo)), x)])
    J_fd = ca.Function('J_fd', [x],
                       [ca.jacobian(cost_fd(CostInput(q, s, mo, fo)), x)])

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


############################
# BODY SCALE ISOTROPY COST #
############################

# The isotropy cost is a `SymbolicCost`, so it is a pure function of the flat
# `body_scales` input and needs no model. `body_scales` is triplet-contiguous, i.e.
# [s0x, s0y, s0z, s1x, s1y, s1z, ...].
def _isotropy_cost(scales, weight=1.0):
    cost = BodyScaleIsotropyCost(weight=weight)
    return float(cost.evaluate(
        CostInput(body_scales=ca.DM(np.asarray(scales, dtype=float)))))


def _manual_isotropy_cost(scales, weight=1.0):
    """
    Independent numpy evaluation: each group's three factors are penalized against
    their own mean.
    """
    groups = np.asarray(scales, dtype=float).reshape(-1, 3)
    return weight * float(np.sum((groups - groups.mean(axis=1, keepdims=True))**2))


def test_isotropy_cost_requires_body_scales():
    assert (BodyScaleIsotropyCost(weight=1.0).required_inputs ==
            frozenset({'body_scales'}))


def test_isotropy_cost_rejects_negative_weight():
    with pytest.raises(ValueError, match='non-negative'):
        BodyScaleIsotropyCost(weight=-1.0)


def test_isotropy_cost_is_zero_for_identity_scaling():
    assert _isotropy_cost([1.0] * 6) == pytest.approx(0.0, abs=1e-12)


def test_isotropy_cost_is_zero_for_uniform_non_identity_scaling():
    # The cost constrains shape, not size: a group scaled equally along X, Y, and Z is
    # free no matter how far from 1.0 it is.
    assert _isotropy_cost([1.2, 1.2, 1.2, 0.7, 0.7, 0.7]) == pytest.approx(
        0.0, abs=1e-12)


def test_isotropy_cost_penalizes_anisotropic_group():
    # scales [1.0, 1.0, 1.3] -> mean 1.1, deviations [-0.1, -0.1, 0.2].
    assert _isotropy_cost([1.0, 1.0, 1.3]) == pytest.approx(0.06, abs=1e-12)


def test_isotropy_cost_sums_over_groups_independently():
    # Group 0 is anisotropic, group 1 is uniform, so only group 0 contributes. This also
    # pins down the triplet-contiguous layout: a group's three axes are adjacent.
    scales = [1.0, 1.0, 1.3, 0.8, 0.8, 0.8]
    assert _isotropy_cost(scales) == pytest.approx(0.06, abs=1e-12)
    assert _isotropy_cost(scales) == pytest.approx(
        _manual_isotropy_cost(scales), abs=1e-12)


def test_isotropy_cost_matches_manual_evaluation():
    rng = np.random.default_rng(0)
    for num_groups in (1, 2, 5):
        scales = 0.5 + rng.random(3 * num_groups)
        assert _isotropy_cost(scales, weight=0.25) == pytest.approx(
            _manual_isotropy_cost(scales, weight=0.25), abs=1e-12)


def test_isotropy_cost_scales_linearly_with_weight():
    scales = [1.0, 1.0, 1.3, 0.9, 1.1, 1.0]
    unit = _isotropy_cost(scales, weight=1.0)
    assert unit > 0.0
    assert _isotropy_cost(scales, weight=3.5) == pytest.approx(3.5 * unit, abs=1e-12)


def test_isotropy_cost_is_invariant_to_axis_permutation():
    # Deviations from the group mean do not depend on which axis is the odd one out.
    assert _isotropy_cost([1.3, 1.0, 1.0]) == pytest.approx(
        _isotropy_cost([1.0, 1.0, 1.3]), abs=1e-12)


def test_isotropy_cost_rejects_body_scales_not_a_multiple_of_three():
    with pytest.raises(ValueError, match='multiple of 3'):
        _isotropy_cost([1.0, 1.0, 1.0, 1.0])


def test_isotropy_cost_gradient_is_symbolic_and_matches_finite_difference():
    # Being a SymbolicCost, it must differentiate through CasADi without a callback.
    scales = np.array([1.0, 1.0, 1.3, 0.9, 1.1, 1.0])
    x = ca.MX.sym('body_scales', scales.size)
    expr = BodyScaleIsotropyCost(weight=2.0).evaluate(CostInput(body_scales=x))
    grad = ca.Function('grad', [x], [ca.gradient(expr, x)])

    analytic = np.asarray(grad(ca.DM(scales))).reshape(-1)
    eps = 1e-7
    numeric = np.zeros_like(scales)
    for k in range(scales.size):
        forward, backward = scales.copy(), scales.copy()
        forward[k] += eps
        backward[k] -= eps
        numeric[k] = (_isotropy_cost(forward, weight=2.0) -
                      _isotropy_cost(backward, weight=2.0)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


######################################
# ANTHROPOMETRIC REGULARIZATION COST #
######################################

def create_two_link_model():
    """
    A two-link pin chain with a station on each body at a non-zero body-frame offset, so
    both mobilizer-frame scaling (chain) and station-offset scaling contribute to a
    measurement spanning the two bodies.
    """
    model = osim.Model()
    model.setName('two_link')
    ground = model.getGround()

    b0 = osim.Body('b0', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(b0)
    j0 = osim.PinJoint('j0', ground, osim.Vec3(0), osim.Vec3(0),
                       b0, osim.Vec3(0, -0.5, 0), osim.Vec3(0))
    model.addJoint(j0)

    b1 = osim.Body('b1', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(b1)
    j1 = osim.PinJoint('j1', b0, osim.Vec3(0), osim.Vec3(0),
                       b1, osim.Vec3(0, -0.5, 0), osim.Vec3(0))
    model.addJoint(j1)

    s0 = osim.Station(b0, osim.Vec3(0.1, 0.2, 0.0))
    s0.setName('S0')
    model.addComponent(s0)
    s1 = osim.Station(b1, osim.Vec3(0.3, 0.0, 0.0))
    s1.setName('S1')
    model.addComponent(s1)

    model.finalizeConnections()
    return model


def register_body_scales(mc, body_paths):
    """Register one BodyScale group per body on the ModelCache, in order."""
    for path in body_paths:
        bs = BodyScale(path, Bounds(0.5, 2.0), np.ones(3))
        bs.validate(mc)
        mc.add_parameter_group(bs.to_group())


def cache_group_joints(mc):
    """Cache each group's mobilizer joints (as BilevelCostRep does)."""
    mc.cache_body_scale_group_joints()


def station_ground_under_scale(mc, station_path, s):
    """
    Station ground position under the solver's scaling model for flat scales `s`:
    scaled mobilizer frames (set_scaled_mobilizer_frame_positions) plus the station's
    own base-frame location scaled by its body's group scale.
    """
    station = osim.Station.safeDownCast(mc.model.getComponent(station_path))
    base_frame = osim.PhysicalFrame.safeDownCast(
        station.getParentFrame().findBaseFrame())
    mobod = int(base_frame.getMobilizedBodyIndex())
    base_loc = station.findLocationInFrame(mc.state, base_frame).to_numpy()

    group = next((g for g, grp in enumerate(mc.body_scale_groups)
                  if mobod in [int(k) for k in grp.mobod_indexes]), None)
    scaled_loc = base_loc.copy()
    if group is not None:
        scaled_loc = base_loc * np.asarray(s[3*group:3*group+3])

    mc.set_scaled_mobilizer_frame_positions(mc.state, np.asarray(s, dtype=float))
    mc.model.realizePosition(mc.state)
    p = base_frame.findStationLocationInGround(
        mc.state, osim.Vec3(*[float(v) for v in scaled_loc])).to_numpy()
    return p

# The cost fits its distribution from the ANSUR II dataset, so measurement names must be
# real ANSUR labels. The station pairs are the synthetic model's stations — anatomical
# correctness is irrelevant here; we exercise the cost mechanics.
def _build_rep(label='stature', axis=Axis.YAxis, sex='female', weight=1.0):
    """
    Build an AnthropometricRegularizationCost and its rep, as a solver does in solve().
    """
    model = create_two_link_model()
    mc = ModelCache(model)
    register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
    measurements = [AnthropometricMeasurement(label, '/S0', '/S1', axis)]
    cost = AnthropometricRegularizationCost(measurements, sex=sex, weight=weight)
    rep = cost.create_rep(mc)
    n = 3 * len(mc.body_scale_groups)
    return rep, n


def _manual_cost(rep, s):
    """
    Independent numpy evaluation of the Mahalanobis penalty: measurements are recomputed
    from explicit station ground positions under the scaling model (bypassing the rep's
    own callback), then combined with its cost's fitted mean and precision.
    """
    s = np.asarray(s, dtype=float)
    cost = rep.cost
    measurements = []
    for (sc1, sc2, axis), m in zip(rep.station_caches, cost.measurements):
        d = (station_ground_under_scale(sc2.mc, m.station2_path, s) -
             station_ground_under_scale(sc1.mc, m.station1_path, s))
        measurements.append(np.abs(d[axis]) if axis is not None else np.linalg.norm(d))
    residual = np.asarray(measurements) - cost.mean
    return cost.weight * 0.5 * residual @ cost.precision @ residual


def test_station_position_jacobian_matches_finite_difference():
    model = create_two_link_model()
    mc = ModelCache(model)
    register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
    cache_group_joints(mc)
    mc.model.realizePosition(mc.state)

    n = 3 * len(mc.body_scale_groups)
    for station_path in ('/S0', '/S1'):
        station = osim.Station.safeDownCast(mc.model.getComponent(station_path))
        station_cache = StationCache.from_station(mc, station)

        # Analytical Jacobian (computed at baseline before any state scaling).
        J = station_cache.calc_position_jacobian_wrt_body_scales(mc.state)

        # Finite-difference the same scaling model.
        p0 = station_ground_under_scale(mc, station_path, np.ones(n))
        eps = 1e-6
        J_fd = np.zeros((3, n))
        for k in range(n):
            s = np.ones(n)
            s[k] += eps
            pk = station_ground_under_scale(mc, station_path, s)
            J_fd[:, k] = (pk - p0) / eps

        np.testing.assert_allclose(J, J_fd, atol=1e-5,
                                   err_msg=f'Jacobian mismatch for {station_path}')


def test_distribution_mean_is_in_meters():
    # A female stature is ~1.6 m; without the mm->m conversion it would be ~1600.
    rep, n = _build_rep(label='stature')
    assert 1.0 < rep.cost.mean[0] < 2.5


def test_cost_matches_manual_mahalanobis():
    rep, n = _build_rep(label='stature', weight=2.0)
    for s in (np.ones(n), np.array([1.1, 1.0, 1.0, 0.9, 1.0, 1.0])):
        value = float(rep(CostInput(body_scales=ca.DM(s))))
        np.testing.assert_allclose(value, _manual_cost(rep, s), rtol=1e-9)


def test_cost_gradient_matches_finite_difference():
    rep, n = _build_rep(label='stature')
    s = ca.MX.sym('s', n)
    grad = ca.Function('grad', [s], [ca.gradient(rep(CostInput(body_scales=s)), s)])
    s0 = np.ones(n)
    g = np.array(grad(s0)).flatten()
    eps = 1e-6
    g_fd = np.zeros(n)
    for k in range(n):
        sp, sm = s0.copy(), s0.copy()
        sp[k] += eps
        sm[k] -= eps
        g_fd[k] = (float(rep(CostInput(body_scales=ca.DM(sp)))) -
                   float(rep(CostInput(body_scales=ca.DM(sm))))) / (2 * eps)
    np.testing.assert_allclose(g, g_fd, atol=1e-6)


def test_euclidean_measurement_builds_and_evaluates():
    rep, n = _build_rep(label='biacromialbreadth', axis=None)
    value = float(rep(CostInput(body_scales=ca.DM(np.ones(n)))))
    assert np.isfinite(value)


def test_cost_is_a_description_and_only_its_rep_is_callable():
    """
    A Cost carries no model state and is not evaluable; the rep it creates is.
    """
    measurements = [AnthropometricMeasurement('stature', '/S0', '/S1', Axis.YAxis)]
    cost = AnthropometricRegularizationCost(measurements, sex='female')
    assert not isinstance(cost, CostRep)
    assert not callable(cost)

    mc = ModelCache(create_two_link_model())
    register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
    rep = cost.create_rep(mc)
    assert isinstance(rep, CostRep)
    assert rep.cost is cost
    assert rep.mc is mc


def test_creating_a_rep_twice_yields_independent_reps():
    """
    A solver builds a fresh rep on every solve, so a Cost must support create_rep more
    than once. A CasADi callback can only be constructed once per proxy, so each rep
    has to be a distinct object rather than the cost itself.
    """
    measurements = [AnthropometricMeasurement('stature', '/S0', '/S1', Axis.YAxis)]
    cost = AnthropometricRegularizationCost(measurements, sex='female', weight=2.0)

    caches, reps = [], []
    for _ in range(2):
        mc = ModelCache(create_two_link_model())
        register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
        caches.append(mc)
        reps.append(cost.create_rep(mc))

    assert reps[0] is not reps[1]
    assert [rep.mc for rep in reps] == caches
    assert all(rep.cost is cost for rep in reps)

    # The models are identical copies, so both reps evaluate to the same value, and
    # each agrees with an independent numpy evaluation through its own caches.
    s = np.array([1.1, 1.0, 1.0, 0.9, 1.0, 1.0])
    values = [float(rep(CostInput(body_scales=ca.DM(s)))) for rep in reps]
    np.testing.assert_allclose(values[0], values[1], rtol=1e-9)
    for rep, value in zip(reps, values):
        np.testing.assert_allclose(value, _manual_cost(rep, s), rtol=1e-9)


def test_symbolic_cost_rep_delegates_to_its_cost():
    cost = BodyScaleRegularizationCost(2.0, target=1.1)
    mc = ModelCache(create_two_link_model())
    rep = cost.create_rep(mc)

    assert isinstance(rep, SymbolicCostRep)
    assert rep.cost is cost

    s = ca.DM([1.2, 0.9, 1.0])
    assert float(rep(CostInput(body_scales=s))) == pytest.approx(
        float(cost.evaluate(CostInput(body_scales=s))))


def test_body_scale_groups_may_be_shared_across_model_caches():
    """
    A `BodyScaleGroup` holds only model-independent descriptors, so the same group may
    be registered on several `ModelCache`s (several tests above do exactly that). The
    `Joint`s that scale with a group belong to one model, so each `ModelCache` must
    cache its own; if they were stored on the group, registering the group on a second
    `ModelCache` would rebind the first cost's `Joint`s to the second model, and that
    cost would then write its own `State` through another model's `Joint`s.
    """
    model = create_two_link_model()
    mc1, mc2 = ModelCache(model), ModelCache(model)
    assert mc1.model is not mc2.model

    def pointers(mc):
        """The C++ addresses of each group's cached outboard Joints."""
        return [[int(j.this) for j in joints]
                for joints in mc.body_scale_group_outboard_joints]

    register_body_scales(mc1, ['/bodyset/b0', '/bodyset/b1'])
    groups = mc1.body_scale_groups
    mc1.cache_body_scale_group_joints()
    before = pointers(mc1)

    # Register the very same group objects on a second ModelCache.
    mc2.body_scale_groups = list(groups)
    mc2.cache_body_scale_group_joints()

    # Caching on mc2 must leave mc1's joints alone, and the two caches must hold
    # different C++ Joints, one set per model copy.
    assert pointers(mc1) == before
    assert before and all(a != b for ja, jb in zip(before, pointers(mc2))
                          for a, b in zip(ja, jb))

    # The shared group descriptors carry no model-specific state at all.
    for group in groups:
        assert not hasattr(group, 'outboard_joints')
        assert not hasattr(group, 'inboard_joints')


def test_set_scaled_mobilizer_frames_requires_cached_joints():
    mc = ModelCache(create_two_link_model())
    register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
    n = 3 * len(mc.body_scale_groups)
    with pytest.raises(RuntimeError, match='cache_body_scale_group_joints'):
        mc.set_scaled_mobilizer_frame_positions(mc.state, np.ones(n))
