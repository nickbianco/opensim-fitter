"""
End-to-end regression test mirroring examples/example_pendulum/example_pendulum.py.

Synthesizes marker data from a double pendulum with known body lengths
(1.25 m and 0.75 m), then runs SplinedKinematicsSolver against an unscaled
model (both lengths = 1.0 m) and asserts the recovered body scales recover
the ground-truth lengths.
"""

import pytest
import opensim as osim
import numpy as np

from osimfit.data_sources import MarkerSource, Trial
from osimfit.solvers import InverseKinematicsSolver, SplinedKinematicsSolver
from osimfit.model import BodyScale
from osimfit.costs import BodyScaleRegularizationCost
from osimfit.bounds import Bounds


###########
# HELPERS #
###########

def create_double_pendulum(length1: float, length2: float) -> osim.Model:
    """
    Build a double-pendulum model with the given body lengths.
    """
    model = osim.Model()
    model.setName("double_pendulum")
    ground = model.getGround()

    b0 = osim.Body("b0", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(b0)
    j0 = osim.PinJoint(
        "j0",
        ground, osim.Vec3(0), osim.Vec3(0),
        b0, osim.Vec3(-length1, 0, 0), osim.Vec3(0),
    )
    j0.updCoordinate().setName("q0")
    model.addJoint(j0)

    b1 = osim.Body("b1", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(b1)
    j1 = osim.PinJoint(
        "j1",
        b0, osim.Vec3(0), osim.Vec3(0),
        b1, osim.Vec3(-length2, 0, 0), osim.Vec3(0),
    )
    j1.updCoordinate().setName("q1")
    model.addJoint(j1)

    model.addMarker(osim.Marker("m0", b0, osim.Vec3(0)))
    model.addMarker(osim.Marker("m1", b1, osim.Vec3(0)))

    model.finalizeConnections()
    return model


def create_synthetic_markers_file(trc_path: str, length1: float,
                                  length2: float, duration: float = 2.0,
                                  initial_q0: float = 0.0) -> None:
    """
    Forward-simulate the truth pendulum and write marker positions to a TRC. `duration`
    and `initial_q0` vary the length and the initial pose of the simulated motion, so
    that distinct trials can be synthesized from the same truth model.
    """
    model = create_double_pendulum(length1, length2)
    state = model.initSystem()
    model.setStateVariableValue(state, '/jointset/j0/q0/value', initial_q0)

    manager = osim.Manager(model)
    manager.setIntegratorFixedStepSize(0.01)
    manager.initialize(state)
    manager.integrate(duration)
    states = manager.getStatesTable()

    controls = osim.TimeSeriesTable(states.getIndependentColumn())
    output_paths = osim.StdVectorString()
    output_paths.append('/markerset/.*location')
    markers = osim.analyzeVec3(model, states, controls, output_paths)

    markers.addTableMetaDataString('DataRate', '100.0')
    markers.addTableMetaDataString('Units', 'm')
    osim.TRCFileAdapter().write(markers, trc_path)


##################
# PENDULUM TESTS #
##################

def test_pendulum_bilevel_recovers_ground_truth_lengths(tmp_path):
    """
    Bilevel solver must recover the body lengths used to synthesize the data.
    """
    true_b0_length = 1.25
    true_b1_length = 0.75

    trc_path = str(tmp_path / "markers.trc")
    create_synthetic_markers_file(trc_path, true_b0_length, true_b1_length)

    # Strip the '|location' suffix that analyzeVec3 appends to marker column
    # labels so they match the marker names ('m0', 'm1') in the unscaled model.
    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}

    # Solve against an unscaled model (both lengths = 1.0 m).
    unscaled_model = create_double_pendulum(1.0, 1.0)
    unscaled_model.initSystem()

    marker_source = MarkerSource('markers', trc_path, label_map=label_map)

    solver = SplinedKinematicsSolver(
        unscaled_model,
        convergence_tolerance=1e-5,
        knot_interval=0.05,
        position_weight=5.0,
    )
    solver.add_trial(Trial('pendulum', [marker_source]))
    solver.add_cost(BodyScaleRegularizationCost(1e-2))
    solver.add_parameter(BodyScale('/bodyset/b0', Bounds(0.5, 2.0), np.ones(3)))
    solver.add_parameter(BodyScale('/bodyset/b1', Bounds(0.5, 2.0), np.ones(3)))

    solution = solver.solve()

    # The single registered trial's solution table covers the simulated 2.0 s @ 100 Hz
    # (201 samples) and exposes both joint coordinates.
    states_table = solution.states_tables['pendulum']
    assert states_table.getNumRows() == 201
    state_labels = list(states_table.getColumnLabels())
    assert '/jointset/j0/q0/value' in state_labels
    assert '/jointset/j1/q1/value' in state_labels

    # The recovered X-axis body scales match the ground-truth lengths. Y and Z scales
    # should stay near 1.0 since the truth model only varies length along the local X
    # axis.
    scales = [p for p in solution.parameters if isinstance(p, BodyScale)]
    assert [p.paths for p in scales] == [['/bodyset/b0'], ['/bodyset/b1']]
    assert abs(scales[0].value[0] - true_b0_length) < 0.02
    assert abs(scales[1].value[0] - true_b1_length) < 0.02
    for body_idx in (0, 1):
        for axis in (1, 2):
            assert abs(scales[body_idx].value[axis] - 1.0) < 0.05


def test_pendulum_bilevel_recovers_shared_length_under_asymmetric_truth(
        tmp_path):
    """
    A single body scale shared across both pendulum bodies must converge to
    a compromise value strictly between the two ground-truth lengths. This
    proves the grouped-scale machinery wires through end-to-end: one
    optimization variable, broadcast to two mobilized bodies, with a
    chain-rule Jacobian column.
    """
    true_b0_length = 1.25
    true_b1_length = 0.75

    trc_path = str(tmp_path / "markers.trc")
    create_synthetic_markers_file(trc_path, true_b0_length, true_b1_length)

    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}

    unscaled_model = create_double_pendulum(1.0, 1.0)
    unscaled_model.initSystem()

    marker_source = MarkerSource('markers', trc_path, label_map=label_map)

    solver = SplinedKinematicsSolver(
        unscaled_model,
        convergence_tolerance=1e-5,
        knot_interval=0.05,
        position_weight=5.0,
    )
    solver.add_trial(Trial('pendulum', [marker_source]))
    solver.add_cost(BodyScaleRegularizationCost(1e-2))
    solver.add_parameter(BodyScale(
        ['/bodyset/b0', '/bodyset/b1'], Bounds(0.5, 2.0), np.ones(3)))

    solution = solver.solve()

    # One shared body scale parameter → one XYZ value shared across both bodies.
    scales = [p for p in solution.parameters if isinstance(p, BodyScale)]
    assert len(scales) == 1
    assert scales[0].value.shape == (3,)
    assert scales[0].paths == ['/bodyset/b0', '/bodyset/b1']

    # The shared scale must lie strictly between the two ground-truth lengths.
    shared_sx = scales[0].value[0]
    assert true_b1_length < shared_sx < true_b0_length

    # Y and Z scales remain near 1.0 since the truth varies length along X.
    for axis in (1, 2):
        assert abs(scales[0].value[axis] - 1.0) < 0.05


def test_pendulum_update_model_applies_recovered_body_scales(tmp_path):
    """
    update_model must scale a model with the bilevel-recovered factors. At the
    default state (q0=q1=0) the pendulum hangs straight along +X, so the marker
    ground positions are exact functions of the per-body scales:

        m0_ground.x = length1 * sx_b0
        m1_ground.x = length1 * sx_b0 + length2 * sx_b1

    With unscaled lengths of 1.0 m, those reduce to the recovered scales — which
    in turn match the ground-truth body lengths to within the same tolerance the
    solver achieves on the raw body scales.
    """
    true_b0_length = 1.25
    true_b1_length = 0.75

    trc_path = str(tmp_path / "markers.trc")
    create_synthetic_markers_file(trc_path, true_b0_length, true_b1_length)

    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}

    unscaled_model = create_double_pendulum(1.0, 1.0)
    unscaled_model.initSystem()

    marker_source = MarkerSource('markers', trc_path, label_map=label_map)
    solver = SplinedKinematicsSolver(
        unscaled_model,
        convergence_tolerance=1e-5,
        knot_interval=0.05,
        position_weight=5.0,
    )
    solver.add_trial(Trial('pendulum', [marker_source]))
    solver.add_cost(BodyScaleRegularizationCost(1e-2))
    solver.add_parameter(BodyScale('/bodyset/b0', Bounds(0.5, 2.0), np.ones(3)))
    solver.add_parameter(BodyScale('/bodyset/b1', Bounds(0.5, 2.0), np.ones(3)))
    solution = solver.solve()

    # Apply the recovered scales to a fresh unscaled model so this assertion is
    # independent of any in-place mutation the solver may have done to its own
    # model handle.
    fresh_model = create_double_pendulum(1.0, 1.0)
    scaled_model = solver.update_model(fresh_model, solution)
    state = scaled_model.initSystem()

    m0 = osim.Marker.safeDownCast(scaled_model.getComponent('/markerset/m0'))
    m1 = osim.Marker.safeDownCast(scaled_model.getComponent('/markerset/m1'))
    m0_x = m0.getLocationInGround(state).to_numpy()[0]
    m1_x = m1.getLocationInGround(state).to_numpy()[0]

    # m0 sits at one b0-length out; m1 sits at b0+b1 lengths out. Tolerance on
    # m1 is doubled because two independent scale errors stack.
    assert abs(m0_x - true_b0_length) < 0.02
    assert abs(m1_x - (true_b0_length + true_b1_length)) < 0.04


######################
# MULTI-TRIAL TESTS  #
######################

def _pendulum_label_map(trc_path: str) -> dict:
    """
    Strip the '|location' suffix that analyzeVec3 appends to marker column labels.
    """
    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    return {label: label.replace('|location', '') for label in raw_labels}


def test_bilevel_over_two_trials_shares_body_scales(tmp_path):
    """
    Two trials of differing duration and initial pose, synthesized from the same truth
    pendulum, must jointly recover the ground-truth body lengths. Because the durations
    differ, the trials carry differently sized blocks of spline control points, which
    exercises the per-trial knot vectors and the offset of the shared parameter block
    past every trial's coefficients in the flat solution vector.
    """
    true_b0_length = 1.25
    true_b1_length = 0.75

    trc_a = str(tmp_path / 'markers_a.trc')
    trc_b = str(tmp_path / 'markers_b.trc')
    create_synthetic_markers_file(trc_a, true_b0_length, true_b1_length,
                                  duration=1.0, initial_q0=0.0)
    create_synthetic_markers_file(trc_b, true_b0_length, true_b1_length,
                                  duration=1.5, initial_q0=0.4)

    unscaled_model = create_double_pendulum(1.0, 1.0)
    unscaled_model.initSystem()

    solver = SplinedKinematicsSolver(
        unscaled_model,
        convergence_tolerance=1e-5,
        knot_interval=0.05,
        position_weight=5.0,
    )
    solver.add_trial(Trial('trial_a', [
        MarkerSource('markers_a', trc_a, label_map=_pendulum_label_map(trc_a))]))
    solver.add_trial(Trial('trial_b', [
        MarkerSource('markers_b', trc_b, label_map=_pendulum_label_map(trc_b))]))
    solver.add_cost(BodyScaleRegularizationCost(1e-2))
    solver.add_parameter(BodyScale('/bodyset/b0', Bounds(0.5, 2.0), np.ones(3)))
    solver.add_parameter(BodyScale('/bodyset/b1', Bounds(0.5, 2.0), np.ones(3)))

    solution = solver.solve()

    # One solution per trial, in registration order, each covering its own time samples
    # with its own number of control points.
    assert list(solution.states_tables) == ['trial_a', 'trial_b']
    assert solution.states_tables['trial_a'].getNumRows() == 101
    assert solution.states_tables['trial_b'].getNumRows() == 151
    nodes_a = solution.outputs['spline_nodes']['trial_a']
    nodes_b = solution.outputs['spline_nodes']['trial_b']
    assert nodes_a.shape == (20, 2)
    assert nodes_b.shape == (30, 2)

    # The body scales are shared across both trials, so there is still one XYZ value per
    # body, and it must recover the ground truth both trials were generated from.
    scales = [p for p in solution.parameters if isinstance(p, BodyScale)]
    assert [p.paths for p in scales] == [['/bodyset/b0'], ['/bodyset/b1']]
    assert abs(scales[0].value[0] - true_b0_length) < 0.02
    assert abs(scales[1].value[0] - true_b1_length) < 0.02


def test_inverse_kinematics_solves_each_trial_independently(tmp_path):
    """
    The frame-by-frame IK solver couples nothing across time or trials, so registering
    two trials must simply produce one states table per trial, each spanning that
    trial's own time samples.
    """
    trc_a = str(tmp_path / 'markers_a.trc')
    trc_b = str(tmp_path / 'markers_b.trc')
    create_synthetic_markers_file(trc_a, 1.0, 1.0, duration=0.2, initial_q0=0.0)
    create_synthetic_markers_file(trc_b, 1.0, 1.0, duration=0.3, initial_q0=0.4)

    model = create_double_pendulum(1.0, 1.0)
    model.initSystem()

    solver = InverseKinematicsSolver(model, convergence_tolerance=1e-3)
    solver.add_trial(Trial('trial_a', [
        MarkerSource('markers_a', trc_a, label_map=_pendulum_label_map(trc_a))]))
    solver.add_trial(Trial('trial_b', [
        MarkerSource('markers_b', trc_b, label_map=_pendulum_label_map(trc_b))]))

    solution = solver.solve()

    assert list(solution.states_tables) == ['trial_a', 'trial_b']
    assert solution.states_tables['trial_a'].getNumRows() == 21
    assert solution.states_tables['trial_b'].getNumRows() == 31

    # Each trial's states track the pose the data was synthesized from, so the first
    # sample recovers that trial's distinct initial angle.
    for name, initial_q0 in (('trial_a', 0.0), ('trial_b', 0.4)):
        table = solution.states_tables[name]
        q0 = table.getDependentColumn('/jointset/j0/q0/value').to_numpy()
        assert abs(q0[0] - initial_q0) < 1e-2


##############################
# TRIAL REGISTRATION #
##############################

def test_add_trial_rejects_duplicate_names_and_empty_trials(tmp_path):
    model = create_double_pendulum(1.0, 1.0)
    model.initSystem()
    solver = SplinedKinematicsSolver(model)

    # A trial with no data sources has no time vector to solve over.
    with pytest.raises(ValueError, match='no reference data'):
        solver.add_trial(Trial('empty'))

    # Solutions are looked up by trial name, so names must be unique.
    trc_path = str(tmp_path / 'markers.trc')
    create_synthetic_markers_file(trc_path, 1.0, 1.0, duration=0.1)
    source = MarkerSource('markers', trc_path, label_map=_pendulum_label_map(trc_path))
    solver.add_trial(Trial('dup', [source]))
    with pytest.raises(ValueError, match='already registered'):
        solver.add_trial(Trial('dup', [source]))


def test_solve_raises_when_a_trial_is_too_short_for_the_spline(tmp_path):
    """
    A degree-p spline needs at least p+1 control points, so a trial spanning fewer than
    p+1 knot intervals cannot be splined. The error must name the offending trial, since
    with several trials registered only one of them may be too short.
    """
    trc_long = str(tmp_path / 'markers_long.trc')
    trc_short = str(tmp_path / 'markers_short.trc')
    create_synthetic_markers_file(trc_long, 1.0, 1.0, duration=1.0)
    create_synthetic_markers_file(trc_short, 1.0, 1.0, duration=0.1)

    model = create_double_pendulum(1.0, 1.0)
    model.initSystem()

    # At a 0.05 s knot interval, a cubic spline needs a trial spanning at least 0.2 s.
    solver = SplinedKinematicsSolver(model, knot_interval=0.05, degree=3)
    solver.add_trial(Trial('long', [
        MarkerSource('markers_long', trc_long, label_map=_pendulum_label_map(trc_long))]))
    solver.add_trial(Trial('short', [
        MarkerSource('markers_short', trc_short,
                     label_map=_pendulum_label_map(trc_short))]))

    with pytest.raises(ValueError, match="Trial 'short'.*degree-3 spline requires"):
        solver.solve()

    # The same short trial is fine once the knot interval is small enough to yield at
    # least four control points.
    solver.knot_interval = 0.02
    solver.solve()


def test_solve_without_trials_raises():
    model = create_double_pendulum(1.0, 1.0)
    model.initSystem()
    with pytest.raises(ValueError, match='no reference data'):
        SplinedKinematicsSolver(model).solve()
    with pytest.raises(ValueError, match='no reference data'):
        InverseKinematicsSolver(model).solve()
