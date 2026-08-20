"""
End-to-end regression test mirroring examples/example_pendulum/example_pendulum.py.

Synthesizes marker data from a double pendulum with known body lengths
(1.25 m and 0.75 m), then runs SplinedKinematicsSolver against an unscaled
model (both lengths = 1.0 m) and asserts the recovered body scales recover
the ground-truth lengths.
"""

import opensim as osim
import numpy as np

from osimfit.data_sources import MarkerSource
from osimfit.solvers import SplinedKinematicsSolver
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
                                  length2: float) -> None:
    """
    Forward-simulate the truth pendulum and write marker positions to a TRC.
    """
    model = create_double_pendulum(length1, length2)
    state = model.initSystem()

    manager = osim.Manager(model)
    manager.setIntegratorFixedStepSize(0.01)
    manager.initialize(state)
    manager.integrate(2.0)
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

    marker_source = MarkerSource(trc_path, label_map=label_map)

    solver = SplinedKinematicsSolver(
        unscaled_model,
        convergence_tolerance=1e-5,
        knot_interval=0.05,
        position_weight=5.0,
    )
    solver.add_marker_reference_data(marker_source)
    solver.add_cost(BodyScaleRegularizationCost(1e-2))
    solver.add_parameter(BodyScale('/bodyset/b0', Bounds(0.5, 2.0), np.ones(3)))
    solver.add_parameter(BodyScale('/bodyset/b1', Bounds(0.5, 2.0), np.ones(3)))

    solution = solver.solve()

    # Solution table covers the simulated 2.0 s @ 100 Hz (201 samples) and exposes both
    # joint coordinates.
    assert solution.states_table.getNumRows() == 201
    state_labels = list(solution.states_table.getColumnLabels())
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

    marker_source = MarkerSource(trc_path, label_map=label_map)

    solver = SplinedKinematicsSolver(
        unscaled_model,
        convergence_tolerance=1e-5,
        knot_interval=0.05,
        position_weight=5.0,
    )
    solver.add_marker_reference_data(marker_source)
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

    marker_source = MarkerSource(trc_path, label_map=label_map)
    solver = SplinedKinematicsSolver(
        unscaled_model,
        convergence_tolerance=1e-5,
        knot_interval=0.05,
        position_weight=5.0,
    )
    solver.add_marker_reference_data(marker_source)
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
