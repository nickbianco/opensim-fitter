"""
Unit and end-to-end tests for marker and frame offset optimization in
SplinedKinematicsSolver.
"""

import pytest
import numpy as np
import opensim as osim

from osimfit.data_sources import MarkerSource
from osimfit.solvers import SplinedKinematicsSolver, SplinedKinematicsSolution
from osimfit.model import MarkerOffsetGroup, MarkerOffset, FrameOffset
from osimfit.costs import OffsetRegularizationCost
from osimfit.bounds import Bounds

from tests.test_double_pendulum import create_double_pendulum

###########
# HELPERS #
###########

def create_offset_test_model() -> osim.Model:
    """
    A one-body model with a PhysicalOffsetFrame on the body, a marker attached
    directly to the body (parent frame == base frame), and a marker attached to the
    offset frame (parent frame != base frame).
    """
    model = osim.Model()
    model.setName('offset_test')
    body = osim.Body('body', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body)
    joint = osim.PinJoint(
        'joint', model.getGround(), osim.Vec3(0), osim.Vec3(0),
        body, osim.Vec3(0), osim.Vec3(0))
    model.addJoint(joint)
    pof = osim.PhysicalOffsetFrame('pof', body, osim.Transform(osim.Vec3(0.1, 0, 0)))
    body.addComponent(pof)
    model.addMarker(osim.Marker('m_body', body, osim.Vec3(0.2, 0, 0)))
    model.addMarker(osim.Marker('m_pof', pof, osim.Vec3(0.0, 0, 0)))
    model.finalizeConnections()
    return model


def create_synthetic_markers(model: osim.Model, trc_path: str,
                             duration: float = 2.0) -> None:
    """
    Forward-simulate `model` and write its marker positions to a TRC file.
    """
    state = model.initSystem()
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


################
# REGISTRATION #
################

def test_add_marker_offset_registers_single_and_shared_groups():
    model = create_offset_test_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(model)
    solver.add_parameter(
        MarkerOffset('/markerset/m_body', Bounds(-0.02, 0.02), np.zeros(3)))
    solver.add_parameter(
        MarkerOffset(['/markerset/m_body'], Bounds(-0.05, 0.05), np.zeros(3)))

    assert len(solver.marker_offsets) == 2
    assert solver.marker_offsets[0].paths == ['/markerset/m_body']
    assert solver.marker_offsets[0].bounds.lower_bound == -0.02
    assert solver.marker_offsets[0].bounds.upper_bound == 0.02
    # A list argument shares one offset group across the listed markers.
    assert solver.marker_offsets[1].to_group() == MarkerOffsetGroup(
        ['/markerset/m_body'], [1])


def test_add_frame_offset_registers():
    model = create_offset_test_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(model)
    solver.add_parameter(
        FrameOffset('/bodyset/body/pof', Bounds(-0.03, 0.03), np.zeros(3)))

    assert len(solver.frame_offsets) == 1
    assert solver.frame_offsets[0].paths == ['/bodyset/body/pof']


def test_add_marker_offset_rejects_empty_and_parent_not_base():
    model = create_offset_test_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(model)
    with pytest.raises(ValueError, match='non-empty'):
        solver.add_parameter(MarkerOffset([], Bounds(-0.02, 0.02), np.zeros(3)))
    # A marker on an offset frame (parent frame != base frame) is rejected.
    with pytest.raises(ValueError, match='parent frame'):
        solver.add_parameter(
            MarkerOffset('/markerset/m_pof', Bounds(-0.02, 0.02), np.zeros(3)))


def test_add_frame_offset_rejects_non_offset_frame():
    model = create_offset_test_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(model)
    # A Body is not a PhysicalOffsetFrame, so it has no translation to offset.
    with pytest.raises(ValueError, match='PhysicalOffsetFrame'):
        solver.add_parameter(
            FrameOffset('/bodyset/body', Bounds(-0.02, 0.02), np.zeros(3)))


################
# UPDATE MODEL #
################

def test_update_model_bakes_marker_and_frame_offsets():
    model = create_offset_test_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(model)

    solution = SplinedKinematicsSolution(
        states_table=None,
        parameters=[
            MarkerOffset('/markerset/m_body', Bounds(-1.0, 1.0),
                         value=np.array([0.1, -0.2, 0.3])),
            FrameOffset('/bodyset/body/pof', Bounds(-1.0, 1.0),
                        value=np.array([0.05, 0.0, -0.1])),
        ],
    )

    updated = solver.update_model(create_offset_test_model(), solution)

    marker = osim.Marker.safeDownCast(updated.getComponent('/markerset/m_body'))
    np.testing.assert_allclose(
        marker.get_location().to_numpy(), np.array([0.2, 0, 0]) + [0.1, -0.2, 0.3])

    pof = osim.PhysicalOffsetFrame.safeDownCast(
        updated.getComponent('/bodyset/body/pof'))
    np.testing.assert_allclose(
        pof.get_translation().to_numpy(), np.array([0.1, 0, 0]) + [0.05, 0.0, -0.1])


####################
# GUESS VALIDATION #
####################

def _make_offset_solver_and_states_table(tmp_path):
    """
    Build a SplinedKinematicsSolver on the offset test model with a marker offset and a
    frame offset registered (canonical order), plus a states_table matching the
    reference data's time samples for use as a guess.
    """
    trc_path = str(tmp_path / 'markers.trc')
    create_synthetic_markers(create_offset_test_model(), trc_path)
    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}
    marker_source = MarkerSource(trc_path, label_map=label_map)

    solver = SplinedKinematicsSolver(create_offset_test_model())
    solver.add_marker_reference_data(marker_source)
    solver.add_parameter(
        MarkerOffset('/markerset/m_body', Bounds(-1.0, 1.0), np.zeros(3)))
    solver.add_parameter(
        FrameOffset('/bodyset/body/pof', Bounds(-1.0, 1.0), np.zeros(3)))

    times = solver.get_times_from_reference_data()
    coords = np.zeros((len(times), len(solver.coordinate_indexes)))
    states_table = SplinedKinematicsSolution.create_states_table(
        solver.mc.model, solver.state, solver.coordinate_indexes, times, coords)
    return solver, states_table


def test_validate_guess_accepts_canonical_parameter_order(tmp_path):
    solver, states_table = _make_offset_solver_and_states_table(tmp_path)
    guess = SplinedKinematicsSolution(states_table=states_table, parameters=[
        MarkerOffset('/markerset/m_body', Bounds(-1.0, 1.0), np.zeros(3)),
        FrameOffset('/bodyset/body/pof', Bounds(-1.0, 1.0), np.zeros(3)),
    ])
    solver._validate_guess(guess)  # canonical order: no raise


def test_validate_guess_rejects_out_of_order_parameters(tmp_path):
    solver, states_table = _make_offset_solver_and_states_table(tmp_path)
    # frame_offsets before marker_offsets violates CostInput.INPUT_ORDER.
    guess = SplinedKinematicsSolution(states_table=states_table, parameters=[
        FrameOffset('/bodyset/body/pof', Bounds(-1.0, 1.0), np.zeros(3)),
        MarkerOffset('/markerset/m_body', Bounds(-1.0, 1.0), np.zeros(3)),
    ])
    with pytest.raises(ValueError, match='ordered by CostInput.INPUT_ORDER'):
        solver._validate_guess(guess)


####################
# END-TO-END TESTS #
####################

def test_pendulum_bilevel_recovers_marker_offset(tmp_path):
    """
    Synthesize marker data from a pendulum whose m1 marker carries a known
    body-frame offset, then solve against a model with that marker at its nominal
    location while optimizing its offset. The recovered offset must match the truth.
    """
    def add_b1_anchor_marker(model: osim.Model) -> None:
        b1 = osim.Body.safeDownCast(model.getComponent('/bodyset/b1'))
        model.addMarker(osim.Marker('m2', b1, osim.Vec3(0.5, 0, 0)))
        model.finalizeConnections()

    true_offset = np.array([0.1, 0.05, 0.0])
    truth = create_double_pendulum(1.0, 1.0)
    add_b1_anchor_marker(truth)
    truth.initSystem()
    m1 = osim.Marker.safeDownCast(truth.getComponent('/markerset/m1'))
    m1.set_location(osim.Vec3(*[float(v) for v in true_offset]))
    truth.finalizeConnections()
    truth.initSystem()

    trc_path = str(tmp_path / 'markers.trc')
    create_synthetic_markers(truth, trc_path)

    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}

    # Solve against the nominal model (m1 at the body origin).
    model = create_double_pendulum(1.0, 1.0)
    add_b1_anchor_marker(model)
    model.initSystem()
    marker_source = MarkerSource(trc_path, label_map=label_map)

    solver = SplinedKinematicsSolver(
        model, convergence_tolerance=1e-5, knot_interval=0.05,
        position_weight=5.0)
    solver.add_marker_reference_data(marker_source)
    solver.add_cost(OffsetRegularizationCost(1e-4))
    solver.add_parameter(
        MarkerOffset('/markerset/m1', Bounds(-0.5, 0.5), np.zeros(3)))

    solution = solver.solve()

    marker_offsets = [p for p in solution.parameters
                      if isinstance(p, MarkerOffset)]
    assert len(marker_offsets) == 1
    assert marker_offsets[0].paths == ['/markerset/m1']
    assert not any(isinstance(p, FrameOffset) for p in solution.parameters)
    np.testing.assert_allclose(marker_offsets[0].value, true_offset, atol=0.02)
