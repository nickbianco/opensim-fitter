"""
Tests for the `MarkerPlacer` solver.
"""

import pytest
import numpy as np
import opensim as osim
import casadi as ca

from osimfit.data_sources import MarkerSource, Trial
from osimfit.solvers import MarkerPlacer
from osimfit.costs import CostInput, SymbolicCost
from osimfit.model import MarkerOffset
from osimfit.bounds import Bounds

from tests.test_double_pendulum import create_double_pendulum


###########
# HELPERS #
###########

class MarkerOffsetPenalty(SymbolicCost):
    """
    A minimal cost whose required inputs MarkerPlacer's `add_cost` accepts, so that
    solve() is what rejects it rather than registration.
    """
    required_inputs = frozenset({'marker_offsets'})

    def evaluate(self, input: CostInput) -> ca.MX:
        return ca.sumsqr(input.marker_offsets)


def create_placer_model(tracking_marker_location: np.ndarray) -> osim.Model:
    """
    A double pendulum carrying two fixed markers that pose the model ('m0' on b0 and
    'm2' on b1) and one tracking marker ('m1' on b1) whose placement is optimized.

    Parameters
    ----------
    tracking_marker_location: np.ndarray
        The body-frame location of the tracking marker 'm1'.
    """
    model = create_double_pendulum(1.0, 1.0)
    b1 = osim.Body.safeDownCast(model.getComponent('/bodyset/b1'))
    model.addMarker(osim.Marker('m2', b1, osim.Vec3(0.5, 0, 0)))
    model.finalizeConnections()
    model.initSystem()

    markerset = model.updMarkerSet()
    for name in ('m0', 'm2'):
        markerset.get(name).set_fixed(True)
    m1 = markerset.get('m1')
    m1.set_fixed(False)
    m1.set_location(osim.Vec3(*[float(v) for v in tracking_marker_location]))

    model.finalizeConnections()
    model.initSystem()
    return model


class PosedMarkerSource(MarkerSource):
    """
    A MarkerSource that reports `model`'s marker positions at a fixed pose rather than
    reading a TRC, so a trial can be built for an exact known posture without a file on
    disk. Column labels are the markers' absolute paths, as the solvers expect.

    Parameters
    ----------
    model: osim.Model
        The model to pose and read marker positions from.
    q_values: dict[str, float]
        Coordinate state-variable paths mapped to the values defining the pose.
    times: tuple[float, ...], optional
        The sample times of the emitted table. MarkerPlacer reads only the first;
        the rest exist so the table is a valid time series.
    """
    def __init__(self, model, q_values, times=(0.0, 0.01)):
        super().__init__('posed', 'unused.trc')
        self.model = model
        self.q_values = q_values
        self.sample_times = times

    def _create_positions_table(self) -> osim.TimeSeriesTableVec3:
        state = self.model.initSystem()
        for coord_path, value in self.q_values.items():
            self.model.setStateVariableValue(state, coord_path, value)
        self.model.realizePosition(state)

        markerset = self.model.getMarkerSet()
        labels = [markerset.get(i).getAbsolutePathString()
                  for i in range(markerset.getSize())]
        table = osim.TimeSeriesTableVec3()
        for time in self.sample_times:
            row = osim.RowVectorVec3(len(labels), osim.Vec3(0))
            for i in range(markerset.getSize()):
                row[i] = markerset.get(i).getLocationInGround(state)
            table.appendRow(time, row)
        table.setColumnLabels(labels)
        table.addTableMetaDataString('Units', 'm')
        return table


def make_trial(name, model, q0, q1, first_time=0.0) -> Trial:
    return Trial(name, [PosedMarkerSource(
        model,
        {'/jointset/j0/q0/value': q0, '/jointset/j1/q1/value': q1},
        times=(first_time, first_time + 0.01))])


####################
# END-TO-END TESTS #
####################

def test_marker_placer_recovers_offset_shared_across_trials():
    """
    Test that MarkerPlacer recovers the ground truth location of the tracking marker
    'm1' given two trials with distinct poses.
    """
    true_offset = np.array([0.1, 0.05, -0.02])
    truth = create_placer_model(true_offset)

    placer = MarkerPlacer(create_placer_model(np.zeros(3)),
                          convergence_tolerance=1e-6)
    placer.add_trial(make_trial('pose_a', truth, 0.0, 0.0, first_time=0.0))
    placer.add_trial(make_trial('pose_b', truth, 0.4, -0.3, first_time=1.0))

    solution = placer.solve()

    # One shared offset, recovering the truth the trials were generated from.
    offsets = [p for p in solution.parameters if isinstance(p, MarkerOffset)]
    assert len(offsets) == 1
    assert offsets[0].paths == ['/markerset/m1']
    np.testing.assert_allclose(offsets[0].value, true_offset, atol=1e-3)

    # get_parameter reaches the same offset by path from the base Solution.
    assert solution.get_parameter('/markerset/m1', MarkerOffset) is offsets[0]


def test_marker_placer_returns_a_one_row_states_table_per_trial():
    """
    Test that the MarkerPlacer solution contains the original poses and the correct
    initial time on each trial.
    """
    truth = create_placer_model(np.array([0.1, 0.05, 0.0]))

    placer = MarkerPlacer(create_placer_model(np.zeros(3)),
                          convergence_tolerance=1e-6)
    placer.add_trial(make_trial('pose_a', truth, 0.0, 0.0, first_time=0.0))
    placer.add_trial(make_trial('pose_b', truth, 0.4, -0.3, first_time=1.0))

    solution = placer.solve()

    assert list(solution.states_tables) == ['pose_a', 'pose_b']
    for name, first_time, (q0, q1) in (('pose_a', 0.0, (0.0, 0.0)),
                                       ('pose_b', 1.0, (0.4, -0.3))):
        table = solution.states_tables[name]
        assert table.getNumRows() == 1
        assert table.getIndependentColumn()[0] == pytest.approx(first_time)
        assert table.getDependentColumn(
            '/jointset/j0/q0/value').to_numpy()[0] == pytest.approx(q0, abs=1e-3)
        assert table.getDependentColumn(
            '/jointset/j1/q1/value').to_numpy()[0] == pytest.approx(q1, abs=1e-3)


def test_marker_placer_update_model_applies_the_shared_offset():
    true_offset = np.array([0.1, 0.05, -0.02])
    truth = create_placer_model(true_offset)

    placer = MarkerPlacer(create_placer_model(np.zeros(3)),
                          convergence_tolerance=1e-6)
    placer.add_trial(make_trial('pose_a', truth, 0.0, 0.0))
    solution = placer.solve()

    updated = placer.update_model(create_placer_model(np.zeros(3)), solution)
    marker = osim.Marker.safeDownCast(updated.getComponent('/markerset/m1'))
    np.testing.assert_allclose(marker.get_location().to_numpy(), true_offset,
                               atol=1e-3)


##############
# VALIDATION #
##############

def test_marker_placer_solve_without_trials_raises():
    placer = MarkerPlacer(create_placer_model(np.zeros(3)))
    with pytest.raises(ValueError, match='no reference data'):
        placer.solve()


def test_marker_placer_rejects_labels_that_are_not_model_markers():
    truth = create_placer_model(np.zeros(3))
    placer = MarkerPlacer(create_placer_model(np.zeros(3)))

    # A trial whose data labels a marker the model does not have.
    trial = make_trial('pose_a', truth, 0.0, 0.0)
    trial.marker_data[0].labels = ['/markerset/m0', '/markerset/nope',
                                   '/markerset/m2']
    placer.add_trial(trial)
    with pytest.raises(ValueError, match='not markers in'):
        placer.solve()


def test_marker_placer_rejects_a_trial_carrying_frame_data():
    truth = create_placer_model(np.zeros(3))
    placer = MarkerPlacer(create_placer_model(np.zeros(3)))

    trial = make_trial('pose_a', truth, 0.0, 0.0)
    trial.frame_data.append(object())     # stand-in; the guard only checks emptiness
    placer.add_trial(trial)
    with pytest.raises(ValueError, match='carries frame data'):
        placer.solve()


def test_marker_placer_rejects_a_trial_without_marker_data():
    truth = create_placer_model(np.zeros(3))
    placer = MarkerPlacer(create_placer_model(np.zeros(3)))

    trial = make_trial('pose_a', truth, 0.0, 0.0)
    trial.frame_data.append(object())     # so base add_trial still sees reference data
    trial.marker_data = []
    placer.add_trial(trial)
    with pytest.raises(ValueError, match='carries no marker data'):
        placer.solve()


def test_marker_placer_rejects_additional_costs_at_solve():
    """
    MarkerPlacer's objective is the placement error alone, so a registered cost would
    have nothing to contribute to.
    """
    truth = create_placer_model(np.zeros(3))
    placer = MarkerPlacer(create_placer_model(np.zeros(3)))
    placer.add_trial(make_trial('pose_a', truth, 0.0, 0.0))
    placer.add_cost(MarkerOffsetPenalty())

    with pytest.raises(ValueError, match='does not accept additional costs'):
        placer.solve()


def test_marker_placer_offset_bounds_constrain_the_solution():
    # The truth offset lies outside the bounds, so the optimizer must stop at the bound.
    truth = create_placer_model(np.array([0.3, 0.0, 0.0]))
    placer = MarkerPlacer(create_placer_model(np.zeros(3)),
                          offset_bounds=Bounds(-0.05, 0.05),
                          convergence_tolerance=1e-6)
    placer.add_trial(make_trial('pose_a', truth, 0.0, 0.0))

    offsets = placer.solve().parameters
    assert offsets[0].value[0] == pytest.approx(0.05, abs=1e-3)
