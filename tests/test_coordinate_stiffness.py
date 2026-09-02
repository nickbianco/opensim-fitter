"""
Unit and end-to-end tests for `CoordinateStiffnessCost`.
"""

import pytest
import numpy as np
import casadi as ca
import opensim as osim

from osimfit.model import ModelCache
from osimfit.costs import CostInput, CoordinateStiffnessCost
from osimfit.solvers import InverseKinematicsSolver, SplinedKinematicsSolver

from tests.test_double_pendulum import create_double_pendulum

Q0 = '/jointset/j0/q0'
Q1 = '/jointset/j1/q1'


@pytest.fixture
def pendulum():
    model = create_double_pendulum(1.0, 1.0)
    model.initSystem()
    return model


##############
# VALIDATION #
##############

def test_rejects_negative_weight():
    with pytest.raises(ValueError, match='non-negative'):
        CoordinateStiffnessCost({Q0: 1.0}, weight=-1.0)


def test_rejects_negative_stiffness():
    with pytest.raises(ValueError, match='non-negative'):
        CoordinateStiffnessCost({Q0: -1.0})


def test_rejects_empty_stiffnesses():
    with pytest.raises(ValueError, match='at least one'):
        CoordinateStiffnessCost({})


def test_rejects_unknown_coordinate(pendulum):
    cost = CoordinateStiffnessCost({'/jointset/j0/nope': 1.0})
    with pytest.raises(ValueError, match='not a coordinate in the model'):
        cost.create_rep(ModelCache(pendulum))


def test_required_inputs_is_coordinates():
    assert CoordinateStiffnessCost({Q0: 1.0}).required_inputs == frozenset(
        {'coordinates'})


##############
# EVALUATION #
##############

def test_penalizes_only_the_named_coordinates(pendulum):
    mc = ModelCache(pendulum)
    rep = CoordinateStiffnessCost({Q1: 2.0}).create_rep(mc)
    order = list(mc.coordinate_map)

    # Moving the unpenalized coordinate leaves the cost at zero; moving the penalized
    # one gives stiffness * deviation^2.
    q = np.zeros(len(order))
    q[order.index(Q0)] = 0.5
    assert float(rep(CostInput(coordinates=ca.DM(q)))) == pytest.approx(0.0)

    q = np.zeros(len(order))
    q[order.index(Q1)] = 0.5
    assert float(rep(CostInput(coordinates=ca.DM(q)))) == pytest.approx(2.0 * 0.25)


def test_weight_scales_the_penalty(pendulum):
    mc = ModelCache(pendulum)
    order = list(mc.coordinate_map)
    q = np.zeros(len(order))
    q[order.index(Q0)] = 0.3

    plain = CoordinateStiffnessCost({Q0: 1.0}).create_rep(mc)
    scaled = CoordinateStiffnessCost({Q0: 1.0}, weight=5.0).create_rep(mc)
    assert float(scaled(CostInput(coordinates=ca.DM(q)))) == pytest.approx(
        5.0 * float(plain(CostInput(coordinates=ca.DM(q)))))


def test_target_defaults_to_the_model_default_value(pendulum):
    """
    A coordinate whose default value is non-zero is pulled toward that value, not
    toward zero, so the penalty vanishes at the model's neutral posture.
    """
    coordinate = osim.Coordinate.safeDownCast(pendulum.getComponent(Q0))
    coordinate.setDefaultValue(0.4)
    pendulum.finalizeConnections()

    mc = ModelCache(pendulum)
    rep = CoordinateStiffnessCost({Q0: 3.0}).create_rep(mc)
    order = list(mc.coordinate_map)

    q = np.zeros(len(order))
    q[order.index(Q0)] = 0.4
    assert float(rep(CostInput(coordinates=ca.DM(q)))) == pytest.approx(0.0)

    q[order.index(Q0)] = 0.5
    assert float(rep(CostInput(coordinates=ca.DM(q)))) == pytest.approx(
        3.0 * (0.5 - 0.4)**2)


def test_explicit_target_overrides_the_model_default(pendulum):
    mc = ModelCache(pendulum)
    rep = CoordinateStiffnessCost({Q0: 1.0}, targets={Q0: 0.25}).create_rep(mc)
    order = list(mc.coordinate_map)
    q = np.zeros(len(order))
    q[order.index(Q0)] = 0.25
    assert float(rep(CostInput(coordinates=ca.DM(q)))) == pytest.approx(0.0)


def test_gradient_matches_the_analytic_spring(pendulum):
    """
    The cost is a plain CasADi expression, so CasADi differentiates it symbolically.
    Its gradient must be the spring gradient 2 * weight * k * (q - target).
    """
    mc = ModelCache(pendulum)
    order = list(mc.coordinate_map)
    rep = CoordinateStiffnessCost({Q0: 2.0, Q1: 0.5}, weight=3.0).create_rep(mc)

    x = ca.SX.sym('x', len(order))
    gradient = ca.Function('g', [x],
                           [ca.jacobian(rep(CostInput(coordinates=x)), x)])
    q = np.zeros(len(order))
    q[order.index(Q0)] = 0.3
    q[order.index(Q1)] = -0.2

    expected = np.zeros(len(order))
    expected[order.index(Q0)] = 2.0 * 3.0 * 2.0 * 0.3
    expected[order.index(Q1)] = 2.0 * 3.0 * 0.5 * -0.2
    np.testing.assert_allclose(np.squeeze(gradient(q).full()), expected, atol=1e-12)


####################
# SOLVER INTEGRATION #
####################

def test_inverse_kinematics_accepts_and_applies_stiffness(pendulum):
    """
    With no reference data driving a coordinate, a large stiffness must hold it at its
    target rather than leaving it free.
    """
    solver = InverseKinematicsSolver(pendulum)
    solver.add_cost(CoordinateStiffnessCost({Q0: 1.0}))
    assert len(solver.costs) == 1


def test_stiffness_pulls_an_underdetermined_coordinate_toward_its_target(pendulum):
    """
    Drive the objective with the stiffness alone and confirm the minimizer sits at the
    target: the penalty is what determines an otherwise unconstrained coordinate.
    """
    mc = ModelCache(pendulum)
    order = list(mc.coordinate_map)
    rep = CoordinateStiffnessCost({Q0: 1.0, Q1: 1.0},
                                  targets={Q0: 0.2, Q1: -0.3}).create_rep(mc)

    x = ca.SX.sym('x', len(order))
    nlp = {'x': x, 'f': rep(CostInput(coordinates=x))}
    solver = ca.nlpsol('solver', 'ipopt', nlp,
                       {'ipopt': {'print_level': 0}, 'print_time': False})
    optimal = np.squeeze(solver(x0=np.zeros(len(order)))['x'].full())

    assert optimal[order.index(Q0)] == pytest.approx(0.2, abs=1e-6)
    assert optimal[order.index(Q1)] == pytest.approx(-0.3, abs=1e-6)


def test_splined_solver_applies_stiffness_per_time_sample(tmp_path):
    """
    A coordinate-dependent cost is a new code path in `SplinedKinematicsSolver`: it is
    evaluated at every time sample rather than once per solve. Solve the same problem
    with and without a stiffness pulling one coordinate away from what the markers
    alone imply, and confirm the stiffened solution moves toward the target.
    """
    from osimfit.data_sources import MarkerSource, Trial
    from tests.test_mobilizer_parameters import (create_beam_recovery_model,
                                                 create_prescribed_markers)

    coordinate = '/jointset/elbow_r/elbow_r_coord_2'
    target = 0.4

    truth = create_beam_recovery_model()
    trc_path = str(tmp_path / 'markers.trc')
    create_prescribed_markers(truth, trc_path)
    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}

    def solve(stiffness):
        model = create_beam_recovery_model()
        model.initSystem()
        solver = SplinedKinematicsSolver(model, convergence_tolerance=1e-6,
                                         knot_interval=0.1, position_weight=5.0)
        solver.add_trial(Trial('beam', [MarkerSource('markers', trc_path,
                                                     label_map=label_map)]))
        if stiffness is not None:
            solver.add_cost(CoordinateStiffnessCost({coordinate: stiffness},
                                                    targets={coordinate: target}))
        solution = solver.solve()
        column = solution.states_tables['beam'].getDependentColumn(
            coordinate + '/value').to_numpy()
        return float(np.mean(column))

    plain = solve(None)
    stiffened = solve(1e3)

    # The stiffened solve must sit closer to the target than the unstiffened one.
    assert abs(stiffened - target) < abs(plain - target)
