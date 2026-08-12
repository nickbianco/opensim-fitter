"""
Unit tests for `Cost`s.
"""

import pytest
import casadi as ca

from osimfit.solvers import InverseKinematicsSolver, SplinedKinematicsSolver
from osimfit.costs import (Cost, CostInput, BodyScaleRegularizationCost,
                           OffsetRegularizationCost)
from test_scaled_double_pendulum import create_double_pendulum


class CoordinatePenalty(Cost):
    """A minimal cost that depends only on the coordinates."""
    required_inputs = frozenset({'coordinates'})

    def __call__(self, input: CostInput) -> ca.MX:
        return ca.sumsqr(input.coordinates)


@pytest.fixture
def model():
    m = create_double_pendulum(1.0, 1.0)
    m.initSystem()
    return m


def test_inverse_kinematics_accepts_coordinate_cost(model):
    solver = InverseKinematicsSolver(model)
    solver.add_cost(CoordinatePenalty())
    assert len(solver.costs) == 1


def test_inverse_kinematics_rejects_parameter_cost(model):
    solver = InverseKinematicsSolver(model)
    with pytest.raises(ValueError, match='body_scales'):
        solver.add_cost(BodyScaleRegularizationCost(1.0))
    assert solver.costs == []


def test_splined_accepts_parameter_costs(model):
    solver = SplinedKinematicsSolver(model)
    solver.add_cost(BodyScaleRegularizationCost(1.0))
    solver.add_cost(OffsetRegularizationCost(1.0))
    assert len(solver.costs) == 2


def test_splined_rejects_coordinate_cost(model):
    solver = SplinedKinematicsSolver(model)
    with pytest.raises(ValueError, match='coordinates'):
        solver.add_cost(CoordinatePenalty())
    assert solver.costs == []
