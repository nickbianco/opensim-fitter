import os
import numpy as np
import opensim as osim
from osimfit.data_sources import MarkerSource, Trial
from osimfit.solvers import SplinedKinematicsSolver
from osimfit.model import BodyScale
from osimfit.costs import BodyScaleRegularizationCost
from osimfit.bounds import Bounds

# EXAMPLE PENDULUM
# ----------------
# This example is a simple demonstration of bilevel optimization where both generalized
# coordinates (i.e., joint angles) and body scales are optimized to fit marker
# data. This is a "round-trip" example, where we first create synthetic marker data by
# simulating a double pendulum with known body lengths, and then we run the
# optimization to see if we can recover the original kinematics and body lengths of the
# double pendulum.

# Helper function for create double pendulum OpenSim models with varying body lengths.
def create_double_pendulum(length1: float, length2: float) -> osim.Model:
    model = osim.Model()
    model.setName("double_pendulum")
    ground = model.getGround()

    # Create the first pendulum body and joint.
    b0 = osim.Body("b0", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(b0)
    j0 = osim.PinJoint("j0",
        ground, osim.Vec3(0), osim.Vec3(0),
        b0, osim.Vec3(-length1, 0, 0), osim.Vec3(0))
    q0 = j0.updCoordinate()
    q0.setName("q0")
    model.addJoint(j0)

    # Create the second pendulum body and joint.
    b1 = osim.Body("b1", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(b1)
    j1 = osim.PinJoint("j1", b0, osim.Vec3(0), osim.Vec3(0),
                       b1, osim.Vec3(-length2, 0, 0), osim.Vec3(0))
    q1 = j1.updCoordinate()
    q1.setName("q1")
    model.addJoint(j1)

    # Add markers to the origin of each pendulum body.
    m0 = osim.Marker("m0", b0, osim.Vec3(0))
    model.addMarker(m0)
    m1 = osim.Marker("m1", b1, osim.Vec3(0))
    model.addMarker(m1)

    # Attach an ellipsoid to a frame located at the center of each body.
    for body, length in zip([b0, b1], [length1, length2]):
        center_frame = osim.PhysicalOffsetFrame(
            f"{body.getName()}_center", body,
            osim.Transform(osim.Vec3(-0.5*length, 0, 0)))
        model.addComponent(center_frame)
        ellipsoid = osim.Ellipsoid(0.5 * length, 0.1, 0.1)
        center_frame.attachGeometry(ellipsoid)

    model.finalizeConnections()
    return model


# Create synthetic marker data by simulating a double pendulum (with non-uniform
# lengths) and saving the marker trajectories to a .trc file.

# Create a simple pendulum model.
model = create_double_pendulum(1.25, 0.75)
state = model.initSystem()

# Run a forward simulations at a fixed framerate.
manager = osim.Manager(model)
manager.setIntegratorFixedStepSize(0.01)
manager.initialize(state)
manager.integrate(2.0)
states = manager.getStatesTable()

# Extract the marker trajectories.
controls = osim.TimeSeriesTable(states.getIndependentColumn())
output_paths = osim.StdVectorString()
output_paths.append('/markerset/.*location')
markers = osim.analyzeVec3(model, states, controls, output_paths)

# Save the marker trajectories to a .trc file.
markers.addTableMetaDataString('DataRate', '100.0')
markers.addTableMetaDataString('Units', 'm')
trc = osim.TRCFileAdapter()
trc.write(markers, 'markers.trc')

# Now we'll run a bilevel optimization problem tracking the synthetic marker data and
# see if we can recover the original kinematics and body lengths of the double pendulum.

# Load the marker trajectories from the .trc file.
marker_table = osim.TimeSeriesTableVec3('markers.trc')
model = create_double_pendulum(1.0, 1.0)
model.initSystem()

# Create a MarkerSource.
labels = marker_table.getColumnLabels()
label_map = {label: label.replace('|location', '') for label in labels}
marker_source = MarkerSource('markers', 'markers.trc', label_map=label_map)

# Construct a SplinedKinematicsSolver to solve for the model kinematics and body
# lengths that best match the marker data.
solver = SplinedKinematicsSolver(model,
                                 convergence_tolerance=1e-5,
                                 knot_interval=0.05,
                                 position_weight=5.0)
solver.add_trial(Trial('pendulum', [marker_source]))
solver.add_cost(BodyScaleRegularizationCost(1e-2))
# Add body scales for the two bodies including lower and upper bounds for the
# optimization variables.
solver.add_parameter(BodyScale('/bodyset/b0', Bounds(0.5, 2.0), np.ones(3)))
solver.add_parameter(BodyScale('/bodyset/b1', Bounds(0.5, 2.0), np.ones(3)))

# Solve!
solution = solver.solve()
states_table = solution.states_tables['pendulum']

# Write the solution to a .sto file.
sto = osim.STOFileAdapter()
sto.write(states_table, 'double_pendulum_ik_solution.sto')

# Print the optimized body lengths.
print("\nOptimized body lengths")
print("----------------------")
scales = [p for p in solution.parameters if isinstance(p, BodyScale)]
print(f' b0 length = {scales[0].value[0]:.3f} m')
print(f' b1 length = {scales[1].value[0]:.3f} m\n')

# Plot joint kinematics from the solution compared to the original simulation.
import matplotlib.pyplot as plt
coordinates  = ['/jointset/j0/q0/value', '/jointset/j1/q1/value']
ylabels = ['q0 (rad)', 'q1 (rad)']
fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
for i, coord in enumerate(coordinates):
    axes[i].plot(states_table.getIndependentColumn(),
                 states_table.getDependentColumn(coord),
                 label='fitting solution', linewidth=4)
    axes[i].plot(states.getIndependentColumn(),
                 states.getDependentColumn(coord),
                 label='original simulation', linestyle='--', linewidth=3)
    axes[i].set_ylabel(ylabels[i])
    axes[i].grid(True, which='both', ls='--', lw=0.5, alpha=0.75)
    axes[i].set_xlim(0, 1.0)

axes[0].set_ylim(-2.5, 0.5)
axes[1].set_ylim(-0.5, 1.0)
axes[1].set_xlabel('time (s)')
axes[0].legend()
plt.show()
