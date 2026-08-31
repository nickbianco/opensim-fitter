import os
import time
import numpy as np
import matplotlib.pyplot as plt
import opensim as osim
from scipy.interpolate import BSpline
from osimfit.data_sources import MarkerSource, Trial
from osimfit.solvers import InverseKinematicsSolver, SplinedKinematicsSolver

# EXAMPLE SPRINT
# --------------
# This example demonstrates how to use the inverse kinematics solvers, both
# frame-by-frame and spline-based, to find the kinematics for a sprinting motion. The
# marker trajectories for the sprinting motion estimated using OpenCap's marker
# augmenter, which uses an LSTM to predict marker trajectories from keypoints extracted
# from smartphone videos via pose detection (e.g., HRNet).

# Load data
# ---------

# Load the marker data and model.
marker_table = osim.TimeSeriesTableVec3('sprint_markers.trc')
model = osim.Model('LaiUhlrich2022_scaled.osim')
model.initSystem()

# Remove markers associated with pose keypoints.
columns_to_remove = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow',
                     'LWrist', 'midHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee',
                     'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe',
                     'RHeel']

# Create a mapping between marker names and paths (e.g., r_calcn -> /markerset/r_calcn).
labels = marker_table.getColumnLabels()
labels = [label for label in labels if label not in columns_to_remove]
label_map = {label: f'/markerset/{label}' for label in labels}

# Create a MarkerSource with the option to trim the data to a specific time range.
marker_source = MarkerSource('sprint_markers', 'sprint_markers.trc',
                             labels_to_remove=columns_to_remove,
                             label_map=label_map,
                             trim_to_range=(1.5, 2.4))

# Frame-by-frame inverse kinematics
# ---------------------------------
solver = InverseKinematicsSolver(model,
                                 convergence_tolerance=1e-2,
                                 position_weight=1.0)
solver.add_trial(Trial('sprint', [marker_source]))
ik_solution = solver.solve()
sto = osim.STOFileAdapter()
sto.write(ik_solution.states_tables['sprint'], 'sprint_ik_solution.sto')

# Spline-based inverse kinematics
# -------------------------------
# Run the spline IK solver, initialized with the frame-by-frame solution, for varying
# knot densities.
knot_intervals = [0.04, 0.08]
colors = ['blue', 'orange']
for knot_interval in knot_intervals:
    solver = SplinedKinematicsSolver(model,
                                     convergence_tolerance=1e-4,
                                     position_weight=1.0,
                                     knot_interval=knot_interval)
    solver.add_trial(Trial('sprint', [marker_source]))
    spline_ik_solution = solver.solve(ik_solution)
    sto = osim.STOFileAdapter()
    sto.write(spline_ik_solution.states_tables['sprint'],
              f'sprint_spline_based_ik_solution_knot{int(knot_interval*1000)}.sto')

# Plot joint kinematics
# ---------------------
coordinates  = ['/jointset/ground_pelvis/pelvis_list/value',
                '/jointset/hip_r/hip_adduction_r/value',
                '/jointset/subtalar_r/subtalar_angle_r/value']
ylabels = ['pelvis list (deg)', 'hip adduction (deg)', 'subtalar angle (deg)']
fig, axes = plt.subplots(3, 1, figsize=(6, 2.5 * len(coordinates)), sharex=True)

# Plot frame-by-frame IK solution.
solution = osim.TimeSeriesTable('sprint_ik_solution.sto')
for ax, coord in zip(axes.flatten(), coordinates):
    t = np.array(solution.getIndependentColumn())
    ax.plot(solution.getIndependentColumn(),
            np.rad2deg(solution.getDependentColumn(coord).to_numpy()),
            color='black', alpha=0.4, lw=5, label='Frame-by-frame IK')
    ax.set_ylabel(ylabels[coordinates.index(coord)])
    ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.75)

# Plot the spline-based IK solutions.
cmap = plt.get_cmap('viridis')
for i, knot_interval in enumerate(knot_intervals):
    solution = osim.TimeSeriesTable(
        f'sprint_spline_based_ik_solution_knot{int(knot_interval*1000)}.sto')
    for ax, coord in zip(axes.flatten(), coordinates):
        ax.plot(solution.getIndependentColumn(),
                np.rad2deg(solution.getDependentColumn(coord).to_numpy()),
                color=colors[i], lw=3,
                label=f'Spline-based IK (interval={1000*knot_interval:.0f} ms)')

axes.flatten()[-1].set_xlabel('time (s)')
axes.flatten()[0].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('compare_joint_kinematics.png', dpi=150, bbox_inches='tight')
# plt.show()

# Plot hamstring kinematics
# -------------------------
time_range = [1.575, 2.075]

# Load the frame-by-frame IK solution and apply a low-pass filter to smooth the
# kinematics. Then, use a spline fit to compute the derivatives of the coordinate
# values.
tableProcessor = osim.TableProcessor('sprint_ik_solution.sto')
tableProcessor.append(osim.TabOpLowPassFilter(10))
tableProcessor.append(osim.TabOpAppendCoordinateValueDerivativesAsSpeeds())
ik_states_table = tableProcessor.process(model)
ik_states_table.addTableMetaDataString('inDegrees', 'no')
ik_states_table.trim(time_range[0], time_range[1])
ik_states = osim.StatesTrajectory.createFromStatesTable(
    model, ik_states_table, True, False, False)

# Load the spline-based IK solution.
spline_based_ik_solution = osim.TimeSeriesTable(
    'sprint_spline_based_ik_solution_knot80.sto')
spline_based_ik_solution.addTableMetaDataString('inDegrees', 'no')
spline_based_ik_solution.trim(time_range[0], time_range[1])
spline_based_ik_states = osim.StatesTrajectory.createFromStatesTable(
    model, spline_based_ik_solution, True, False, False)

# Compute biceps femoris long head (bflh) lengths and lengthening speeds.
bflh_lengths = {'ik': np.ndarray(ik_states.getSize()),
                'spline_ik': np.ndarray(spline_based_ik_states.getSize())}
bflh_speeds = {'ik': np.ndarray(ik_states.getSize()),
               'spline_ik': np.ndarray(spline_based_ik_states.getSize())}
bflh_r = osim.Millard2012EquilibriumMuscle.safeDownCast(
    model.getComponent('/forceset/bflh_r'))

for istate in range(ik_states.getSize()):
    state = ik_states.get(istate)
    model.realizeVelocity(state)
    bflh_lengths['ik'][istate] = bflh_r.getLength(state)
    bflh_speeds['ik'][istate] = bflh_r.getLengtheningSpeed(state)

for istate in range(spline_based_ik_states.getSize()):
    state = spline_based_ik_states.get(istate)
    model.realizeVelocity(state)
    bflh_lengths['spline_ik'][istate] = bflh_r.getLength(state)
    bflh_speeds['spline_ik'][istate] = bflh_r.getLengtheningSpeed(state)

# Load experimental data from Yu et al. 2008.
lit_lengths_raw = np.loadtxt('BingYuBFLHLengths.csv', delimiter=',')
lit_velocities_raw = np.loadtxt('BingYuBFLHVelocities.csv', delimiter=',')
def prepare_literature_curve(raw):
    order = np.argsort(raw[:, 0])
    x_sorted = raw[order, 0]
    y_sorted = raw[order, 1]
    _, unique_idx = np.unique(x_sorted, return_index=True)
    return x_sorted[unique_idx], y_sorted[unique_idx]
lit_len_x, lit_len_y = prepare_literature_curve(lit_lengths_raw)
lit_vel_x, lit_vel_y = prepare_literature_curve(lit_velocities_raw)

# Plot bflh lengths and lengthening speeds for the frame-by-frame and spline-based IK
# solutions, along with the experimental data from Yu et al. 2008.
ik_times = np.linspace(0, 100, len(ik_states_table.getIndependentColumn()))
spline_ik_times = np.linspace(0, 100,
                              len(spline_based_ik_solution.getIndependentColumn()))
fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
axes[0].plot(lit_len_x, lit_len_y, color='black', lw=3, ls='--',
             label='Literature (Yu et al. 2008)')
axes[0].plot(ik_times, bflh_lengths['ik'], color='blue', lw=3,
             label='Frame-by-frame IK')
axes[0].plot(spline_ik_times, bflh_lengths['spline_ik'], color='orange', lw=3,
             label='Spline-based IK (interval=80 ms)')

axes[0].set_ylabel('bflh length (m)')
axes[0].grid(True, which='both', ls='--', lw=0.5, alpha=0.75)
axes[0].legend(loc='best', fontsize=8)

axes[1].plot(lit_vel_x, lit_vel_y, color='black', lw=3, ls='--',
             label='Literature (Yu et al. 2008)')
axes[1].plot(ik_times, bflh_speeds['ik'], color='blue', lw=3,
             label='Frame-by-frame IK')
axes[1].plot(spline_ik_times, bflh_speeds['spline_ik'], color='orange', lw=3,
             label='Spline-based IK (interval=80 ms)')
axes[1].set_ylabel('bflh lengthening speed (m/s)')
axes[1].set_xlabel('time (s)')
axes[1].grid(True, which='both', ls='--', lw=0.5, alpha=0.75)

plt.tight_layout()
plt.savefig('compare_bflh_kinematics.png', dpi=150, bbox_inches='tight')
# plt.show()
