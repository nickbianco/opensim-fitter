import os
import time
import numpy as np
import matplotlib.pyplot as plt
import opensim as osim
from osimfit.data_sources import TheiaFrameSource
from osimfit.scaling import PositionBasedScaler, FrameMeasurement, Axis, \
                            AnthropometricScaler, AnthropometricMeasurement
from osimfit.solvers import (InverseKinematicsSolver,
                             SplineBasedInverseKinematicsSolver,
                             SplineTrackingSolution)
from osimfit.model import BodyScale
from osimfit.bounds import Bounds

# EXAMPLE JUMP
# ------------
# This example demonstrates how to go from body position and orientation data collected
# from the Theia markerless motion capture system to a scaled OpenSim model and
# corresponding joint kinematics.

# Position-based scaling
# ----------------------
# First, we will scale a generic OpenSim model based on the positions of body segments
# measured by Theia. This is similar to the approach used by the OpenSim scale tool,
# which computes body scales for each body segment based on the ratio of measured
# distances between experimental markers and the corresponding distances between
# virtual markers on the model.

# Define a mapping between experimental data labels and model frames.
# (data label --> model frame path)
frame_map = {
    'l_thigh': '/jointset/hip_l/femur_l_offset/l_thigh',
    'l_shank': '/jointset/walker_knee_l/tibia_l_offset/l_shank',
    'l_foot': '/jointset/ankle_l/talus_l_offset/l_foot',
    'l_toes': '/jointset/mtp_l/toes_l_offset/l_toes',
    'r_thigh': '/jointset/hip_r/femur_r_offset/r_thigh',
    'r_shank': '/jointset/walker_knee_r/tibia_r_offset/r_shank',
    'r_foot': '/jointset/ankle_r/talus_r_offset/r_foot',
    'r_toes': '/jointset/mtp_r/toes_r_offset/r_toes',
    'l_uarm': '/jointset/acromial_l/humerus_l_offset/l_uarm',
    'l_larm': '/jointset/elbow_l/ulna_l_offset/l_larm',
    'l_hand': '/jointset/radius_hand_l/hand_l_offset/l_hand',
    'r_uarm': '/jointset/acromial_r/humerus_r_offset/r_uarm',
    'r_larm': '/jointset/elbow_r/ulna_r_offset/r_larm',
    'r_hand': '/jointset/radius_hand_r/hand_r_offset/r_hand',
    'pelvis': '/bodyset/pelvis/pelvis',
    'torso': '/bodyset/torso/torso',
}

# Define a mapping between model body segments and the scaling "rules" to apply during
# model scaling. Each rule consists of two Theia frame data labels and the axis
# along which to apply the body scale. The body scale is computed as the ratio of
# the distance between the two Theia frames in the experimental data and the distance
# between the corresponding points on the model.
# (segment_name --> [data_label_1, data_label_2, axis])
scale_map = {
    'pelvis': ['pelvis', 'torso', Axis.YAxis],
    'pelvis': ['l_thigh', 'r_thigh', Axis.ZAxis],
    'torso': ['torso', 'pelvis', Axis.YAxis],
    'torso': ['l_uarm', 'r_uarm', Axis.ZAxis],
    'humerus_r': ['r_uarm', 'r_larm', Axis.YAxis],
    'humerus_l': ['l_uarm', 'l_larm', Axis.YAxis],
    'radius_r': ['r_larm', 'r_hand', Axis.YAxis],
    'radius_l': ['l_larm', 'l_hand', Axis.YAxis],
    'femur_r': ['r_thigh', 'r_shank', Axis.YAxis],
    'femur_l': ['l_thigh', 'l_shank', Axis.YAxis],
    'tibia_r': ['r_shank', 'r_foot', Axis.YAxis],
    'tibia_l': ['l_shank', 'l_foot', Axis.YAxis],
    'calcn_r': ['r_foot', 'r_toes', Axis.XAxis],
    'calcn_r': ['r_foot', 'r_toes', Axis.YAxis],
    'calcn_l': ['l_foot', 'l_toes', Axis.XAxis],
    'calcn_l': ['l_foot', 'l_toes', Axis.YAxis]
}

# Load the model and Theia frame data, and create a PositionBasedScaler object.
model = osim.Model('unscaled_generic.osim')
model.initSystem()

# Save a clone of the unscaled model for the bilevel optimization below.
unscaled_model = osim.Model(model)

c3d_source = TheiaFrameSource('pose_0.c3d')
position_scaler = PositionBasedScaler(model, c3d_source)

# Add scaling rules to the PositionBasedScaler based on the mapping above.
for segment_name, (data_label_1, data_label_2, axis) in scale_map.items():
    measurement = FrameMeasurement(frame_map[data_label_1], frame_map[data_label_2])
    position_scaler.add_measurement_body_scale(
        segment_name, axis, measurement, data_label_1, data_label_2)

# Add symmetry pairs. Internally, the PositionBasedScaler will average the body scales
# computed for each pair of symmetric segments to ensure left-right symmetry.
position_scaler.add_symmetry_pair('humerus_l', 'humerus_r')
position_scaler.add_symmetry_pair('radius_l', 'radius_r')
position_scaler.add_symmetry_pair('femur_l', 'femur_r')
position_scaler.add_symmetry_pair('tibia_l', 'tibia_r')
position_scaler.add_symmetry_pair('calcn_l', 'calcn_r')

# Scale the model.
scaled_model = position_scaler.scale()
scaled_model.printToXML('jump_1_scaled.osim')

# Anthropometry-based scaling
# ---------------------------
# Next, we will adjust the scaled model based on anthropometric measurements from the
# ANSUR II dataset.

# Define a mapping between ANSUR II measurement labels and pairs of stations (e.g.,
# body-fixed points) representing the measurement, along with the axis along which to
# apply the measurement. If no axis is specified, the measurement will be applied
# isotropically.
# ansur_label --> (station1_path, station2_path, axis)
ansur_measurements = {
    'biacromialbreadth':      ('/acromion_r', '/acromion_l', None),
    'bicristalbreadth':       ('/iliocrestale_r', '/iliocrestale_l', None),
    'bimalleolarbreadth':     ('/lateral_malleolus_r', '/medial_malleolus_r', None),
    'footbreadthhorizontal':  ('/mtp1_r', '/mtp5_r', Axis.ZAxis),
    'footlength':             ('/acropodion_r', '/pternion_r', Axis.XAxis),
    'iliocristaleheight':     ('/iliocrestale_r', '/mtp5_r', Axis.YAxis),
    'lateralmalleolusheight': ('/lateral_malleolus_r', '/mtp5_r', Axis.YAxis),
    'radialestylionlength':   ('/radiale_r', '/stylion_r', None),
    'shoulderelbowlength':    ('/acromion_r', '/olecranon_r', None),
    'stature':                ('/vertex', '/mtp5_r', Axis.YAxis),
    'suprasternaleheight':    ('/suprasternale', '/mtp5_r', Axis.YAxis),
    'tibialheight':           ('/tibiale_r', '/mtp5_r', Axis.YAxis),
    'trochanterionheight':    ('/trochanterion_r', '/mtp5_r', Axis.YAxis),
    'waistbacklength':        ('/cervicale', '/posterior_omphalion', None),
    'waistdepth':             ('/posterior_omphalion', '/anterior_omphalion', None)
}

# Create the AnthropometricScaler and add measurements based on the mapping above.
anthropometric_scaler = AnthropometricScaler(scaled_model, sex='female')

# Build the measurements once so we can reuse them by label below.
ansur_measurement_map = {
    label: AnthropometricMeasurement(station1_path, station2_path, axis)
    for label, (station1_path, station2_path, axis) in ansur_measurements.items()
}

# Register every measurement so it participates in the joint MVN distribution.
for ansur_label, measurement in ansur_measurement_map.items():
    anthropometric_scaler.add_measurement(ansur_label, measurement)

# Select of subset of the measurements that we will use to condition the
# multivariate normal distribution. These measurements are "trustworthy" in the
# sense that we can estimate them relatively well from the Theia frames.
anthropometric_scaler.add_conditional_measurement('iliocristaleheight')
anthropometric_scaler.add_conditional_measurement('radialestylionlength')
anthropometric_scaler.add_conditional_measurement('shoulderelbowlength')
anthropometric_scaler.add_conditional_measurement('stature')
anthropometric_scaler.add_conditional_measurement('suprasternaleheight')
anthropometric_scaler.add_conditional_measurement('tibialheight')
anthropometric_scaler.add_conditional_measurement('trochanterionheight')
anthropometric_scaler.add_conditional_measurement('waistbacklength')

# Define the body scales that will be generated from the conditioned anthropometric
# measurements.
anthro_scale_rules = [
    ('torso',   'biacromialbreadth',     Axis.ZAxis),
    ('pelvis',  'bicristalbreadth',      Axis.ZAxis),
    ('tibia_r', 'bimalleolarbreadth',    Axis.YAxis),
    ('tibia_r', 'bimalleolarbreadth',    Axis.ZAxis),
    ('tibia_l', 'bimalleolarbreadth',    Axis.YAxis),
    ('tibia_l', 'bimalleolarbreadth',    Axis.ZAxis),
    ('calcn_r', 'footlength',            Axis.XAxis),
    ('calcn_r', 'footbreadthhorizontal', Axis.ZAxis),
    ('calcn_l', 'footlength',            Axis.XAxis),
    ('calcn_l', 'footbreadthhorizontal', Axis.ZAxis),
]
for segment, ansur_label, axis in anthro_scale_rules:
    anthropometric_scaler.add_anthropometric_body_scale(segment, axis, ansur_label)

# Scale the model.
anthro_scaled_model = anthropometric_scaler.scale()
anthro_scaled_model.printToXML('jump_1_anthro_scaled.osim')

# Frame-by-frame inverse kinematics
# ---------------------------------
# Reload the Theia frame data, now discarding any unused data columns and updating the
# frame labels to match the model frame paths.
columns_to_remove = ['worldbody', 'head', 'pelvis_shifted', 'l_clavicle', 'r_clavicle']
theia_frame_source = TheiaFrameSource('pose_0.c3d',
                                      labels_to_remove=columns_to_remove,
                                      label_map=frame_map)

# Run the frame-by-frame IK solver.
solver = InverseKinematicsSolver(anthro_scaled_model,
                                 convergence_tolerance=1e-3,
                                 position_weight=1.0,
                                 orientation_weight=2.0)
solver.add_theia_frame_reference_data(theia_frame_source)
ik_solution = solver.solve()
sto = osim.STOFileAdapter()
sto.write(ik_solution.states_table, 'jump_1_ik_solution.sto')

# Spline-based bilevel optimization
# ----------------------------------
# Construct a SplineBasedBilevelSolver to solve for the model kinematics and body
# scales that best match the Theia frame data, initialized from the unscaled model.
solver = SplineBasedInverseKinematicsSolver(anthro_scaled_model,
                                            convergence_tolerance=1e-4,
                                            knot_interval=0.05,
                                            position_weight=5.0,
                                            orientation_weight=10.0)
solver.add_theia_frame_reference_data(theia_frame_source)

# Create an initial guess based on the kinematics from the frame-by-frame inverse
# kinematics solution and the combined body scales set above.
guess = SplineTrackingSolution(
    states_table=osim.TimeSeriesTable('jump_1_ik_solution.sto'))
spline_ik_solution = solver.solve(guess)
sto = osim.STOFileAdapter()
sto.write(spline_ik_solution.states_table, 'jump_1_spline_ik_solution.sto')

# Visualization
# -------------
modelProcessor = osim.ModelProcessor('jump_1_anthro_scaled.osim')
modelProcessor.append(osim.ModOpRemoveMuscles())
model = modelProcessor.process()
model.initSystem()

states = osim.TimeSeriesTable('jump_1_spline_ik_solution.sto')
states.addTableMetaDataString('inDegrees', 'no')
osim.VisualizerUtilities.showMotion(model, states)

# Plot joint kinematics
# ---------------------
coordinate = '/jointset/hip_r/hip_flexion_r/value'
ylabel = 'hip flexion (deg)'
fig = plt.figure(figsize=(6.5, 4))
ax = fig.subplots(1, 1)

# Plot frame-by-frame IK solution.
solution = osim.TimeSeriesTable('jump_1_ik_solution.sto')
t = np.array(solution.getIndependentColumn())
ax.plot(solution.getIndependentColumn(),
        np.rad2deg(solution.getDependentColumn(coordinate).to_numpy()),
        color='black', lw=5.0, alpha=0.5,
        label='Frame-by-frame IK')
ax.set_ylabel(ylabel)
ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.75)

# Plot the bilevel optimization solution.
solution = osim.TimeSeriesTable(
    f'jump_1_spline_ik_solution.sto')
t_bilevel = np.array(solution.getIndependentColumn())
y_bilevel = np.rad2deg(solution.getDependentColumn(coordinate).to_numpy())
ax.plot(t_bilevel, y_bilevel,
        color='darkorange', lw=3.5,
        label=f'Spline-based IK optimization')
ax.set_ylim(-5, 85)
ax.set_xlabel('time (s)')
ax.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('compare_joint_kinematics.png', dpi=150, bbox_inches='tight')
plt.show()
