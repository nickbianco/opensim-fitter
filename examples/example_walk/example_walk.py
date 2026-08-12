import os
import time
import numpy as np
import matplotlib.pyplot as plt
import opensim as osim
from osimfit.data_sources import MarkerSource
from osimfit.scaling import (Axis, PositionBasedScaler, MarkerMeasurement,
                             AnthropometricMeasurement, AnthropometricScaler)
from osimfit.solvers import (InverseKinematicsSolver, MarkerPlacer,
                             SplinedKinematicsSolver, SplinedKinematicsSolution)
from osimfit.model import BodyScale, MarkerOffset
from osimfit.costs import BodyScaleRegularizationCost, OffsetRegularizationCost
from osimfit.bounds import Bounds
from osimfit.utilities import (compute_marker_errors, plot_marker_errors,
                               plot_coordinates)

# EXAMPLE WALK
# ------------
# This example demonstrates how to perform scaling and inverse kinematics for a walking
# motion using OpenSim Fitter. The example data comes from the Rajagopal et al. (2016)
# model distribution. The model has been modified from the original for the purposes of
# the example: the wrist joints have been welded, the subtalar and toe joints have been
# unlocked, and stations corresponding to anatomical locations have been added (to
# support the anthropometric scaling step).

# Load data
# ---------
# Load the marker data and model.
markers_fpath = 'motion_capture_walk.trc'
markers_table = osim.TimeSeriesTableVec3(markers_fpath)
marker_labels = markers_table.getColumnLabels()
model = osim.Model('RajagopalLaiUhlrich2023.osim')
model.initSystem()

# Append the markerset used for scaling and inverse kinematics to the model.
markerset = model.updMarkerSet()
markerset.clearAndDestroy()
mset = osim.MarkerSet('markerset_walk.xml')
for i in range(mset.getSize()):
    markerset.cloneAndAppend(mset.get(i))

# Save a clone of the unscaled model.
unscaled_model = osim.Model(model)

# Define a mapping between marker names and marker paths.
# (marker_name --> /marker/path)
marker_map = {label: f'/markerset/{label}' for label in marker_labels}

# Marker-based scaling
# --------------------
# Define scaling rules as a list of (segment, marker_1, marker_2, axis) tuples.
# Each rule specifies a segment to scale, two markers whose inter-distance defines
# the body scale, and the axis along which to apply it.
scale_rules = [
    ('torso', 'R.PSIS', 'R.Shoulder', Axis.YAxis),
    ('torso', 'L.PSIS', 'L.Shoulder', Axis.YAxis),
    ('torso', 'R.Shoulder', 'L.Shoulder', Axis.ZAxis),

    ('pelvis', 'R.ASIS', 'L.ASIS', Axis.ZAxis),
    ('pelvis', 'R.PSIS', 'L.PSIS', Axis.ZAxis),
    ('pelvis', 'R.PSIS', 'R.ASIS', Axis.XAxis),
    ('pelvis', 'L.PSIS', 'L.ASIS', Axis.XAxis),

    ('humerus_r', 'R.Shoulder', 'R.Elbow', Axis.YAxis),
    ('humerus_l', 'L.Shoulder', 'L.Elbow', Axis.YAxis),

    ('radius_r', 'R.Elbow', 'R.Wrist', Axis.YAxis),
    ('radius_l', 'L.Elbow', 'L.Wrist', Axis.YAxis),

    ('ulna_r', 'R.Elbow', 'R.Wrist', Axis.YAxis),
    ('ulna_l', 'L.Elbow', 'L.Wrist', Axis.YAxis),

    ('hand_r', 'R.Elbow', 'R.Wrist', Axis.YAxis),
    ('hand_l', 'L.Elbow', 'L.Wrist', Axis.YAxis),

    ('femur_r', 'R.ASIS', 'R.Knee', Axis.YAxis),
    ('femur_l', 'L.ASIS', 'L.Knee', Axis.YAxis),

    ('patella_r', 'R.ASIS', 'R.Knee', Axis.YAxis),
    ('patella_l', 'L.ASIS', 'L.Knee', Axis.YAxis),

    ('tibia_r', 'R.Knee', 'R.Ankle', Axis.YAxis),
    ('tibia_l', 'L.Knee', 'L.Ankle', Axis.YAxis),

    ('calcn_r', 'R.Heel', 'R.Toe', Axis.XAxis),
    ('calcn_r', 'R.Heel', 'R.MT5', Axis.XAxis),
    ('calcn_r', 'R.Toe', 'R.MT5', Axis.ZAxis),
    ('calcn_r', 'R.Heel', 'R.Ankle', Axis.YAxis),
    ('toes_r', 'R.Heel', 'R.Toe', Axis.XAxis),
    ('toes_r', 'R.Heel', 'R.MT5', Axis.XAxis),
    ('toes_r', 'R.Toe', 'R.MT5', Axis.ZAxis),
    ('toes_r', 'R.Heel', 'R.Ankle', Axis.YAxis),

    ('calcn_l', 'L.Heel', 'L.Toe', Axis.XAxis),
    ('calcn_l', 'L.Heel', 'L.MT5', Axis.XAxis),
    ('calcn_l', 'L.Toe', 'L.MT5', Axis.ZAxis),
    ('calcn_l', 'L.Heel', 'L.Ankle', Axis.YAxis),
    ('toes_l', 'L.Heel', 'L.Toe', Axis.XAxis),
    ('toes_l', 'L.Heel', 'L.MT5', Axis.XAxis),
    ('toes_l', 'L.Toe', 'L.MT5', Axis.ZAxis),
    ('toes_l', 'L.Heel', 'L.Ankle', Axis.YAxis),
]


# Create a MarkerSource and PositionBasedScaler.
marker_source = MarkerSource(markers_fpath)
position_scaler = PositionBasedScaler(model, marker_source)

# Add scaling rules to the PositionBasedScaler.
for segment_name, marker_1, marker_2, axis in scale_rules:
    measurement = MarkerMeasurement(marker_map[marker_1], marker_map[marker_2])
    position_scaler.add_measurement_body_scale(
        segment_name, axis, measurement, marker_1, marker_2)

# Add symmetry pairs. Internally, the PositionBasedScaler will average the body scales
# computed for each pair of symmetric segments to ensure left-right symmetry.
position_scaler.add_symmetry_pair('humerus_l', 'humerus_r')
position_scaler.add_symmetry_pair('radius_l', 'radius_r')
position_scaler.add_symmetry_pair('ulna_l', 'ulna_r')
position_scaler.add_symmetry_pair('hand_l', 'hand_r')
position_scaler.add_symmetry_pair('femur_l', 'femur_r')
position_scaler.add_symmetry_pair('patella_l', 'patella_r')
position_scaler.add_symmetry_pair('tibia_l', 'tibia_r')
position_scaler.add_symmetry_pair('calcn_l', 'calcn_r')
position_scaler.add_symmetry_pair('toes_l', 'toes_r')

# Scale the model.
scaled_model = position_scaler.scale()
scaled_model.printToXML('subject_marker_scaled_walk.osim')


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
    'waistdepth':             ('/posterior_omphalion', '/anterior_omphalion', None),
}

# Create the AnthropometricScaler and add measurements based on the mapping above.
anthropometric_scaler = AnthropometricScaler(scaled_model, sex='female')

# Build the measurements once so we can reuse them by label below.
ansur_measurement_map = {
    label: AnthropometricMeasurement(station1_path, station2_path, axis)
    for label, (station1_path, station2_path, axis) in ansur_measurements.items()
}

# Register every measurement so it participates in the joint MVN distribution.
# Measurements directly used by body scales will be registered redundantly by the
# harvest step inside scale() — that's harmless.
for ansur_label, measurement in ansur_measurement_map.items():
    anthropometric_scaler.add_measurement(ansur_label, measurement)

# Select a subset of the measurements that we will use to condition the
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
    ('toes_r',  'footlength',            Axis.XAxis),
    ('toes_r',  'footbreadthhorizontal', Axis.ZAxis),
    ('calcn_l', 'footlength',            Axis.XAxis),
    ('calcn_l', 'footbreadthhorizontal', Axis.ZAxis),
    ('toes_l',  'footlength',            Axis.XAxis),
    ('toes_l',  'footbreadthhorizontal', Axis.ZAxis),
]
for segment, ansur_label, axis in anthro_scale_rules:
    anthropometric_scaler.add_anthropometric_body_scale(
        segment, axis, ansur_label)

# Scale the model.
anthro_scaled_model = anthropometric_scaler.scale()
anthro_scaled_model.printToXML('subject_anthro_scaled_walk.osim')

# Place markers
# -------------
# Create a new marker source with updated column labels representing the full path to
# each marker.
marker_source = MarkerSource(markers_fpath, label_map=marker_map)
placer = MarkerPlacer(anthro_scaled_model, marker_source)
solution = placer.solve()
# Update both 'anthro_scaled_model', which we'll use to generate a guess via inverse
# kinematics, and 'unscaled_model' which we'll use in the final bilevel optimization.
anthro_scaled_model = placer.update_model(anthro_scaled_model, solution)
unscaled_model = placer.update_model(unscaled_model, solution)

# Frame-by-frame inverse kinematics
# ---------------------------------
# Run the frame-by-frame IK solver.
solver = InverseKinematicsSolver(anthro_scaled_model,
                                 convergence_tolerance=1e-2,
                                 position_weight=1.0)
solver.add_marker_reference_data(marker_source)
ik_solution = solver.solve()
sto = osim.STOFileAdapter()
sto.write(ik_solution.states_table, 'walk_ik_solution.sto')

# Spline-based inverse kinematics
# -------------------------------
# Construct a SplinedKinematicsSolver to solve for the model kinematics and body
# lengths that best match the marker data.
solver = SplinedKinematicsSolver(unscaled_model,
                                 convergence_tolerance=1e-2,
                                 knot_interval=0.05,
                                 position_weight=5.0)
solver.add_marker_reference_data(marker_source)
solver.add_cost(BodyScaleRegularizationCost(1e-1))
solver.add_cost(OffsetRegularizationCost(1e-3))
# Add body scales for each body in the model. Apply the same scales to groups of bodies,
# including those that should share left-right symmetry.
bounds = Bounds(0.5, 2.0)
solver.add_parameter(BodyScale('/bodyset/torso', bounds, np.ones(3)))
solver.add_parameter(BodyScale('/bodyset/pelvis', bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/humerus_r', '/bodyset/humerus_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/radius_r', '/bodyset/radius_l',
                                '/bodyset/ulna_r', '/bodyset/ulna_l',
                                '/bodyset/hand_r', '/bodyset/hand_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/femur_r', '/bodyset/femur_l',
                                '/bodyset/patella_r', '/bodyset/patella_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/tibia_r', '/bodyset/tibia_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/calcn_r', '/bodyset/calcn_l',
                                '/bodyset/toes_r', '/bodyset/toes_l'],
                               bounds, np.ones(3)))
# Add marker offset parameters for the tracking markers.
bounds = Bounds(-0.25, 0.25)
for i in range(unscaled_model.getMarkerSet().getSize()):
    marker = unscaled_model.getMarkerSet().get(i)
    if not marker.get_fixed():
        path = marker.getAbsolutePathString()
        solver.add_parameter(MarkerOffset(path, bounds, np.zeros(3)))

# Combine the per-body XYZ body scales from the two scaling stages above by
# element-wise multiplication.
def per_body_factors(scaleset, body_name):
    factors = scaleset.get(body_name).getScaleFactors()
    return np.array([factors[0], factors[1], factors[2]])

parameters_guess = [p.with_value(p.value) for p in solver.parameters]
body_scales = [p for p in parameters_guess if isinstance(p, BodyScale)]
for scale in body_scales:
    per_body = []
    for body_path in scale.paths:
        body_name = body_path.rsplit('/', 1)[-1]
        per_body.append(
            per_body_factors(position_scaler.scaleset, body_name)
            * per_body_factors(anthropometric_scaler.scaleset, body_name))
    scale.value = np.mean(per_body, axis=0)

# Create an initial guess based on the the kinematics from the inverse kinematics
# solution and the combined body scales set above.
guess = SplinedKinematicsSolution(
    states_table=osim.TimeSeriesTable('walk_ik_solution.sto'),
    parameters=parameters_guess)
bilevel_solution = solver.solve(guess)
sto.write(bilevel_solution.states_table, 'walk_bilevel_solution.sto')
bilevel_scaled_model = solver.update_model(unscaled_model, bilevel_solution)
bilevel_scaled_model.printToXML('subject_bilevel_scaled_walk.osim')

# Convert the solution to a StatesTrajectory for computing marker errors.
states_table = osim.TimeSeriesTable('walk_bilevel_solution.sto')
states_table.addTableMetaDataString('inDegrees', 'no')
states_traj = osim.StatesTrajectory.createFromStatesTable(bilevel_scaled_model,
                                                          states_table, True)

# Plotting
# --------
# Plot the coordinates.
coordinates_pdf_fpath = 'walk_bilevel_solution_coordinates.pdf'
coordinate_ranges = {
    'pelvis_tilt':      (-40, 40),
    'pelvis_list':      (-40, 40),
    'pelvis_rotation':  (-40, 40),
    'pelvis_tx':        (-7.5, 2.5),
    'pelvis_ty':        (0, 2.5),
    'pelvis_tz':        (-1.0, 1.0),
    'hip_rotation_r':   (-30, 30),
    'hip_rotation_l':   (-30, 30),
    'lumbar_extension': (-50, 50),
    'lumbar_bending':   (-50, 50),
    'lumbar_rotation':  (-50, 50),
    'arm_flex_r':       (-100, 100),
    'arm_add_r':        (-100, 100),
    'arm_rot_r':        (-100, 100),
    'arm_flex_l':       (-100, 100),
    'arm_add_l':        (-100, 100),
    'arm_rot_l':        (-100, 100),
}
plot_coordinates(bilevel_scaled_model, states_traj,
                 'walk_bilevel_solution_coordinates.pdf',
                 convert_radians_to_degrees=True,
                 coordinate_ranges=coordinate_ranges)

# Plot the marker errors.
errors = compute_marker_errors(bilevel_scaled_model, states_traj, marker_source)
plot_marker_errors(errors, 'walk_bilevel_solution_marker_errors.pdf')
