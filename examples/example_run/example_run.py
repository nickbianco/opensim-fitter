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
from osimfit.costs import (AnthropometricRegularizationCost, OffsetRegularizationCost,
                           BodyScaleIsotropyCost)
from osimfit.bounds import Bounds
from osimfit.utilities import (compute_marker_errors, plot_marker_errors,
                               plot_coordinates)

# EXAMPLE RUN
# -----------
# This example demonstrates how to perform scaling and inverse kinematics for a running
# motion using OpenSim Fitter.

# Load data
# ---------
# Load the marker data and model.
markers_fpath = 'run_5_5_1_1.trc'
markers_table = osim.TimeSeriesTableVec3(markers_fpath)
marker_labels = markers_table.getColumnLabels()
modelProcessor = osim.ModelProcessor('FB_Model_WuTsai.osim')
modelProcessor.append(osim.ModOpRemoveMuscles())
model = modelProcessor.process()
model.initSystem()

# Widen the bounds on the pelvis rotational coordinates.
coordset = model.updCoordinateSet()
for coord_name in ['pelvis_rotation', 'pelvis_list', 'pelvis_tilt']:
    coord = coordset.get(coord_name)
    coord.setRangeMin(-2.5)
    coord.setRangeMax(2.5)

# Define the tracking markers.
tracking_markers = ['RBACK', 'T10']
for marker in ['UARM', 'FARM',
               'THI1', 'THI2', 'THI3', 'THI4', 'THI5',
               'SHA1', 'SHA2', 'SHA3', 'SHA4', 'SHA5']:
    for side in ['L', 'R']:
        tracking_markers.append(f'{side}{marker}')

# Set markers as fixed or unfixed.
markerset = model.updMarkerSet()
for imarker in range(markerset.getSize()):
    marker = markerset.get(imarker)
    marker_name = marker.getName()
    marker.set_fixed(marker_name not in tracking_markers)

# Add anthropometric stations to the model.
markers = osim.MarkerSet('anthropometric_stations.xml')
for i in range(markers.getSize()):
    marker = markers.get(i)
    frame_path = marker.getSocket('parent_frame').getConnecteePath()
    parent_frame = osim.PhysicalFrame.safeDownCast(model.getComponent(frame_path))
    station = osim.Station(parent_frame, marker.get_location())
    station.setName(marker.getName())
    model.addComponent(station)

model.finalizeFromProperties()
model.initSystem()

# Save a clone of the unscaled model.
model.setName('subject_example_run')
unscaled_model = osim.Model(model)
unscaled_model.printToXML('unscaled_generic.osim')

# Define a mapping between marker names and marker paths.
# (marker_name --> /marker/path)
marker_map = {label: f'/markerset/{label}' for label in marker_labels}

# Marker-based scaling
# --------------------
# Define scaling rules as a list of (segment, marker_1, marker_2, axis) tuples.
# Each rule specifies a segment to scale, two markers whose inter-distance defines
# the body scale, and the axis along which to apply it.
scale_rules = [
    ('torso', 'RPSIS', 'RSHO', Axis.YAxis),
    ('torso', 'LPSIS', 'LSHO', Axis.YAxis),
    ('torso', 'RSHO', 'LSHO', Axis.ZAxis),
    ('pelvis', 'RASIS', 'LASIS', Axis.ZAxis),
    ('pelvis', 'RPSIS', 'LPSIS', Axis.ZAxis),
    ('pelvis', 'RPSIS', 'RASIS', Axis.XAxis),
    ('pelvis', 'LPSIS', 'LASIS', Axis.XAxis),
]
for s in [('L', '_l'), ('R', '_r')]:
    # upper body
    scale_rules.append((f'humerus{s[1]}', f'{s[0]}SHO', f'{s[0]}LELB', Axis.YAxis))
    scale_rules.append((f'humerus{s[1]}', f'{s[0]}LELB', f'{s[0]}MELB', Axis.XAxis))
    scale_rules.append((f'humerus{s[1]}', f'{s[0]}LELB', f'{s[0]}MELB', Axis.ZAxis))
    scale_rules.append((f'radius{s[1]}', f'{s[0]}LELB', f'{s[0]}LWRI', Axis.YAxis))
    scale_rules.append((f'radius{s[1]}', f'{s[0]}LELB', f'{s[0]}MELB', Axis.XAxis))
    scale_rules.append((f'radius{s[1]}', f'{s[0]}LELB', f'{s[0]}MELB', Axis.ZAxis))
    scale_rules.append((f'ulna{s[1]}', f'{s[0]}LELB', f'{s[0]}LWRI', Axis.YAxis))
    scale_rules.append((f'ulna{s[1]}', f'{s[0]}LELB', f'{s[0]}MELB', Axis.XAxis))
    scale_rules.append((f'ulna{s[1]}', f'{s[0]}LELB', f'{s[0]}MELB', Axis.ZAxis))
    scale_rules.append((f'hand{s[1]}', f'{s[0]}LELB', f'{s[0]}LWRI', Axis.YAxis))
    scale_rules.append((f'hand{s[1]}', f'{s[0]}MWRI', f'{s[0]}LWRI', Axis.XAxis))
    scale_rules.append((f'hand{s[1]}', f'{s[0]}MWRI', f'{s[0]}LWRI', Axis.ZAxis))
    # lower body
    scale_rules.append((f'femur{s[1]}', f'{s[0]}ASIS', f'{s[0]}LKNE', Axis.YAxis))
    scale_rules.append((f'femur{s[1]}', f'{s[0]}LKNE', f'{s[0]}MKNE', Axis.XAxis))
    scale_rules.append((f'femur{s[1]}', f'{s[0]}LKNE', f'{s[0]}MKNE', Axis.ZAxis))
    scale_rules.append((f'tibia{s[1]}', f'{s[0]}LKNE', f'{s[0]}LANK', Axis.YAxis))
    scale_rules.append((f'tibia{s[1]}', f'{s[0]}LANK', f'{s[0]}MANK', Axis.XAxis))
    scale_rules.append((f'tibia{s[1]}', f'{s[0]}LANK', f'{s[0]}MANK', Axis.ZAxis))
    scale_rules.append((f'calcn{s[1]}', f'{s[0]}HEEL', f'{s[0]}MT1', Axis.XAxis))
    scale_rules.append((f'calcn{s[1]}', f'{s[0]}HEEL', f'{s[0]}MT5', Axis.XAxis))
    scale_rules.append((f'calcn{s[1]}', f'{s[0]}MT1', f'{s[0]}MT5', Axis.ZAxis))
    scale_rules.append((f'calcn{s[1]}', f'{s[0]}HEEL', f'{s[0]}LANK', Axis.YAxis))
    scale_rules.append((f'toes{s[1]}', f'{s[0]}HEEL', f'{s[0]}MT1', Axis.XAxis))
    scale_rules.append((f'toes{s[1]}', f'{s[0]}HEEL', f'{s[0]}MT5', Axis.XAxis))
    scale_rules.append((f'toes{s[1]}', f'{s[0]}MT1', f'{s[0]}MT5', Axis.ZAxis))
    scale_rules.append((f'toes{s[1]}', f'{s[0]}HEEL', f'{s[0]}LANK', Axis.YAxis))
    scale_rules.append((f'talus{s[1]}', f'{s[0]}HEEL', f'{s[0]}MT1', Axis.XAxis))
    scale_rules.append((f'talus{s[1]}', f'{s[0]}HEEL', f'{s[0]}MT5', Axis.XAxis))
    scale_rules.append((f'talus{s[1]}', f'{s[0]}MT1', f'{s[0]}MT5', Axis.ZAxis))
    scale_rules.append((f'talus{s[1]}', f'{s[0]}HEEL', f'{s[0]}LANK', Axis.YAxis))


# Create a MarkerSource and PositionBasedScaler.
markers_to_remove = ['IMUBACK', 'IMULFOOT', 'IMULSHA', 'IMULTHI', 'IMUPELVIS',
                     'IMURFOOT', 'IMURSHA', 'IMURTHI']
time_range = (28.885, 30.1)
marker_source = MarkerSource(markers_fpath,
                             labels_to_remove=markers_to_remove,
                             trim_to_range=time_range)
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
position_scaler.add_symmetry_pair('tibia_l', 'tibia_r')
position_scaler.add_symmetry_pair('calcn_l', 'calcn_r')
position_scaler.add_symmetry_pair('talus_l', 'talus_r')
position_scaler.add_symmetry_pair('toes_l', 'toes_r')

# Scale the model.
scaled_model = position_scaler.scale()
scaled_model.printToXML('subject_marker_scaled_run.osim')

# Anthropometric measurements
# ---------------------------
# Define the list of anthropometric measurements from the ANSUR II dataset that will
# regularize the body scales during the bilevel optimization below. Each
# `AnthropometricMeasurement` object contains the name of the measurement, paths to two
# `Station`s in the model from which the measurement is computed, and the axis along
# which the measurement is taken. If no axis is specified, the measurement is the
# Euclidean distance between the two stations.
ansur_measurements_map = {
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
ansur_measurements: list[AnthropometricMeasurement] = []
for name, (station1_path, station2_path, axis) in ansur_measurements_map.items():
    ansur_measurements.append(
        AnthropometricMeasurement(name, station1_path, station2_path, axis))

# Place markers
# -------------
# Create a new marker source with updated column labels representing the full path to
# each marker.
marker_source = MarkerSource(markers_fpath, label_map=marker_map,
                             labels_to_remove=markers_to_remove,
                             trim_to_range=time_range)
marker_source.write('run_markers.trc')
placer = MarkerPlacer(scaled_model, marker_source)
solution = placer.solve()
# Update both 'scaled_model', which we'll use to generate a guess via inverse
# kinematics, and 'unscaled_model' which we'll use in the final bilevel optimization.
scaled_model = placer.update_model(scaled_model, solution)
unscaled_model = placer.update_model(unscaled_model, solution)

# Frame-by-frame inverse kinematics
# ---------------------------------
# Run the frame-by-frame IK solver.
solver = InverseKinematicsSolver(scaled_model,
                                 convergence_tolerance=1e-2,
                                 position_weight=1.0)
solver.add_marker_reference_data(marker_source)
ik_solution = solver.solve()
sto = osim.STOFileAdapter()
sto.write(ik_solution.states_table, 'run_ik_solution.sto')

# Convert the solution to a StatesTrajectory for computing marker errors.
states_table = osim.TimeSeriesTable('run_ik_solution.sto')
states_table.addTableMetaDataString('inDegrees', 'no')
states_traj = osim.StatesTrajectory.createFromStatesTable(scaled_model,
                                                          states_table, True)

# Spline-based inverse kinematics
# -------------------------------
# Construct a SplinedKinematicsSolver to solve for the model kinematics and body
# lengths that best match the marker data.
solver = SplinedKinematicsSolver(unscaled_model,
                                 convergence_tolerance=1e-3,
                                 knot_interval=0.05,
                                 position_weight=1.0)
solver.add_marker_reference_data(marker_source)

# Add additional cost terms to the solver.
#
# A regularization penalty on body-scale factors that maximizes the log-likelihood that
# a set of anthropometric measurements (that are function of body scales) fall within a
# distribution fit to the ANSUR II dataset.
solver.add_cost(AnthropometricRegularizationCost(
    ansur_measurements, sex='female', weight=1e-3))
# Penalize component (i.e., X, Y, or Z) body scales that deviate far from the mean
# across the three component scales. In other words, encourage the body to scale
# isotropically.
solver.add_cost(BodyScaleIsotropyCost(weight=1e-1))
# Penalize marker and frame offsets that deviate far from their nominal values.
solver.add_cost(OffsetRegularizationCost(weight=1e-3))

# Add body scales for each body in the model. Apply the same scales to groups of bodies,
# including those that should share left-right symmetry.
bounds = Bounds(0.5, 1.5)
solver.add_parameter(BodyScale('/bodyset/torso', bounds, np.ones(3)))
solver.add_parameter(BodyScale('/bodyset/pelvis', bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/humerus_r', '/bodyset/humerus_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/radius_r', '/bodyset/radius_l',
                                '/bodyset/ulna_r', '/bodyset/ulna_l',
                                '/bodyset/hand_r', '/bodyset/hand_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/femur_r', '/bodyset/femur_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/tibia_r', '/bodyset/tibia_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/calcn_r', '/bodyset/calcn_l',
                                '/bodyset/toes_r', '/bodyset/toes_l',
                                '/bodyset/talus_r', '/bodyset/talus_l'],
                               bounds, np.ones(3)))
# Add marker offset parameters for the tracking markers.
tracking_bounds = Bounds(-0.25, 0.25)
anatomical_bounds = Bounds(-0.02, 0.02)
for i in range(unscaled_model.getMarkerSet().getSize()):
    marker = unscaled_model.getMarkerSet().get(i)
    path = marker.getAbsolutePathString()
    if not marker.get_fixed():
        solver.add_parameter(MarkerOffset(path, tracking_bounds, np.zeros(3)))
    elif not path.endswith('JC'):
        solver.add_parameter(MarkerOffset(path, anatomical_bounds, np.zeros(3)))

# Gather the per-body XYZ body scales from the position-based scaling stage above,
# averaging over the bodies in each parameter group.
scaleset = position_scaler.scaleset
parameters_guess = [p.with_value(p.value) for p in solver.parameters]
for scale in parameters_guess:
    if isinstance(scale, BodyScale):
        factors = [scaleset.get(path.rsplit('/', 1)[-1]).getScaleFactors().to_numpy()
                   for path in scale.paths]
        scale.value = np.mean(factors, axis=0)

# Create an initial guess based on the the kinematics from the inverse kinematics
# solution and the combined body scales set above.
guess = SplinedKinematicsSolution(
    states_table=osim.TimeSeriesTable('run_ik_solution.sto'),
    parameters=parameters_guess)
bilevel_solution = solver.solve(guess)
sto.write(bilevel_solution.states_table, 'run_bilevel_solution.sto')
bilevel_scaled_model = solver.update_model(unscaled_model, bilevel_solution)
bilevel_scaled_model.printToXML('subject_bilevel_scaled_run.osim')

# Convert the solution to a StatesTrajectory for computing marker errors.
states_table = osim.TimeSeriesTable('run_bilevel_solution.sto')
states_table.addTableMetaDataString('inDegrees', 'no')
states_traj = osim.StatesTrajectory.createFromStatesTable(bilevel_scaled_model,
                                                          states_table, True)

# Plot the coordinates.
coordinate_ranges = {
    'pelvis_tilt':      (-40, 40),
    'pelvis_list':      (-40, 40),
    'pelvis_rotation':  (0, 200),
    'pelvis_tx':        (-7.5, 2.5),
    'pelvis_ty':        (0, 2.5),
    'pelvis_tz':        (-2.0, 1.0),
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
coordinates_pdf_fpath = 'run_bilevel_solution_coordinates.pdf'
plot_coordinates(bilevel_scaled_model, states_traj,
                 'run_bilevel_solution_coordinates.pdf',
                 convert_radians_to_degrees=True,
                 coordinate_ranges=coordinate_ranges)

# Plot the marker errors.
errors = compute_marker_errors(bilevel_scaled_model, states_traj, marker_source)
plot_marker_errors(errors, 'run_bilevel_solution_marker_errors.pdf')
