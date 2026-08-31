import os
import time
import numpy as np
import opensim as osim
import matplotlib.pyplot as plt

from osimfit.data_sources import MarkerSource, Trial
from osimfit.scaling import (Axis, PositionBasedScaler, MarkerMeasurement,
                             AnthropometricMeasurement)
from osimfit.solvers import (InverseKinematicsSolver, MarkerPlacer,
                             SplinedKinematicsSolver, Solution)
from osimfit.model import BodyScale, MarkerOffset
from osimfit.costs import (AnthropometricRegularizationCost, OffsetRegularizationCost,
                           BodyScaleIsotropyCost)
from osimfit.bounds import Bounds
from osimfit.utilities import (compute_marker_errors, plot_marker_errors,
                               plot_coordinates)

# EXAMPLE MULTIPLE TRIALS
# -----------------------
# This example demonstrates how to perform scaling and inverse kinematics across
# multiple trials simultaneously using OpenSim Fitter. The example data comes from
# subject #2 in the OpenCap validation dataset (https://simtk.org/projects/opencap).

# Results directory.
if not os.path.exists('results'):
    os.mkdir('results')

# Load data
# ---------
# Load the marker data and model.
markers_table = osim.TimeSeriesTableVec3('walking1.trc')
marker_labels = markers_table.getColumnLabels()
model = osim.Model('LaiArnoldModified2017_poly_withArms_weldHand_generic.osim')
model.initSystem()

# Define the tracking markers.
tracking_markers = []
for marker in ['thigh1', 'thigh2', 'thigh3', 'thigh4', 'thigh5',
               'shank_antsup', 'sh2', 'sh3', 'sh4']:
    for side in ['L_', 'r_']:
        tracking_markers.append(f'{side}{marker}')
tracking_markers.remove('L_thigh5')
tracking_markers.remove('L_sh4')

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
    ('torso', 'r.PSIS', 'R_Shoulder', Axis.YAxis),
    ('torso', 'L.PSIS', 'L_Shoulder', Axis.YAxis),
    ('torso', 'R_Shoulder', 'L_Shoulder', Axis.ZAxis),

    ('pelvis', 'r.ASIS', 'L.ASIS', Axis.ZAxis),
    ('pelvis', 'r.PSIS', 'L.PSIS', Axis.ZAxis),
    ('pelvis', 'R_HJC', 'L_HJC', Axis.ZAxis),
    ('pelvis', 'r.PSIS', 'r.ASIS', Axis.XAxis),
    ('pelvis', 'L.PSIS', 'L.ASIS', Axis.XAxis),

    ('humerus_r', 'R_Shoulder', 'R_elbow_lat', Axis.YAxis),
    ('humerus_l', 'L_Shoulder', 'L_elbow_lat', Axis.YAxis),

    ('radius_r', 'R_elbow_lat', 'R_wrist_radius', Axis.YAxis),
    ('radius_l', 'L_elbow_lat', 'L_wrist_radius', Axis.YAxis),

    ('ulna_r', 'R_elbow_med', 'R_wrist_ulna', Axis.YAxis),
    ('ulna_l', 'L_elbow_med', 'L_wrist_ulna', Axis.YAxis),

    ('hand_r', 'R_elbow_lat', 'R_wrist_radius', Axis.YAxis),
    ('hand_l', 'L_elbow_lat', 'L_wrist_radius', Axis.YAxis),

    ('femur_r', 'r.ASIS', 'r_knee', Axis.YAxis),
    ('femur_l', 'L.ASIS', 'L_knee', Axis.YAxis),

    ('patella_r', 'r.ASIS', 'r_knee', Axis.YAxis),
    ('patella_l', 'L.ASIS', 'L_knee', Axis.YAxis),

    ('tibia_r', 'r_knee', 'r_ankle', Axis.YAxis),
    ('tibia_l', 'L_knee', 'L_ankle', Axis.YAxis),

    ('calcn_r', 'r_calc', 'r_toe', Axis.XAxis),
    ('calcn_r', 'r_calc', 'r_5meta', Axis.XAxis),
    ('calcn_r', 'r_toe', 'r_5meta', Axis.ZAxis),
    ('calcn_r', 'r_calc', 'r_ankle', Axis.YAxis),
    ('toes_r', 'r_calc', 'r_toe', Axis.XAxis),
    ('toes_r', 'r_calc', 'r_5meta', Axis.XAxis),
    ('toes_r', 'r_toe', 'r_5meta', Axis.ZAxis),
    ('toes_r', 'r_calc', 'r_ankle', Axis.YAxis),

    ('calcn_l', 'L_calc', 'L_toe', Axis.XAxis),
    ('calcn_l', 'L_calc', 'L_5meta', Axis.XAxis),
    ('calcn_l', 'L_toe', 'L_5meta', Axis.ZAxis),
    ('calcn_l', 'L_calc', 'L_ankle', Axis.YAxis),
    ('toes_l', 'L_calc', 'L_toe', Axis.XAxis),
    ('toes_l', 'L_calc', 'L_5meta', Axis.XAxis),
    ('toes_l', 'L_toe', 'L_5meta', Axis.ZAxis),
    ('toes_l', 'L_calc', 'L_ankle', Axis.YAxis),
]

# Create a MarkerSource and PositionBasedScaler.
markers_to_remove = ['L_HJC_reg', 'L_forearm', 'L_humerus',
                     'R_HJC_reg', 'R_forearm', 'R_humerus']
marker_source = MarkerSource('walking1_markers', 'walking1.trc',
                             labels_to_remove=markers_to_remove)
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
scaled_model.printToXML(os.path.join('results', 'subject_marker_scaled.osim'))

# Assemble trials.
# ----------------
trial_ranges = {
    'walking1': (0, 1.570),
    'squats1': (0, 1.6),
    'DJ1': (1.25, 2.0),
    'STS1': (0, 1.825)
}

trials = []
for trial_name, trial_range in trial_ranges.items():
    marker_source = MarkerSource(f'{trial_name}_markers', f'{trial_name}.trc',
                                 label_map=marker_map,
                                 labels_to_remove=markers_to_remove,
                                 trim_to_range=trial_range)
    trials.append(Trial(trial_name, [marker_source]))

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
# Create a MarkerPlacer.
placer = MarkerPlacer(scaled_model)
# Add the trials.
for trial in trials:
    placer.add_trial(trial)
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
for trial in trials:
    solver.add_trial(trial)
ik_solution = solver.solve()
sto = osim.STOFileAdapter()
for trial in trials:
    sto.write(ik_solution.states_tables[trial.name],
              os.path.join('results', f'{trial.name}_ik_solution.sto'))

# Spline-based inverse kinematics
# -------------------------------
# Construct a SplinedKinematicsSolver to solve for the model kinematics and body
# lengths that best match the marker data.
solver = SplinedKinematicsSolver(unscaled_model,
                                 convergence_tolerance=1e-2,
                                 knot_interval=0.05,
                                 position_weight=5.0)
for trial in trials:
    solver.add_trial(trial)

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

# Gather the per-body XYZ body scales from the position-based scaling stage above,
# averaging over the bodies in each parameter group.
scaleset = position_scaler.scaleset
parameters_guess = [p.with_value(p.value) for p in solver.parameters]
for scale in parameters_guess:
    if isinstance(scale, BodyScale):
        factors = [scaleset.get(path.rsplit('/', 1)[-1]).getScaleFactors().to_numpy()
                   for path in scale.paths]
        scale.value = np.mean(factors, axis=0)

# Create an initial guess based on the kinematics from the inverse kinematics
# solution and the position-based body scales set above.
guess = Solution(states_tables=ik_solution.states_tables,
                 parameters=parameters_guess)
bilevel_solution = solver.solve(guess)
for trial in trials:
    sto.write(bilevel_solution.states_tables[trial.name],
              os.path.join('results', f'{trial.name}_bilevel_solution.sto'))
bilevel_scaled_model = solver.update_model(unscaled_model, bilevel_solution)
bilevel_scaled_model.printToXML(os.path.join('results', 'subject_bilevel_scaled.osim'))

# Plotting
# --------
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

bilevel_scaled_model = osim.Model(os.path.join('results',
                                               'subject_bilevel_scaled.osim'))
bilevel_scaled_model.initSystem();

# Convert the solution to a StatesTrajectory for computing marker errors.
for trial in trials:
    states_table = osim.TimeSeriesTable(
        os.path.join('results', f'{trial.name}_bilevel_solution.sto'))
    states_table.addTableMetaDataString('inDegrees', 'no')
    states_traj = osim.StatesTrajectory.createFromStatesTable(bilevel_scaled_model,
                                                              states_table, True)

    # Plot the coordinates.
    plot_coordinates(bilevel_scaled_model, states_traj,
                    os.path.join('results',
                                 f'{trial.name}_bilevel_solution_coordinates.pdf'),
                    convert_radians_to_degrees=True,
                    coordinate_ranges=coordinate_ranges)

    # Plot the marker errors.
    errors = compute_marker_errors(bilevel_scaled_model, states_traj,
                                   trial.get_data_source(f'{trial.name}_markers'))
    plot_marker_errors(errors,
        os.path.join('results', f'{trial.name}_bilevel_solution_marker_errors.pdf'))
