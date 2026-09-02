import os
import json
import numpy as np
import opensim as osim

from osimfit.data_sources import MarkerSource, Trial
from osimfit.scaling import (Axis, PositionBasedScaler, MarkerMeasurement,
                             AnthropometricMeasurement)
from osimfit.solvers import (InverseKinematicsSolver, MarkerPlacer,
                             SplinedKinematicsSolver, Solution)
from osimfit.model import BodyScale, MarkerOffset, EllipsoidRadii, BeamLength
from osimfit.costs import (AnthropometricRegularizationCost, BodyScaleRegularizationCost, OffsetRegularizationCost,
                           BodyScaleIsotropyCost, CoordinateStiffnessCost,
                           MobilizerDimensionRegularizationCost)
from osimfit.bounds import Bounds
from osimfit.utilities import (compute_marker_errors, plot_marker_errors,
                               plot_coordinates)

# EXAMPLE ATHLETE
# ---------------
# This example demonstrates multi-trial bilevel fitting of a detailed athlete model to
# marker data from eight movement tasks performed by a single subject (WT02) in the Wu
# Tsai Human Performance Alliance dataset.

TARGET_RATE = 50.0      # Hz; marker data is downsampled to this rate.
KNOT_INTERVAL = 0.05    # s; B-spline knot spacing for the kinematics.

# Results directory.
if not os.path.exists('results'):
    os.mkdir('results')


def downsample_markers(source_path: str, target_path: str, time_range: tuple,
                       target_rate: float) -> None:
    """
    Write the rows of a ``.trc`` file that fall inside `time_range`, decimated to
    approximately `target_rate`, to a new ``.trc`` file. The source files are 200 Hz
    recordings of a whole session; trimming and decimating here keeps the fitting
    problem to a size the solver can handle.
    """
    table = osim.TimeSeriesTableVec3(source_path)
    times = np.asarray(table.getIndependentColumn())
    rate = float(table.getTableMetaDataString('DataRate'))
    step = max(1, int(round(rate / target_rate)))

    downsampled = osim.TimeSeriesTableVec3()
    downsampled.setColumnLabels(table.getColumnLabels())
    for i in range(len(times)):
        if i % step == 0 and time_range[0] <= times[i] <= time_range[1]:
            downsampled.appendRow(times[i], table.getRowAtIndex(i))

    downsampled.addTableMetaDataString('DataRate', str(rate / step))
    downsampled.addTableMetaDataString(
        'Units', table.getTableMetaDataString('Units'))
    osim.TRCFileAdapter().write(downsampled, target_path)


# Load data
# ---------
model = osim.Model('AthleteModelNoMusclesNoContact.osim')
model.initSystem()

# The eight tasks and the time range of each, in seconds.
with open(os.path.join('WT02', 'bilevel_ranges.json')) as f:
    task_ranges = json.load(f)

# The marker labels recorded in the data.
data_labels = set(osim.TimeSeriesTableVec3(
    os.path.join('WT02', f'{next(iter(task_ranges))}.trc')).getColumnLabels())

# Apply the markerset to the model.
markers = osim.MarkerSet('WuTsai_markerset.xml')
for i in range(markers.getSize()):
    marker = markers.get(i)
    name = marker.getName()
    frame_path = marker.getSocket('parent_frame').getConnecteePath()
    location = marker.get_location()
    if 'torso' in frame_path:
        if 'LSHO' in name:
            frame_path = frame_path.replace('torso', 'scapula_l')
            location = osim.Vec3(0)
        elif 'RSHO' in name:
            frame_path = frame_path.replace('torso', 'scapula_r')
            location = osim.Vec3(0)
        elif 'C7' in name:
            frame_path = frame_path.replace('torso', 'head')
            location[1] -= 0.5
        else:
            frame_path = frame_path.replace('torso', 'thorax')
            location[1] -= 0.375

    parent_frame = osim.PhysicalFrame.safeDownCast(model.getComponent(frame_path))
    marker = osim.Marker(name, parent_frame, location)
    model.addMarker(marker)

model.finalizeConnections()
model_markers = {model.getMarkerSet().get(i).getName()
                 for i in range(model.getMarkerSet().getSize())}

# Remove any non-model markers in the data (e.g., IMU locations).
markers_to_remove = sorted(data_labels - model_markers)

# Define the tracking markers.
tracking_markers = ['RBACK', 'T10']
for marker in ['UARM', 'FARM',
               'THI1', 'THI2', 'THI3', 'THI4', 'THI5',
               'SHA1', 'SHA2', 'SHA3', 'SHA4', 'SHA5']:
    for side in ['L', 'R']:
        tracking_markers.append(f'{side}{marker}')

markerset = model.updMarkerSet()
for imarker in range(markerset.getSize()):
    marker = markerset.get(imarker)
    marker.set_fixed(marker.getName() not in tracking_markers)

model.finalizeFromProperties()
model.initSystem()

# Save a clone of the unscaled model.
unscaled_model = osim.Model(model)
unscaled_model.printToXML(os.path.join('results', 'unscaled_athlete.osim'))

# Trim and downsample each task's marker data.
downsampled_paths = {}
for task_name, time_range in task_ranges.items():
    target_path = os.path.join('results', f'{task_name}_{int(TARGET_RATE)}hz.trc')
    downsample_markers(os.path.join('WT02', f'{task_name}.trc'), target_path,
                       time_range, TARGET_RATE)
    downsampled_paths[task_name] = target_path

# Define a mapping between marker names and marker paths.
# (marker_name --> /marker/path)
marker_map = {label: f'/markerset/{label}' for label in sorted(model_markers)}

# Marker-based scaling
# --------------------
# Define scaling rules as a list of (segment, marker_1, marker_2, axis) tuples.
# Each rule specifies a segment to scale, two markers whose inter-distance defines
# the body scale, and the axis along which to apply it.
#
# The trunk measurements (PSIS to shoulder vertically, shoulder to shoulder laterally)
# drive every segment between the pelvis and the arms. `head`, `clavicle` and `scapula`
# carry no marker pair of their own, so they inherit the trunk measurements rather than
# being left unscaled while their neighbors scale.
scale_rules = [
    ('pelvis', 'RASIS', 'LASIS', Axis.ZAxis),
    ('pelvis', 'RPSIS', 'LPSIS', Axis.ZAxis),
    ('pelvis', 'RPSIS', 'RASIS', Axis.XAxis),
    ('pelvis', 'LPSIS', 'LASIS', Axis.XAxis),
]
for segment in ['lumbar_spine', 'thorax', 'head',
                'clavicle_r', 'clavicle_l', 'scapula_r', 'scapula_l']:
    scale_rules.append((segment, 'RPSIS', 'RSHO', Axis.YAxis))
    scale_rules.append((segment, 'LPSIS', 'LSHO', Axis.YAxis))
    scale_rules.append((segment, 'RSHO', 'LSHO', Axis.ZAxis))
for prefix, suffix in [('L', '_l'), ('R', '_r')]:
    # Upper body.
    scale_rules.append((f'humerus{suffix}', f'{prefix}SHO', f'{prefix}LELB',
                        Axis.YAxis))
    scale_rules.append((f'humerus{suffix}', f'{prefix}LELB', f'{prefix}MELB',
                        Axis.XAxis))
    scale_rules.append((f'humerus{suffix}', f'{prefix}LELB', f'{prefix}MELB',
                        Axis.ZAxis))
    for segment in [f'radius{suffix}', f'ulna{suffix}']:
        scale_rules.append((segment, f'{prefix}LELB', f'{prefix}LWRI', Axis.YAxis))
        scale_rules.append((segment, f'{prefix}LELB', f'{prefix}MELB', Axis.XAxis))
        scale_rules.append((segment, f'{prefix}LELB', f'{prefix}MELB', Axis.ZAxis))
    scale_rules.append((f'hand{suffix}', f'{prefix}LELB', f'{prefix}LWRI', Axis.YAxis))
    scale_rules.append((f'hand{suffix}', f'{prefix}MWRI', f'{prefix}LWRI', Axis.XAxis))
    scale_rules.append((f'hand{suffix}', f'{prefix}MWRI', f'{prefix}LWRI', Axis.ZAxis))
    # Lower body.
    scale_rules.append((f'femur{suffix}', f'{prefix}ASIS', f'{prefix}LKNE', Axis.YAxis))
    scale_rules.append((f'femur{suffix}', f'{prefix}LKNE', f'{prefix}MKNE', Axis.XAxis))
    scale_rules.append((f'femur{suffix}', f'{prefix}LKNE', f'{prefix}MKNE', Axis.ZAxis))
    scale_rules.append((f'tibia{suffix}', f'{prefix}LKNE', f'{prefix}LANK', Axis.YAxis))
    scale_rules.append((f'tibia{suffix}', f'{prefix}LANK', f'{prefix}MANK', Axis.XAxis))
    scale_rules.append((f'tibia{suffix}', f'{prefix}LANK', f'{prefix}MANK', Axis.ZAxis))
    for segment in [f'talus{suffix}', f'calcn{suffix}', f'toes{suffix}']:
        scale_rules.append((segment, f'{prefix}HEEL', f'{prefix}MT1', Axis.XAxis))
        scale_rules.append((segment, f'{prefix}HEEL', f'{prefix}MT5', Axis.XAxis))
        scale_rules.append((segment, f'{prefix}MT1', f'{prefix}MT5', Axis.ZAxis))
        scale_rules.append((segment, f'{prefix}HEEL', f'{prefix}LANK', Axis.YAxis))

# Create a MarkerSource and PositionBasedScaler. The walking task is used to compute
# the marker-based scale factors.
scaling_source = MarkerSource('scaling_markers', downsampled_paths['normal_walk_1_1'],
                              labels_to_remove=markers_to_remove)
position_scaler = PositionBasedScaler(model, scaling_source)

# Add scaling rules to the PositionBasedScaler.
for segment_name, marker_1, marker_2, axis in scale_rules:
    measurement = MarkerMeasurement(marker_map[marker_1], marker_map[marker_2])
    position_scaler.add_measurement_body_scale(
        segment_name, axis, measurement, marker_1, marker_2)

# Add symmetry pairs. Internally, the PositionBasedScaler will average the body scales
# computed for each pair of symmetric segments to ensure left-right symmetry.
for segment in ['clavicle', 'scapula', 'humerus', 'radius', 'ulna', 'hand',
                'femur', 'tibia', 'talus', 'calcn', 'toes']:
    position_scaler.add_symmetry_pair(f'{segment}_l', f'{segment}_r')

# Scale the model.
scaled_model = position_scaler.scale()
scaled_model.printToXML(os.path.join('results', 'subject_marker_scaled_athlete.osim'))

# Assemble trials
# ---------------
# Bundle each task's marker data into its own Trial. Every trial contributes its own
# block of spline control points to the bilevel problem below, while the body scales,
# marker offsets, and mobilizer dimensions are shared across all of them.
trials = []
for task_name, path in downsampled_paths.items():
    marker_source = MarkerSource(f'{task_name}_markers', path,
                                 label_map=marker_map,
                                 labels_to_remove=markers_to_remove)
    trials.append(Trial(task_name, [marker_source]))

# Anthropometric measurements
# ---------------------------
# Define the list of anthropometric measurements from the ANSUR II dataset that will
# regularize the body scales during the bilevel optimization below. Each
# `AnthropometricMeasurement` object contains the name of the measurement, paths to two
# `Station`s in the model from which the measurement is computed, and the axis along
# which the measurement is taken. If no axis is specified, the measurement is the
# Euclidean distance between the two stations.
ansur_measurements_map = {
    'biacromialbreadth':      ('/bodyset/scapula_r/acromion_r',
                               '/bodyset/scapula_l/acromion_l', None),
    'bicristalbreadth':       ('/bodyset/pelvis/iliocrestale_r',
                               '/bodyset/pelvis/iliocrestale_l', None),
    'bimalleolarbreadth':     ('/bodyset/tibia_r/lateral_malleolus_r',
                               '/bodyset/tibia_r/medial_malleolus_r', None),
    'footbreadthhorizontal':  ('/bodyset/calcn_r/mtp1_r',
                               '/bodyset/calcn_r/mtp5_r', Axis.ZAxis),
    'footlength':             ('/bodyset/toes_r/acropodion_r',
                               '/bodyset/calcn_r/pternion_r', Axis.XAxis),
    'iliocristaleheight':     ('/bodyset/pelvis/iliocrestale_r',
                               '/bodyset/calcn_r/mtp5_r', Axis.YAxis),
    'lateralmalleolusheight': ('/bodyset/tibia_r/lateral_malleolus_r',
                               '/bodyset/calcn_r/mtp5_r', Axis.YAxis),
    'radialestylionlength':   ('/bodyset/radius_r/radiale_r',
                               '/bodyset/radius_r/stylion_r', None),
    'shoulderelbowlength':    ('/bodyset/scapula_r/acromion_r',
                               '/bodyset/ulna_r/olecranon_r', None),
    'stature':                ('/bodyset/head/vertex',
                               '/bodyset/calcn_r/mtp5_r', Axis.YAxis),
    'suprasternaleheight':    ('/bodyset/thorax/suprasternale',
                               '/bodyset/calcn_r/mtp5_r', Axis.YAxis),
    'trochanterionheight':    ('/bodyset/femur_r/trochanterion_r',
                               '/bodyset/calcn_r/mtp5_r', Axis.YAxis),
}
ansur_measurements: list[AnthropometricMeasurement] = []
for name, (station1_path, station2_path, axis) in ansur_measurements_map.items():
    ansur_measurements.append(
        AnthropometricMeasurement(name, station1_path, station2_path, axis))

# Place markers
# -------------
placer = MarkerPlacer(scaled_model)
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
# Construct a SplinedKinematicsSolver to solve for the model kinematics, body scales,
# marker offsets, and mobilizer dimensions that best match the marker data across all
# eight trials simultaneously.
solver = SplinedKinematicsSolver(unscaled_model,
                                 convergence_tolerance=1e-3,
                                 knot_interval=KNOT_INTERVAL,
                                 position_weight=1.0)
for trial in trials:
    solver.add_trial(trial)

# Add additional cost terms to the solver.
#
# A regularization penalty on body-scale factors that maximizes the log-likelihood that
# a set of anthropometric measurements (that are function of body scales) fall within a
# distribution fit to the ANSUR II dataset. The subject's sex is not recorded with the
# data, so the combined male-and-female dataset is used.
solver.add_cost(AnthropometricRegularizationCost(
    ansur_measurements, weight=1e-3))
# Penalize component (i.e., X, Y, or Z) body scales that deviate far from the mean
# across the three component scales. In other words, encourage the body to scale
# isotropically.
solver.add_cost(BodyScaleIsotropyCost(weight=1e-1))

# Penalize marker and frame offsets that deviate far from their nominal values.
solver.add_cost(OffsetRegularizationCost(weight=1e-3))

# Penalize the mobilizer dimension factors away from 1.0. Without this the spine beam
# lengths and scapulothoracic radii tend to absorb marker error and run to their
# bounds, since a joint's internal geometry can partly mimic a change in pose.
solver.add_cost(MobilizerDimensionRegularizationCost(weight=1e-3))

# # Apply a pseudo-stiffness to the scapula and spine coordinates, which surface markers
# # determine only weakly. Each is pulled toward its default value in the model, so the
# # stiffness resists drift away from the neutral posture without narrowing the ranges.
# coordinate_stiffnesses = {}
# for coordinate, stiffness in [('cervical_bending', 1.0),
#                               ('cervical_extension', 1.0),
#                               ('cervical_rotation', 1.0),
#                               ('lumbar_bending', 0.1),
#                               ('lumbar_extension', 0.1),
#                               ('lumbar_rotation', 0.1),
#                               ('thorax_bending', 0.1),
#                               ('thorax_extension', 0.1),
#                               ('thorax_rotation', 0.1)]:
#     joint = coordinate.split('_')[0]
#     coordinate_stiffnesses[f'/jointset/{joint}/{coordinate}'] = stiffness
# for side in ['r', 'l']:
#     for motion in ['abduction', 'elevation', 'rotation']:
#         coordinate_stiffnesses[
#             f'/jointset/scapulothoracic_{side}/scapula_{motion}_{side}'] = 0.1
# solver.add_cost(CoordinateStiffnessCost(coordinate_stiffnesses, weight=1e-3))

# Add body scales for each body in the model. Apply the same scales to groups of bodies,
# including those that should share left-right symmetry.
bounds = Bounds(0.5, 2.0)
solver.add_parameter(BodyScale('/bodyset/pelvis', bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/lumbar_spine', '/bodyset/thorax',
                                '/bodyset/head'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/clavicle_r', '/bodyset/clavicle_l',
                                '/bodyset/scapula_r', '/bodyset/scapula_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/humerus_r', '/bodyset/humerus_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/ulna_r', '/bodyset/ulna_l',
                                '/bodyset/radius_r', '/bodyset/radius_l',
                                '/bodyset/hand_r', '/bodyset/hand_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/femur_r', '/bodyset/femur_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/tibia_r', '/bodyset/tibia_l'],
                               bounds, np.ones(3)))
solver.add_parameter(BodyScale(['/bodyset/talus_r', '/bodyset/talus_l',
                                '/bodyset/calcn_r', '/bodyset/calcn_l',
                                '/bodyset/toes_r', '/bodyset/toes_l'],
                               bounds, np.ones(3)))

# Add the mobilizer geometry parameters.
bounds = Bounds(0.75, 1.25)
solver.add_parameter(BeamLength(['/jointset/lumbar', '/jointset/thorax',
                                '/jointset/cervical'], bounds))
solver.add_parameter(EllipsoidRadii(['/jointset/scapulothoracic_r',
                                     '/jointset/scapulothoracic_l'], bounds))

# Add marker offset parameters for the markers.
for i in range(unscaled_model.getMarkerSet().getSize()):
    marker = unscaled_model.getMarkerSet().get(i)
    path = marker.getAbsolutePathString()
    if path.endswith('JC'): continue

    bounds = Bounds(-0.02, 0.02) if marker.get_fixed() else Bounds(-0.25, 0.25)
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
# solution and the position-based body scales set above. The mobilizer dimensions start
# from their nominal factors of 1.0.
guess = Solution(states_tables=ik_solution.states_tables,
                 parameters=parameters_guess)
bilevel_solution = solver.solve(guess)
for trial in trials:
    sto.write(bilevel_solution.states_tables[trial.name],
              os.path.join('results', f'{trial.name}_bilevel_solution.sto'))
bilevel_scaled_model = solver.update_model(unscaled_model, bilevel_solution)
bilevel_scaled_model.printToXML(
    os.path.join('results', 'subject_bilevel_scaled_athlete.osim'))

# Report the optimized mobilizer dimensions.
print('\nOptimized mobilizer dimensions (factor on the nominal value):')
for parameter in bilevel_solution.parameters:
    if isinstance(parameter, (BeamLength, EllipsoidRadii)):
        print(f'  {type(parameter).__name__:15s} {parameter.paths} '
              f'{np.round(parameter.value, 4)}')

# Plotting
# --------
coordinate_ranges = {
    'pelvis_tilt':          (-40, 40),
    'pelvis_list':          (-40, 40),
    'pelvis_rotation':      (-180, 180),
    'hip_rotation_r':       (-40, 40),
    'hip_rotation_l':       (-40, 40),
    'lumbar_bending':       (-30, 30),
    'lumbar_extension':     (-70, 30),
    'lumbar_rotation':      (-15, 15),
    'thorax_bending':       (-30, 30),
    'thorax_rotation':      (-50, 50),
    'cervical_bending':     (-50, 50),
    'cervical_rotation':    (-85, 85),
    'scapula_abduction_r':  (-30, 30),
    'scapula_abduction_l':  (-30, 30),
    'scapula_elevation_r':  (-10, 10),
    'scapula_elevation_l':  (-10, 10),
    'scapula_rotation_r':   (-30, 60),
    'scapula_rotation_l':   (-30, 60),
    'shoulder_flexion_r':   (-115, 115),
    'shoulder_flexion_l':   (-115, 115),
    'shoulder_rotation_r':  (-30, 30),
    'shoulder_rotation_l':  (-30, 30),
}

bilevel_scaled_model = osim.Model(
    os.path.join('results', 'subject_bilevel_scaled_athlete.osim'))
bilevel_scaled_model.initSystem()

# Convert each solution to a StatesTrajectory for computing marker errors.
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
