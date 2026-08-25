"""
Unit tests for src/osimfit/scaling.py.
"""

import pytest
import numpy as np
import opensim as osim
from pathlib import Path

from osimfit.scaling import (
    AnthropometricMeasurement,
    AnthropometricScaler,
    Axis,
    FrameMeasurement,
    MarkerMeasurement,
    MeasurementBodyScale,
    PositionBasedScaler,
)
from osimfit.data_sources import DataSource

MODEL_FPATH = str(Path(__file__).parent / 'subject_scale_walk.osim')


###########
# HELPERS #
###########

class PositionsOnlySource(DataSource):
    """
    In-test DataSource that returns a pre-built positions table as-is.
    Used to construct a PositionBasedScaler without touching the filesystem.
    """
    def __init__(self, positions_table):
        super().__init__()
        self._positions_table = positions_table

    def _create_positions_table(self):
        return self._positions_table


def load_full_body_model():
    """
    Load subject_scale_walk.osim and return (model, initialized state).
    """
    model = osim.Model(MODEL_FPATH)
    state = model.initSystem()
    return model, state


def create_one_body_test_model():
    """
    Build a one-body model welded to ground with two markers and two
    stations placed 0.5 m apart along the body-frame X axis.
    """
    model = osim.Model()
    model.setName('test_rig')
    ground = model.getGround()

    body = osim.Body('rig_body', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body)

    joint = osim.WeldJoint(
        'rig_weld',
        ground, osim.Vec3(0), osim.Vec3(0),
        body, osim.Vec3(0), osim.Vec3(0),
    )
    model.addJoint(joint)

    model.addMarker(osim.Marker('m0', body, osim.Vec3(0)))
    model.addMarker(osim.Marker('m1', body, osim.Vec3(0.5, 0, 0)))

    s0 = osim.Station(body, osim.Vec3(0))
    s0.setName('s0')
    model.addComponent(s0)
    s1 = osim.Station(body, osim.Vec3(0.5, 0, 0))
    s1.setName('s1')
    model.addComponent(s1)

    model.finalizeConnections()
    return model


def empty_positions_table():
    """
    Minimal positions table just for constructing MeasurementBodyScales in
    the container-shape tests.
    """
    table = osim.TimeSeriesTableVec3()
    row = osim.RowVectorVec3(2, osim.Vec3(0))
    for t in (0.0, 0.1):
        table.appendRow(t, row)
    table.setColumnLabels(['a', 'b'])
    return table


#####################
# FRAME MEASUREMENT #
#####################

def test_frame_measurement_same_frame_yields_zero():
    model, state = load_full_body_model()
    measurement = FrameMeasurement('/bodyset/pelvis', '/bodyset/pelvis')
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.0, abs=1e-12)


def test_frame_measurement_matches_direct_calc():
    model, state = load_full_body_model()
    path1 = '/bodyset/femur_r'
    path2 = '/bodyset/tibia_r'

    frame1 = osim.Frame.safeDownCast(model.getComponent(path1))
    frame2 = osim.Frame.safeDownCast(model.getComponent(path2))
    expected = float(np.linalg.norm(
        frame1.getPositionInGround(state).to_numpy()
        - frame2.getPositionInGround(state).to_numpy()))

    measurement = FrameMeasurement(path1, path2)
    actual = measurement.compute_measurement(model, state)
    assert actual == pytest.approx(expected, rel=1e-12)


######################
# MARKER MEASUREMENT #
######################

def test_marker_measurement_same_marker_yields_zero():
    model = create_one_body_test_model()
    state = model.initSystem()
    measurement = MarkerMeasurement('/markerset/m0', '/markerset/m0')
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.0, abs=1e-12)


def test_marker_measurement_returns_offset_distance():
    model = create_one_body_test_model()
    state = model.initSystem()
    measurement = MarkerMeasurement('/markerset/m0', '/markerset/m1')
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.5, abs=1e-12)


##############################
# ANTHROPOMETRIC MEASUREMENT #
##############################

def test_anthropometric_no_axis_returns_meter_magnitude():
    model = create_one_body_test_model()
    state = model.initSystem()
    measurement = AnthropometricMeasurement('stature', '/s0', '/s1')
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.5, abs=1e-9)


def test_anthropometric_x_axis_returns_meters_along_x():
    model = create_one_body_test_model()
    state = model.initSystem()
    measurement = AnthropometricMeasurement('stature', '/s0', '/s1', axis=Axis.XAxis)
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.5, abs=1e-9)


def test_anthropometric_y_axis_returns_zero_for_pure_x_offset():
    model = create_one_body_test_model()
    state = model.initSystem()
    measurement = AnthropometricMeasurement('stature', '/s0', '/s1', axis=Axis.YAxis)
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.0, abs=1e-9)


def test_anthropometric_z_axis_returns_zero_for_pure_x_offset():
    model = create_one_body_test_model()
    state = model.initSystem()
    measurement = AnthropometricMeasurement('stature', '/s0', '/s1', axis=Axis.ZAxis)
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.0, abs=1e-9)


#########################
# POSITION-BASED SCALER #
#########################

def create_position_based_scaler():
    """
    Build a PositionBasedScaler over the rig model, backed by a positions table
    with columns 'a' and 'b' so add_measurement_body_scale calls can resolve
    their data labels.
    """
    return PositionBasedScaler(create_one_body_test_model(),
                               PositionsOnlySource(empty_positions_table()))


def create_measurement_body_scale(body_name='rig_body'):
    """
    Build a throwaway MeasurementBodyScale — its data isn't exercised by the
    add_body_scale / add_symmetry_pair container tests.
    """
    table = empty_positions_table()
    measurement = FrameMeasurement('/bodyset/rig_body', '/bodyset/rig_body')
    return MeasurementBodyScale(
        body_name, Axis.XAxis, measurement,
        table.getDependentColumn('a'),
        table.getDependentColumn('b'))


def test_position_based_scaler_add_body_scale_appends_to_list():
    scaler = create_position_based_scaler()
    sf = create_measurement_body_scale(body_name='rig_body')
    scaler.add_body_scale(sf)
    assert scaler.body_scales == [sf]


def test_position_based_scaler_add_body_scale_preserves_order():
    scaler = create_position_based_scaler()
    sf1 = create_measurement_body_scale(body_name='rig_body')
    sf2 = create_measurement_body_scale(body_name='rig_body')
    scaler.add_body_scale(sf1)
    scaler.add_body_scale(sf2)
    assert scaler.body_scales == [sf1, sf2]


def test_position_based_scaler_seeds_unity_scaleset_per_body():
    scaler = create_position_based_scaler()
    scaler.populate_scaleset()
    assert scaler.scaleset.getSize() == 1
    entry = scaler.scaleset.get('rig_body')
    factors = entry.getScaleFactors()
    assert factors[0] == pytest.approx(1.0, abs=1e-12)
    assert factors[1] == pytest.approx(1.0, abs=1e-12)
    assert factors[2] == pytest.approx(1.0, abs=1e-12)


def test_position_based_scaler_add_measurement_body_scale_constructs_and_appends():
    scaler = create_position_based_scaler()
    measurement = FrameMeasurement('/bodyset/rig_body', '/bodyset/rig_body')
    scaler.add_measurement_body_scale(
        'rig_body', Axis.YAxis, measurement, 'a', 'b')
    assert len(scaler.body_scales) == 1
    sf = scaler.body_scales[0]
    assert isinstance(sf, MeasurementBodyScale)
    assert sf.body_name == 'rig_body'
    assert sf.axis is Axis.YAxis
    # The data labels were resolved against the scaler's positions table; both
    # columns have the same number of rows as that table.
    assert sf.position1_data.size() == 2
    assert sf.position2_data.size() == 2


def test_position_based_scaler_add_symmetry_pair_appends_tuple():
    scaler = create_position_based_scaler()
    scaler.add_symmetry_pair('femur_l', 'femur_r')
    scaler.add_symmetry_pair('tibia_l', 'tibia_r')
    assert scaler.symmetry_pairs == [
        ('femur_l', 'femur_r'),
        ('tibia_l', 'tibia_r'),
    ]


#############################
# ANTHROPOMETRIC BODY SCALE #
#############################

def test_anthropometric_body_scale_applies_expected_scale():
    model = create_one_body_test_model()
    scaler = AnthropometricScaler(model)
    scaler.context.model_values['mylabel'] = 100.0
    scaler.context.conditioned_values['mylabel'] = 200.0
    measurement = AnthropometricMeasurement('mylabel', '/s0', '/s1')
    scaler.add_measurement(measurement)
    scaler.add_anthropometric_body_scale('rig_body', Axis.YAxis, 'mylabel')

    scaler.populate_scaleset()

    assert scaler.scaleset.getSize() == 1
    entry = scaler.scaleset.get('rig_body')
    assert entry.getSegmentName() == 'rig_body'
    factors = entry.getScaleFactors()
    assert factors[0] == pytest.approx(1.0, abs=1e-12)
    assert factors[1] == pytest.approx(2.0, abs=1e-12)
    assert factors[2] == pytest.approx(1.0, abs=1e-12)


def test_anthropometric_body_scale_requires_registered_measurement():
    model = create_one_body_test_model()
    scaler = AnthropometricScaler(model)
    with pytest.raises(ValueError, match="No anthropometric measurement"):
        scaler.add_anthropometric_body_scale(
            'rig_body', Axis.YAxis, 'unregistered_label')
