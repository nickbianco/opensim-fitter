"""
Unit tests for src/osimfit/data_sources.py.
"""

import pytest
import opensim as osim
from osimfit.data_sources import DataSource, MarkerSource, Trial


###########
# HELPERS #
###########

def create_Vec3_table(times, labels):
    """
    Build a TimeSeriesTableVec3 with rows of zero-valued Vec3 entries.
    """
    table = osim.TimeSeriesTableVec3()
    for t in times:
        row = osim.RowVectorVec3(len(labels), osim.Vec3(0))
        table.appendRow(t, row)
    table.setColumnLabels(labels)
    return table


def create_Quaternion_table(times, labels):
    """
    Build a TimeSeriesTableQuaternion with identity-quaternion rows.
    """
    table = osim.TimeSeriesTableQuaternion()
    for t in times:
        row = osim.RowVectorQuaternion(len(labels), osim.Quaternion())
        table.appendRow(t, row)
    table.setColumnLabels(labels)
    return table


class PositionsSource(DataSource):
    """
    Test subclass that provides positions only.
    """
    def __init__(self):
        super().__init__('positions')

    def _provides_positions(self):
        return True

    def _provides_orientations(self):
        return False

    def _create_positions_table(self):
        return create_Vec3_table([0.0, 0.1], ['a', 'b'])

    def _create_orientations_table(self):
        raise NotImplementedError(
            f"PositionsSource does not provide orientation data.")


class OrientationsSource(DataSource):
    """
    Test subclass that provides orientations only.
    """
    def __init__(self):
        super().__init__('orientations')

    def _provides_positions(self):
        return False

    def _provides_orientations(self):
        return True

    def _create_positions_table(self):
        raise NotImplementedError(
            f"OrientationsSource does not provide position data.")

    def _create_orientations_table(self):
        return create_Quaternion_table([0.0, 0.1], ['a', 'b'])


class PositionsAndOrientationsSource(DataSource):
    """
    Test subclass that provides both positions and orientations.
    """
    def __init__(self):
        super().__init__('positions_and_orientations')

    def _provides_positions(self):
        return True

    def _provides_orientations(self):
        return True

    def _create_positions_table(self):
        return create_Vec3_table([0.0, 0.1], ['a', 'b'])

    def _create_orientations_table(self):
        return create_Quaternion_table([0.0, 0.1], ['a', 'b'])


class NoSource(DataSource):
    """
    Test subclass that provides neither positions nor orientations.
    """
    def __init__(self, trim_to_range=None):
        super().__init__('none', trim_to_range=trim_to_range)

    def _provides_positions(self):
        return False

    def _provides_orientations(self):
        return False

    def _create_positions_table(self):
        raise NotImplementedError(
            f"NoSource does not provide position data.")

    def _create_orientations_table(self):
        raise NotImplementedError(
            f"NoSource does not provide orientation data.")


#########################
# DATA SOURCE INTERFACE #
#########################

def test_provides_positions_only():
    src = PositionsSource()
    assert src.provides_positions() is True
    assert src.provides_orientations() is False


def test_provides_orientations_only():
    src = OrientationsSource()
    assert src.provides_positions() is False
    assert src.provides_orientations() is True


def test_provides_positions_and_orientations():
    src = PositionsAndOrientationsSource()
    assert src.provides_positions() is True
    assert src.provides_orientations() is True


def test_no_source_provided():
    src = NoSource()
    assert src.provides_positions() is False
    assert src.provides_orientations() is False


def test_marker_source_provides_positions_only():
    src = MarkerSource('nonexistent', 'nonexistent.trc')
    assert src.provides_positions() is True
    assert src.provides_orientations() is False


def test_get_positions_raises_with_subclass_name_when_not_provided():
    with pytest.raises(NotImplementedError, match='NoSource'):
        NoSource().get_positions_table()


def test_get_orientations_raises_with_subclass_name_when_not_provided():
    with pytest.raises(NotImplementedError, match='NoSource'):
        NoSource().get_orientations_table()


def test_get_orientations_raises_on_positions_only_subclass():
    with pytest.raises(NotImplementedError, match='PositionsSource'):
        PositionsSource().get_orientations_table()


def test_get_positions_raises_on_orientations_only_subclass():
    with pytest.raises(NotImplementedError, match='OrientationsSource'):
        OrientationsSource().get_positions_table()


################
# MODIFY TABLE #
################

def test_remove_columns_drops_listed_columns():
    table = create_Vec3_table([0.0, 0.1], ['a', 'b', 'c'])
    DataSource.remove_columns(table, ['b'])
    assert list(table.getColumnLabels()) == ['a', 'c']


def test_update_column_labels_renames_via_mapping():
    table = create_Vec3_table([0.0, 0.1], ['a', 'b', 'c'])
    DataSource.update_column_labels(table, {'a': 'x', 'c': 'z'})
    assert list(table.getColumnLabels()) == ['x', 'b', 'z']


def test_update_column_labels_no_op_for_empty_map():
    table = create_Vec3_table([0.0, 0.1], ['a', 'b'])
    DataSource.update_column_labels(table, {})
    assert list(table.getColumnLabels()) == ['a', 'b']


def test_trim_table_to_range_keeps_inclusive_window():
    table = create_Vec3_table([0.0, 0.5, 1.0, 1.5, 2.0], ['a'])
    DataSource.trim_table_to_range(table, (0.5, 1.5))
    assert list(table.getIndependentColumn()) == [0.5, 1.0, 1.5]


def test_assert_position_orientation_consistent_raises_on_label_mismatch():
    table = create_Vec3_table([0.0, 0.5, 1.0, 1.5, 2.0], ['a'])
    with pytest.raises(ValueError, match='end time in trim_to_range'):
        DataSource.trim_table_to_range(table, (1.5, 0.5))


def test_init_raises_for_non_tuple_trim_to_range():
    with pytest.raises(ValueError, match='tuple'):
        NoSource(trim_to_range=[0.0, 1.0])


def test_init_raises_for_wrong_length_trim_to_range():
    with pytest.raises(ValueError, match='tuple'):
        NoSource(trim_to_range=(0.0, 0.5, 1.0))


###############
# CONSISTENCY #
###############

def test_assert_position_orientation_consistent_happy_path():
    positions = create_Vec3_table([0.0, 0.1], ['a', 'b'])
    orientations = create_Quaternion_table([0.0, 0.1], ['a', 'b'])
    DataSource.assert_position_orientation_consistent(positions, orientations)


def test_assert_position_orientation_consistent_raises_on_label_mismatch():
    positions = create_Vec3_table([0.0, 0.1], ['a', 'b'])
    orientations = create_Quaternion_table([0.0, 0.1], ['a', 'x'])
    with pytest.raises(ValueError, match='mismatched column'):
        DataSource.assert_position_orientation_consistent(
            positions, orientations)


def test_assert_position_orientation_consistent_raises_on_row_count_mismatch():
    positions = create_Vec3_table([0.0, 0.1, 0.2], ['a', 'b'])
    orientations = create_Quaternion_table([0.0, 0.1], ['a', 'b'])
    with pytest.raises(ValueError, match='mismatched row'):
        DataSource.assert_position_orientation_consistent(
            positions, orientations)


def test_assert_position_orientation_consistent_raises_on_time_mismatch():
    positions = create_Vec3_table([0.0, 0.1], ['a', 'b'])
    orientations = create_Quaternion_table([0.0, 0.2], ['a', 'b'])
    with pytest.raises(ValueError, match='mismatched time'):
        DataSource.assert_position_orientation_consistent(
            positions, orientations)


def test_assert_tables_share_times_returns_shared_time_vector():
    a = create_Vec3_table([0.0, 0.1, 0.2], ['x'])
    b = create_Vec3_table([0.0, 0.1, 0.2], ['y'])
    assert DataSource.assert_tables_share_times([a, b]) == [0.0, 0.1, 0.2]


def test_assert_tables_share_times_raises_on_mismatch():
    a = create_Vec3_table([0.0, 0.1], ['x'])
    b = create_Vec3_table([0.0, 0.2], ['y'])
    with pytest.raises(ValueError, match='differs from'):
        DataSource.assert_tables_share_times([a, b])


def test_assert_sources_share_times_falls_back_to_orientations():
    # PositionsSource and OrientationsSource both expose times [0.0, 0.1] —
    # the orientation-only source must contribute via its orientations table.
    sources = [PositionsSource(), OrientationsSource()]
    assert DataSource.assert_sources_share_times(sources) == [0.0, 0.1]


def test_assert_sources_share_times_raises_when_source_hasNoSource():
    sources = [PositionsSource(), NoSource()]
    with pytest.raises(ValueError, match='NoSource'):
        DataSource.assert_sources_share_times(sources)


#########
# TRIAL #
#########

class StubMarkerSource(MarkerSource):
    """
    A MarkerSource that returns a synthetic positions table rather than reading a TRC,
    so Trial's type dispatch can be exercised without a file on disk.
    """
    def __init__(self, name: str, sample_times: list[float], labels=('a', 'b')):
        super().__init__(name, 'unused.trc')
        self.sample_times = sample_times
        self.labels = list(labels)

    def _create_positions_table(self):
        return create_Vec3_table(self.sample_times, self.labels)


def test_trial_rejects_empty_name():
    with pytest.raises(ValueError, match='non-empty string'):
        Trial('')


def test_trial_adopts_shared_times_from_its_sources():
    trial = Trial('trial', [StubMarkerSource('stub1', [0.0, 0.1, 0.2]),
                            StubMarkerSource('stub2', [0.0, 0.1, 0.2], labels=('c',))])
    assert len(trial.marker_data) == 2
    assert trial.times == [0.0, 0.1, 0.2]
    assert trial.num_times == 3


def test_trial_rejects_source_with_mismatched_times():
    trial = Trial('trial', [StubMarkerSource('stub1', [0.0, 0.1])])
    with pytest.raises(ValueError, match='same times'):
        trial.add_data_source(StubMarkerSource('stub2', [0.0, 0.2]))


def test_trial_times_raises_without_reference_data():
    with pytest.raises(ValueError, match='no reference data'):
        Trial('trial').times
