"""
Unit tests for src/osimfit/data_sources.py.
"""

import pytest
import opensim as osim
from osimfit.data_sources import DataSource, MarkerSource


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
    Throwaway subclass that provides positions only.
    """
    def _create_positions_table(self):
        return create_Vec3_table([0.0, 0.1], ['a', 'b'])


class OrientationsSource(DataSource):
    """
    Throwaway subclass that provides orientations only.
    """
    def _create_orientations_table(self):
        return create_Quaternion_table([0.0, 0.1], ['a', 'b'])


class PositionsAndOrientationsSource(DataSource):
    """
    Throwaway subclass that provides both positions and orientations.
    """
    def _create_positions_table(self):
        return create_Vec3_table([0.0, 0.1], ['a', 'b'])

    def _create_orientations_table(self):
        return create_Quaternion_table([0.0, 0.1], ['a', 'b'])


class NoSource(DataSource):
    """
    Throwaway subclass that provides neither positions nor orientations.
    """
    pass


#########################
# DATA SOURCE INTERFACE #
#########################

def test_provides_positions_only():
    src = PositionsSource()
    assert src.provides_positions is True
    assert src.provides_orientations is False


def test_provides_orientations_only():
    src = OrientationsSource()
    assert src.provides_positions is False
    assert src.provides_orientations is True


def test_provides_positions_and_orientations():
    src = PositionsAndOrientationsSource()
    assert src.provides_positions is True
    assert src.provides_orientations is True


def test_no_source_provided():
    src = NoSource()
    assert src.provides_positions is False
    assert src.provides_orientations is False


def test_marker_source_provides_positions_only():
    src = MarkerSource('nonexistent.trc')
    assert src.provides_positions is True
    assert src.provides_orientations is False


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
