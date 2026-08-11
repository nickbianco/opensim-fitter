import ezc3d
import numpy as np
import opensim as osim
from abc import ABC


class DataSource(ABC):
    """
    Abstract base class for time-series data sources.

    Subclasses parse a source file (e.g., a TRC or C3D file) and expose the data as
    OpenSim TimeSeriesTables of positions and/or orientations. Tables can be accessed
    via the base class methods ``get_positions_table`` and ``get_orientations_table``;
    calling these methods triggers table construction and modifications based on the
    (optional) input parameters ``labels_to_remove``, ``label_map``, and
    ``trim_to_range`` are applied.

    Subclasses signal which data they provide simply by overriding the corresponding
    protected hook: override ``_create_positions_table`` to provide positions, and/or
    ``_create_orientations_table`` to provide orientations. The base class detects
    overrides via the read-only properties ``provides_positions`` and
    ``provides_orientations``; if a hook is not overridden, calling the matching
    ``get_*_table`` method raises ``NotImplementedError``.

    Consistency between paired position/orientation tables is not enforced here —
    callers that need it should invoke ``assert_position_orientation_consistent``
    after pulling both tables.

    Parameters
    ----------
    labels_to_remove: list[str], optional
        Column labels to drop from each constructed table.
    label_map: dict, optional
        Mapping from existing column labels to new column labels. Columns
        not listed are left unchanged.
    trim_to_range: tuple[float, float], optional
        The time range used to trim each constructed table to a sub-range of
        the original time vector.
    """

    def __init__(self, labels_to_remove=None, label_map=None, trim_to_range=None):
        super().__init__()
        self.labels_to_remove = labels_to_remove
        self.label_map = label_map

        if trim_to_range is not None:
            if not (isinstance(trim_to_range, tuple) and len(trim_to_range) == 2):
                raise ValueError(
                    "trim_to_range must be a tuple of (start_time, end_time).")
            if not trim_to_range[0] < trim_to_range[1]:
                raise ValueError(
                    f"Expected the end time in trim_to_range to be greater than the "
                    f"start time, but received {trim_to_range[1]} and "
                    f"{trim_to_range[0]}, respectively.")
        self.trim_to_range = trim_to_range

    @property
    def provides_positions(self) -> bool:
        return (type(self)._create_positions_table
                is not DataSource._create_positions_table)

    @property
    def provides_orientations(self) -> bool:
        return (type(self)._create_orientations_table
                is not DataSource._create_orientations_table)

    def get_positions_table(self) -> osim.TimeSeriesTableVec3:
        """
        Build and return the positions table.

        The table is constructed on every call via
        ``_create_positions_table`` and then post-processed (remove
        columns, relabel, trim).

        Raises
        ------
        NotImplementedError
            If the source does not provide position data (i.e.,
            ``_create_positions_table`` is not overridden).
        """
        if not self.provides_positions:
            raise NotImplementedError(
                f"{type(self).__name__} does not provide position data.")
        table = self._create_positions_table()
        if self.labels_to_remove:
            self.remove_columns(table, self.labels_to_remove)
        if self.label_map:
            self.update_column_labels(table, self.label_map)
        if self.trim_to_range:
            self.trim_table_to_range(table, self.trim_to_range)
        return table

    def get_orientations_table(self) -> osim.TimeSeriesTableQuaternion:
        """
        Build and return the orientations table.

        The table is constructed on every call via
        ``_create_orientations_table`` and then post-processed (remove
        columns, relabel, trim).

        Raises
        ------
        NotImplementedError
            If the source does not provide orientation data (i.e.,
            ``_create_orientations_table`` is not overridden).
        """
        if not self.provides_orientations:
            raise NotImplementedError(
                f"{type(self).__name__} does not provide orientation data.")
        table = self._create_orientations_table()
        if self.labels_to_remove:
            self.remove_columns(table, self.labels_to_remove)
        if self.label_map:
            self.update_column_labels(table, self.label_map)
        if self.trim_to_range:
            self.trim_table_to_range(table, self.trim_to_range)
        return table

    def _create_positions_table(self) -> osim.TimeSeriesTableVec3:
        raise NotImplementedError(
            f"{type(self).__name__} does not provide position data.")

    def _create_orientations_table(self) -> osim.TimeSeriesTableQuaternion:
        raise NotImplementedError(
            f"{type(self).__name__} does not provide orientation data.")

    @staticmethod
    def assert_position_orientation_consistent(positions, orientations):
        """
        Verify that a paired positions/orientations table pair is consistent.

        Checks that the two tables share the same column labels, row count,
        and independent (time) column. Intended for callers that have
        pulled both tables from a single :py:class:`DataSource` and need
        to confirm they line up before consuming them together.

        Parameters
        ----------
        positions : osim.TimeSeriesTableVec3
        orientations : osim.TimeSeriesTableQuaternion

        Raises
        ------
        ValueError
            If column labels, row counts, or time columns differ.
        """
        pos_labels = list(positions.getColumnLabels())
        ori_labels = list(orientations.getColumnLabels())
        if pos_labels != ori_labels:
            raise ValueError(
                "Position and orientation tables have mismatched column "
                f"labels (positions={pos_labels}, orientations={ori_labels}).")

        if positions.getNumRows() != orientations.getNumRows():
            raise ValueError(
                "Position and orientation tables have mismatched row "
                f"counts ({positions.getNumRows()} vs "
                f"{orientations.getNumRows()}).")

        pos_times = np.asarray(positions.getIndependentColumn())
        ori_times = np.asarray(orientations.getIndependentColumn())
        if not np.array_equal(pos_times, ori_times):
            raise ValueError(
                "Position and orientation tables have mismatched time "
                "columns.")

    @staticmethod
    def remove_columns(table, columns_to_remove):
        """
        Drop the given columns from ``table`` in place.

        Parameters
        ----------
        table : osim.TimeSeriesTable
            Table to modify; modified in place.
        columns_to_remove : iterable of str
            Column labels to remove.

        Returns
        -------
        osim.TimeSeriesTable
            The same table instance, returned for convenience.
        """
        for col in columns_to_remove:
            table.removeColumn(col)
        return table

    @staticmethod
    def update_column_labels(table, label_map):
        """
        Rename columns of ``table`` in place using ``label_map``.

        Columns not present in ``label_map`` are left unchanged.

        Parameters
        ----------
        table : osim.TimeSeriesTable
            Table to modify; modified in place.
        label_map : dict
            Mapping from existing column labels to new column labels.

        Returns
        -------
        osim.TimeSeriesTable
            The same table instance, returned for convenience.
        """
        if not label_map:
            return table

        labels = list(table.getColumnLabels())
        for ilabel in range(len(labels)):
            labels[ilabel] = label_map.get(labels[ilabel], labels[ilabel])
        table.setColumnLabels(labels)
        return table

    @staticmethod
    def trim_table_to_range(table, time_range):
        """
        Trim ``table`` to the given ``(start_time, end_time)`` window in place.

        Parameters
        ----------
        table : osim.TimeSeriesTable
            Table to modify; modified in place.
        time_range : tuple of (float, float)
            Inclusive ``(start_time, end_time)`` window.

        Returns
        -------
        osim.TimeSeriesTable
            The same table instance, returned for convenience.
        """
        table.trim(time_range[0], time_range[1])
        return table

    @staticmethod
    def assert_sources_share_times(sources: list["DataSource"]) -> list[float]:
        """
        Verify that all data sources share the same time vector.

        For each source, the comparison uses its positions table when available;
        otherwise it falls back to its orientations table. A source that provides
        neither raises ``ValueError``.

        Parameters
        ----------
        sources: list[DataSource]
            Data sources to compare.

        Returns
        -------
        list of float
            The shared time vector (taken from the first source).

        Raises
        ------
        ValueError
            If any source's time vector differs from the first source's, or if
            any source provides neither positions nor orientations.
        """
        tables = []
        for source in sources:
            if source.provides_positions:
                tables.append(source.get_positions_table())
            elif source.provides_orientations:
                tables.append(source.get_orientations_table())
            else:
                raise ValueError(
                    f"{type(source).__name__} provides neither position nor "
                    "orientation data.")
        return DataSource.assert_tables_share_times(tables)

    @staticmethod
    def assert_tables_share_times(tables) -> list[float]:
        """
        Verify that all tables share the same independent (time) column.

        Parameters
        ----------
        tables : iterable of TimeSeriesTable
            Tables to compare. Any OpenSim TimeSeriesTable variant is accepted.

        Returns
        -------
        list of float
            The shared time vector (taken from the first table).

        Raises
        ------
        ValueError
            If any table's time column differs from the first table's.
        """
        times = None
        for itab, table in enumerate(tables):
            col = list(table.getIndependentColumn())
            if times is None:
                times = col
            elif not np.array_equal(np.asarray(times), np.asarray(col)):
                raise ValueError(
                    f"Table at index {itab} has a time column that differs from "
                    f"the first table's time column.")
        return times


class MarkerSource(DataSource):
    """
    Marker position data source backed by an OpenSim ``.trc`` file.

    Reads the file directly with ``opensim.TimeSeriesTableVec3`` and
    converts the marker coordinates from millimeters to meters when the
    file metadata reports units of ``mm``. Orientation data is not
    available; calling :py:meth:`get_orientations_table` raises
    ``NotImplementedError``.

    Parameters
    ----------
    trc_filepath: str
        Path to the ``.trc`` file.
    labels_to_remove: list[str], optional
        See :py:class:`DataSource`.
    label_map: dict, optional
        See :py:class:`DataSource`.
    trim_to_range: tuple[float, float], optional
        See :py:class:`DataSource`.
    """

    def __init__(self, trc_filepath, labels_to_remove=None, label_map=None,
                 trim_to_range=None):
        super().__init__(labels_to_remove=labels_to_remove, label_map=label_map,
                         trim_to_range=trim_to_range)
        self.trc_filepath = trc_filepath

    def _create_positions_table(self) -> osim.TimeSeriesTableVec3:
        table = osim.TimeSeriesTableVec3(self.trc_filepath)
        units = table.getTableMetaDataString("Units")
        if units == "mm":
            for icol in range(table.getNumColumns()):
                columnData = table.updDependentColumnAtIndex(icol)
                columnData.multiplyAssign(0.001)
            table.addTableMetaDataString("Units", "m")

        return table

    def validate_marker_paths(self, model: osim.Model):
        """
        Validate that each marker label in this source matches the absolute path of a
        marker in ``model``. Does not disallow extra markers in the model.

        Parameters
        ----------
        model : osim.Model
            Model to compare against.

        Raises
        ------
        ValueError
            If any marker label in this source is not the absolute path of a marker in
            ``model``.
        """
        table = self.get_positions_table()
        source_labels = set(table.getColumnLabels())
        model_labels = set()
        markerset = model.getMarkerSet()
        for i in range(markerset.getSize()):
            model_labels.add(markerset.get(i).getAbsolutePathString())
        missing_in_model = sorted(source_labels - model_labels)
        if missing_in_model:
            raise ValueError(
                f"Marker labels in {self.trc_filepath} are not all present in "
                f"{model.getName()}. Missing from model: {missing_in_model}.")

    def write(self, trc_filename: str):
        markers = self.get_positions_table()
        marker_names = osim.StdVectorString()
        for label in markers.getColumnLabels():
            marker_names.append(label.replace('/markerset/', ''))
        markers.setColumnLabels(marker_names)
        trc = osim.TRCFileAdapter()
        trc.write(markers, trc_filename)


class TheiaFrameSource(DataSource):
    """
    Data source for Theia markerless motion capture data. Theia outputs 4x4
    homogeneous transformation matrices for various frames from its internal
    representation of the human skeletal model. This class extracts the
    position and orientation of each frame in Theia's output and converts
    them to OpenSim's coordinate system. Callers can verify that the
    positions and orientations tables line up via
    :py:meth:`DataSource.assert_position_orientation_consistent`.

    Parameters
    ----------
    c3d_filepath: str
        Path to the C3D file containing Theia's output. The C3D file should
        contain 4x4 homogeneous transformation matrices for each frame,
        stored in the 'rotations' field. Each frame's transformation matrix
        should be labeled with a unique name in the C3D file.
    labels_to_remove: list[str], optional
        See :py:class:`DataSource`.
    label_map: dict, optional
        See :py:class:`DataSource`.
    trim_to_range: tuple[float, float], optional
        See :py:class:`DataSource`.
    """

    def __init__(self, c3d_filepath, labels_to_remove=None, label_map=None,
                 trim_to_range=None):
        super().__init__(labels_to_remove=labels_to_remove, label_map=label_map,
                         trim_to_range=trim_to_range)

        self.filepath = c3d_filepath
        self.c3d = ezc3d.c3d(c3d_filepath)

        # This is a Y-Z space-fixed rotation needed to convert data collected from Theia
        # to OpenSim's ground reference frame convention (X forward, Y up, Z right).
        osim_rotation = osim.Rotation()
        osim_rotation.setRotationFromTwoAnglesTwoAxes(1, # space-fixed
                -0.5*np.pi, osim.CoordinateAxis(1), # Y rotation
                -0.5*np.pi, osim.CoordinateAxis(2)) # Z rotation
        self.osim_rotation = osim_rotation

        # This is an additional body-fixed rotation that effectively swaps the axes of
        # the rotations collected from Theia to match OpenSim's ground reference frame
        # convention (X forward, Y up, Z right), which is the convention used by the
        # matching Frame elements in the generic model.
        frame_rotation = osim.Rotation()
        frame_rotation.setRotationToBodyFixedXY(osim.Vec2(0.5*np.pi))
        self.frame_rotation = frame_rotation

        # Extract the column labels from the C3D file and remove the '_4X4' suffix.
        raw_labels = self.c3d.parameters['ROTATION']['LABELS']['value']
        self.labels = [label.replace('_4X4', '') for label in raw_labels]

        self.data = self.c3d.data['rotations']
        self.rate = self.c3d.parameters['ROTATION']['RATE']['value'][0]
        self.num_frames = self.data.shape[3]
        self.times = np.array([i/self.rate for i in range(self.num_frames)])


    def _create_positions_table(self) -> osim.TimeSeriesTableVec3:
        table = osim.TimeSeriesTableVec3()
        for iframe in range(self.num_frames):
            row = osim.RowVectorVec3(len(self.labels), osim.Vec3(0))
            for ilabel, label in enumerate(self.labels):
                position = self.data[:, 3, ilabel, iframe] / 1000.0  # mm to m
                row[ilabel] = osim.Vec3(position[0], position[1], position[2])
                row[ilabel] = self.osim_rotation.multiply(row[ilabel])

            table.appendRow(self.times[iframe], row)

        table.setColumnLabels(self.labels)
        table.addTableMetaDataString("Units", "m")
        table.addTableMetaDataString("DataRate", str(self.rate))

        return table


    def _create_orientations_table(self) -> osim.TimeSeriesTableQuaternion:
        table = osim.TimeSeriesTableQuaternion()
        for iframe in range(self.num_frames):
            row = osim.RowVectorQuaternion(len(self.labels), osim.Quaternion())
            for ilabel, label in enumerate(self.labels):
                rot = self.data[:3, :3, ilabel, iframe]
                data_rotation = osim.Rotation()
                data_rotation.set(0,0, rot[0,0])
                data_rotation.set(1,0, rot[1,0])
                data_rotation.set(2,0, rot[2,0])
                data_rotation.set(0,1, rot[0,1])
                data_rotation.set(1,1, rot[1,1])
                data_rotation.set(2,1, rot[2,1])
                data_rotation.set(0,2, rot[0,2])
                data_rotation.set(1,2, rot[1,2])
                data_rotation.set(2,2, rot[2,2])
                rotation = self.osim_rotation.multiply(data_rotation)
                rotation = rotation.multiply(self.frame_rotation)

                # Store as a quaternion.
                new_quat = rotation.convertRotationToQuaternion()
                upd_quat = row.updElt(0, ilabel)
                upd_quat.set(0, new_quat.get(0))
                upd_quat.set(1, new_quat.get(1))
                upd_quat.set(2, new_quat.get(2))
                upd_quat.set(3, new_quat.get(3))

            table.appendRow(self.times[iframe], row)

        table.setColumnLabels(self.labels)
        table.addTableMetaDataString("DataRate", str(self.rate))

        return table
