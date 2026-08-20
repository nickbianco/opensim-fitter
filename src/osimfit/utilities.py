import os
import copy
import numpy as np
import opensim as osim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .data_sources import MarkerSource


def plot_coordinates(model: osim.Model, states: osim.StatesTrajectory,
                     pdf_fpath: str, convert_radians_to_degrees: bool=False,
                     coordinate_ranges: dict = None):
    """
    Plot coordinate trajectories across time.

    Parameters
    ----------
    model : osim.Model
        OpenSim Model containing the coordinates to plot.
    states : osim.StatesTrajectory
        A trajectory of states for which to plot the coordinate trajectories.
    convert_radians_to_degrees : bool
        (Optional) Whether to convert values for rotational coordinates from radians to
        degrees.
    coordinate_ranges : dict
        (Optional) A dictionary contained specified ranges for specific coordinates.
        Assumes that the provided values match the units designated by the
        `convert_radians_to_degrees` flag. If a range is not provided for a coordinate,
        the range from the OpenSim Coordinate definition will be used by default.
    """

    # Extract the coordinate values from the states trajectory.
    coordset = model.getCoordinateSet()
    coordinate_values = dict()
    coordinate_ranges = copy.deepcopy(coordinate_ranges)
    coordinate_units = dict()
    coordinate_names = list()
    for icoord in range(coordset.getSize()):

        # Get the coordinate and its motion type to determine the units (rad or m). Skip
        # coupled coordinates.
        coord = coordset.get(icoord)
        motion_type = coord.getMotionType()
        if convert_radians_to_degrees and motion_type == 1:
            coordinate_units[coord.getName()] = 'deg'
        elif not convert_radians_to_degrees and motion_type == 1:
            coordinate_units[coord.getName()] = 'rad'
        elif  motion_type == 2:
            coordinate_units[coord.getName()] = 'm'
        else:
            continue

        # Coordinate name and range.
        coordinate_names.append(coord.getName())
        if not coord.getName() in coordinate_ranges:
            coordinate_ranges[coord.getName()] = (coord.getRangeMin(),
                                                  coord.getRangeMax())
            if convert_radians_to_degrees and motion_type == 1:
                coordinate_ranges[coord.getName()] = (np.degrees(coord.getRangeMin()),
                                                      np.degrees(coord.getRangeMax()))

        # Coordinate values.
        values = np.zeros(states.getSize())
        for istate in range(states.getSize()):
            state = states.get(istate)
            values[istate] = coord.getValue(state)
            if convert_radians_to_degrees and motion_type == 1:
                values[istate] = np.degrees(values[istate])
        coordinate_values[coord.getName()] = values

    # Time vector.
    time = np.array([states.get(i).getTime() for i in range(states.getSize())])

    # Plot coordinate trajectories to PDF, 12 per page (4 rows x 3 cols).
    PLOTS_PER_PAGE = 12
    ROWS, COLS = 4, 3

    n_coords = len(coordinate_names)
    with PdfPages(pdf_fpath) as pdf:
        n_pages = int(np.ceil(n_coords / PLOTS_PER_PAGE))
        for page in range(n_pages):
            fig, axes = plt.subplots(ROWS, COLS, figsize=(11, 8.5))
            axes_flat = axes.flatten()

            for plot_idx in range(PLOTS_PER_PAGE):
                ax = axes_flat[plot_idx]
                coord_idx = page * PLOTS_PER_PAGE + plot_idx

                if coord_idx >= n_coords:
                    ax.set_visible(False)
                    continue

                coord_name = coordinate_names[coord_idx]
                ax.plot(time, coordinate_values[coord_name], linewidth=2)
                ax.set_xlim(time[0], time[-1])
                ax.set_ylim(coordinate_ranges[coord_name])
                ax.set_xlabel('time (s)', fontsize=8)
                ax.set_ylabel(f'{coord_name} value ({coordinate_units[coord_name]})',
                              fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, linestyle='--', alpha=0.5)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def compute_marker_errors(model: osim.Model, states: osim.StatesTrajectory,
                          marker_source: MarkerSource) -> osim.TimeSeriesTableVec3:
    """
    Compute the error between experimental marker data and model marker positions across
    a trajectory of states.

    The time vector in the marker data is expected to match the time vector in the
    states trajectory, and the marker labels in the marker data are expected to match
    the paths of virtual markers in the model. The error is computed as the difference
    between the position of each virtual marker in the model and the corresponding
    experimental marker position at each time step.

    Parameters
    ----------
    model : osim.Model
        OpenSim Model with virtual markers matching the experimental marker data.
    states : osim.StatesTrajectory
        A trajectory of states for which to compute the marker errors.
    marker_source : osim.MarkerSource
        A data source containing the experimental marker positions.

    Returns
    -------
    osim.TimeSeriesTableVec3
        A table containing the marker errors for each marker at each time step. Each
        column corresponds to a marker, and each row corresponds to a time step. The
        entries are Vec3 objects representing the error in the X, Y, and Z directions
        in ground for each marker.
    """
    marker_data = marker_source.get_positions_table()
    marker_paths = marker_data.getColumnLabels()
    errors = osim.TimeSeriesTableVec3()

    # Verify that the time vector in the marker data matches the time vector in the
    # states trajectory.
    state_times = [states.get(i).getTime() for i in range(states.getSize())]
    marker_times = marker_data.getIndependentColumn()
    if not np.allclose(state_times, marker_times):
        raise ValueError('Expected the time vector in the marker data to match the time '
                         'vector in the states trajectory, but it does not.')

    for i in range(states.getSize()):
        state = states.get(i)
        model.realizePosition(state)
        data = marker_data.getRowAtIndex(i)
        errors_row = osim.RowVectorVec3(len(marker_paths), osim.Vec3(0))
        for imarker, marker_path in enumerate(marker_paths):
            marker = osim.Marker.safeDownCast(model.getComponent(marker_path))
            if marker is None:
                raise ValueError(f'Marker {marker_path} not found in model.')

            position = marker.getLocationInGround(state).to_numpy()
            error = position - data.getElt(0, imarker).to_numpy()
            errors_row.updElt(0, imarker).set(0, error[0])
            errors_row.updElt(0, imarker).set(1, error[1])
            errors_row.updElt(0, imarker).set(2, error[2])

        errors.appendRow(state_times[i], errors_row)

    errors.setColumnLabels(marker_paths)
    errors.addTableMetaDataString("Units", "m")
    return errors


def plot_marker_errors(errors: osim.TimeSeriesTableVec3, pdf_fpath: str):
    """
    Plot marker errors across time and save to a PDF.

    Parameters
    ----------
    errors: osim.TimeSeriesTableVec3
        A table containing the marker errors for each marker at each time step. Each
        column corresponds to a marker, and each row corresponds to a time step. The
        entries are Vec3 objects representing the error in the X, Y, and Z directions
        in ground for each marker. Marker errors are expected to be in meters.
    pdf_fpath: str
        The file path where the PDF of marker error plots should be saved.

    Returns
    -------
    mean_errors: dict[str, float]
        Per-marker mean error magnitude (cm), keyed by marker label.
    max_errors: dict[str, float]
        Per-marker maximum error magnitude (cm), keyed by marker label.
    """

    # Extract error magnitudes (m -> cm).
    time = np.array(errors.getIndependentColumn())
    marker_labels = list(errors.getColumnLabels())
    n_markers = len(marker_labels)
    n_times = errors.getNumRows()

    error_norms = np.zeros((n_times, n_markers))
    for i in range(n_times):
        row = errors.getRowAtIndex(i)
        for j in range(n_markers):
            vec = row.getElt(0, j)
            error_norms[i, j] = np.sqrt(vec.get(0)**2 + vec.get(1)**2 + vec.get(2)**2)
    error_norms *= 100  # m -> cm

    # Per-marker mean and max error magnitudes (cm), keyed by marker label.
    mean_errors = {label: float(np.mean(error_norms[:, j]))
                   for j, label in enumerate(marker_labels)}
    max_errors = {label: float(np.max(error_norms[:, j]))
                  for j, label in enumerate(marker_labels)}

    # Plot marker errors to PDF, 12 per page (4 rows x 3 cols).
    PLOTS_PER_PAGE = 12
    ROWS, COLS = 4, 3
    MAX_ERROR = 10.0

    with PdfPages(pdf_fpath) as pdf:
        n_pages = int(np.ceil(n_markers / PLOTS_PER_PAGE))
        for page in range(n_pages):
            fig, axes = plt.subplots(ROWS, COLS, figsize=(11, 8.5))
            axes_flat = axes.flatten()

            for plot_idx in range(PLOTS_PER_PAGE):
                ax = axes_flat[plot_idx]
                marker_idx = page * PLOTS_PER_PAGE + plot_idx

                if marker_idx >= n_markers:
                    ax.set_visible(False)
                    continue

                label = marker_labels[marker_idx]
                err = error_norms[:, marker_idx]
                mean_err = mean_errors[label]
                max_err = max_errors[label]
                y_max = MAX_ERROR if max_err <= MAX_ERROR else max_err * 1.1

                if max_err > MAX_ERROR:
                    ax.axhspan(MAX_ERROR, y_max, color='lightcoral',
                               alpha=0.4, zorder=0)

                ax.plot(time, err, linewidth=2.0)
                ax.axhline(2.0, color='black', linestyle='--', linewidth=0.5)
                ax.axhline(4.0, color='red', linestyle='--', linewidth=0.5)
                ax.set_xlim(time[0], time[-1])
                ax.set_ylim(0, y_max)
                ax.set_title(label.split('/')[-1], fontsize=8)
                ax.set_xlabel('time (s)', fontsize=8)
                ax.set_ylabel('error (cm)', fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(True, linestyle='--', alpha=0.5)

                # Annotate each plot with its mean and max error.
                ax.text(0.97, 0.95,
                        f'mean error: {mean_err:.2f} cm\n max error: {max_err:.2f} cm',
                        transform=ax.transAxes, ha='right', va='top', fontsize=7,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                                  edgecolor='0.7'))

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return mean_errors, max_errors
