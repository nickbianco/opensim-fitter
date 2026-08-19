import os
import copy
import scipy
import numpy as np
import pandas as pd
import opensim as osim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .data_sources import MarkerSource


class MultivariateNormal:
    """
    A utility class for modeling a multivariate normal distribution. Uses internally
    scipy.stats.multivariate_normal to store fitted distributions.
    """

    @classmethod
    def from_data(cls, variables: list[str], data: np.ndarray):
        """
        Create a MultivariateNormal from data.

        Parameters
        ----------
        variables : list of str
            The names of the variables in the distribution.
        data : ndarray (m, n)
            Data the distribution is fitted to.

        Returns
        -------
        MultivariateNormal
            The fitted distribution.
        """
        mean, cov = scipy.stats.multivariate_normal.fit(data)
        return cls(variables, mean, cov)

    def __init__(self, variables: list[str], mean: np.ndarray, cov: np.ndarray):
        """
        Create a MultivariateNormal directly from the mean and covariance.

        Parameters
        ----------
        variables : list of str
            The names of the variables in the distribution.
        mean : ndarray (n,)
            The mean of the distribution.
        cov : ndarray (n, n)
            The covariance matrix of the distribution.

        Returns
        -------
        MultivariateNormal
            The created distribution.
        """
        assert(len(variables) == mean.shape[0] == cov.shape[0] == cov.shape[1])
        self.variables = variables
        self.mean = mean
        self.cov = cov
        self.multivariate_normal = scipy.stats.multivariate_normal(
            mean=self.mean, cov=self.cov)

    def _convert_values_from_dict(self, values: dict[str, float]):
        """
        Convert a dictionary of variable values to an ndarray.

        Parameters
        ----------
        values : dict[str, float]
            A dictionary mapping variable names to their values.

        Returns
        -------
        ndarray (n,)
            The values in the order of self.variables.
        """
        x = np.zeros(len(self.variables))
        for var in values.keys():
            if var not in self.variables:
                raise ValueError(f'Variable {var} not in distribution.')
            x[self.variables.index(var)] = values[var]
        return x

    def get_variables(self):
        """
        Get the names of the variables in the distribution.

        Returns
        -------
        list of str
            The names of the variables.
        """
        return self.variables

    def get_mean(self):
        """
        Get the mean of the distribution.

        Returns
        -------
        ndarray (n,)
            The mean of the distribution.
        """
        return self.mean

    def get_covariance(self):
        """
        Get the covariance matrix of the distribution.

        Returns
        -------
        ndarray (n, n)
            The covariance matrix of the distribution.
        """
        return self.cov

    def get_dimension(self):
        """
        Get the dimensionality of the distribution.

        Returns
        -------
        int
            The number of variables in the distribution.
        """
        return len(self.variables)

    def get_random_sample(self):
        """
        Get a random sample from the distribution.

        Returns
        -------
        ndarray (n,)
            A random sample from the distribution.
        """
        return self.multivariate_normal.rvs()

    def get_pdf(self, values: np.ndarray):
        """
        Get the probability density function (PDF) value at a given point.

        Parameters
        ----------
        values : ndarray (n,)
            The point at which to evaluate the PDF.

        Returns
        -------
        float
            The PDF value at the given point.
        """
        return self.multivariate_normal.pdf(self._convert_values_from_dict(values))

    def get_logpdf(self, values: np.ndarray):
        """
        Get the log of the probability density function (PDF) value at a given point.

        Parameters
        ----------
        values : ndarray (n,)
            The point at which to evaluate the log PDF.

        Returns
        -------
        float
            The log PDF value at the given point.
        """
        return self.multivariate_normal.logpdf(self._convert_values_from_dict(values))

    def get_cdf(self, values: np.ndarray):
        """
        Get the cumulative density function (CDF) value at a given point.

        Parameters
        ----------
        values : ndarray (n,)
            The point at which to evaluate the CDF.

        Returns
        -------
        float
            The CDF value at the given point.
        """
        return self.multivariate_normal.cdf(self._convert_values_from_dict(values))

    def get_logcdf(self, values: np.ndarray):
        """
        Get the log of the cumulative density function (CDF) value at a given point.

        Parameters
        ----------
        values : ndarray (n,)
            The point at which to evaluate the log CDF.

        Returns
        -------
        float
            The log CDF value at the given point.
        """
        return self.multivariate_normal.logcdf(self._convert_values_from_dict(values))

    def condition(self, values: dict[str, float]):
        """
        Condition the multivariate normal distribution on a subset of variables
        being equal to the specified 'values'. Returns a new MultivariateNormal instance
        for the conditioned distribution.

        Parameters
        ----------
        values : dict[str, float]
            A dictionary mapping variable names to their observed values.

        Returns
        -------
        MultivariateNormal
            The conditioned distribution.
        """
        idx_b = [self.variables.index(var) for var in values.keys()]
        idx_a = [i for i in range(len(self.variables)) if i not in idx_b]

        mean_a = self.mean[idx_a]
        mean_b = self.mean[idx_b]

        cov_aa = self.cov[np.ix_(idx_a, idx_a)]
        cov_bb = self.cov[np.ix_(idx_b, idx_b)]
        cov_ab = self.cov[np.ix_(idx_a, idx_b)]
        cov_ba = self.cov[np.ix_(idx_b, idx_a)]

        cov_bb_inv = np.linalg.inv(cov_bb)

        values_array = np.array(list(values.values()))
        mean_cond = mean_a + cov_ab @ cov_bb_inv @ (values_array - mean_b)
        cov_cond = cov_aa - cov_ab @ cov_bb_inv @ cov_ba

        variables_a = [self.variables[i] for i in idx_a]
        return MultivariateNormal(variables_a, mean_cond, cov_cond)

    def get_variable_percentile(self, variable_name: str, value: float):
        """
        Calculate the percentile rank of a specific value for an individual variable.

        Parameters
        ----------
        variable_name : str
            The name of the variable.
        value : float
            The value to find the percentile for.

        Returns
        -------
        float
            The percentile rank (0-100) of the given value.

        Raises
        ------
        ValueError
            If the variable is not in the distribution.
        """
        if variable_name not in self.variables:
            raise ValueError(f"Variable '{variable_name}' not found in the distribution")

        # Get index of the variable
        var_idx = self.variables.index(variable_name)

        # Get mean and standard deviation for this variable
        var_mean = self.mean[var_idx]
        var_std = np.sqrt(self.cov[var_idx, var_idx])

        # Create a normal distribution for this variable
        var_dist = scipy.stats.norm(var_mean, var_std)

        # Calculate the percentile (CDF * 100)
        percentile = var_dist.cdf(value) * 100.0

        return percentile

    def plot_variable_distribution(self, data_df: pd.DataFrame, variable_name: str,
                                   bins: int=30):
        """
        Plot the fitted distribution versus actual data for a specific variable.

        Parameters
        ----------
        data_df: pandas.DataFrame
            The original data
        variable_name: str
            Name of the variable to plot
        bins: int, optional
            Number of histogram bins (default: 30)
        """
        import matplotlib.pyplot as plt

        # Check if variable exists in the distribution
        if variable_name not in self.variables:
            raise ValueError(f"Variable '{variable_name}' not found in the distribution")

        # Get index of the variable
        var_idx = self.variables.index(variable_name)

        # Get mean and standard deviation for this variable
        var_mean = self.mean[var_idx]
        var_std = np.sqrt(self.cov[var_idx, var_idx])

        # Create a normal distribution for this variable
        var_dist = scipy.stats.norm(var_mean, var_std)

        # Plot histogram of actual data
        plt.figure(figsize=(10, 6))
        plt.hist(data_df[variable_name], bins=bins, density=True, alpha=0.6,
                label='Actual data', color='skyblue')

        # Plot the fitted distribution
        x = np.linspace(var_mean - 4*var_std, var_mean + 4*var_std, 1000)
        plt.plot(x, var_dist.pdf(x), 'r-', lw=2, label='Fitted normal distribution')

        plt.title(f'Distribution of {variable_name}')
        plt.xlabel(variable_name)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


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
