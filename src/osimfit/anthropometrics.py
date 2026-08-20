import os
import numpy as np
import pandas as pd
import scipy.stats


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
            raise ValueError(
                f"Variable '{variable_name}' not found in the distribution")

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
            raise ValueError(
                f"Variable '{variable_name}' not found in the distribution")

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


def build_ansur_distribution(labels: list[str], sex: str = None) -> MultivariateNormal:
    """
    Fit a `MultivariateNormal` to a subset of ANSUR II measurements, in meters.

    The distribution is fit over the requested measurement columns of the ANSUR II
    dataset selected by `sex`. The dataset's millimeter values are converted to meters
    so the distribution matches model-side measurements (which are computed in meters).

    Parameters
    ----------
    labels: list[str]
        ANSUR II measurement names selecting the distribution's variables (columns).
        Every name must be present in the dataset.
    sex: str, optional
        Subject sex ('male' or 'female') selecting the ANSUR II subset. Defaults to None
        (the combined male-and-female dataset).

    Returns
    -------
    MultivariateNormal
        The fitted distribution over `labels`, in meters.

    Raises
    ------
    ValueError
        If any name in `labels` is not present in the ANSUR II dataset.
    """
    sex_tag = 'BOTH'
    if sex and sex.lower() == 'male':
        sex_tag = 'MALE'
    elif sex and sex.lower() == 'female':
        sex_tag = 'FEMALE'
    csv_path = os.path.join(os.path.dirname(__file__), 'resources',
                            f'ANSUR_II_{sex_tag}_Public.csv')

    df = pd.read_csv(csv_path, encoding_errors='replace')
    labels = list(labels)
    missing = [label for label in labels if label not in df.columns]
    if missing:
        raise ValueError(
            f'Measurement name(s) not present in the ANSUR II dataset: {missing}.')

    data = df[labels].to_numpy(dtype=float) * 1e-3  # mm -> m
    return MultivariateNormal.from_data(labels, data)
