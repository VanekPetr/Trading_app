import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class MomentGenerator:
    """
    Provides methods for mean, variance generation.
    """

    @staticmethod
    def _alpha_numerator(Z, S):
        s = 0
        T = Z.shape[1]
        for k in range(T):
            z = Z[:, k][:, np.newaxis]
            X = z @ z.T - S
            s += np.trace(X @ X)
        s /= T**2
        return s

    @staticmethod
    def _ledoit_wolf_shrinkage(X, S):
        """
        Computes the Ledoit--Wolf shrinkage, using a target of scaled identity.
        """
        N = len(X.columns)
        # In case only one asset in the matrix, for example for benchmark with one asset, no shrinkage is needed
        if N == 1:
            return S

        # Center the data
        X = (X - X.mean(0)).to_numpy().T

        # Target.
        s_avg2 = np.trace(S) / N
        B = s_avg2 * np.eye(N)

        # Shrinkage coefficient.
        alpha_num = MomentGenerator._alpha_numerator(X, S)
        alpha_den = np.trace((S - B) @ (S - B))
        alpha = alpha_num / alpha_den

        # Shrunk covariance
        shrunk = (1 - alpha) * S + alpha * B

        return shrunk

    @staticmethod
    def generate_sigma_mu_for_test_periods(
        data: pd.DataFrame, n_test: int
    ) -> Tuple[List, List]:
        logger.info(
            "⏳ Computing covariance matrix and mean array for each investment period"
        )

        # Initialize variables
        sigma_lst = []
        mu_lst = []

        n_iter = 4  # we work with 4-week periods
        n_train_weeks = len(data.index) - n_test
        n_rolls = math.floor(n_test / n_iter) + 1

        for p in range(int(n_rolls)):
            rolling_train_dataset = data.iloc[
                (n_iter * p) : (n_train_weeks + n_iter * p), :
            ]

            sigma = np.atleast_2d(
                np.cov(rolling_train_dataset, rowvar=False, bias=True)
            )  # The sample covariance matrix

            # Add a shrinkage term (Ledoit--Wolf multiple of identity)
            sigma = MomentGenerator._ledoit_wolf_shrinkage(rolling_train_dataset, sigma)

            # Make sure sigma is positive semidefinite
            # sigma = np.atleast_2d(0.5 * (sigma + sigma.T))
            # min_eig = np.min(np.linalg.eigvalsh(sigma))
            # if min_eig < 0:
            #     sigma -= 5 * min_eig * np.eye(*sigma.shape)

            # RHO = np.corrcoef(ret_train, rowvar=False)            # The correlation matrix
            mu = np.mean(rolling_train_dataset, axis=0)  # The mean array
            # sd = np.sqrt(np.diagonal(SIGMA))                      # The standard deviation

            sigma_lst.append(sigma)
            mu_lst.append(mu)

        return sigma_lst, mu_lst

    @staticmethod
    def generate_annual_sigma_mu_with_risk_free(
        data: pd.DataFrame, risk_free_rate_annual: float = 0.02
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Computes the annualized and weekly covariance matrix (sigma) and mean return array (mu)
        for the entire historical dataset, including a risk-free asset.

        Parameters:
        - data: A pandas DataFrame with weekly returns for each asset.
        - risk_free_rate_annual: Annual return rate of the risk-free asset, default is 2%.

        Returns:
        - A tuple containing:
            - sigma_annual: Annualized covariance matrix including the risk-free asset.
            - mu_annual: Annualized mean return vector including the risk-free asset.
            - sigma_weekly: Weekly covariance matrix including the risk-free asset.
            - mu_weekly: Weekly mean return vector including the risk-free asset.
        """
        logger.debug(
            "⏳ Generating annual Sigma and Mu parameter estimations for the optimization model."
        )
        # Compute the sample covariance matrix for the entire dataset
        sigma_weekly_np = np.atleast_2d(
            np.cov(data, rowvar=False, bias=True)
        )  # The sample covariance matrix

        # Add a shrinkage term (Ledoit--Wolf multiple of identity)
        sigma_weekly_np = MomentGenerator._ledoit_wolf_shrinkage(data, sigma_weekly_np)

        # Compute the mean return array for the entire dataset
        mu_weekly_np = np.mean(data, axis=0)

        # Convert the annual risk-free rate to a weekly rate
        risk_free_rate_weekly = (1 + risk_free_rate_annual) ** (1 / 52) - 1

        # Append the risk-free rate to the weekly mean return array
        mu_weekly_np = np.append(mu_weekly_np, risk_free_rate_weekly)

        # Append a row and column of zeros for the risk-free asset in the covariance matrix
        sigma_weekly_np = np.pad(sigma_weekly_np, ((0, 1), (0, 1)), "constant")

        # Convert numpy arrays to pandas DataFrame/Series and set appropriate asset names
        assets_with_rf = data.columns.tolist() + ["Cash"]
        sigma_weekly = pd.DataFrame(
            sigma_weekly_np, index=assets_with_rf, columns=assets_with_rf
        )
        mu_weekly = pd.Series(mu_weekly_np, index=assets_with_rf)

        # Annualize the covariance matrix and mean return array
        sigma_annual = sigma_weekly * 52
        mu_annual = mu_weekly.copy()
        mu_annual.iloc[:-1] = (
            mu_annual.iloc[:-1] * 52
        )  # Annualize only the risky assets

        return sigma_annual, mu_annual, sigma_weekly, mu_weekly


class ScenarioGenerator:
    """
    Provides methods for scenario generation.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    # ----------------------------------------------------------------------
    # Scenario Generation: THE MONTE CARLO METHOD
    # ----------------------------------------------------------------------
    def monte_carlo(
        self,
        data: pd.DataFrame,
        n_simulations: int,
        n_test: int,
        sigma_lst: list,
        mu_lst: list,
    ) -> np.ndarray:
        logger.info(
            f"⏳ Generating {n_simulations} scenarios for each investment period with Monte Carlo method"
        )

        n_iter = 4  # we work with 4-week periods
        n_indices = data.shape[1]
        n_rolls = math.floor(n_test / n_iter) + 1
        sim = np.zeros(
            (n_rolls * 4, n_simulations, n_indices), dtype=float
        )  # Match GAMS format

        # First generate the weekly simulations for each rolling period
        for p in range(int(n_rolls)):
            sigma = sigma_lst[p]
            mu = mu_lst[p]

            for week in range(n_iter * p, n_iter * p + n_iter):
                sim[week, :, :] = self.rng.multivariate_normal(
                    mean=mu, cov=sigma, size=n_simulations
                )

        # Now create the monthly (4-weeks) simulations for each rolling period
        monthly_sim = np.zeros((n_rolls, n_simulations, n_indices))
        for roll in range(n_rolls):
            roll_mult = roll * n_iter
            for s in range(n_simulations):
                for index in range(n_indices):
                    tmp_rets = 1 + sim[roll_mult : (roll_mult + n_iter), s, index]
                    monthly_sim[roll, s, index] = np.prod(tmp_rets) - 1

        return monthly_sim

    # ----------------------------------------------------------------------
    # Scenario Generation: THE BOOTSTRAPPING METHOD
    # ----------------------------------------------------------------------
    def bootstrapping(
        self, data: pd.DataFrame, n_simulations: int, n_test: int
    ) -> np.ndarray:
        logger.info(
            f"⏳ Generating {n_simulations} scenarios for each investment period with Bootstrapping method"
        )

        n_iter = 4  # 4 weeks compounded in our scenario
        n_train_weeks = len(data.index) - n_test
        n_indices = data.shape[1]
        n_simulations = n_simulations
        n_rolls = math.floor(n_test / n_iter) + 1

        sim = np.zeros((int(n_rolls), n_simulations, n_indices, n_iter), dtype=float)
        monthly_sim = np.ones(
            (
                int(n_rolls),
                n_simulations,
                n_indices,
            )
        )
        for p in range(int(n_rolls)):
            for s in range(n_simulations):
                for w in range(n_iter):
                    random_num = self.rng.integers(
                        n_iter * p, n_train_weeks + n_iter * p
                    )
                    sim[p, s, :, w] = data.iloc[random_num, :]
                    monthly_sim[p, s, :] *= 1 + sim[p, s, :, w]
                monthly_sim[p, s, :] += -1

        return monthly_sim

    def MC_simulation_annual_from_weekly(
        self,
        weekly_mu: pd.Series,
        weekly_sigma: pd.DataFrame,
        n_simulations: int,
        n_years: int,
        cash_return_annual: float = 0.02,
    ) -> np.ndarray:
        """
        Generates Monte Carlo simulations for annual returns based on provided weekly mu and sigma.
        Assumes 'Cash' or risk-free asset is already included and sets its annual return to a constant value.

        Parameters:
        - weekly_mu: Weekly mean returns as a pandas Series, including 'Cash'.
        - weekly_sigma: Weekly covariance matrix as a pandas DataFrame, including 'Cash'.
        - n_simulations: Number of simulations to generate.
        - n_years: Number of years to simulate.
        - cash_return_annual: Annual return rate of the 'Cash' or risk-free asset, default is 2%.

        Returns:
        - annual_simulations: An array of simulated annual returns (n_simulations, n_years, n_assets).
        """
        logger.debug(
            f"⏳ Simulating annual returns with Monte Carlo method based on weekly mu and weekly sigma. "
            f"We are generating {n_simulations} simulations for {n_years} years."
        )
        n_assets = len(weekly_mu)
        weeks_per_year = 52
        weekly_scenarios = np.zeros(
            (n_simulations, n_years * weeks_per_year, n_assets), dtype=float
        )

        # Generate weekly simulations
        for week in range(n_years * weeks_per_year):
            weekly_returns = self.rng.multivariate_normal(
                mean=weekly_mu.values, cov=weekly_sigma.values, size=n_simulations
            )
            weekly_scenarios[:, week, :] = weekly_returns

        # Convert weekly simulations to annual simulations
        annual_simulations = np.zeros((n_simulations, n_years, n_assets), dtype=float)
        for year in range(n_years):
            start_week = year * weeks_per_year
            end_week = (year + 1) * weeks_per_year
            # Accumulate weekly returns to get annual returns
            for simulation in range(n_simulations):
                # Convert weekly returns to cumulative product for each asset
                for asset in range(n_assets):
                    if (
                        weekly_mu.index[asset] == "Cash"
                    ):  # Assume 'Cash' represents the risk-free asset
                        # Set 'Cash' returns to a constant annual rate
                        annual_simulations[simulation, year, asset] = cash_return_annual
                    else:
                        annual_simulations[simulation, year, asset] = (
                            np.prod(
                                1
                                + weekly_scenarios[
                                    simulation, start_week:end_week, asset
                                ]
                            )
                            - 1
                        )

        return annual_simulations
