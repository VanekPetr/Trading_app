import os
from itertools import cycle
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from .Clustering import cluster, pick_cluster
from .CVaRmodel import cvar_model
from .CVaRtargets import get_cvar_targets
from .dataAnalyser import final_stats, mean_an_returns
from .lifecycle.glidePathCreator import generate_risk_profiles
from .lifecycle.MVOlifecycleModel import (
    get_port_allocations,
    riskadjust_model_scen,
)
from .MST import minimum_spanning_tree
from .MVOmodel import mvo_model
from .MVOtargets import get_mvo_targets
from .ScenarioGeneration import MomentGenerator, ScenarioGenerator

pio.renderers.default = "browser"

# that's unfortunate but will be addressed later
ROOT_DIR = Path(__file__).parent.parent
# Load our data
weekly_returns = pd.read_parquet(
    os.path.join(ROOT_DIR, "financial_data/all_etfs_rets.parquet.gzip")
)
tickers = [pair[0] for pair in weekly_returns.columns.values]
names = [pair[1] for pair in weekly_returns.columns.values]


class TradeBot:
    """
    Python class analysing financial products and based on machine learning algorithms and mathematical
    optimization suggesting optimal portfolio of assets.
    """

    def __init__(self):
        self.tickers = tickers
        self.names = names
        self.weeklyReturns = weekly_returns
        self.min_date = str(weekly_returns.index[0])
        self.max_date = str(weekly_returns.index[-1])

        weekly_returns.columns = tickers

    #def f(self, logger=None):
    #    logger = logger or logging.getLogger(__name__)
    #    logger.info("Hello World")
        
        
    @staticmethod
    def __plot_backtest(
        performance: pd.DataFrame,
        performance_benchmark: pd.DataFrame,
        composition: pd.DataFrame,
        names: list,
        tickers: list,
    ) -> Tuple[px.line, go.Figure]:
        """METHOD TO PLOT THE BACKTEST RESULTS"""

        performance.index = pd.to_datetime(performance.index.values, utc=True)

        # ** PERFORMANCE GRAPH **
        try:
            df_to_plot = pd.concat([performance, performance_benchmark], axis=1)
        except Exception:
            logger.warning("⚠️ Old data format.")
            performance.index = [
                date.date() for date in performance.index
            ]  # needed for old data
            df_to_plot = pd.concat([performance, performance_benchmark], axis=1)

        color_discrete_map = {
            "Portfolio_Value": "#21304f",
            "Benchmark_Value": "#f58f02",
        }
        fig = px.line(
            df_to_plot,
            x=df_to_plot.index,
            y=df_to_plot.columns,
            title="Comparison of different strategies",
            color_discrete_map=color_discrete_map,
        )
        fig_performance = fig

        # ** COMPOSITION GRAPH **
        # change ISIN to NAMES in allocation df
        composition_names = []
        for ticker in composition.columns:
            ticker_index = list(tickers).index(ticker)
            composition_names.append(list(names)[ticker_index])
        composition.columns = composition_names

        composition = composition.loc[:, (composition != 0).any(axis=0)]
        data = []
        idx_color = 0
        composition_color = (
            px.colors.sequential.turbid
            + px.colors.sequential.Brwnyl
            + px.colors.sequential.YlOrBr
            + px.colors.sequential.gray
            + px.colors.sequential.Mint
            + px.colors.sequential.dense
            + px.colors.sequential.Plasma
            + px.colors.sequential.Viridis
            + px.colors.sequential.Cividis
        )
        for isin in composition.columns:
            trace = go.Bar(
                x=composition.index,
                y=composition[isin],
                name=str(isin),
                marker_color=composition_color[
                    idx_color % len(composition_color)
                ],  # custom color
            )
            data.append(trace)
            idx_color += 1

        layout = go.Layout(barmode="stack")
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            title="Portfolio Composition",
            xaxis_title="Number of the Investment Period",
            yaxis_title="Composition",
            legend_title="Name of the Fund",
        )
        fig.layout.yaxis.tickformat = ",.1%"
        fig_composition = fig

        # Show figure if needed
        # fig.show()

        return fig_performance, fig_composition

    @staticmethod
    def __plot_portfolio_densities(
        portfolio_performance_dict: dict,
        compositions: Dict[str, pd.DataFrame],
        tickers: list,
        names: list,
    ) -> Tuple[go.Figure, Dict[str, go.Figure], go.Figure]:
        """METHOD TO PLOT THE LIFECYCLE SIMULATION RESULTS"""

        # Define colors
        colors = [
            "#99A4AE",  # gray50
            "#3b4956",  # dark
            "#b7ada5",  # secondary
            "#4099da",  # blue
            "#8ecdc8",  # aqua
            "#e85757",  # coral
            "#fdd779",  # sun
            "#644c76",  # eggplant
            "#D8D1CA",  # warmGray50
        ]

        color_cycle = cycle(colors)  # To cycle through colors
        fig = go.Figure()

        max_density_across_all_datasets = 0  # Initialize max density tracker

        for label, df in portfolio_performance_dict.items():
            # Kernel Density Estimation for each dataset
            kde = gaussian_kde(df["Terminal Wealth"])

            # Generating a range of values to evaluate the KDE
            x_min = df["Terminal Wealth"].min()
            x_max = df["Terminal Wealth"].max()
            x = np.linspace(x_min, x_max, 1000)

            # Evaluate the KDE
            density = kde(x)

            # Update max density if current density peak is higher
            max_density_across_all_datasets = max(
                max_density_across_all_datasets, max(density)
            )

            # Create line plot trace for this dataset
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=density,
                    mode="lines",
                    name=label,  # Use the dictionary key as the label
                    line=dict(
                        width=2.5, color=next(color_cycle)
                    ),  # Assign color from Orsted-Colors
                )
            )

        # Add a dashed vertical line at x=0
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=max_density_across_all_datasets,  # Use the max density across all datasets
            line=dict(
                color="Black",
                width=3,
                dash="dash",  # Define dash pattern
            ),
        )
        """
        # Update the layout
        fig.update_layout(
            title_text='Density function(s) of terminal wealth for risk classes in 1000 different scenarios.',
            xaxis_title='Terminal Wealth',
            yaxis_title='Density',
            legend_title='Risk Class',
            template='plotly_white'
        )
        """
        # Update the layout with larger fonts
        fig.update_layout(
            title_text="Density function(s) of the end portfolio value for various glide paths.",
            title_font=dict(size=24),  # Increase title font size
            xaxis_title="Target date portfolio value",
            xaxis_title_font=dict(size=18),  # Increase x-axis title font size
            xaxis_tickfont=dict(size=16),  # Increase x-axis tick label font size
            yaxis_title="Density",
            yaxis_title_font=dict(size=18),  # Increase y-axis title font size
            yaxis_tickfont=dict(size=16),  # Increase y-axis tick label font size
            legend_title="Risb Budget glide path",
            legend_title_font=dict(size=18),  # Increase legend title font size
            legend_font=dict(size=16),  # Increase legend text font size
            template="plotly_white",
        )

        # Show the figure in a browser
        # fig.show(renderer="browser")

        composition_figures = {}
        filtered_compositions = {
            name: comp for name, comp in compositions.items() if "reverse" not in name
        }
        num_portfolios = len(filtered_compositions)
        cols = 2 if num_portfolios > 1 else 1
        rows = ceil(
            num_portfolios / cols
        )  # Calculate the number of rows needed based on the total number of compositions

        subplot_titles = [
            f"Portfolio Composition: {name}" for name in filtered_compositions.keys()
        ]
        fig_subplots = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

        tickers_in_legend = set()
        current_plot = (
            1  # Keep track of the current plot index to correctly calculate row and col
        )

        for portfolio_name, composition in filtered_compositions.items():
            composition_names = []
            for ticker in composition.columns[:-1]:
                ticker_index = list(tickers).index(ticker)
                composition_names.append(list(names)[ticker_index])
            if "Cash" not in composition_names:
                composition_names.append("Cash")
            composition.columns = composition_names
            composition = composition.loc[:, (composition != 0).any(axis=0)]

            idx_color = 0
            composition_color = (
                px.colors.sequential.turbid
                + px.colors.sequential.Brwnyl
                + px.colors.sequential.YlOrBr
                + px.colors.sequential.gray
                + px.colors.sequential.Mint
                + px.colors.sequential.dense
                + px.colors.sequential.Plasma
                + px.colors.sequential.Viridis
                + px.colors.sequential.Cividis
            )

            # Create an individual figure for the current portfolio
            individual_fig = go.Figure()

            for isin in composition.columns:
                show_legend = isin not in tickers_in_legend
                tickers_in_legend.add(isin)

                trace = go.Bar(
                    x=composition.index,
                    y=composition[isin],
                    name=str(isin),
                    marker_color=composition_color[idx_color % len(composition_color)],
                    showlegend=show_legend,
                )

                # Add trace to both the subplot and the individual figure
                row, col = divmod(current_plot - 1, cols)
                fig_subplots.add_trace(trace, row=row + 1, col=col + 1)
                individual_fig.add_trace(trace)

                idx_color += 1

            # Configure the individual figure layout
            individual_fig.update_layout(
                title=f"Portfolio Composition: {portfolio_name}",
                plot_bgcolor="white",
                barmode="stack",
            )
            individual_fig["layout"]["yaxis"].tickformat = ",.1%"

            # Store the individual figure in the dictionary
            composition_figures[portfolio_name] = individual_fig

            current_plot += 1

        fig_subplots.update_layout(
            title="Portfolio Compositions",
            height=500 * rows,
            width=1000 * cols,
            plot_bgcolor="white",
            barmode="stack",
        )
        # Update y-axis tick format for all subplots
        for i in range(1, cols * rows + 1):
            fig_subplots["layout"][f"yaxis{i}"].tickformat = ",.1%"

        # fig_subplots.show()

        return fig, composition_figures, fig_subplots

    def get_stat(self, start_date: str, end_date: str) -> pd.DataFrame:
        """METHOD COMPUTING ANNUAL RETURNS, ANNUAL STD. DEV. & SHARPE RATIO OF ASSETS"""

        # ANALYZE THE DATA for a given time period
        weekly_data = self.weeklyReturns[
            (self.weeklyReturns.index >= start_date)
            & (self.weeklyReturns.index <= end_date)
        ].copy()

        # Create table with summary statistics
        mu_ga = mean_an_returns(weekly_data)  # Annualised geometric mean of returns
        std_dev_a = weekly_data.std(axis=0) * np.sqrt(
            52
        )  # Annualised standard deviation of returns
        sharpe = round(mu_ga / std_dev_a, 2)  # Sharpe ratio of each financial product

        # Write all results into a data frame
        stat_df = pd.concat([mu_ga, std_dev_a, sharpe], axis=1)
        stat_df.columns = [
            "Average Annual Returns",
            "Standard Deviation of Returns",
            "Sharpe Ratio",
        ]
        stat_df["ISIN"] = stat_df.index  # Add names into the table
        stat_df["Name"] = self.names
        stat_df["Size"] = 1
        stat_df["Type"] = "ETF"

        return stat_df

    def get_top_performing_assets(
        self, time_periods: List[Tuple[str, str]], top_percent: float = 0.2
    ) -> List[str]:
        stats_for_periods = {
            f"period_{i}": self.get_stat(*period)
            for i, period in enumerate(time_periods, 1)
        }

        # Create 'Risk class' column where the value is
        # 'Risk Class 1' if Standard Deviation of Returns <= 0.005
        # 'Risk Class 2' if > 0.005 and < 0.02
        # 'Risk Class 3' if > 0.02 and < 0.05
        # 'Risk Class 4' if > 0.05 and < 0.1
        # 'Risk Class 5' if > 0.1 and < 0.15
        # 'Risk Class 6' if > 0.15 and < 0.25 then
        # 'Risk Class 7' if > 0.25
        risk_level = {
            "Risk Class 1": 0.005,
            "Risk Class 2": 0.02,
            "Risk Class 3": 0.05,
            "Risk Class 4": 0.10,
            "Risk Class 5": 0.15,
            "Risk Class 6": 0.25,
            "Risk Class 7": 1,
        }
        for data in stats_for_periods.values():
            data["Risk Class"] = pd.cut(
                data["Standard Deviation of Returns"],
                bins=[-1] + list(risk_level.values()),
                labels=list(risk_level.keys()),
                right=True,
            )
        # For each data_period and each risk class, find the top 20% best performing assets
        # mark them as True in column 'Top Performer'
        for data in stats_for_periods.values():
            for risk_class in risk_level.keys():
                data.loc[
                    data["Risk Class"] == risk_class,
                    "Top Performer",
                ] = data.loc[
                    data["Risk Class"] == risk_class, "Sharpe Ratio"
                ].rank(pct=True) > (1 - top_percent)
        # for each period, save the pandas dataframe into excel files
        # for index, data in enumerate(stats_for_periods.values()):
        #     data.to_excel(f"top_performers_{time_periods[index]}.xlsx")

        # ISIN codes for assets which were top performers in all n periods
        top_isins = (
            stats_for_periods["period_1"]
            .loc[stats_for_periods["period_1"]["Top Performer"], "ISIN"]
            .values
        )
        for data in stats_for_periods.values():
            top_isins = np.intersect1d(
                top_isins, data.loc[data["Top Performer"], "ISIN"].values
            )

        top_names = [self.names[self.tickers.index(isin)] for isin in top_isins]

        return top_names

    def plot_dots(
        self,
        start_date: str,
        end_date: str,
        ml: str = "",
        ml_subset: Union[list, pd.DataFrame] = None,
        fund_set: Union[list, None] = None,
        top_performers: Union[list, None] = None,
        optimal_portfolio: Union[list, None] = None,
        benchmark: Union[list, None] = None,
    ) -> px.scatter:
        """METHOD TO PLOT THE OVERVIEW OF THE FINANCIAL PRODUCTS IN TERMS OF RISK AND RETURNS"""
        fund_set = fund_set if fund_set else []
        top_performers = top_performers if top_performers else []

        # Get statistics for a given time period
        data = self.get_stat(start_date, end_date)

        # Add data about the optimal portfolio and benchmark for plotting
        if optimal_portfolio:
            data.loc[optimal_portfolio[4]] = optimal_portfolio
        if benchmark:
            data.loc[benchmark[4]] = benchmark

        # IF WE WANT TO HIGHLIGHT THE SUBSET OF ASSETS BASED ON ML
        if ml == "MST":
            data.loc[:, "Type"] = "Funds"
            for fund in ml_subset:
                data.loc[fund, "Type"] = "MST subset"
        if ml == "Clustering":
            data.loc[:, "Type"] = ml_subset.loc[:, "Cluster"]

        # If selected any fund for comparison
        for fund in fund_set:
            isin_idx = list(self.names).index(fund)
            data.loc[self.tickers[isin_idx], "Type"] = str(
                data.loc[self.tickers[isin_idx], "Name"]
            )
            data.loc[self.tickers[isin_idx], "Size"] = 3

        for fund in top_performers:
            isin_idx = list(self.names).index(fund)
            data.loc[self.tickers[isin_idx], "Type"] = "Top Performer"
            data.loc[self.tickers[isin_idx], "Size"] = 3

        # PLOTTING Data
        color_discrete_map = {
            "ETF": "#21304f",
            "Mutual Fund": "#f58f02",
            "Funds": "#21304f",
            "MST subset": "#f58f02",
            "Top Performer": "#f58f02",
            "Cluster 1": "#21304f",
            "Cluster 2": "#f58f02",
            "Benchmark Portfolio": "#f58f02",
            "Optimal Portfolio": "olive",
        }
        fig = px.scatter(
            data,
            x="Standard Deviation of Returns",
            y="Average Annual Returns",
            color="Type",
            size="Size",
            size_max=8,
            hover_name="Name",
            hover_data={"Sharpe Ratio": True, "ISIN": True, "Size": False},
            color_discrete_map=color_discrete_map,
            title="Annual Returns and Standard Deviation of Returns from "
            + start_date[:10]
            + " to "
            + end_date[:10],
        )

        # AXIS IN PERCENTAGES
        fig.layout.yaxis.tickformat = ",.1%"
        fig.layout.xaxis.tickformat = ",.1%"

        # RISK LEVEL MARKER
        min_risk = data["Standard Deviation of Returns"].min()
        max_risk = data["Standard Deviation of Returns"].max()
        risk_level = {
            "Risk Class 1": 0.005,
            "Risk Class 2": 0.02,
            "Risk Class 3": 0.05,
            "Risk Class 4": 0.10,
            "Risk Class 5": 0.15,
            "Risk Class 6": 0.25,
            "Risk Class 7": max_risk,
        }
        # Initialize dynamic risk levels
        actual_risk_level = set()
        for i in range(1, 8):
            k = "Risk Class " + str(i)
            if (risk_level[k] >= min_risk) and (risk_level[k] <= max_risk):
                actual_risk_level.add(i)

        if max(actual_risk_level) < 7:
            actual_risk_level.add(
                max(actual_risk_level) + 1
            )  # Add the final risk level

        for level in actual_risk_level:
            k = "Risk Class " + str(level)
            fig.add_vline(
                x=risk_level[k], line_width=1, line_dash="dash", line_color="#7c90a0"
            )  # annotation_text=k, annotation_position="top left")
            fig.add_annotation(
                x=risk_level[k] - 0.01,
                y=max(data["Average Annual Returns"]),
                text=k,
                textangle=-90,
                showarrow=False,
            )

        # RETURN LEVEL MARKER
        fig.add_hline(y=0, line_width=1.5, line_color="rgba(233, 30, 99, 0.5)")

        # TITLES
        fig.update_annotations(font_color="#000000")
        fig.update_layout(
            xaxis_title="Annualised standard deviation of returns (Risk)",
            yaxis_title="Annualised average returns",
        )
        # Position of legend
        fig.update_layout(legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01))
        # fig.show()
        return fig

    def mst(
        self, start_date: str, end_date: str, n_mst_runs: int, plot: bool = False
    ) -> Tuple[Union[None, px.scatter], list]:
        """METHOD TO RUN MST METHOD AND PRINT RESULTS"""
        fig, subset_mst = None, []

        # Starting subset of data for MST
        subset_mst_df = self.weeklyReturns[
            (self.weeklyReturns.index >= start_date)
            & (self.weeklyReturns.index <= end_date)
        ].copy()

        for i in range(n_mst_runs):
            subset_mst, subset_mst_df, corr_mst_avg, pdi_mst = minimum_spanning_tree(
                subset_mst_df
            )

        # PLOTTING RESULTS
        if plot and len(subset_mst) > 0:
            end_df_date = str(subset_mst_df.index.date[-1])
            fig = self.plot_dots(
                start_date=start_date,
                end_date=end_df_date,
                ml="MST",
                ml_subset=subset_mst,
            )

        return fig, subset_mst

    def clustering(
        self,
        start_date: str,
        end_date: str,
        n_clusters: int,
        n_assets: int,
        plot: bool = False,
    ) -> Tuple[Union[None, px.scatter], list]:
        """
        METHOD TO RUN MST METHOD AND PRINT RESULTS
        """
        fig = None
        dataset = self.weeklyReturns[
            (self.weeklyReturns.index >= start_date)
            & (self.weeklyReturns.index <= end_date)
        ].copy()
        # CLUSTER DATA
        clusters = cluster(dataset, n_clusters)

        # SELECT ASSETS
        end_dataset_date = str(dataset.index.date[-1])
        clustering_stats = self.get_stat(start_date, end_dataset_date)
        subset_clustering, subset_clustering_df = pick_cluster(
            data=dataset, stat=clustering_stats, ml=clusters, n_assets=n_assets
        )  # Number of assets from each cluster

        # PLOTTING DATA
        if plot:
            fig = self.plot_dots(
                start_date=start_date,
                end_date=end_dataset_date,
                ml="Clustering",
                ml_subset=clusters,
            )

            # fig.show()

        return fig, subset_clustering

    def backtest(
        self,
        start_train_date: str,
        start_test_date: str,
        end_test_date: str,
        subset_of_assets: list,
        benchmarks: list,
        scenarios_type: str,
        n_simulations: int,
        model: str,
        solver: str = "CLARABEL",
        lower_bound: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, px.line, go.Figure]:
        """METHOD TO COMPUTE THE BACKTEST"""

        # Find Benchmarks' ISIN codes
        benchmark_isin = [
            self.tickers[list(self.names).index(name)] for name in benchmarks
        ]

        # Get train and testing datasets
        whole_dataset = self.weeklyReturns[
            (self.weeklyReturns.index >= start_train_date)
            & (self.weeklyReturns.index <= end_test_date)
        ].copy()
        test_dataset = self.weeklyReturns[
            (self.weeklyReturns.index > start_test_date)
            & (self.weeklyReturns.index <= end_test_date)
        ].copy()

        # SCENARIO GENERATION
        # ---------------------------------------------------------------------------------------------------
        # Create scenario generator
        sg = ScenarioGenerator(np.random.default_rng())

        if model == "Markowitz model" or scenarios_type == "MonteCarlo":
            sigma_lst, mu_lst = MomentGenerator.generate_sigma_mu_for_test_periods(
                data=whole_dataset[subset_of_assets], n_test=len(test_dataset.index)
            )

        if scenarios_type == "MonteCarlo":
            scenarios = sg.monte_carlo(
                data=whole_dataset[subset_of_assets],  # subsetMST_df or subsetCLUST_df
                n_simulations=n_simulations,
                n_test=len(test_dataset.index),
                sigma_lst=sigma_lst,
                mu_lst=mu_lst,
            )
        else:
            scenarios = sg.bootstrapping(
                data=whole_dataset[subset_of_assets],  # subsetMST or subsetCLUST
                n_simulations=n_simulations,  # number of scenarios per period
                n_test=len(test_dataset.index),
            )  # number of periods

        # TARGETS GENERATION
        # ---------------------------------------------------------------------------------------------------
        start_of_test_dataset = str(test_dataset.index.date[0])
        if model == "Markowitz model":
            targets, benchmark_port_val = get_mvo_targets(
                test_date=start_of_test_dataset,
                benchmark=benchmark_isin,
                budget=100,
                data=whole_dataset,
            )

        else:
            targets, benchmark_port_val = get_cvar_targets(
                test_date=start_of_test_dataset,
                benchmark=benchmark_isin,
                budget=100,
                cvar_alpha=0.05,
                data=whole_dataset,
                scgen=sg,
                n_simulations=n_simulations,
            )

        # MATHEMATICAL MODELING
        # ---------------------------------------------------------------------------------------------------
        if model == "Markowitz model":
            port_allocation, port_value, port_cvar = mvo_model(
                test_ret=test_dataset[subset_of_assets],
                mu_lst=mu_lst,
                sigma_lst=sigma_lst,
                targets=targets,
                budget=100,
                trans_cost=0.001,
                max_weight=1,
                solver=solver,
                lower_bound=lower_bound,
            )
        #                                                      inaccurate=inaccurate_solution)

        else:
            port_allocation, port_value, port_cvar = cvar_model(
                test_ret=test_dataset[subset_of_assets],
                scenarios=scenarios,  # Scenarios
                targets=targets,  # Target
                budget=100,
                cvar_alpha=0.05,
                trans_cost=0.001,
                max_weight=1,
                solver=solver,
                lower_bound=lower_bound,
            )
        #                                                       inaccurate=inaccurate_solution)

        # PLOTTING
        # ------------------------------------------------------------------
        fig_performance, fig_composition = self.__plot_backtest(
            performance=port_value.copy(),
            performance_benchmark=benchmark_port_val.copy(),
            composition=port_allocation,
            names=self.names,
            tickers=self.tickers,
        )

        # RETURN STATISTICS
        # ------------------------------------------------------------------
        optimal_portfolio_stat = final_stats(port_value)
        benchmark_stat = final_stats(benchmark_port_val)

        return optimal_portfolio_stat, benchmark_stat, fig_performance, fig_composition

    def lifecycle_scenario_analysis(
        self,
        subset_of_assets: list,
        scenarios_type: str,
        n_simulations: int,
        end_year: int,
        withdrawals: int,
        initial_risk_appetite: float,
        initial_budget: int,
        rng_seed=0,
        test_split: float = False,
    ) -> Tuple[dict, pd.DataFrame, go.Figure, go.Figure, dict, dict, go.Figure]:
        """METHOD TO COMPUTE THE LIFECYCLE SCENARIO ANALYSIS"""

        # ------------------------------- INITIALIZE FUNCTION -------------------------------
        n_periods = end_year - 2023
        withdrawal_lst = [withdrawals * (1 + 0.04) ** i for i in range(n_periods)]

        # ------------------------------- PARAMETER INITIALIZATION -------------------------------
        if test_split != 0:
            sampling_set, estimating_set = MomentGenerator.split_dataset(
                data=self.weeklyReturns[subset_of_assets], sampling_ratio=test_split
            )

            _, _, sigma_weekly, mu_weekly = (
                MomentGenerator.generate_annual_sigma_mu_with_risk_free(
                    data=sampling_set
                )
            )

            sigma, mu, _, _ = MomentGenerator.generate_annual_sigma_mu_with_risk_free(
                data=estimating_set
            )
        else:
            sigma, mu, sigma_weekly, mu_weekly = (
                MomentGenerator.generate_annual_sigma_mu_with_risk_free(
                    data=self.weeklyReturns[subset_of_assets]
                )
            )

        # ------------------------------- SCENARIO GENERATION -------------------------------
        if rng_seed == 0:
            sg = ScenarioGenerator(np.random.default_rng())
        else:
            sg = ScenarioGenerator(np.random.default_rng(rng_seed))

        if scenarios_type == "MonteCarlo":
            scenarios = sg.MC_simulation_annual_from_weekly(
                weekly_mu=mu_weekly,
                weekly_sigma=sigma_weekly,
                n_simulations=n_simulations,
                n_years=n_periods,
            )

        elif scenarios_type == "Bootstrap":
            scenarios = sg.bootstrap_simulation_annual_from_weekly(
                historical_weekly_returns=self.weeklyReturns[subset_of_assets],
                n_simulations=n_simulations,
                n_years=n_periods,
            )

        else:
            raise ValueError(
                "It appears that a scenario method other than MonteCarlo or Bootstrap has been chosen. "
                "Please check for spelling mistakes."
            )

        # ------------------------------- Allocation Target Generation -------------------------------
        glide_paths_df, fig_glidepaths = generate_risk_profiles(
            n_periods=n_periods, initial_risk=initial_risk_appetite, minimum_risk=0.01
        )

        allocation_targets = {}
        for r in glide_paths_df.columns:
            targets = get_port_allocations(
                mu_lst=mu,
                sigma_lst=sigma,
                targets=glide_paths_df[r],
                max_weight=1 / 4,
                solver="CLARABEL",
            )
            allocation_targets[f"{r}"] = targets

        # ------------------------------- MATHEMATICAL MODELING -------------------------------
        exhibition_summary = pd.DataFrame()
        terminal_wealth_dict = {}

        for key, df in allocation_targets.items():
            logger.info(
                f"Optimizing portfolio for {key} over {n_simulations} scenarios. An info message will "
                f"appear, when we are halfway through the scenarios for the current strategy."
            )
            portfolio_df, mean_allocations_df, analysis_metrics = riskadjust_model_scen(
                scen=scenarios[:, :, :],
                targets=df,
                budget=initial_budget,
                trans_cost=0.002,
                withdrawal_lst=withdrawal_lst,
                interest_rate=0.04,
            )

            # Add the analysis_metrics DataFrame as a new column in the storage DataFrame
            exhibition_summary[key] = analysis_metrics.squeeze()

            portfolio_df["Terminal Wealth"] = pd.to_numeric(
                portfolio_df["Terminal Wealth"], errors="coerce"
            )
            terminal_wealth_dict[f"{key}"] = portfolio_df

        # ------------------------------- PLOTTING -------------------------------
        fig_performance, fig_compositions, fig_compositions_all = (
            self.__plot_portfolio_densities(
                portfolio_performance_dict=terminal_wealth_dict,
                compositions=allocation_targets,
                tickers=self.tickers,
                names=self.names,
            )
        )

        # ------------------------------- RETURN STATISTICS -------------------------------
        return (
            terminal_wealth_dict,
            exhibition_summary,
            fig_performance,
            fig_glidepaths,
            allocation_targets,
            fig_compositions,
            fig_compositions_all,
        )


if __name__ == "__main__":
    # INITIALIZATION OF THE CLASS
    algo = TradeBot()

    # Get top performing assets for given periods and measure
    top_assets = algo.get_top_performing_assets(
        time_periods=[
            (algo.min_date, "2017-01-01"),
            ("2017-01-02", "2020-01-01"),
            ("2020-01-02", algo.max_date),
        ],
        top_percent=0.2,
    )

    # PLOT INTERACTIVE GRAPH
    algo.plot_dots(
        start_date=algo.min_date, end_date=algo.max_date, top_performers=top_assets
    )

    # RUN THE MINIMUM SPANNING TREE METHOD
    _, mst_subset_of_assets = algo.mst(
        start_date="2000-01-01", end_date="2024-01-01", n_mst_runs=5, plot=False
    )

    # RUN THE CLUSTERING METHOD
    _, clustering_subset_of_assets = algo.clustering(
        start_date="2015-12-23",
        end_date="2017-07-01",
        n_clusters=3,
        n_assets=10,
        plot=True,
    )

    # RUN THE LIFECYCLE
    lifecycle = algo.lifecycle_scenario_analysis(
        subset_of_assets=mst_subset_of_assets,
        scenarios_type="MonteCarlo",
        n_simulations=1000,
        end_year=2050,
        withdrawals=51000,
        initial_risk_appetite=0.15,
        initial_budget=137000,
    )

    # RUN THE BACKTEST
    backtest = algo.backtest(
        start_train_date="2015-12-23",
        start_test_date="2018-09-24",
        end_test_date="2019-09-01",
        subset_of_assets=mst_subset_of_assets,
        benchmarks=["BankInvest Danske Aktier W"],
        scenarios_type="Bootstrapping",
        n_simulations=500,
        model="Markowitz model",
        lower_bound=0,
    )
