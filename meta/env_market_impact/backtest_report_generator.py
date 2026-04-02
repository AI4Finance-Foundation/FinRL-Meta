"""
Backtest Report Generator

A class-based report generator for creating interactive HTML reports from backtest results.
Generates comprehensive visualizations and performance metrics tables.

Usage:
    # Full HTML report
    gen = BacktestReportGenerator("path/to/summary.json")
    gen.generate_report()

    # Single chart in a Jupyter notebook
    from backtest_report_generator import BacktestReportGenerator, Chart
    gen = BacktestReportGenerator("path/to/summary.json")
    gen.make_chart(Chart.PORTFOLIO_VALUE)
"""

import json
import math
import os
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .envs.utils import compute_performance_stats
from .envs.utils import get_logger
from .envs.utils import rolling_sharpe

log = get_logger()


class Chart(str, Enum):
    """Available charts for the backtest report."""

    PORTFOLIO_VALUE = "portfolio_value"
    TURNOVER = "turnover"
    CUMULATIVE_COST = "cumulative_cost"
    ROLLING_SHARPE = "rolling_sharpe"
    DRAWDOWN = "drawdown"
    TRADING_VOLUME = "trading_volume"
    PORTFOLIO_VALUE_BLANK_SLATE = "portfolio_value_blank_slate"
    ROLLING_SHARPE_BLANK_SLATE = "rolling_sharpe_blank_slate"
    TRADING_VOLUME_BLANK_SLATE = "trading_volume_blank_slate"
    DRAWDOWN_BLANK_SLATE = "drawdown_blank_slate"
    POV_HISTOGRAM_TRAIN = "pov_histogram_train"
    POV_HISTOGRAM_TEST = "pov_histogram_test"
    POV_HISTOGRAM_BLANK_SLATE = "pov_histogram_blank_slate"
    POV_TIME_SERIES = "pov_time_series"
    POV_TIME_SERIES_BLANK_SLATE = "pov_time_series_blank_slate"
    EPOCH_STATS = "epoch_stats"
    EPOCH_ANNUALIZED_RETURN = "epoch_annualized_return"
    EPOCH_ANNUALIZED_SHARPE = "epoch_annualized_sharpe"
    EPOCH_ANNUALIZED_VOLATILITY = "epoch_annualized_volatility"
    EPOCH_SORTINO = "epoch_sortino"
    EPOCH_CALMAR = "epoch_calmar"
    EPOCH_MAX_DRAWDOWN = "epoch_max_drawdown"
    EPOCH_AVG_TURNOVER = "epoch_avg_turnover"
    EPOCH_AVG_COST = "epoch_avg_cost"
    EPOCH_AVG_POV = "epoch_avg_pov"
    EPOCH_AVG_REWARD = "epoch_avg_reward"
    PERFORMANCE_TABLE = "performance_table"


class BacktestReportGenerator:
    """
    Generates interactive HTML reports from backtest summary JSON files.

    All data is loaded and parsed in the constructor.  Individual charts can
    be obtained as standalone ``plotly.graph_objects.Figure`` objects via
    :meth:`make_chart`, or the complete report can be written to an HTML file
    via :meth:`generate_report`.
    """

    colors = plotly.colors.qualitative.Plotly
    blue_colors = plotly.colors.sequential.Blues
    red_colors = plotly.colors.sequential.Reds

    _CHART_HEIGHTS = {
        Chart.PORTFOLIO_VALUE: 500,
        Chart.TURNOVER: 500,
        Chart.CUMULATIVE_COST: 500,
        Chart.ROLLING_SHARPE: 500,
        Chart.DRAWDOWN: 500,
        Chart.TRADING_VOLUME: 500,
        Chart.PORTFOLIO_VALUE_BLANK_SLATE: 500,
        Chart.ROLLING_SHARPE_BLANK_SLATE: 500,
        Chart.TRADING_VOLUME_BLANK_SLATE: 500,
        Chart.DRAWDOWN_BLANK_SLATE: 500,
        Chart.POV_HISTOGRAM_TRAIN: 500,
        Chart.POV_HISTOGRAM_TEST: 500,
        Chart.POV_HISTOGRAM_BLANK_SLATE: 500,
        Chart.POV_TIME_SERIES: 500,
        Chart.POV_TIME_SERIES_BLANK_SLATE: 500,
        Chart.EPOCH_STATS: 1400,
        Chart.EPOCH_ANNUALIZED_RETURN: 400,
        Chart.EPOCH_ANNUALIZED_SHARPE: 400,
        Chart.EPOCH_ANNUALIZED_VOLATILITY: 400,
        Chart.EPOCH_SORTINO: 400,
        Chart.EPOCH_CALMAR: 400,
        Chart.EPOCH_MAX_DRAWDOWN: 400,
        Chart.EPOCH_AVG_TURNOVER: 400,
        Chart.EPOCH_AVG_COST: 400,
        Chart.EPOCH_AVG_POV: 400,
        Chart.EPOCH_AVG_REWARD: 400,
        Chart.PERFORMANCE_TABLE: 1000,
    }

    _CHART_BUILDERS: dict[Chart, str] = {
        Chart.PORTFOLIO_VALUE: "_make_portfolio_value",
        Chart.TURNOVER: "_make_turnover",
        Chart.CUMULATIVE_COST: "_make_cumulative_cost",
        Chart.ROLLING_SHARPE: "_make_rolling_sharpe",
        Chart.DRAWDOWN: "_make_drawdown",
        Chart.TRADING_VOLUME: "_make_trading_volume",
        Chart.PORTFOLIO_VALUE_BLANK_SLATE: "_make_portfolio_value_blank_slate",
        Chart.ROLLING_SHARPE_BLANK_SLATE: "_make_rolling_sharpe_blank_slate",
        Chart.TRADING_VOLUME_BLANK_SLATE: "_make_trading_volume_blank_slate",
        Chart.DRAWDOWN_BLANK_SLATE: "_make_drawdown_blank_slate",
        Chart.POV_HISTOGRAM_TRAIN: "_make_pov_histogram_train",
        Chart.POV_HISTOGRAM_TEST: "_make_pov_histogram_test",
        Chart.POV_HISTOGRAM_BLANK_SLATE: "_make_pov_histogram_blank_slate",
        Chart.POV_TIME_SERIES: "_make_pov_time_series",
        Chart.POV_TIME_SERIES_BLANK_SLATE: "_make_pov_time_series_blank_slate",
        Chart.EPOCH_STATS: "_make_epoch_stats",
        Chart.EPOCH_ANNUALIZED_RETURN: "_make_epoch_annualized_return",
        Chart.EPOCH_ANNUALIZED_SHARPE: "_make_epoch_annualized_sharpe",
        Chart.EPOCH_ANNUALIZED_VOLATILITY: "_make_epoch_annualized_volatility",
        Chart.EPOCH_SORTINO: "_make_epoch_sortino",
        Chart.EPOCH_CALMAR: "_make_epoch_calmar",
        Chart.EPOCH_MAX_DRAWDOWN: "_make_epoch_max_drawdown",
        Chart.EPOCH_AVG_TURNOVER: "_make_epoch_avg_turnover",
        Chart.EPOCH_AVG_COST: "_make_epoch_avg_cost",
        Chart.EPOCH_AVG_POV: "_make_epoch_avg_pov",
        Chart.EPOCH_AVG_REWARD: "_make_epoch_avg_reward",
        Chart.PERFORMANCE_TABLE: "_make_performance_table",
    }

    _REPORT_CHARTS = [
        Chart.PORTFOLIO_VALUE,
        Chart.TURNOVER,
        Chart.CUMULATIVE_COST,
        Chart.ROLLING_SHARPE,
        Chart.DRAWDOWN,
        Chart.TRADING_VOLUME,
        Chart.PORTFOLIO_VALUE_BLANK_SLATE,
        Chart.ROLLING_SHARPE_BLANK_SLATE,
        Chart.TRADING_VOLUME_BLANK_SLATE,
        Chart.DRAWDOWN_BLANK_SLATE,
        Chart.POV_HISTOGRAM_TRAIN,
        Chart.POV_HISTOGRAM_TEST,
        Chart.POV_HISTOGRAM_BLANK_SLATE,
        Chart.POV_TIME_SERIES,
        Chart.POV_TIME_SERIES_BLANK_SLATE,
        Chart.EPOCH_STATS,
        Chart.PERFORMANCE_TABLE,
    ]

    _MAX_TRACE_NAME_LEN = 90
    _MAX_SHORT_NAME_LEN = 60

    def __init__(
        self,
        summary_json_path: str,
        filters: Optional[dict] = None,
        exclude_filters: Optional[dict] = None,
        display_keys: Optional[list[str]] = None,
    ):
        """
        Load and parse all backtest data from a summary JSON file.

        Args:
            summary_json_path: Path to the summary JSON file containing
                backtest metadata.
            filters: Optional dict of ``{key: value}`` pairs.  Only backtests
                whose metadata matches every filter are kept.
            exclude_filters: Optional dict of ``{key: value}`` pairs.
                Backtests whose metadata matches **all** entries are excluded.
            display_keys: Optional list of metadata keys to include in trace
                names and table labels.  Overrides the automatic varying-key
                detection when provided.
        """
        log.info(f"Loading backtest data from {summary_json_path}")
        self._summary_dir = os.path.dirname(summary_json_path)

        with open(summary_json_path, "r") as f:
            summary_data = json.load(f)

        backtests_metadata = summary_data["backtests"]
        if filters:
            prev_count = len(backtests_metadata)
            backtests_metadata = [
                m
                for m in backtests_metadata
                if all(m.get(k) == v for k, v in filters.items())
            ]
            log.info(f"{len(backtests_metadata)} filtered backtests of {prev_count}")
        if exclude_filters:
            prev_count = len(backtests_metadata)
            backtests_metadata = [
                m
                for m in backtests_metadata
                if not all(m.get(k) == v for k, v in exclude_filters.items())
            ]
            log.info(
                f"{len(backtests_metadata)} backtests after exclusion filter ({prev_count - len(backtests_metadata)} excluded)"
            )

        self._benchmark_ticker: str = summary_data["benchmark_ticker"]
        if display_keys is not None:
            self._varying_keys: set[str] = set(display_keys)
        else:
            self._varying_keys: set[str] = self._get_varying_param_keys(
                backtests_metadata
            )
        self._split_date = None

        self._backtests: list[dict] = []
        benchmark_caps_seen: set = set()

        for i, metadata in enumerate(backtests_metadata):
            (
                df_train,
                df_test,
                df_test_blank,
                trades_train,
                trades_test,
                trades_blank,
            ) = self._load_and_prepare_data(
                metadata["results_csv_train"],
                metadata["results_csv_test"],
                metadata["results_csv_test_blank"],
                metadata.get("trades_csv_train"),
                metadata.get("trades_csv_test"),
                metadata.get("trades_csv_test_blank"),
            )

            rgb = self._get_color_info(i)
            trace_name = self._build_trace_name(metadata, self._varying_keys)
            short_name = self._build_short_name(metadata, self._varying_keys)

            train_metrics = self._calculate_performance_metrics(df_train, trades_train)
            test_metrics = self._calculate_performance_metrics(df_test, trades_test)
            blank_metrics = self._calculate_performance_metrics(
                df_test_blank, trades_blank
            )

            capital = metadata["initial_capital"]
            add_benchmark = capital not in benchmark_caps_seen
            if add_benchmark:
                benchmark_caps_seen.add(capital)

            self._backtests.append(
                {
                    "metadata": metadata,
                    "df_train": df_train,
                    "df_test": df_test,
                    "df_test_blank": df_test_blank,
                    "trades_train": trades_train,
                    "trades_test": trades_test,
                    "trades_blank": trades_blank,
                    "trace_name": trace_name,
                    "short_name": short_name,
                    "rgb": rgb,
                    "index": i,
                    "epoch_stats": {
                        "train": metadata.get("epoch_stats_train", []),
                        "test_blank": metadata.get("epoch_stats_test_blank", []),
                    },
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "blank_metrics": blank_metrics,
                    "add_benchmark": add_benchmark,
                    "capital": capital,
                }
            )

            if self._split_date is None and not df_train.empty:
                self._split_date = df_train["date"].max()

        # Build performance table data (train rows, then test, then blank slate)
        self._table_headers = [
            "Backtest",
            "Period",
            "Total Return (%)",
            "Annualized Return (%)",
            "Total Sharpe Ratio",
            "Annualized Sharpe Ratio",
            "Ann. Volatility (%)",
            "Sortino Ratio",
            "Calmar Ratio",
            "Avg Daily Turnover (%)",
            "Max Drawdown (%)",
            "Total Trading Cost ($)",
            "Avg Daily Trading Cost ($)",
            "Avg POV (%)",
            "Wtd Avg Turnover Pctl",
            "Avg Step Reward",
        ]
        self._table_data = (
            [
                [bt["short_name"], "Train"] + bt["train_metrics"]
                for bt in self._backtests
            ]
            + [
                [bt["short_name"], "Test"] + bt["test_metrics"]
                for bt in self._backtests
            ]
            + [
                [bt["short_name"], "Blank Slate"] + bt["blank_metrics"]
                for bt in self._backtests
            ]
        )

        log.info(f"Loaded {len(self._backtests)} backtests")

    def make_chart(
        self,
        chart: Chart,
        legend_position: str = "outside",
        legend_columns: int = 1,
        height: Optional[int] = None,
        legend_gap: Optional[float] = None,
    ) -> go.Figure:
        """
        Create and return a standalone Plotly figure for the given chart type.

        Args:
            chart: A :class:`Chart` enum value specifying which chart to create.
            legend_position: ``"outside"`` (default, standard Plotly behaviour),
                ``"inside"`` (overlaid top-left with semi-transparent background),
                or ``"below"`` (vertical stack under the chart).
            legend_columns: Number of columns for ``"below"`` legends.
                Defaults to 1 (single column).  Ignored for other positions.
            height: Chart height in pixels.  When *None* (default) the
                built-in height for each chart type is used.
            legend_gap: Distance between the chart and the ``"below"`` legend,
                expressed as a fraction of the plot area (0.0–1.0).  Defaults
                to 0.08.  Ignored for other legend positions.

        Returns:
            A ``plotly.graph_objects.Figure`` ready for ``.show()`` in Jupyter
            or any other Plotly rendering context.
        """
        builder_name = self._CHART_BUILDERS.get(chart)
        if builder_name is None:
            raise ValueError(f"Unknown chart type: {chart}")
        fig = getattr(self, builder_name)()
        if height is not None:
            fig.update_layout(height=height)
        self._apply_legend_position(fig, legend_position, legend_columns, legend_gap)
        return fig

    def generate_report(
        self,
        output_path: str = None,
        legend_position: str = "outside",
        legend_columns: int = 1,
    ) -> str:
        """
        Generate a full HTML report by assembling all charts.

        Calls :meth:`make_chart` for every :class:`Chart` member, converts
        each figure to an HTML ``<div>``, and writes them into a single page.

        Args:
            output_path: Path for the output HTML file.  Defaults to
                ``backtest_report.html`` in the summary directory.
            legend_position: Passed through to :meth:`make_chart`.
            legend_columns: Passed through to :meth:`make_chart`.

        Returns:
            Path to the generated HTML report file.
        """
        if output_path is None:
            output_path = os.path.join(self._summary_dir, "backtest_report.html")

        log.info("Generating report …")
        chart_divs = []
        for chart in self._REPORT_CHARTS:
            fig = self.make_chart(
                chart, legend_position=legend_position, legend_columns=legend_columns
            )
            div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            chart_divs.append(div)

        html = self._wrap_html(chart_divs)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        log.info(f"Report generated at {output_path}")
        return output_path

    def _make_portfolio_value(self) -> go.Figure:
        fig = go.Figure()
        for bt in self._backtests:
            tc, ttc = self._train_test_colors(bt["rgb"])
            fig.add_trace(
                go.Scatter(
                    x=bt["df_train"]["date"],
                    y=bt["df_train"]["portfolio_value"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=tc),
                    legendgroup=bt["trace_name"],
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bt["df_test"]["date"],
                    y=bt["df_test"]["portfolio_value"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=ttc),
                    legendgroup=bt["trace_name"],
                    showlegend=False,
                )
            )
        for bt in self._backtests:
            if bt["add_benchmark"]:
                self._add_benchmark_scatter(fig, bt, "benchmark_value")
        self._add_split_vline(fig)
        fig.update_layout(
            title="Portfolio Value Over Time",
            yaxis_title="Portfolio Value ($)",
            height=self._CHART_HEIGHTS[Chart.PORTFOLIO_VALUE],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    def _make_turnover(self) -> go.Figure:
        fig = go.Figure()
        for bt in self._backtests:
            tc, ttc = self._train_test_colors(bt["rgb"])
            fig.add_trace(
                go.Scatter(
                    x=bt["df_train"]["date"],
                    y=bt["df_train"]["turnover"] * 100,
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=tc),
                    legendgroup=bt["trace_name"],
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bt["df_test"]["date"],
                    y=bt["df_test"]["turnover"] * 100,
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=ttc),
                    legendgroup=bt["trace_name"],
                    showlegend=False,
                )
            )
        self._add_split_vline(fig)
        fig.update_layout(
            title="Daily Turnover (%)",
            yaxis_title="Turnover (%)",
            height=self._CHART_HEIGHTS[Chart.TURNOVER],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    def _make_cumulative_cost(self) -> go.Figure:
        fig = go.Figure()
        for bt in self._backtests:
            tc, ttc = self._train_test_colors(bt["rgb"])
            fig.add_trace(
                go.Scatter(
                    x=bt["df_train"]["date"],
                    y=bt["df_train"]["cumulative_cost"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=tc),
                    legendgroup=bt["trace_name"],
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bt["df_test"]["date"],
                    y=bt["df_test"]["cumulative_cost"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=ttc),
                    legendgroup=bt["trace_name"],
                    showlegend=False,
                )
            )
        self._add_split_vline(fig)
        fig.update_layout(
            title="Cumulative Trading Cost ($)",
            yaxis_title="Cumulative Cost ($)",
            height=self._CHART_HEIGHTS[Chart.CUMULATIVE_COST],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    def _make_rolling_sharpe(self) -> go.Figure:
        fig = go.Figure()
        for bt in self._backtests:
            tc, ttc = self._train_test_colors(bt["rgb"])
            fig.add_trace(
                go.Scatter(
                    x=bt["df_train"]["date"],
                    y=bt["df_train"]["rolling_sharpe"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=tc),
                    legendgroup=bt["trace_name"],
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bt["df_test"]["date"],
                    y=bt["df_test"]["rolling_sharpe"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=ttc),
                    legendgroup=bt["trace_name"],
                    showlegend=False,
                )
            )
        for bt in self._backtests:
            if bt["add_benchmark"]:
                self._add_benchmark_scatter(fig, bt, "benchmark_rolling_sharpe")
        self._add_split_vline(fig)
        fig.update_layout(
            title="Rolling Sharpe Ratio",
            yaxis_title="Rolling Sharpe Ratio",
            height=self._CHART_HEIGHTS[Chart.ROLLING_SHARPE],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    def _make_drawdown(self) -> go.Figure:
        fig = go.Figure()
        for bt in self._backtests:
            tc, ttc = self._train_test_colors(bt["rgb"])
            fig.add_trace(
                go.Scatter(
                    x=bt["df_train"]["date"],
                    y=bt["df_train"]["drawdown"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=tc),
                    legendgroup=bt["trace_name"],
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bt["df_test"]["date"],
                    y=bt["df_test"]["drawdown"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=ttc),
                    legendgroup=bt["trace_name"],
                    showlegend=False,
                )
            )
        self._add_split_vline(fig)
        fig.update_layout(
            title="Portfolio Drawdown (%)",
            yaxis_title="Drawdown (%)",
            height=self._CHART_HEIGHTS[Chart.DRAWDOWN],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    def _make_trading_volume(self) -> go.Figure:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for bt in self._backtests:
            tc, ttc = self._train_test_colors(bt["rgb"])
            sc_train, sc_test = self._sell_colors(bt["rgb"])
            grp = bt["trace_name"]
            cash_grp = f"{grp} - Cash %"
            og = f"bt{bt['index']}"
            gc_train, gc_test = self._cash_green(bt["index"])

            fig.add_trace(
                go.Bar(
                    x=bt["df_train"]["date"],
                    y=bt["df_train"]["total_buy_value"],
                    name=f"{grp} - Buys",
                    marker_color=tc,
                    legendgroup=grp,
                    offsetgroup=og,
                )
            )
            fig.add_trace(
                go.Bar(
                    x=bt["df_train"]["date"],
                    y=-bt["df_train"]["total_sell_value"],
                    name=f"{grp} - Sells",
                    marker_color=sc_train,
                    legendgroup=grp,
                    offsetgroup=og,
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Bar(
                    x=bt["df_test"]["date"],
                    y=bt["df_test"]["total_buy_value"],
                    name=grp,
                    marker_color=ttc,
                    legendgroup=grp,
                    offsetgroup=og,
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Bar(
                    x=bt["df_test"]["date"],
                    y=-bt["df_test"]["total_sell_value"],
                    name=grp,
                    marker_color=sc_test,
                    legendgroup=grp,
                    offsetgroup=og,
                    showlegend=False,
                )
            )

            train_cash_pct = (
                bt["df_train"]["cash"] / bt["df_train"]["portfolio_value"] * 100
            )
            test_cash_pct = (
                bt["df_test"]["cash"] / bt["df_test"]["portfolio_value"] * 100
            )
            fig.add_trace(
                go.Scatter(
                    x=bt["df_train"]["date"],
                    y=train_cash_pct,
                    mode="lines",
                    name=cash_grp,
                    line=dict(color=gc_train),
                    legendgroup=cash_grp,
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=bt["df_test"]["date"],
                    y=test_cash_pct,
                    mode="lines",
                    name=cash_grp,
                    line=dict(color=gc_test),
                    legendgroup=cash_grp,
                    showlegend=False,
                ),
                secondary_y=True,
            )

        self._add_split_vline(fig)
        fig.update_layout(
            title="Daily Trading Volume ($)",
            barmode="group",
            height=self._CHART_HEIGHTS[Chart.TRADING_VOLUME],
            hoverlabel=dict(namelength=-1),
        )
        fig.update_yaxes(title_text="Daily Trading Volume ($)", secondary_y=False)
        fig.update_yaxes(title_text="Cash (% of Portfolio)", secondary_y=True)
        return fig

    # -- Blank-slate variants --

    def _make_portfolio_value_blank_slate(self) -> go.Figure:
        fig = go.Figure()
        for bt in self._backtests:
            _, ttc = self._train_test_colors(bt["rgb"])
            fig.add_trace(
                go.Scatter(
                    x=bt["df_test_blank"]["date"],
                    y=bt["df_test_blank"]["portfolio_value"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=ttc),
                    legendgroup=bt["trace_name"],
                )
            )
        for bt in self._backtests:
            if bt["add_benchmark"]:
                self._add_benchmark_scatter_blank(fig, bt, "benchmark_value")
        fig.update_layout(
            title="Portfolio Value Over Time (Blank Slate)",
            yaxis_title="Portfolio Value ($)",
            height=self._CHART_HEIGHTS[Chart.PORTFOLIO_VALUE_BLANK_SLATE],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    def _make_rolling_sharpe_blank_slate(self) -> go.Figure:
        fig = go.Figure()
        for bt in self._backtests:
            _, ttc = self._train_test_colors(bt["rgb"])
            fig.add_trace(
                go.Scatter(
                    x=bt["df_test_blank"]["date"],
                    y=bt["df_test_blank"]["rolling_sharpe"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=ttc),
                    legendgroup=bt["trace_name"],
                )
            )
        for bt in self._backtests:
            if bt["add_benchmark"]:
                self._add_benchmark_scatter_blank(fig, bt, "benchmark_rolling_sharpe")
        fig.update_layout(
            title="Rolling Sharpe Ratio (Blank Slate, 63d)",
            yaxis_title="Rolling Sharpe Ratio",
            height=self._CHART_HEIGHTS[Chart.ROLLING_SHARPE_BLANK_SLATE],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    def _make_trading_volume_blank_slate(self) -> go.Figure:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for bt in self._backtests:
            _, ttc = self._train_test_colors(bt["rgb"])
            grp = bt["trace_name"]
            cash_grp = f"{grp} - Cash %"
            og = f"bt{bt['index']}"
            _, gc = self._cash_green(bt["index"])

            _, sc = self._sell_colors(bt["rgb"])

            fig.add_trace(
                go.Bar(
                    x=bt["df_test_blank"]["date"],
                    y=bt["df_test_blank"]["total_buy_value"],
                    name=f"{grp} - Buys",
                    marker_color=ttc,
                    legendgroup=grp,
                    offsetgroup=og,
                )
            )
            fig.add_trace(
                go.Bar(
                    x=bt["df_test_blank"]["date"],
                    y=-bt["df_test_blank"]["total_sell_value"],
                    name=f"{grp} - Sells",
                    marker_color=sc,
                    legendgroup=grp,
                    offsetgroup=og,
                    showlegend=False,
                )
            )

            blank_cash_pct = (
                bt["df_test_blank"]["cash"]
                / bt["df_test_blank"]["portfolio_value"]
                * 100
            )
            fig.add_trace(
                go.Scatter(
                    x=bt["df_test_blank"]["date"],
                    y=blank_cash_pct,
                    mode="lines",
                    name=cash_grp,
                    line=dict(color=gc),
                    legendgroup=cash_grp,
                ),
                secondary_y=True,
            )

        fig.update_layout(
            title="Daily Trading Volume (Blank Slate) ($)",
            barmode="group",
            height=self._CHART_HEIGHTS[Chart.TRADING_VOLUME_BLANK_SLATE],
            hoverlabel=dict(namelength=-1),
        )
        fig.update_yaxes(title_text="Daily Trading Volume ($)", secondary_y=False)
        fig.update_yaxes(title_text="Cash (% of Portfolio)", secondary_y=True)
        return fig

    def _make_drawdown_blank_slate(self) -> go.Figure:
        fig = go.Figure()
        for bt in self._backtests:
            _, ttc = self._train_test_colors(bt["rgb"])
            fig.add_trace(
                go.Scatter(
                    x=bt["df_test_blank"]["date"],
                    y=bt["df_test_blank"]["drawdown"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=ttc),
                    legendgroup=bt["trace_name"],
                )
            )
        fig.update_layout(
            title="Portfolio Drawdown (Blank Slate) (%)",
            yaxis_title="Drawdown (%)",
            height=self._CHART_HEIGHTS[Chart.DRAWDOWN_BLANK_SLATE],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    # -- POV / turnover-percentile histograms --

    def _make_pov_histogram_train(self) -> go.Figure:
        return self._make_pov_histogram("train")

    def _make_pov_histogram_test(self) -> go.Figure:
        return self._make_pov_histogram("test")

    def _make_pov_histogram_blank_slate(self) -> go.Figure:
        return self._make_pov_histogram("blank_slate")

    def _make_pov_histogram(self, period: str) -> go.Figure:
        """Shared builder for POV + turnover-percentile histograms (1×2)."""
        trades_key = {
            "train": "trades_train",
            "test": "trades_test",
            "blank_slate": "trades_blank",
        }[period]
        title_period = {"train": "Train", "test": "Test", "blank_slate": "Blank Slate"}[
            period
        ]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"POV Distribution ({title_period})",
                f"Turnover Percentile ({title_period})",
            ),
        )

        pov_bins = np.linspace(0, 10, 21)
        pctl_bins = np.linspace(0, 100, 21)
        pov_bw_full = pov_bins[1] - pov_bins[0]
        pctl_bw_full = pctl_bins[1] - pctl_bins[0]

        trades_list = [
            (bt[trades_key], bt["index"], bt["trace_name"])
            for bt in self._backtests
            if not bt[trades_key].empty
        ]
        n = len(trades_list)
        if n == 0:
            fig.update_layout(
                height=self._CHART_HEIGHTS[Chart.POV_HISTOGRAM_TRAIN],
                hoverlabel=dict(namelength=-1),
            )
            return fig

        pov_bw = pov_bw_full * 0.8 / n
        pctl_bw = pctl_bw_full * 0.8 / n

        for j, (trades_df, bt_idx, trace_name) in enumerate(trades_list):
            rgb = self._get_color_info(bt_idx)
            color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.7)"
            pov_offset = (j - (n - 1) / 2) * pov_bw
            pctl_offset = (j - (n - 1) / 2) * pctl_bw
            notional = trades_df["notional"]

            # POV histogram (col 1 – carries the legend entry)
            hw, be = np.histogram(
                trades_df["pov"] * 100, bins=pov_bins, weights=notional
            )
            centers = (be[:-1] + be[1:]) / 2
            fig.add_trace(
                go.Bar(
                    x=centers + pov_offset,
                    y=hw,
                    width=pov_bw,
                    name=trace_name,
                    marker_color=color,
                    legendgroup=trace_name,
                    customdata=centers,
                    hovertemplate="POV=%{customdata:.1f}%<br>Notional=%{y:,.0f}"
                    "<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=1,
            )

            # Turnover-percentile histogram (col 2)
            hw, be = np.histogram(
                trades_df["turnover_percentile"], bins=pctl_bins, weights=notional
            )
            centers = (be[:-1] + be[1:]) / 2
            fig.add_trace(
                go.Bar(
                    x=centers + pctl_offset,
                    y=hw,
                    width=pctl_bw,
                    name=trace_name,
                    marker_color=color,
                    legendgroup=trace_name,
                    showlegend=False,
                    customdata=centers,
                    hovertemplate="Percentile=%{customdata:.0f}<br>Notional=%{y:,.0f}"
                    "<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=2,
            )

        fig.update_yaxes(title_text="Gross Notional ($)", row=1, col=1)
        fig.update_yaxes(title_text="Gross Notional ($)", row=1, col=2)
        fig.update_xaxes(title_text="POV (%)", row=1, col=1)
        fig.update_xaxes(title_text="Turnover Percentile", row=1, col=2)
        fig.update_layout(
            height=self._CHART_HEIGHTS[Chart.POV_HISTOGRAM_TRAIN],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    # -- POV time series --

    def _make_pov_time_series(self) -> go.Figure:
        fig = go.Figure()
        for bt in self._backtests:
            tc, ttc = self._train_test_colors(bt["rgb"])
            shown = False
            for trades_df, color in [
                (bt["trades_train"], tc),
                (bt["trades_test"], ttc),
            ]:
                if trades_df.empty:
                    continue
                daily_pov = self._daily_avg_pov(trades_df)
                fig.add_trace(
                    go.Scatter(
                        x=daily_pov["date"],
                        y=daily_pov["avg_pov"],
                        mode="lines",
                        name=bt["trace_name"],
                        line=dict(color=color),
                        legendgroup=bt["trace_name"],
                        showlegend=not shown,
                    )
                )
                shown = True
        self._add_split_vline(fig)
        fig.update_layout(
            title="Daily Avg POV (%)",
            yaxis_title="Avg POV (%)",
            height=self._CHART_HEIGHTS[Chart.POV_TIME_SERIES],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    def _make_pov_time_series_blank_slate(self) -> go.Figure:
        fig = go.Figure()
        for bt in self._backtests:
            _, ttc = self._train_test_colors(bt["rgb"])
            if bt["trades_blank"].empty:
                continue
            daily_pov = self._daily_avg_pov(bt["trades_blank"])
            fig.add_trace(
                go.Scatter(
                    x=daily_pov["date"],
                    y=daily_pov["avg_pov"],
                    mode="lines",
                    name=bt["trace_name"],
                    line=dict(color=ttc),
                    legendgroup=bt["trace_name"],
                )
            )
        fig.update_layout(
            title="Daily Avg POV (Blank Slate) (%)",
            yaxis_title="Avg POV (%)",
            height=self._CHART_HEIGHTS[Chart.POV_TIME_SERIES_BLANK_SLATE],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    # -- Epoch stats --

    _EPOCH_STAT_CONFIGS = [
        ("annualized_return", "Annualized Return per Epoch (%)", "Ann. Return (%)"),
        ("annualized_sharpe", "Annualized Sharpe Ratio per Epoch", "Ann. Sharpe"),
        (
            "annualized_volatility",
            "Annualized Volatility per Epoch (%)",
            "Ann. Volatility (%)",
        ),
        ("sortino_ratio", "Sortino Ratio per Epoch", "Sortino"),
        ("calmar_ratio", "Calmar Ratio per Epoch", "Calmar"),
        ("max_drawdown", "Max Drawdown per Epoch (%)", "Max DD (%)"),
        ("avg_daily_turnover", "Avg Daily Turnover per Epoch (%)", "Avg Turnover (%)"),
        (
            "avg_daily_trading_cost",
            "Avg Daily Trading Cost per Epoch ($)",
            "Avg Cost ($)",
        ),
        ("avg_order_pov", "Avg Order POV per Epoch (%)", "Avg POV (%)"),
        ("avg_step_reward", "Avg Step Reward per Epoch", "Avg Reward"),
    ]

    def _make_epoch_stats(self) -> go.Figure:
        nrows = 5
        ncols = 2
        subtitles = [
            f"{title} — solid: train, dashed: OOS"
            for _, title, _ in self._EPOCH_STAT_CONFIGS
        ]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=subtitles,
            vertical_spacing=0.06,
            horizontal_spacing=0.08,
        )

        for stat_idx, (stat_key, _, y_label) in enumerate(self._EPOCH_STAT_CONFIGS):
            row = stat_idx // 2 + 1
            col = stat_idx % 2 + 1
            show_legend = stat_idx == 0

            for bt in self._backtests:
                rgb = bt["rgb"]
                epoch_stats = bt["epoch_stats"]

                train_stats = epoch_stats.get("train", [])
                if train_stats:
                    tc = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.7)"
                    fig.add_trace(
                        go.Scatter(
                            x=[s["epoch"] for s in train_stats],
                            y=[s[stat_key] for s in train_stats],
                            mode="lines+markers",
                            name=f"{bt['trace_name']} (Train)",
                            line=dict(color=tc),
                            marker=dict(size=5),
                            legendgroup=bt["trace_name"],
                            showlegend=show_legend,
                        ),
                        row=row,
                        col=col,
                    )

                test_stats = epoch_stats.get("test_blank", [])
                if test_stats:
                    ttc = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
                    fig.add_trace(
                        go.Scatter(
                            x=[s["epoch"] for s in test_stats],
                            y=[s[stat_key] for s in test_stats],
                            mode="lines+markers",
                            name=f"{bt['trace_name']} (OOS)",
                            line=dict(color=ttc, dash="dash"),
                            marker=dict(size=5, symbol="diamond"),
                            legendgroup=bt["trace_name"],
                            showlegend=show_legend,
                        ),
                        row=row,
                        col=col,
                    )

            fig.update_yaxes(title_text=y_label, row=row, col=col)
            fig.update_xaxes(title_text="Epoch", row=row, col=col)

        fig.update_layout(
            height=self._CHART_HEIGHTS[Chart.EPOCH_STATS],
            hoverlabel=dict(namelength=-1),
        )
        return fig

    def _make_single_epoch_stat(self, stat_index: int) -> go.Figure:
        """Build a standalone chart for one epoch stat by index into
        ``_EPOCH_STAT_CONFIGS``."""
        stat_key, title, y_label = self._EPOCH_STAT_CONFIGS[stat_index]
        fig = go.Figure()
        for bt in self._backtests:
            rgb = bt["rgb"]
            epoch_stats = bt["epoch_stats"]

            train_stats = epoch_stats.get("train", [])
            if train_stats:
                tc = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.7)"
                fig.add_trace(
                    go.Scatter(
                        x=[s["epoch"] for s in train_stats],
                        y=[s[stat_key] for s in train_stats],
                        mode="lines+markers",
                        name=f"{bt['trace_name']} (Train)",
                        line=dict(color=tc),
                        marker=dict(size=5),
                        legendgroup=bt["trace_name"],
                    )
                )

            test_stats = epoch_stats.get("test_blank", [])
            if test_stats:
                ttc = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
                fig.add_trace(
                    go.Scatter(
                        x=[s["epoch"] for s in test_stats],
                        y=[s[stat_key] for s in test_stats],
                        mode="lines+markers",
                        name=f"{bt['trace_name']} (OOS)",
                        line=dict(color=ttc, dash="dash"),
                        marker=dict(size=5, symbol="diamond"),
                        legendgroup=bt["trace_name"],
                    )
                )

        fig.update_layout(
            title=f"{title} — solid: train, dashed: OOS",
            xaxis_title="Epoch",
            yaxis_title=y_label,
            height=400,
            hoverlabel=dict(namelength=-1),
        )
        return fig

    def _make_epoch_annualized_return(self):
        return self._make_single_epoch_stat(0)

    def _make_epoch_annualized_sharpe(self):
        return self._make_single_epoch_stat(1)

    def _make_epoch_annualized_volatility(self):
        return self._make_single_epoch_stat(2)

    def _make_epoch_sortino(self):
        return self._make_single_epoch_stat(3)

    def _make_epoch_calmar(self):
        return self._make_single_epoch_stat(4)

    def _make_epoch_max_drawdown(self):
        return self._make_single_epoch_stat(5)

    def _make_epoch_avg_turnover(self):
        return self._make_single_epoch_stat(6)

    def _make_epoch_avg_cost(self):
        return self._make_single_epoch_stat(7)

    def _make_epoch_avg_pov(self):
        return self._make_single_epoch_stat(8)

    def _make_epoch_avg_reward(self):
        return self._make_single_epoch_stat(9)

    # -- Performance table --

    def _make_performance_table(self) -> go.Figure:
        fig = go.Figure()
        if not self._table_data:
            fig.update_layout(height=self._CHART_HEIGHTS[Chart.PERFORMANCE_TABLE])
            return fig

        table_values = [list(col) for col in zip(*self._table_data)]
        num_rows = len(self._table_data)
        rows_per_section = num_rows // 3 if num_rows > 0 else 0

        section_colors = [
            ("#f7fafc", "#edf2f7"),  # Train
            ("#f7faf7", "#edf5ed"),  # Test
            ("#fdfcfa", "#f9f7f3"),  # Blank Slate
        ]
        row_colors = []
        for i in range(num_rows):
            sec = min(i // rows_per_section if rows_per_section > 0 else 0, 2)
            bt_idx = i % rows_per_section if rows_per_section > 0 else i
            pair = section_colors[sec]
            row_colors.append(pair[0] if bt_idx % 2 == 0 else pair[1])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=self._table_headers,
                    fill_color="lightblue",
                    align="left",
                    font=dict(size=12),
                    line=dict(color="#cccccc", width=1),
                ),
                cells=dict(
                    values=table_values,
                    fill_color=[row_colors] * len(self._table_headers),
                    align="left",
                    font=dict(size=11),
                    line=dict(color="#cccccc", width=1),
                ),
            )
        )

        fig.update_layout(
            title="Performance Metrics",
            height=self._CHART_HEIGHTS[Chart.PERFORMANCE_TABLE],
        )
        return fig

    # ------------------------------------------------------------------ #
    #  Trace / layout helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _train_test_colors(rgb: tuple) -> tuple[str, str]:
        """Return ``(train_color, test_color)`` RGBA/RGB strings."""
        return (
            f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.7)",
            f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})",
        )

    @staticmethod
    def _sell_colors(rgb: tuple) -> tuple[str, str]:
        """Return slightly transparent sell-bar colors for train and test."""
        return (
            f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.5)",
            f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.7)",
        )

    _GREEN_SHADES = [
        (0, 128, 0),  # green
        (34, 139, 34),  # forest green
        (0, 100, 0),  # dark green
        (60, 179, 113),  # medium sea green
        (46, 139, 87),  # sea green
        (0, 155, 72),  # jade
        (85, 107, 47),  # dark olive green
        (107, 142, 35),  # olive drab
    ]

    @classmethod
    def _cash_green(cls, index: int) -> tuple[str, str]:
        """Return ``(train_green, test_green)`` for the cash % line."""
        r, g, b = cls._GREEN_SHADES[index % len(cls._GREEN_SHADES)]
        return (
            f"rgba({r}, {g}, {b}, 0.5)",
            f"rgb({r}, {g}, {b})",
        )

    def _add_benchmark_scatter(self, fig, bt: dict, column: str):
        """Add train + test benchmark scatter traces."""
        capital_str = self._format_capital(bt["capital"])
        label = f"{self._benchmark_ticker} | Capital={capital_str}"
        fig.add_trace(
            go.Scatter(
                x=bt["df_train"]["date"],
                y=bt["df_train"][column],
                mode="lines",
                name=label,
                line=dict(color="rgba(0, 0, 0, 0.7)"),
                legendgroup=label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bt["df_test"]["date"],
                y=bt["df_test"][column],
                mode="lines",
                name=label,
                line=dict(color="rgb(0, 0, 0)"),
                legendgroup=label,
                showlegend=False,
            )
        )

    def _add_benchmark_scatter_blank(self, fig, bt: dict, column: str):
        """Add blank-slate benchmark scatter trace."""
        capital_str = self._format_capital(bt["capital"])
        label = f"{self._benchmark_ticker} | Capital={capital_str}"
        fig.add_trace(
            go.Scatter(
                x=bt["df_test_blank"]["date"],
                y=bt["df_test_blank"][column],
                mode="lines",
                name=label,
                line=dict(color="rgb(0, 0, 0)"),
                legendgroup=label,
            )
        )

    def _add_split_vline(self, fig):
        """Add a vertical dashed line at the train/test split date."""
        if self._split_date:
            fig.add_vline(
                x=self._split_date,
                line_width=2,
                line_dash="dash",
                line_color="grey",
            )

    @staticmethod
    def _apply_legend_position(
        fig: go.Figure, position: str, columns: int = 1, gap: Optional[float] = None
    ):
        if position == "inside":
            fig.update_layout(
                legend=dict(
                    x=0.01,
                    y=0.99,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="rgba(0, 0, 0, 0.2)",
                    borderwidth=1,
                    font=dict(size=10),
                )
            )
        elif position == "below":
            gap = gap if gap is not None else 0.08
            n_entries = sum(1 for t in fig.data if t.showlegend is not False)
            columns = max(columns, 1)
            n_rows = math.ceil(n_entries / columns)
            row_height = 18
            extra_height = max(n_rows * row_height, 60)
            current_height = fig.layout.height or 500
            legend_opts = dict(
                x=0.0,
                y=-gap,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
                tracegroupgap=2,
            )
            if columns > 1:
                legend_opts.update(
                    orientation="h",
                    entrywidth=0.9 / columns,
                    entrywidthmode="fraction",
                )
            else:
                legend_opts["orientation"] = "v"
            fig.update_layout(
                legend=legend_opts,
                height=current_height + extra_height,
                margin=dict(b=extra_height + 30),
            )

    @staticmethod
    def _daily_avg_pov(trades_df: pd.DataFrame) -> pd.DataFrame:
        """Compute daily notional-weighted average POV (%)."""
        daily = (
            trades_df.groupby("date")
            .apply(
                lambda x: (
                    (x["pov"] * x["notional"]).sum() / x["notional"].sum() * 100
                    if x["notional"].sum() > 0
                    else 0
                )
            )
            .reset_index()
        )
        daily.columns = ["date", "avg_pov"]
        daily["date"] = pd.to_datetime(daily["date"])
        return daily

    @staticmethod
    def _wrap_html(chart_divs: list[str]) -> str:
        """Wrap individual chart HTML divs into a complete HTML page."""
        body = "\n<hr>\n".join(chart_divs)
        return (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            '  <meta charset="utf-8">\n'
            "  <title>Backtest Performance Report</title>\n"
            '  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
            "  <style>\n"
            "    body { font-family: Arial, sans-serif; margin: 20px; }\n"
            "    hr { border: none; border-top: 1px solid #ddd; margin: 10px 0; }\n"
            "  </style>\n"
            "</head>\n<body>\n"
            "  <h1>Backtest Performance Comparison</h1>\n"
            f"  {body}\n"
            "</body>\n</html>"
        )

    def _load_and_prepare_data(
        self,
        train_csv_path: str,
        test_csv_path: str,
        test_blank_csv_path: str,
        trades_train_path: str = None,
        trades_test_path: str = None,
        trades_blank_path: str = None,
    ):
        """Load and prepare training and testing data with calculated metrics."""
        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)
        df_test_blank = pd.read_csv(test_blank_csv_path)

        trades_train = (
            pd.read_csv(trades_train_path)
            if trades_train_path and os.path.exists(trades_train_path)
            else pd.DataFrame()
        )
        trades_test = (
            pd.read_csv(trades_test_path)
            if trades_test_path and os.path.exists(trades_test_path)
            else pd.DataFrame()
        )
        trades_blank = (
            pd.read_csv(trades_blank_path)
            if trades_blank_path and os.path.exists(trades_blank_path)
            else pd.DataFrame()
        )

        df_train["date"] = pd.to_datetime(df_train["date"])
        df_test["date"] = pd.to_datetime(df_test["date"])
        df_test_blank["date"] = pd.to_datetime(df_test_blank["date"])

        if not df_test.empty:
            test_start_date = df_test["date"].min()
            df_test_blank = df_test_blank[
                df_test_blank["date"] >= test_start_date
            ].copy()
            df_test_blank.reset_index(drop=True, inplace=True)

        # --- Process Train and Test data ---
        df_train["cumulative_cost"] = df_train["cost"].cumsum()
        df_test["cumulative_cost"] = df_test["cost"].cumsum()
        if not df_train.empty:
            df_test["cumulative_cost"] += df_train["cumulative_cost"].iloc[-1]

        df_train["daily_return"] = df_train["portfolio_value"].pct_change()
        df_test["daily_return"] = df_test["portfolio_value"].pct_change()

        df_train["benchmark_daily_return"] = df_train["benchmark_value"].pct_change()
        df_test["benchmark_daily_return"] = df_test["benchmark_value"].pct_change()

        for df in [df_train, df_test]:
            rolling_max = df["portfolio_value"].expanding(min_periods=1).max()
            df["drawdown"] = (df["portfolio_value"] - rolling_max) / rolling_max * 100

        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        df_combined = df_combined.sort_values("date").reset_index(drop=True)
        df_combined["rolling_sharpe"] = rolling_sharpe(df_combined["daily_return"])
        df_combined["benchmark_rolling_sharpe"] = rolling_sharpe(
            df_combined["benchmark_daily_return"]
        )

        train_dates = set(df_train["date"])
        train_mask = df_combined["date"].isin(train_dates)

        df_train["rolling_sharpe"] = df_combined[train_mask]["rolling_sharpe"].values
        df_test["rolling_sharpe"] = df_combined[~train_mask]["rolling_sharpe"].values
        df_train["benchmark_rolling_sharpe"] = df_combined[train_mask][
            "benchmark_rolling_sharpe"
        ].values
        df_test["benchmark_rolling_sharpe"] = df_combined[~train_mask][
            "benchmark_rolling_sharpe"
        ].values

        # --- Process Blank Slate data ---
        df_test_blank["cumulative_cost"] = df_test_blank["cost"].cumsum()
        df_test_blank["daily_return"] = df_test_blank["portfolio_value"].pct_change()
        df_test_blank["benchmark_daily_return"] = df_test_blank[
            "benchmark_value"
        ].pct_change()

        rolling_max_blank = (
            df_test_blank["portfolio_value"].expanding(min_periods=1).max()
        )
        df_test_blank["drawdown"] = (
            (df_test_blank["portfolio_value"] - rolling_max_blank)
            / rolling_max_blank
            * 100
        )

        df_test_blank["rolling_sharpe"] = rolling_sharpe(
            df_test_blank["daily_return"], window=63, min_periods=21
        )
        df_test_blank["benchmark_rolling_sharpe"] = rolling_sharpe(
            df_test_blank["benchmark_daily_return"], window=63, min_periods=21
        )

        return df_train, df_test, df_test_blank, trades_train, trades_test, trades_blank

    def _get_color_info(self, i: int):
        """Get RGB color information for plotting."""
        color_hex = self.colors[i % len(self.colors)]
        h = color_hex.lstrip("#")
        rgb = tuple(int(h[j : j + 2], 16) for j in (0, 2, 4))
        return rgb

    @staticmethod
    def _format_capital(capital: float) -> str:
        """Format capital amount for display."""
        if capital >= 1e9:
            return f"{capital / 1e9:.0f}B"
        return f"{capital / 1e6:.0f}M"

    def _calculate_performance_metrics(self, df, trades_df=None):
        """Calculate performance metrics for a dataframe.

        Delegates to the unified ``compute_performance_stats`` utility and
        returns the values as a list of formatted strings for the table.
        """
        if df.empty:
            return ["N/A"] * 14

        stats = compute_performance_stats(
            df["portfolio_value"],
            df["turnover"],
            df["cost"],
            trades_df,
            rewards=df["reward"] if "reward" in df.columns else None,
        )
        return [
            f"{stats['total_return']:.2f}",
            f"{stats['annualized_return']:.2f}",
            f"{stats['total_sharpe']:.3f}",
            f"{stats['annualized_sharpe']:.3f}",
            f"{stats['annualized_volatility']:.2f}",
            f"{stats['sortino_ratio']:.3f}",
            f"{stats['calmar_ratio']:.3f}",
            f"{stats['avg_daily_turnover']:.2f}",
            f"{stats['max_drawdown']:.2f}",
            f"{stats['total_trading_cost']:,.2f}",
            f"{stats['avg_daily_trading_cost']:,.2f}",
            f"{stats['avg_order_pov']:.4f}",
            f"{stats['wtd_avg_turnover_percentile']:.2f}",
            f"{stats['avg_step_reward']:.6f}",
        ]

    _KEY_PRIORITY = [
        "run_type",
        "drl_agent",
        "impact_model",
        "initial_capital",
        "eta_dd",
        "horizon",
        "reward_scaling",
        "with_tbill",
        "with_perm",
        "with_cooldown",
        "use_obs_normalizer",
        "obs_clip",
        "learning_rate",
        "gamma",
        "ent_coef",
        "net_arch",
    ]

    @classmethod
    def _get_varying_param_keys(cls, metadatas: list[dict]) -> set[str]:
        """Return the minimal set of parameter keys that vary across runs.

        Keys whose value partitioning is identical to a higher-priority key
        (i.e. they are perfectly correlated) are dropped so that trace names
        stay concise.
        """
        candidate_keys = set(cls._KEY_PRIORITY)
        present_keys = [k for k in candidate_keys if any(k in m for m in metadatas)]

        varying: list[str] = []
        for k in present_keys:
            values = {str(m.get(k)) for m in metadatas}
            if len(values) > 1:
                varying.append(k)

        if len(varying) <= 1:
            return set(varying)

        priority_map = {k: i for i, k in enumerate(cls._KEY_PRIORITY)}

        def _partition(key: str) -> tuple:
            """Map raw values to group indices so that two keys with the
            same grouping produce the same tuple regardless of their
            actual values."""
            mapping: dict[str, int] = {}
            result: list[int] = []
            for m in metadatas:
                v = str(m.get(key))
                if v not in mapping:
                    mapping[v] = len(mapping)
                result.append(mapping[v])
            return tuple(result)

        sig_to_keys: dict[tuple, list[str]] = {}
        for k in varying:
            sig_to_keys.setdefault(_partition(k), []).append(k)

        result: set[str] = set()
        for keys in sig_to_keys.values():
            best = min(keys, key=lambda k: priority_map.get(k, len(cls._KEY_PRIORITY)))
            result.add(best)
        return result

    def _build_trace_name(self, metadata: dict, varying_keys: set[str]) -> str:
        """Build the legend trace name including only varying parameters."""
        parts: list[str] = []

        def add(label: str, key: str, fmt=lambda x: x):
            if key in varying_keys and key in metadata and metadata[key] is not None:
                parts.append(f"{label}={fmt(metadata[key])}")

        add("Type", "run_type", str)
        add("Agent", "drl_agent", lambda x: str(x).upper())
        add("Impact", "impact_model", str)
        add("Capital", "initial_capital", lambda x: self._format_capital(float(x)))
        add("Eta", "eta_dd", lambda x: f"{float(x):.2f}")
        add("Hz", "horizon", lambda x: str(x))
        add("RewScale", "reward_scaling", lambda x: f"{float(x):.4g}")
        add("Tbill", "with_tbill", lambda x: str(bool(x)))
        add("Perm", "with_perm", lambda x: str(bool(x)))
        add("Cool", "with_cooldown", lambda x: str(bool(x)))
        add("Norm", "use_obs_normalizer", lambda x: str(bool(x)))
        add("Clip", "obs_clip", lambda x: f"{float(x):.0f}")
        add("LR", "learning_rate", lambda x: f"{float(x):.2e}")
        add("Gamma", "gamma", lambda x: f"{float(x):.3f}")
        add(
            "Ent",
            "ent_coef",
            lambda x: f"{float(x):.3f}" if isinstance(x, (int, float)) else str(x),
        )
        add("Arch", "net_arch", str)

        if not parts and "drl_agent" in metadata:
            parts.append(f"Agent={str(metadata['drl_agent']).upper()}")

        return (
            self._truncate_parts(parts, " | ", self._MAX_TRACE_NAME_LEN) or "Backtest"
        )

    def _build_short_name(self, metadata: dict, varying_keys: set[str]) -> str:
        """Build a concise name for the performance table including only varying params."""
        parts: list[str] = []

        def add(key: str, fmt=lambda x: x):
            if key in varying_keys and key in metadata and metadata[key] is not None:
                parts.append(str(fmt(metadata[key])))

        add("run_type", str)
        add("drl_agent", lambda x: str(x).upper())
        add("impact_model", str)
        add("initial_capital", lambda x: self._format_capital(float(x)))
        add("eta_dd", lambda x: f"eta={float(x):.2f}")
        add("horizon", lambda x: f"hz={x}")
        add("reward_scaling", lambda x: f"rs={float(x):.4g}")
        add("with_tbill", lambda x: f"tb={bool(x)}")
        add("with_perm", lambda x: f"pm={bool(x)}")
        add("with_cooldown", lambda x: f"cd={bool(x)}")
        add("use_obs_normalizer", lambda x: f"nm={bool(x)}")
        add("obs_clip", lambda x: f"cl={float(x):.0f}")
        add("learning_rate", lambda x: f"lr={float(x):.1e}")
        add("gamma", lambda x: f"g={float(x):.3f}")
        add(
            "ent_coef",
            lambda x: f"e={float(x):.3f}" if isinstance(x, (int, float)) else f"e={x}",
        )
        add("net_arch", lambda x: f"a={x}")

        return (
            self._truncate_parts(parts, "-", self._MAX_SHORT_NAME_LEN)
            or str(metadata.get("drl_agent", "Backtest")).upper()
        )

    @staticmethod
    def _truncate_parts(parts: list[str], sep: str, max_len: int) -> str:
        """Join *parts* with *sep*, dropping trailing parts to stay under *max_len*."""
        if not parts:
            return ""
        full = sep.join(parts)
        if len(full) <= max_len:
            return full
        for n in range(len(parts), 0, -1):
            candidate = sep.join(parts[:n])
            suffix = f"{sep}+{len(parts) - n}" if n < len(parts) else ""
            if len(candidate) + len(suffix) <= max_len:
                return candidate + suffix
        return parts[0][:max_len]
