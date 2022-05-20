"""
main module to create plot figure from previously created dataframe
"""
# pylint: disable=C0103

import pickle
from pathlib import Path
from typing import Any, Iterable, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from matplotlib.lines import Line2D  # type: ignore
from matplotlib.patches import Patch  # type: ignore

from helper_funcs import plot_colors_dict, poly_fit, read_config_file


class PumpFigure:
    """
    Main pump plot object.
    """

    def __init__(self, plot_config_dict: dict):
        self.plot_config_dict = plot_config_dict
        self.speeds = plot_config_dict["plot_speeds"]
        self.ax: plt.Axes
        self.fig: plt.Figure
        self.iteration = 0  # used to track which color to plot
        self.colors = list(plot_colors_dict().values())
        self.AOR_plot = False
        self.POR_plot = False
        self.BEP_plot = False

    def build_base_fig(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        builds the base figure with gridlines, axe ticks, title, and axis labels
        """
        SMALL_SIZE = 12
        MEDIUM_SIZE = 16
        plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the figure title
        figure_size = (
            float(self.plot_config_dict["figure_width"]),
            float(self.plot_config_dict["figure_height"]),
        )

        fig, ax = plt.subplots(figsize=figure_size, ncols=1, nrows=1)
        ax.set(
            ylabel=f"Head ({self.plot_config_dict['head_units']})",
            xlabel=f"Flow ({self.plot_config_dict['flow_units']})",
            title=self.plot_config_dict["plot_title"],
        )
        ax.set_ylim(bottom=0, top=self.plot_config_dict["ymax"])
        ax.set_xlim(left=0, right=self.plot_config_dict["xmax"])
        ax.minorticks_on()
        ax.grid(
            which="minor",
            alpha=0.25,
            color="gray",
            linestyle="dashed",
        )
        ax.grid(
            which="major",
            alpha=0.5,
            color="gray",
            linestyle="dashed",
        )
        self.ax = ax
        self.fig = fig
        return self.ax

    def plot_system_curves(
        self,
        flow: Iterable,
        head: Union[Iterable, list[Iterable]],
        fill: bool = False,
        **kwargs,
    ):
        """
        Adds system curve plot to fig.
        Flow data should be a list of pandas series of flow points.
        If plotting a single curve, head data should be an iterable
        i.e a list of list or  pandas series of head points.
        If filling between two curves (fill = True), head should be supplied
        as list of either pandas series' or a list of head points, i.e two curve's
        worth of head points
        """
        if fill:
            if len(head) != 2:  # type: ignore
                raise IndexError(
                    "Please provide two arrays of head data to fill between."
                )
            head1, head2 = np.array(head[0]), np.array(head[1])  # type: ignore
            flow = np.array(flow)
            fill_xrange = np.linspace(flow.min(), flow.max(), len(flow))
            self.ax.fill_between(
                x=fill_xrange,
                y1=head1,
                y2=head2,
                linewidth=0,
                **kwargs,
            )
            return self.ax

        self.ax.plot(flow, head, **kwargs)
        return self.ax

    def plot_speed_curves(self, pump_df: pd.DataFrame, **kwargs):
        """
        add speed lines for all speeds defined under config.yml -> plot_config -> plot_speeds
        """
        for speed in self.speeds:
            speed_linestyle = "dashed" if speed != 100 else "solid"
            self.ax.plot(
                pump_df[f"flow_{speed}"],
                pump_df[f"head_{speed}"],
                linestyle=speed_linestyle,
                color=self.colors[self.iteration],
                **kwargs,
            )
        return self.ax

    def plot_BEP_markers(self, pump_df: pd.DataFrame, **kwargs):
        """
        adds best efficiency markers for each speed defined under
        config.yml -> plot_config -> plot_speeds
        """
        for speed in self.speeds:
            self.ax.scatter(
                pump_df[f"bep_flow_{speed}"],
                pump_df[f"bep_head_{speed}"],
                color="red",
                zorder=5,
                marker="x",
                **kwargs,
            )
        self.BEP_plot = True
        return self.ax

    def plot_aor(self, pump_df: pd.DataFrame, **kwargs):
        """
        plots AOR filled area for range defined under config.yml -> pump_config -> aor_range
        sometimes the AOR fill area overlaps the actual pump curve. This can be mitigated by
        increasing the aor_por_fit_degree in the config file, but may not eliminate the overlap
        entirely
        """
        aor_upper_range = np.linspace(
            min(pump_df["aor_upper_flow"].values), max(pump_df["aor_upper_flow"].values)
        )
        aor_upper_line = poly_fit(
            pump_df["aor_upper_flow"].dropna(),
            pump_df["aor_upper_head"].dropna(),
            x_new=aor_upper_range,
        )
        aor_lower_range = np.linspace(
            min(pump_df["aor_lower_flow"].values), max(pump_df["aor_lower_flow"].values)
        )
        aor_lower_line = poly_fit(
            pump_df["aor_lower_flow"].dropna(),
            pump_df["aor_lower_head"].dropna(),
            x_new=aor_lower_range,
        )
        self.ax.fill(
            np.append(aor_upper_range, aor_lower_range[::-1]),
            np.append(aor_upper_line, aor_lower_line[::-1]),
            color=self.colors[self.iteration],
            alpha=0.25,
            linewidth=0,
            **kwargs,
        )
        fill_xs_range = np.linspace(max(aor_lower_range), max(aor_upper_range), 30)
        y1 = poly_fit(
            x=[max(aor_lower_range), max(aor_upper_range)],
            y=[max(aor_lower_line), max(aor_upper_line)],
            deg=1,
            x_new=fill_xs_range,
        )
        y2_xs = pump_df[
            (pump_df["flow_100"] >= max(aor_lower_range))
            & (pump_df["flow_100"] <= max(aor_upper_range))
        ]["flow_100"]
        y2_ys = pump_df[pump_df["flow_100"].isin(y2_xs)]["head_100"]

        assert len(y2_xs) == len(y2_ys), "x and y must have same length"
        y2 = poly_fit(x=y2_xs, y=y2_ys, deg=2, x_new=fill_xs_range)
        self.ax.fill_between(
            fill_xs_range,
            y1,
            y2,
            color=self.colors[self.iteration],
            alpha=0.25,
            linewidth=0,
            **kwargs,
        )
        self.AOR_plot = True
        return self.ax

    def plot_por(self, pump_df: pd.DataFrame, **kwargs):
        """
        plots POR filled area for range defined under config.yml -> pump_config -> por_range
        sometimes the POR fill area overlaps the actual pump curve. This can be mitigated by
        increasing the aor_por_fit_degree in the config file, but may not eliminate the overlap
        entirely
        """
        por_upper_range = np.linspace(
            pump_df["por_upper_flow"].dropna().min(),
            pump_df["por_upper_flow"].dropna().max(),
        )
        por_upper_line = poly_fit(
            pump_df["por_upper_flow"].dropna(),
            pump_df["por_upper_head"].dropna(),
            x_new=por_upper_range,
        )
        por_lower_range = np.linspace(
            pump_df["por_lower_flow"].dropna().min(),
            pump_df["por_lower_flow"].dropna().max(),
        )
        por_lower_line = poly_fit(
            pump_df["por_lower_flow"].dropna(),
            pump_df["por_lower_head"].dropna(),
            x_new=por_lower_range,
        )
        self.ax.fill(
            np.append(por_upper_range, por_lower_range[::-1]),
            np.append(por_upper_line, por_lower_line[::-1]),
            color=self.colors[self.iteration],
            alpha=0.35,
            linewidth=0,
            **kwargs,
        )
        fill_xs_range = np.linspace(max(por_lower_range), max(por_upper_range), 30)
        y1 = poly_fit(
            x=[max(por_lower_range), max(por_upper_range)],
            y=[max(por_lower_line), max(por_upper_line)],
            deg=1,
            x_new=fill_xs_range,
        )

        y2_xs = pump_df[
            (pump_df["flow_100"] >= max(por_lower_range))
            & (pump_df["flow_100"] <= max(por_upper_range))
        ]["flow_100"]
        y2_ys = pump_df[pump_df["flow_100"].isin(y2_xs)]["head_100"]

        y2 = poly_fit(x=y2_xs, y=y2_ys, deg=2, x_new=fill_xs_range)
        self.ax.fill_between(
            fill_xs_range,
            y1,
            y2,
            color=self.colors[self.iteration],
            alpha=0.35,
            linewidth=0,
            **kwargs,
        )
        self.POR_plot = True
        return self.ax

    def add_bep_aor_por_legend(self):
        """
        Builds legend for specific plots added.
        """
        legend_handlers = []
        if self.AOR_plot:
            aor_patch = Patch(
                facecolor="grey", edgecolor="white", label="AOR", alpha=0.25
            )
            legend_handlers.append(aor_patch)
        if self.POR_plot:
            por_patch = Patch(
                facecolor="grey", edgecolor="white", label="POR", alpha=0.35
            )
            legend_handlers.append(por_patch)
        if self.BEP_plot:
            bep_patch = Line2D(
                [0],
                [0],
                marker="x",
                color="w",
                label="Best Efficiency Point",
                markeredgecolor="r",
                markersize=10,
                lw=5,
            )
            legend_handlers.append(bep_patch)

        self.ax.legend(
            handles=legend_handlers, loc="upper right", fancybox=True, framealpha=1
        )
        return self.ax

    def create_legend_entry(
        self, legend_text: str, color: str, fill: bool = False, **kwargs
    ):
        """
        adds a manual legend entry.
        Fill creates a filled square, otherwise a line is plotted.
        """
        if fill:
            patch = Patch(
                facecolor=color, edgecolor="white", label=legend_text, **kwargs
            )
        else:
            patch = Line2D([0], [0], color=color, label=legend_text, **kwargs)
        try:
            legend_handles = self.ax.get_legend().axes.legend_.legendHandles
            legend_handles.append(patch)
            self.ax.legend(
                handles=legend_handles, loc="upper right", fancybox=True, framealpha=1
            )
        except AttributeError:
            self.ax.legend(
                handles=[patch], loc="upper right", fancybox=True, framealpha=1
            )
        return self.ax

    def add_vertical_line(self, x_pos: Union[float, int], **kwargs):
        """
        plots vertical line
        """
        self.ax.axvline(x=x_pos, **kwargs)
        return self.ax

    def add_horizontal_line(self, y_pos: Union[float, int], **kwargs):
        """
        plots horizontal line
        """
        self.ax.axhline(y=y_pos, **kwargs)
        return self.ax

    def annotate(self, label: str, pos_to_annotate: Tuple, label_pos: Tuple):
        """
        adds text annotation with label and arrow
        """
        self.ax.annotate(
            label,
            xy=pos_to_annotate,
            xytext=label_pos,
            xycoords="data",
            ha="left",
            arrowprops={"arrowstyle": "-|>"},
        )
        return self.ax


def load_pickle(filename: Union[str, Path]) -> Any:
    """
    loads in pickle files from filename
    """
    if Path(filename).suffix != ".pkl":
        filename = str(filename) + ".pkl"
    with open(filename, "rb") as f:
        _file = pickle.load(f)
    return _file


def save_figure(filename: str):
    """
    save plot to a file directory
    """
    plt.savefig(
        fname=Path("outputs", filename + ".pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Figure Saved as {filename + '.pdf'} in {str(Path('outputs'))}.")


def main(config_file: dict):
    """
    main func
    # TODO This does too much, split up a little
    """
    colors_dict = plot_colors_dict()
    df_list = load_pickle(Path("outputs", "pump_df_list.pkl"))
    sys_curve_df = load_pickle(Path("outputs", "sys_curve_df.pkl"))
    figure = PumpFigure(config_file["plot_config"])  # create base figure object
    figure.build_base_fig()  # build base fig
    figure.plot_system_curves(
        flow=sys_curve_df["system flow"],
        head=[sys_curve_df["max system head"], sys_curve_df["min system head"]],
        fill=True,
        color=colors_dict["purple"],
        alpha=0.65,
        zorder=3,
    )
    figure.plot_system_curves(
        flow=sys_curve_df["system flow"],
        head=[sys_curve_df["1mm max system head"], sys_curve_df["1mm min system head"]],
        fill=True,
        color=colors_dict["purple"],
        alpha=1,
        zorder=3,
    )
    figure.plot_system_curves(
        flow=sys_curve_df["system flow"],
        head=[
            sys_curve_df["forcemain1 max system head"],
            sys_curve_df["forcemain1 min system head"],
        ],
        fill=True,
        color="mediumaquamarine",
        alpha=0.75,
        zorder=1,
    )

    # create lines for each different num of paralle pumps (df)
    for i, df in enumerate(df_list):
        figure.iteration = i
        figure.plot_speed_curves(df)
        figure.plot_BEP_markers(df)
        figure.plot_aor(df)
        figure.plot_por(df)

    # manually creating legend entries
    figure.create_legend_entry("One Pump", color=colors_dict["yellow"])
    figure.create_legend_entry("Two Pumps", color=colors_dict["blue"])
    figure.create_legend_entry("Three Pumps", color=colors_dict["red"])
    figure.create_legend_entry("Four Pumps", color=colors_dict["green"])
    figure.create_legend_entry("30Hz Speed", color='black', linestyle = 'dashed')
    figure.create_legend_entry("FM2 System Curves", color=colors_dict["purple"], fill=True, alpha=0.65)
    figure.create_legend_entry("FM2 1mm System\nCurves", color=colors_dict["purple"], fill=True, alpha=1)
    figure.create_legend_entry("FM1 System Curves", color="mediumaquamarine", fill=True, alpha=0.75)
    figure.create_legend_entry("AOR", color="grey", alpha=0.35, fill=True)
    figure.create_legend_entry("POR", color="grey", alpha=0.5, fill=True)

    # adding annotations
    figure.add_vertical_line(
        x_pos=1350, color=colors_dict["purple"], linestyle="dashed",alpha = 0.35
    )
    figure.add_vertical_line(
        x_pos=1645, color=colors_dict["purple"], linestyle="dashdot", alpha = 0.35
    )
    figure.annotate(
        "1,645 L/s", pos_to_annotate=(1645, 68), label_pos=(1750, 76)
    )
    figure.annotate(
        "1,350 L/s",
        pos_to_annotate=(1350, 65),
        label_pos=(1000, 72),
    )

    # Saving
    if config_file["save_config"]["save_figure"]:
        save_figure(config_file["save_config"]["figure_save_name"])
    plt.show()


if __name__ == "__main__":
    config = read_config_file(Path("config.yml"))
    main(config_file=config)
