import pandas as pd
import seaborn as sns
from PyQt5.QtWidgets import QSizePolicy
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from statannotations.Annotator import Annotator

# This module plots through seaborn/matplotlib and draws nothing by hand. It used to also carry
# three pyqtgraph widgets -- BoxPlotWidget, BoxPlotItem and PoissonPlotWidget -- that painted box
# plots and a Poisson comparison with QPainter, manual whisker geometry and a x10 coordinate trick.
# All three were unreachable (nothing constructed them; PoissonPlotWidget's last use was deleted at
# the 1.0 release, and BoxPlotWidget's last substantive commit is titled "Statistics currently
# disabled"), and they were removed rather than repaired: the statistics dialog delegates its
# plotting to seaborn, so if a box plot or a Poisson comparison returns it is written against
# seaborn, not revived from hand-rolled drawing code.


STANDARD_SETTINGS = {
    # --- Figure geometry & layout ---
    "figure.figsize": (8.8, 6.0),      # single-column default
    "figure.dpi": 100,                           # on-screen DPI
    "savefig.dpi": 300,                          # print/export DPI
    "figure.constrained_layout.use": True,       # robust auto-layout engine
    "figure.autolayout": False,                  # avoid conflict with constrained_layout
    "savefig.bbox": "tight",                     # crop to content on export
    "savefig.pad_inches": 0.04,                  # 0.04 in ≈ 1.0 mm padding
    "figure.facecolor": "white",
    "axes.facecolor": "white",

    # --- Fonts & text ---
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 9.0,                 # base font size in pt
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "legend.title_fontsize": 8.0,
    "mathtext.fontset": "dejavusans", # consistent sans math
    "text.color": "#222222",          # near-black for print readability

    # --- Axes & spines ---
    "axes.linewidth": 0.8,            # pt; ~0.28 mm
    "axes.edgecolor": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,           # grid behind data
    "axes.grid": False,               # off by default for publications
    # If you want a light grid globally, set True + grid.* below.

    # --- Grid (used only if axes.grid=True) ---
    "grid.color": "#d0d0d0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,

    # --- Lines & markers ---
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "lines.markeredgewidth": 0.6,

    # --- Ticks ---
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": True,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,

    # --- Legends ---
    "legend.frameon": True,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.6,
    "legend.borderaxespad": 0.4,

    # --- Images / colormaps ---
    "image.cmap": "viridis",          # perceptually uniform default

    # --- Export to vector formats (journal-friendly fonts) ---
    "pdf.fonttype": 42,               # embed TrueType, editable in Illustrator/Inkscape
    "ps.fonttype": 42,
}

# Its own dict, not a second name for the one above. `STANDARD_SETTINGS = PUB_RC = {...}` bound one
# object to two names, so an in-place edit through either changed both -- and, while this was also
# plot()'s default argument, every later call as well.
PUB_RC = dict(STANDARD_SETTINGS)

COLOR_PALETTES = ["deep", "muted","pastel", "bright", "dark", "colorblind",
                  "tab20", "tab20b", "tab20c", "husl", "Set1", "Set2",
                  "Set3", "Paired", "Accent", "Pastel1", "Pastel2", "Dark2"
                  ]

class PlotCanvas(FigureCanvasQTAgg):
    plot_types = (
        "Violin Plot",
        "Boxplot"
    )

    def __init__(self, parent=None, width=10, height=10, dpi=100):
        self.fig = Figure(figsize=(width, height),
                          dpi=dpi,
                          layout="constrained")
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.plot_type = self.plot_types[0]
        # The columns the last plot() call mapped to each axis. display_statistics annotates against
        # these; the vertical mapping is the default so the attributes are never undefined
        self._x_column = "Group"
        self._y_column = "Foci"
        self.setParent(parent)
        # Optional: make it expand with the window
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def resizeEvent(self, event):
        # keep layout consistent with current canvas size
        self.draw_idle()
        super().resizeEvent(event)

    def plot(self,
             data: pd.DataFrame,
             specific_settings: dict,
             plot_type: str = plot_types[0],
             ordering=None,
             title="Violin Plot",
             settings: dict = None) -> None:
        """
        Method to create the violin plot

        :param data: The date to display
        :param specific_settings: The per-plot settings. REQUIRED -- it used to default to None
                                  while the body indexed it unconditionally, so calling this method
                                  with its own defaults raised TypeError. Both call sites always
                                  passed it; the signature now says so rather than advertising an
                                  option that never worked
        :param plot_type: Which of plot_types to draw. Named plot_type, not type, so it does not
                          shadow the builtin
        :param labels: The labels for the data. Determines the plot ordering, if provided
        :param title: The title of the plot
        :param settings: matplotlib rc parameters. Defaults to STANDARD_SETTINGS, resolved here
                         rather than in the signature -- a mutable object as a default argument is
                         shared by every call, so one in-place edit would have changed the defaults
                         for the rest of the session
        :return: None
        """
        with mpl.rc_context(STANDARD_SETTINGS if settings is None else settings):
            # Clear the axis
            self.ax.clear()
            if not ordering:
                ordering = sorted(data["Group"].unique())
            # Remembered so display_statistics can annotate against the same axes. It hardcoded
            # x="Group", y="Foci" and therefore placed every annotation against the wrong axis on a
            # horizontal plot; deriving both here means the two cannot drift apart again
            self._x_column = "Group" if specific_settings["orientation"] == "vertical" else "Foci"
            self._y_column = "Foci" if specific_settings["orientation"] == "vertical" else "Group"
            # Sized by the column it actually colours. It was sized by the number of GROUPS while
            # hue is Channel, so seaborn recycled or truncated colours whenever the two counts
            # differed -- two channels could come out the same colour
            palette = sns.color_palette(specific_settings["palette"], data["Channel"].nunique())
            if plot_type == self.plot_types[0]:
                sns.violinplot(
                    data=data,
                    x=self._x_column,
                    y=self._y_column,
                    hue="Channel",
                    ax=self.ax,
                    split = specific_settings["violin_split"],
                    inner = specific_settings["violin_inner"],
                    order = ordering,
                    palette = palette,
                    linewidth = 2,
                    cut = 0,
                    bw_adjust = 0.75,
                    legend=specific_settings["show_legend"]
                )
            elif plot_type == self.plot_types[1]:
                sns.boxplot(
                    data=data,
                    x=self._x_column,
                    y=self._y_column,
                    hue="Channel",
                    ax=self.ax,
                    order = ordering,
                    palette=palette,
                    linewidth=2,
                    legend=specific_settings["show_legend"]
                )
            else:
                # There was no else. An unrecognised type drew no artist at all and then fell into
                # the legend block below, where get_legend() returns None -- so a typo produced an
                # AttributeError from a line that has nothing to do with the cause. The combo box is
                # populated from plot_types, so this is unreachable from the UI and is a programming
                # error when it happens
                raise ValueError(f"Unknown plot type '{plot_type}'; expected one of {self.plot_types}")
            # Seaborn does not always produce a legend even when asked -- a single hue level is
            # enough for it not to -- so this is checked rather than assumed
            legend = self.ax.get_legend()
            if specific_settings["show_legend"] and legend is not None:
                legend.get_title().set_fontsize(specific_settings["legend_fontsize"] + 2)
                for text in legend.get_texts():
                    text.set_fontsize(specific_settings["legend_fontsize"])
            self.ax.set_title(title)
            self.ax.title.set_visible(specific_settings["show_title"])
            self.ax.set_xlabel("Groups" if specific_settings["orientation"] == "vertical" else "Foci")
            self.ax.xaxis.label.set_visible(specific_settings["show_xlabel"])
            self.ax.set_ylabel("Foci" if specific_settings["orientation"] == "vertical" else "Groups")
            self.ax.yaxis.label.set_visible(specific_settings["show_ylabel"])
            self.ax.grid(visible=specific_settings["show_grid"])
            self.draw()
            self.draw_idle()

    def display_statistics(self,
                           data: pd.DataFrame,
                           statistics: pd.DataFrame,
                           ordering=None,
                           show_ns=True) -> None:
        if statistics is None:
            return
        pairs = []
        pvals = []
        ordering = sorted(statistics["Group"].unique()) if not ordering else ordering
        # Iterate over each group and get the statistics data for it
        for group in ordering:
            group_data = statistics[statistics["Group"] == group]
            pairs_a = group_data[["Group", "Channel"]].to_numpy()
            pairs_b = group_data[["Tested Against", "Channel"]].to_numpy()
            pairs.extend([(tuple(x), tuple(y)) for x, y in zip(pairs_a, pairs_b)])
            pvals.extend(group_data["p-Value"].values)
        # Create an annotator
        # The same axes plot() used, not a hardcoded pair. This is always called on a canvas that
        # has just been plotted -- plot_statistics() calls plot_data() first -- so the remembered
        # mapping describes the figure being annotated
        annot = Annotator(self.ax,
                          pairs,
                          data=data,
                          x=self._x_column,
                          y=self._y_column,
                          hue="Channel",
                          order=ordering)
        # Configure the annotator
        annot.configure(test=None,  # we already have p-values
                        text_format="star",  # ‘star’, ‘simple’, or ‘full’
                        hide_non_significant=not show_ns,
                        loc="outside",  # draw outside the plot area
                        fontsize=12,
                        comparisons_correction=None)
        annot.set_pvalues(pvals)
        annot.annotate()
        self.ax.set_title("")
        self.draw()
