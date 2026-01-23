import math
import numpy as np
import pandas as pd
import pyqtgraph as pg
import seaborn as sns
from typing import List, Dict, Union, Iterable
from PyQt5 import QtGui
from PyQt5.QtCore import QRectF, QLine, QPointF
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import QSizePolicy
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pyqtgraph import InfiniteLine
from statannotations.Annotator import Annotator # TODO entweder in den Requierements erwähnen oder wieder entfernen


STANDARD_SETTINGS = PUB_RC = {
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
             type:str = plot_types[0],
             ordering=None,
             title="Violin Plot",
             settings=STANDARD_SETTINGS,
             specific_settings = None) -> None:
        """
        Method to create the violin plot

        :param data: The date to display
        :param labels: The labels for the data. Determines the plot ordering, if provided
        :param title: The title of the plot
        :return: None
        """
        with mpl.rc_context(settings):
            # Clear the axis
            self.ax.clear()
            if not ordering:
                ordering = sorted(data["Group"].unique())
            if type == self.plot_types[0]:
                sns.violinplot(
                    data=data,
                    x="Group" if specific_settings["orientation"] == "vertical" else "Foci",
                    y="Foci" if specific_settings["orientation"] == "vertical" else "Group",
                    hue="Channel",
                    ax=self.ax,
                    split = specific_settings["violin_split"],
                    inner = specific_settings["violin_inner"],
                    order = ordering,
                    palette = sns.color_palette(specific_settings["palette"], len(data["Group"].unique())),
                    linewidth = 2,
                    cut = 0,
                    bw_adjust = 0.75,
                    legend=specific_settings["show_legend"]
                )
            elif type == self.plot_types[1]:
                sns.boxplot(
                    data=data,
                    x="Group" if specific_settings["orientation"] == "vertical" else "Foci",
                    y="Foci" if specific_settings["orientation"] == "vertical" else "Group",
                    hue="Channel",
                    ax=self.ax,
                    order = ordering,
                    palette=sns.color_palette(specific_settings["palette"], len(data["Group"].unique())),
                    linewidth=2,
                    legend=specific_settings["show_legend"]
                )
            if specific_settings["show_legend"]:
                self.ax.get_legend().get_title().set_fontsize(specific_settings["legend_fontsize"] + 2)
                for text in self.ax.get_legend().get_texts():
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
        annot = Annotator(self.ax,
                          pairs,
                          data=data,
                          x="Group",
                          y="Foci",
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



class BoxPlotWidget(pg.PlotWidget):
    """
    Widget to hold a BoxPlotItem
    """

    # TODO Anpassen um es als Statistik-Plot verwenden zu können

    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.boxPlotItem = BoxPlotItem(**kargs)
        self.p_data = self.boxPlotItem.p_data
        self.laxis = self.plotItem.getAxis("left")
        self.baxis = self.plotItem.getAxis("bottom")
        self.setBackground("w")
        self.showGrid(x=True, y=True)
        self.addItem(self.boxPlotItem)
        self._initialize_axis(**kargs)
        self.setMinimumSize(600, 400)
        self.setToolTip("Median: Red Horizontal Line\nAverage: Yellow Horizontal Line")
        # Get max value of all data
        max = -1
        for dat in kargs["data"]:
            for dat2 in dat:
                if dat2 > max:
                    max = dat2
        self.boxPlotItem.getViewBox().setAspectLocked(True, max * 0.75)

    def _initialize_axis(self, **kargs) -> None:
        """
        Method to initialize the axis of the plot Widget

        :param kargs: Dictionary containing the data and given groups
        :return: None
        """
        self.laxis.setScale(0.1)
        self.baxis.setScale(0.1)
        dummy_groups = ["A", "B", "C", "D", "E", "F"]
        axis_labels = []
        if "groups" not in kargs:
            kargs["groups"] = []
            for i in range(len(kargs["data"])):
                kargs["groups"].append(dummy_groups[i % len(dummy_groups)])
        if len(kargs["groups"]) < len(kargs["data"]):
            # Fill groups if not all data arrays were assigned a group
            for i in range(len(kargs["data"]) - len(kargs["groups"])):
                kargs["groups"].append(dummy_groups[i % len(dummy_groups)])
        for i in range(len(kargs["groups"])):
            axis_labels.append(((i + 1) * 10, kargs["groups"][i]))
        self.baxis.setTicks([axis_labels])


class BoxPlotItem(pg.GraphicsObject):
    """
    Class to create a box plot with pyqtgraph
    """
    FILL_COLORS = [
        QColor(200, 50, 20),  # Red
        QColor(50, 200, 20),  # Blue
        QColor(20, 30, 200)   # Green
    ]

    def __init__(self, **kwargs) -> None:
        pg.GraphicsObject.__init__(self)
        self.kwargs = kwargs
        self.p_data = []
        self.generate_picture(self.kwargs)

    def generate_picture(self, kwargs) -> None:
        """
        Method to plot the dictionary data.
        The dictionary should contain following keys:
        groups: Iterable containing the labels for each group. If the group key is present, data will be assumed to be
        split into tuples containing the data for each group
        data: The data to plot.

        :param data: The data to plot
        :return: None
        """
        # Check if the dict is empty or does not contain data to plot
        if not kwargs or "data" not in kwargs:
            return
        # Iterate over the data
        raw_data = kwargs["data"]
        self.picture = QtGui.QPicture()
        outlines = pg.mkPen("w", width=3)
        median = pg.mkPen("r", width=2)
        average = pg.mkPen("y", width=2)
        fill = pg.mkBrush((150, 150, 30))
        num_data = len(raw_data) if isinstance(raw_data[0], (list, tuple)) else 1
        p = QtGui.QPainter(self.picture)
        #p.scale(10, 10)
        max = -1
        # Get max y value
        if num_data == 1:
            for val in raw_data[0]:
                if val > max:
                    max = val
        else:
            for data in raw_data:
                for val in data:
                    if val > max:
                        max = val
        for i in range(num_data):
            fill.setColor(self.FILL_COLORS[i % 3])
            outlines.setColor(self.FILL_COLORS[i % 3].lighter())
            num = i * 10
            data = self._calculate_plotting_data(raw_data[i] if num_data > 1 else raw_data[0])
            self.p_data.append(data)
            # Set up the painter
            p.setPen(outlines)
            p.setBrush(fill)
            # Get the interquartile range of the box
            iqr = data["iqr"]
            outliers = data["outliers"]
            # Draw the box
            p.drawRect(
                QRectF(num + 8, data["q25"] * 10, 4, iqr * 10)
            )
            p.setPen(median)
            # Draw median
            p.drawLine(
                QLine(num + 8, data["median"] * 10, num + 12, data["median"] * 10)
            )
            p.setPen(average)
            p.drawLine(
                QLine(num + 8, data["average"] * 10, num + 12, data["average"] * 10)
            )
            p.setPen(outlines)
            # Draw Whiskers
            # Top Whisker
            p.drawLine(
                QLine(num + 10, data["max"] * 10, num + 10, data["q75"] * 10)
            )
            p.drawLine(
                QLine(num + 9, data["max"] * 10, num + 11, data["max"] * 10 )
            )
            # Bottom Whisker
            p.drawLine(
                QLine(num + 10, data["min"] * 10, num + 10, data["q25"] * 10)
            )
            p.drawLine(
                QLine(num + 9, data["min"] * 10, num + 11, data["min"] * 10)
            )
            # Draw outliers
            for outlier in sorted(outliers):
                p.drawEllipse(QPointF(num + 10, outlier * 10), 0.15, 0.225 * max/num_data)
        p.end()

    @staticmethod
    def _calculate_plotting_data(data: List[int]) -> Dict:
        """
        Private method to calculate the needed data for plotting such as median, quartiles and outliers.

        :param data: The data points
        :return: The plotting data
        """
        pdata = {
            "median": np.median(data),
            "average": np.average(data),
            "q25": np.quantile(data, 0.25),
            "q75": np.quantile(data, 0.75),
        }
        pdata["iqr"] = pdata["q75"] - pdata["q25"]
        pdata["min"] = pdata["q25"] - 1.5 * pdata["iqr"]
        pdata["max"] = pdata["q75"] + 1.5 * pdata["iqr"]
        pdata["outliers"] = set([x for x in data if x < pdata["min"] or x > pdata["max"]])
        pdata["number"] = len(data) - len(pdata["outliers"])
        return pdata

    def paint(self, p: QPainter, *args) -> None:
        """
        Method to paint this Widget

        :param p: The painter to paint this widget with
        :return: None
        """
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self) -> QRectF:
        """
        Method to get the bounding rect of this widget

        :return: The bounding rect
        """
        return QRectF(self.picture.boundingRect())


class PoissonPlotWidget(pg.PlotWidget):
    """
    Class to compare the given data distribution to a poisson plot
    """

    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.setMinimumSize(600, 400)
        self.setBackground("w")
        self.showGrid(x=True, y=True)
        self.plotItem.setLabel("left", "Probability [%]")
        self.data = kargs.get("data", [])
        self.data_name = kargs.get("label", "Channel")
        self.data_graph = None
        self.data_graph_line = None
        self.poisson_graph = None
        self.poisson_graph_line = None
        self.data_average_line = None
        self.prepare_plot()

    def set_data(self, data: Iterable[float], title: str) -> None:
        """
        Method to change the displayed data

        :param data: The data to display as list of floats
        :param title: The title to display
        :return: None
        """
        self.removeItem(self.data_graph)
        self.removeItem(self.poisson_graph)
        self.removeItem(self.data_average_line)
        self.data = data
        self.setTitle(title)
        self.prepare_plot()

    def prepare_plot(self) -> None:
        """
        Method to prepare the plot

        :return: None
        """
        # If the data array is empty, return
        if not self.data:
            return
        # Calculate the average
        av = np.average(self.data)
        # Get the unique elements of data and their counts
        unique, counts = np.unique(self.data, return_counts=True)
        # Calculate the occurence probability of all unique elements
        prob = [counts[x] / np.sum(counts) for x in range(len(counts))]
        # Calculate the probability of elements to occur according to the poisson distrubution
        poisson = self.poisson(av, np.arange(0, max(unique)))
        # Prepare bar graphs
        self.data_graph = pg.BarGraphItem(x=unique-0.1, height=prob,
                                          width=0.2, brush=pg.mkBrush(color=(255, 50, 30, 255)))
        self.data_graph_line = pg.PlotDataItem(unique-0.1, prob, pen=pg.mkPen(color=(255, 50, 30, 100), width=5))
        self.poisson_graph = pg.BarGraphItem(x=np.arange(0, max(unique))+0.1, height=poisson, width=0.2,
                                             brush=pg.mkBrush(color=(85, 30, 255, 255)))
        self.poisson_graph_line = pg.PlotDataItem(np.arange(0, max(unique))+0.1, poisson,
                                                  pen=pg.mkPen(color=(85, 30, 255, 100), width=5))
        self.data_average_line = InfiniteLine(av, angle=90, pen=pg.mkPen(color=(0, 255, 0, 255)))
        # Add indicator for data average
        self.addItem(self.data_average_line)
        # self.addItem(self.data_graph_line)
        # self.addItem(self.poisson_graph_line)
        self.addItem(self.poisson_graph)
        self.addItem(self.data_graph)

        self.setToolTip("Red: Data Distribution\nBlue: Poisson Distribution\nGreen: Average")
        # Add Legend
        legend = pg.LegendItem((80, 60), offset=(70, 20))
        legend.setParentItem(self.getPlotItem())
        legend.addItem(self.data_graph, f"<font size='4' color=#96961e>▮</font>{self.data_name:>10}")
        legend.addItem(self.poisson_graph, f"<font size='4' color=#96961e>▮</font>{'Poisson':>10}")

    def poisson(self, lam: float, test: Union[list, np.ndarray]):
        """
        Recursive method to calculate the probability of elements to occur according to the poisson distribution

        :param lam: The average of the distribution
        :param test: Either array of elements to test or one element to test
        :return: List of probabilities or the probability for the given element
        """
        if isinstance(test, list) or isinstance(test, np.ndarray):
            res = []
            for el in test:
                res.append(self.poisson(lam, el))
            return res
        else:
            try:
                return ((lam ** test) / math.factorial(test)) * math.exp(-lam)
            except OverflowError:
                return 0
