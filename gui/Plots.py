from typing import List, Dict, Union

from PyQt5 import QtGui
from PyQt5.QtCore import QRectF, QLine, QPointF
from PyQt5.QtGui import QPainter, QColor

import numpy as np
import math
import pyqtgraph as pg
from pyqtgraph import InfiniteLine


class BoxPlotWidget(pg.PlotWidget):
    """
    Widget to hold a BoxPlotItem
    """

    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.boxPlotItem = BoxPlotItem(**kargs)
        self.laxis = self.plotItem.getAxis("left")
        self.baxis = self.plotItem.getAxis("bottom")
        self.setBackground("w")
        self.showGrid(x=True, y=True)
        self.addItem(self.boxPlotItem)
        self._initialize_axis(**kargs)
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
        # TODO test
        #p.scale(10, 10)
        max = -1
        # Get max y value
        if num_data == 1:
            for val in raw_data:
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
            data = self._calculate_plotting_data(raw_data[i] if num_data > 1 else raw_data)
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
        pdata["outliers"] = [x for x in data if x < pdata["min"] or x > pdata["max"]]
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
        self.setBackground("w")
        self.showGrid(x=True, y=True)
        self.plotItem.setLabel("left", "Probability [%]")
        self.data = kargs.get("data", [])
        self.data_name = kargs.get("label", "Channel")
        self.data_graph = None
        self.poisson_graph = None
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
        self.data_graph = pg.BarGraphItem(x=unique, height=prob, width=0.8, brush=pg.mkBrush(color=(150, 50, 30, 125)))
        self.poisson_graph = pg.BarGraphItem(x=np.arange(0, max(unique)), height=poisson, width=0.8,
                                             brush=pg.mkBrush(color=(85, 30, 150, 125)))
        # Add indicator for data average
        self.addItem(InfiniteLine(av, angle=90, pen=pg.mkPen(color=(0, 255, 0, 255))))
        self.addItem(self.poisson_graph)
        self.addItem(self.data_graph)
        self.setToolTip("Red: Data Distribution\nBlue: Poisson Distribution\nGreen: Average")
        # Add Legend
        # TODO
        """
        legend = pg.LegendItem((80, 60), offset=(70, 20))
        legend.setParentItem(self.getPlotItem())
        legend.addItem(self.data_graph, f"<font size='4' color=#96961e>▮</font>{self.data_name:>10}")
        legend.addItem(self.poisson_graph, f"<font size='4' color=#96961e>▮</font>{'Poisson':>10}")
        """

    def poisson(self, lam: float, test: Union[list, np.ndarray, ]):
        """
        Recursive method to calculate the probability of elements to occur according to the poisson distrubution

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
            p = ((lam ** test) / math.factorial(test)) * math.exp(-lam)
            return p
