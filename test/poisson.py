import math
import sys  # We need sys so that we can pass argv to QApplication

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout
from pyqtgraph import InfiniteLine


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.graphWidget = pg.PlotWidget()
        self.view = QVBoxLayout()
        self.view.addWidget(self.graphWidget)
        self.setCentralWidget(self.graphWidget)
        # Set background
        self.graphWidget.setBackground('w')
        # Generate random data
        dat = np.random.randint(0, 20, 100)
        av = np.average(dat)
        unique, counts = np.unique(dat, return_counts=True)
        prob = [counts[x] / np.sum(counts) for x in range(len(counts))]
        prob2 = np.random.rand(len(prob))
        poisson = self.poisson(av, np.arange(0, max(unique)))
        bg1 = pg.BarGraphItem(x=unique, height=prob, width=0.8, brush=pg.mkBrush(color=(150, 150, 30, 85)))
        bg2 = pg.BarGraphItem(x=np.arange(0, max(unique)), height=poisson, width=0.8,
                              brush=pg.mkBrush(color=(150, 30, 85, 85)))
        bg3 = pg.BarGraphItem(x=unique, height=prob2, width=0.8, brush=pg.mkBrush(color=(150, 20, 30, 85)))
        self.graphWidget.addItem(bg1)
        self.graphWidget.addItem(bg2)
        self.graphWidget.addItem(bg3)
        self.graphWidget.addItem(
            InfiniteLine(av, angle=90)
        )
        self.graphWidget.setLabel("left", "Probability")
        self.graphWidget.setLabel("bottom", "Foci/Nucleus")
        legend = pg.LegendItem((80, 60), offset=(70, 20))
        legend.setParentItem(self.graphWidget.getPlotItem())
        pl_item = 'Poisson'
        cl_item = 'Channel_1'
        print(pl_item)
        print(cl_item)
        legend.addItem(bg1, pl_item)
        legend.addItem(bg2, cl_item)
        legend.addItem(bg3, cl_item)

    def poisson(self, lam, test):
        if isinstance(test, list) or isinstance(test, np.ndarray):
            res = []
            for el in test:
                res.append(self.poisson(lam, el))
            return res
        else:
            p = ((lam ** test) / math.factorial(test)) * math.exp(-lam)
            return p

        """
        # Define window and axis titles
        self.graphWidget.setTitle("Comparison to Poisson Distribution", color=(255, 0, 255), size="15pt")
        self.graphWidget.setLabel('left', 'Temperature', units="°C", color=(255, 0, 0), size=30)
        self.graphWidget.setLabel('bottom', 'Hour', units="H", color=(0, 255, 0), size=30)
        # Add background grid
        self.graphWidget.showGrid(x=True, y=True)
        # Define pen for plotting
        pen = pg.mkPen(color=(255, 0, 0), width=3, style=QtCore.Qt.DashLine)
        pen2 = pg.mkPen(color=(0, 255, 0), width=3, style=QtCore.Qt.DotLine)
        # Set the axis limits
        self.graphWidget.setXRange(0, 12)
        self.graphWidget.setYRange(0, 50)
        # Enable menu
        hour = [1,2,3,4,5,6,7,8,9,10]
        temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]
        temp2 = [18, 15, 12, 17, 19, 10, 7, 8, 14, 12]

        # plot data: x, y values &nbsp;&nbsp; -> add two spaces, needed for legend
        g1 = self.graphWidget.plot(hour, temperature, pen=pen, symbol="+", name="&nbsp;&nbsp;Example 1")
        g2 = self.graphWidget.plot(hour, temp2, pen=pen2, symbol="o", name="&nbsp;&nbsp;Example 2")
        # Define bar chart
        y1 = np.linspace(0, 20, num=20)
        #   create horizontal list
        x = np.arange(20)
        # Define pen for bar chart
        pen3 = pg.mkPen(color=(155, 150, 150, 30), width=6)
        # create bar chart
        bg1 = pg.BarGraphItem(x=x, height=y1, width=0.8, brush=pg.mkBrush(color=(150, 150, 30, 100)),
                              name="Test BarChart")
        top = np.arange(0, 20)
        # Define error bars
        err1 = pg.ErrorBarItem(x=x, y=y1, top=top, beam=0.25)
        self.graphWidget.addItem(bg1)
        self.graphWidget.addItem(err1)
        # Add legend
        legend = pg.LegendItem((80, 60), offset=(70, 20))
        legend.setParentItem(self.graphWidget.getPlotItem())
        legend.addItem(g1, "&nbsp;&nbsp;Example 1")
        legend.addItem(g2, "&nbsp;&nbsp;Example 2")
        legend.addItem(bg1, '<font size="4" color=#96961e>▮</font>Example 3')
        """
        self.setLayout(self.view)


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()