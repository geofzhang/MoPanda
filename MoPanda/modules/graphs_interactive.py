import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg


class WellLogViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Create pyqtgraph PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        # Sample plot data
        self.plot_widget.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])

        self.setWindowTitle("Well Log Viewer")
        self.setGeometry(100, 100, 800, 600)


app = QApplication(sys.argv)
window = WellLogViewer()
window.show()
sys.exit(app.exec())
