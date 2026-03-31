"""
Loss Chart Widget - Standalone loss curve widget using matplotlib.
"""

from collections import deque
from PyQt6.QtWidgets import QWidget, QVBoxLayout

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class LossChartWidget(QWidget):
    """Compact loss curve chart for embedding in panels."""

    def __init__(self, parent=None, max_points=5000):
        super().__init__(parent)
        self.max_points = max_points
        self.iterations = deque(maxlen=max_points)
        self.values = {}  # name -> deque

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(5, 2), dpi=100)
        self.fig.set_facecolor('#161b22')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0d1117')
        self.ax.tick_params(colors='#8b949e', labelsize=8)

        layout.addWidget(self.canvas)

    def add_series(self, name: str):
        """Register a new data series."""
        self.values[name] = deque(maxlen=self.max_points)

    def add_point(self, iteration: int, **kwargs):
        """Add data points. kwargs are series_name=value pairs."""
        self.iterations.append(iteration)
        for name, val in kwargs.items():
            if name not in self.values:
                self.values[name] = deque(maxlen=self.max_points)
            self.values[name].append(val)

        if iteration % 20 == 0:
            self._redraw()

    def _redraw(self):
        self.ax.clear()
        iters = list(self.iterations)
        colors = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff']

        for i, (name, vals) in enumerate(self.values.items()):
            color = colors[i % len(colors)]
            self.ax.plot(iters[:len(vals)], list(vals),
                         color=color, linewidth=1, label=name)

        self.ax.legend(fontsize=7, facecolor='#21262d', edgecolor='#30363d',
                       labelcolor='#e6edf3')
        self.ax.set_facecolor('#0d1117')
        self.ax.tick_params(colors='#8b949e', labelsize=8)
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw()

    def clear(self):
        self.iterations.clear()
        self.values.clear()
        self.ax.clear()
        self.canvas.draw()
