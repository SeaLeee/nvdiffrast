"""
Monitor Panel - Real-time optimization metrics and loss curves.
"""

from collections import deque
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QGridLayout,
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class LossPlotCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas embedded in Qt for plotting loss curves."""

    def __init__(self, parent=None, max_points=5000):
        self.fig = Figure(figsize=(6, 2.5), dpi=100)
        self.fig.set_facecolor('#161b22')
        super().__init__(self.fig)

        self.ax_loss = self.fig.add_subplot(121)
        self.ax_psnr = self.fig.add_subplot(122)

        for ax in [self.ax_loss, self.ax_psnr]:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#8b949e', labelsize=8)
            ax.spines['bottom'].set_color('#30363d')
            ax.spines['left'].set_color('#30363d')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        self.ax_loss.set_title('Loss', color='#e6edf3', fontsize=10)
        self.ax_psnr.set_title('PSNR (dB)', color='#e6edf3', fontsize=10)

        self.fig.tight_layout(pad=1.5)

        self.max_points = max_points
        self.iterations = deque(maxlen=max_points)
        self.losses = deque(maxlen=max_points)
        self.psnrs = deque(maxlen=max_points)

        self._loss_line = None
        self._psnr_line = None

    def add_point(self, iteration: int, loss: float, psnr: float):
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.psnrs.append(psnr)

        # Redraw every 10 iterations to avoid UI lag
        if iteration % 10 == 0:
            self._redraw()

    def _redraw(self):
        iters = list(self.iterations)

        self.ax_loss.clear()
        self.ax_loss.plot(iters, list(self.losses), color='#58a6ff', linewidth=1.2)
        self.ax_loss.set_title('Loss', color='#e6edf3', fontsize=10)
        self.ax_loss.set_yscale('log')

        self.ax_psnr.clear()
        self.ax_psnr.plot(iters, list(self.psnrs), color='#3fb950', linewidth=1)
        self.ax_psnr.set_title('PSNR (dB)', color='#e6edf3', fontsize=10)

        for ax in [self.ax_loss, self.ax_psnr]:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#8b949e', labelsize=8)

        self.fig.tight_layout(pad=1.5)
        self.draw()

    def clear(self):
        self.iterations.clear()
        self.losses.clear()
        self.psnrs.clear()
        self.ax_loss.clear()
        self.ax_psnr.clear()
        self.draw()


class MonitorPanel(QWidget):
    """Real-time optimization monitoring with loss curves and metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Stats group
        stats_group = QGroupBox("Current Metrics")
        stats_layout = QGridLayout(stats_group)

        self.lbl_iter = QLabel("0")
        self.lbl_iter.setProperty("cssClass", "accent")
        self.lbl_loss = QLabel("--")
        self.lbl_loss.setProperty("cssClass", "accent")
        self.lbl_psnr = QLabel("--")
        self.lbl_psnr.setProperty("cssClass", "accent")
        self.lbl_lr = QLabel("--")
        self.lbl_gpu = QLabel("--")
        self.lbl_gpu.setProperty("cssClass", "accent")

        stats_layout.addWidget(QLabel("Iteration:"), 0, 0)
        stats_layout.addWidget(self.lbl_iter, 0, 1)
        stats_layout.addWidget(QLabel("Loss:"), 0, 2)
        stats_layout.addWidget(self.lbl_loss, 0, 3)
        stats_layout.addWidget(QLabel("PSNR:"), 1, 0)
        stats_layout.addWidget(self.lbl_psnr, 1, 1)
        stats_layout.addWidget(QLabel("GPU Mem:"), 1, 2)
        stats_layout.addWidget(self.lbl_gpu, 1, 3)

        layout.addWidget(stats_group)

        # Loss plot
        self.plot = LossPlotCanvas()
        layout.addWidget(self.plot)

    def update_metrics(self, iteration: int, loss: float, psnr: float = 0,
                       lr: float = 0):
        self.lbl_iter.setText(str(iteration))
        self.lbl_loss.setText(f"{loss:.6f}")
        self.lbl_psnr.setText(f"{psnr:.2f} dB")

        # GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / (1024 ** 3)
                self.lbl_gpu.setText(f"{mem:.2f} GB")
        except Exception:
            pass

        self.plot.add_point(iteration, loss, psnr)

    def clear(self):
        self.lbl_iter.setText("0")
        self.lbl_loss.setText("--")
        self.lbl_psnr.setText("--")
        self.lbl_gpu.setText("--")
        self.plot.clear()

    def reset(self):
        """Alias for clear - reset all metrics."""
        self.clear()
