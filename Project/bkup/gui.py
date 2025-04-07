import matplotlib
matplotlib.use("Qt5Agg")
import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel
from PyQt5.QtWidgets import QSizePolicy, QComboBox, QCheckBox, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import traceback
from observables import plot_results
from observables import plot_ergotropy_difference
from workers import SimulationWorker


class SER_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.progress_bar = None
        self.status_label = None
        self.ax = None
        self.plot_widget = None
        self.compare_button = None
        self.work_checkbox = None
        self.clear_plot_checkbox = None
        self.feedback_selector = None
        self.slider_coupling = None
        self.label_coupling = None
        self.init_ui()
        self.results_store = {}

    def init_ui(self):
        layout = QVBoxLayout()

        self.label_coupling = QLabel('Coupling Strength: 20 MHz', self)
        layout.addWidget(self.label_coupling)

        self.slider_coupling = QSlider()
        self.slider_coupling.setOrientation(1)  # Horizontal
        self.slider_coupling.setMinimum(10)
        self.slider_coupling.setMaximum(200)
        self.slider_coupling.setValue(20)
        self.slider_coupling.valueChanged.connect(self.update_label)
        layout.addWidget(self.slider_coupling)

        self.feedback_selector = QComboBox()
        self.feedback_selector.addItems(["none", "adaptive", "fixed"])
        layout.addWidget(self.feedback_selector)

        self.clear_plot_checkbox = QCheckBox("Clear previous runs")
        self.clear_plot_checkbox.setChecked(True)  # Default behavior
        layout.addWidget(self.clear_plot_checkbox)

        self.work_checkbox = QCheckBox("Show dErgotropy/dt")
        layout.addWidget(self.work_checkbox)

        self.compare_button = QPushButton("Compare Ergotropy: Adaptive - Fixed")
        self.compare_button.clicked.connect(self.compare_ergotropy)
        layout.addWidget(self.compare_button)

        self.run_button = QPushButton('Run Simulation')
        self.run_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.run_button)

        self.plot_widget = FigureCanvas(Figure(figsize=(8, 4)))
        layout.addWidget(self.plot_widget)
        self.ax = self.plot_widget.figure.subplots()

        self.status_label = QLabel("Idle", self)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # 0 makes it an indeterminate "spinner"
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)
        self.setWindowTitle('SER Simulation GUI')
        self.resize(1000, 600)
        self.show()

    def update_label(self, value):
        self.label_coupling.setText(f'Coupling Strength: {value} MHz')

    def on_simulation_finished(self):
        print("[DEBUG] Entered on_simulation_finished")
        self.status_label.setText("Processing results...")
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)

        if not os.path.exists("results.npz"):
            raise FileNotFoundError("results.npz not found after simulation.")

        data = np.load("results.npz")

        print("[DEBUG] Extracting arrays...")
        t = data['time']
        concurrence = data['concurrence']
        ergotropy = data['ergotropy']
        photons = data['photons']

        # Shape debug
        print(f"[DEBUG] t.shape = {t.shape}")
        print(f"[DEBUG] concurrence.shape = {concurrence.shape}")
        print(f"[DEBUG] ergotropy.shape = {ergotropy.shape}")
        print(f"[DEBUG] photons.shape = {photons.shape}")

        feedback_mode = self.feedback_selector.currentText()
        coupling = self.slider_coupling.value()

        self.results_store[f"{feedback_mode}_{coupling}"] = {
            'time': t,
            'concurrence': concurrence,
            'ergotropy': ergotropy,
            'photons': photons
        }

        if self.clear_plot_checkbox.isChecked():
            print("[DEBUG] Clearing axes...")
            self.ax.clear()

        print("[DEBUG] Plotting...")
        try:
            plot_results(
                self.ax, t, concurrence, ergotropy, photons,
                feedback_mode, coupling,
                show_derivative=self.work_checkbox.isChecked()
            )
            print("[DEBUG] Drawing canvas...")
            self.plot_widget.draw_idle()
            self.plot_widget.flush_events()
        except Exception as e:
            print("⚠️ Crash during plot or canvas draw:", e)

        print("[DEBUG] Plot complete.")
        self.status_label.setText("Done.")

    def on_simulation_error(self, msg):
        self.status_label.setText("⚠️ Simulation failed.")
        self.progress_bar.setVisible(False)
        self.findChild(QPushButton, "Run Simulation").setEnabled(True)
        print("[SIM ERROR] Message from worker:\n", msg)

    def compare_ergotropy(self):
        adaptive_key = f"adaptive_{self.slider_coupling.value()}"
        fixed_key = f"fixed_{self.slider_coupling.value()}"

        if adaptive_key in self.results_store and fixed_key in self.results_store:
            t = self.results_store[adaptive_key]['time']
            erg_adaptive = self.results_store[adaptive_key]['ergotropy']
            erg_fixed = self.results_store[fixed_key]['ergotropy']

            plot_ergotropy_difference(self.ax, t, erg_adaptive, erg_fixed)
            self.plot_widget.draw()
        else:
            print("Run both 'adaptive' and 'fixed' for current coupling first.")

    def run_simulation(self):
        coupling = self.slider_coupling.value()
        feedback_mode = self.feedback_selector.currentText()

        # UI Feedback
        self.status_label.setText("Running simulation...")
        self.progress_bar.setVisible(True)
        self.run_button.setEnabled(False)

        # Start worker thread
        self.worker = SimulationWorker(coupling, feedback_mode)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SER_GUI()
    sys.exit(app.exec_())
