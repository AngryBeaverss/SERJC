from PyQt5.QtCore import QThread, pyqtSignal
import os
import subprocess
import sys


script_path = os.path.join(os.path.dirname(__file__), 'run.py')


class SimulationWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, coupling, feedback_mode, param_overrides=None):
        super().__init__()
        self.coupling = coupling
        self.feedback_mode = feedback_mode
        self.param_overrides = param_overrides or {}

    def run(self):
        try:
            script_path = os.path.join(os.path.dirname(__file__), 'run.py')
            command = [
                sys.executable, script_path,
                f'--coupling={self.coupling}',
                f'--feedback_mode={self.feedback_mode}'
            ]

            for k, v in self.param_overrides.items():
                command.append(f'--{k}={v}')

            print("[WORKER] Launching subprocess with command:", command)

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(script_path)
            )
            stdout, stderr = process.communicate()

            print("[WORKER] Subprocess finished with code", process.returncode)
            print("[WORKER] Command:", command)
            print("[WORKER] STDOUT:\n", stdout.decode())
            print("[WORKER] STDERR:\n", stderr.decode())

            if process.returncode != 0:
                self.error.emit(f"Subprocess failed (code {process.returncode}):\n{stderr.decode()}")
            else:
                self.finished.emit()

        except Exception as e:
            self.error.emit(f"Worker crashed:\n{str(e)}")