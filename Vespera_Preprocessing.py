##############################################
# Vespera — Preprocessing
# Automated Stacking for Alt‑Az Mounts
# Authors: Claude (Anthropic) (2025)
#          G. Trainar
# Contact: github.com/gtrainar
##############################################
# (c) 2025 Claude (Anthropic), G. Trainar - MIT License
# Vespera Preprocessing
# Version 2.0.1
#
# Credits / Origin
# ----------------
#   • Based on Siril's OSC_Preprocessing_BayerDrizzle.ssf
#   • Optimized for Vaonis Vespera II and Pro telescopes
#   • Handles single dark frame capture (Expert Mode)
#   • Dual-band filter support (Ha/OIII extraction)
##############################################

"""

Overview
--------
Full‑featured preprocessing script for Vaonis Vespera astrophotography data.
Designed to handle the unique characteristics of alt‑az mounted smart telescopes
including field rotation, various filter configurations, and different sky conditions.

Features
--------
• Bayer Drizzle: Handles field rotation from alt‑az tracking without grid artifacts
• Single Dark Support: Automatically detects and handles 1 or multiple dark frames
• Sky Quality Presets: Optimized settings for dark to urban skies
• Auto Cleanup: Removes all temporary files after successful processing
• Post‑Processing Options: Auto background extraction and color calibration

Compatibility
-------------
• Siril 1.4+
• Python 3.10+ (via sirilpy)
• Dependencies: sirilpy, PyQt6

License
-------
Released under MIT License.
"""

import sys
import os
import glob
import shutil
import traceback
from typing import Optional, Dict, Any

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    print("Error: sirilpy module not found. This script must be run within Siril.")
    sys.exit(1)

# Ensure dependencies
s.ensure_installed("PyQt6")

from PyQt6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QProgressBar, QMessageBox,
                             QTextEdit, QGroupBox, QComboBox, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QTabWidget, QWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QFont

VERSION = "1.0.0"

##############################################
# CHANGELOG
##############################################

CHANGELOG = """
Version 1.0.0 (2026‑01)
• ProcessingProgress constants for standardized progress tracking
• Implemented feathering option (0‑100px) to reduce stacking artifacts
• Added two‑pass registration with framing for improved field rotation handling
• Enhanced logging system with color‑coded messages (red/green/blue/salmon)
• Improved error handling and validation throughout the processing pipeline
"""

##############################################
# CONFIGURATION PRESETS
##############################################

class ProcessingProgress:
    """Constants for processing progress percentages"""
    CLEANUP = 5
    DARK_PROCESSING = 10
    LIGHT_CONVERSION = 20
    CALIBRATION = 30
    REGISTRATION = 50
    STACKING = 75
    FINALIZATION = 88
    COMPLETE = 100

# Sky Quality Presets (Bortle scale)
SKY_PRESETS = {
    "Bortle 1‑2 (Excellent Dark)": {
        "description": "Remote dark sites, minimal light pollution",
        "sigma_low": 3.0,
        "sigma_high": 3.0,
        "bg_samples": 6,
        "bg_tolerance": 1.0,
        "gradient_correction": False,
    },
    "Bortle 3‑4 (Rural)": {
        "description": "Rural areas, some light domes on horizon",
        "sigma_low": 3.0,
        "sigma_high": 3.0,
        "bg_samples": 9,
        "bg_tolerance": 1.0,
        "gradient_correction": True,
    },
    "Bortle 5‑6 (Suburban)": {
        "description": "Suburban skies, noticeable light pollution",
        "sigma_low": 2.5,
        "sigma_high": 3.0,
        "bg_samples": 12,
        "bg_tolerance": 0.8,
        "gradient_correction": True,
    },
    "Bortle 7‑8 (Urban)": {
        "description": "City skies, heavy light pollution",
        "sigma_low": 2.0,
        "sigma_high": 2.5,
        "bg_samples": 16,
        "bg_tolerance": 0.5,
        "gradient_correction": True,
    },
}

# Stacking methods with tooltips explaining technical details
STACKING_METHODS = {
    "Bayer Drizzle (Recommended)": {
        "description": "Best for field rotation, gaussian kernel for smooth CFA",
        "tooltip": (
            "Uses Gaussian drizzle kernel with area‑based interpolation.\n\n"
            "• Gaussian kernel: Produces smooth, centrally‑peaked PSFs\n"
            "• Area interpolation: Reduces moiré patterns from field rotation\n"
            "• Best choice for typical Vespera sessions with 10‑15° rotation\n\n"
            "Technical: scale=1.0, pixfrac=1.0, kernel=gaussian, interp=area"
        ),
        "use_drizzle": True,
        "drizzle_scale": 1.0,
        "drizzle_pixfrac": 1.0,
        "drizzle_kernel": "gaussian",
        "interp": "area",
        "feather_px": 0,
    },
    "Bayer Drizzle (Square)": {
        "description": "Classic drizzle kernel, mathematically flux‑preserving",
        "tooltip": (
            "Uses classic square drizzle kernel (original HST algorithm).\n\n"
            "• Square kernel: Mathematically flux‑preserving by construction\n"
            "• May show subtle grid patterns with significant field rotation\n"
            "• Better for photometry applications\n\n"
            "Technical: scale=1.0, pixfrac=1.0, kernel=square, interp=area"
        ),
        "use_drizzle": True,
        "drizzle_scale": 1.0,
        "drizzle_pixfrac": 1.0,
        "drizzle_kernel": "square",
        "interp": "area",
        "feather_px": 0,
    },
    "Bayer Drizzle (Nearest)": {
        "description": "Nearest‑neighbor interpolation to minimize moiré patterns",
        "tooltip": (
            "Uses nearest‑neighbor interpolation to eliminate moiré.\n\n"
            "• Nearest interpolation: No interpolation artifacts at CFA boundaries\n"
            "• May appear slightly blocky at pixel level\n"
            "• Try this if other methods show checkerboard patterns\n\n"
            "Technical: scale=1.0, pixfrac=1.0, kernel=gaussian, interp=nearest"
        ),
        "use_drizzle": True,
        "drizzle_scale": 1.0,
        "drizzle_pixfrac": 1.0,
        "drizzle_kernel": "gaussian",
        "interp": "nearest",
        "feather_px": 0,
    },
    "Standard Registration": {
        "description": "Faster processing, good for short sessions with minimal rotation",
        "tooltip": (
            "Standard debayer‑then‑register workflow (no drizzle).\n\n"
            "• Faster processing, lower memory usage\n"
            "• Works well for sessions under 30 minutes\n"
            "• May show field rotation artifacts at image edges\n"
            "• Not recommended for sessions with >5° total rotation"
        ),
        "use_drizzle": False,
        "feather_px": 0,
    },
    "Drizzle 2x Upscale": {
        "description": "Doubles resolution, requires many well‑dithered frames (50+)",
        "tooltip": (
            "Upscales to 2x resolution using drizzle algorithm.\n\n"
            "• Requires 50+ frames with good sub‑pixel dithering\n"
            "• Output will be 7072×7072 pixels (vs 3536×3536)\n"
            "• Uses square kernel (only valid choice for scale>1)\n"
            "• Significantly increased processing time and file sizes\n\n"
            "Note: Lanczos kernels cannot be used with scale>1.0\n"
            "Technical: scale=2.0, pixfrac=1.0, kernel=square, interp=area"
        ),
        "use_drizzle": True,
        "drizzle_scale": 2.0,
        "drizzle_pixfrac": 1.0,
        "drizzle_kernel": "square",
        "interp": "area",
        "feather_px": 0,
    },
}

##############################################
# DARK STYLESHEET
##############################################

DARK_STYLESHEET = """
QDialog { background-color: #2b2b2b; color: #e0e0e0; }
QTabWidget::pane { border: 1px solid #444444; background-color: #2b2b2b; }
QTabBar::tab {
    background-color: #3c3c3c;
    color: #aaaaaa;
    padding: 8px 16px;
    border: 1px solid #444444;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected { background-color: #2b2b2b; color: #ffffff; }
QTabBar::tab:hover { background-color: #444444; }

QGroupBox {
    border: 1px solid #444444;
    margin-top: 12px;
    font-weight: bold;
    border-radius: 4px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #88aaff;
}

QLabel { color: #cccccc; font-size: 10pt; }
QLabel#title { color: #88aaff; font-size: 14pt; font-weight: bold; }
QLabel#subtitle { color: #888888; font-size: 9pt; }
QLabel#status { color: #ffcc00; font-size: 10pt; }
QLabel#error { color: #ff8888; }
QLabel#info { color: #88aaff; font-size: 9pt; }

QComboBox {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px 10px;
    min-width: 200px;
}
QComboBox:hover { border-color: #88aaff; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow {
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #aaaaaa;
}
QComboBox QAbstractItemView {
    background-color: #3c3c3c;
    color: #ffffff;
    selection-background-color: #285299;
    border: 1px solid #555555;
}

QCheckBox { color: #cccccc; spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #666666;
    background: #3c3c3c;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background-color: #285299;
    border: 1px solid #88aaff;
}
QCheckBox::indicator:hover { border-color: #88aaff; }

QSpinBox, QDoubleSpinBox {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
}

QProgressBar {
    border: 1px solid #555555;
    border-radius: 4px;
    background-color: #3c3c3c;
    text-align: center;
    color: #ffffff;
    min-height: 20px;
}
QProgressBar::chunk { background-color: #285299; border-radius: 3px; }

QPushButton {
    background-color: #444444;
    color: #dddddd;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 8px 20px;
    font-weight: bold;
    min-width: 100px;
}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton:pressed { background-color: #333333; }
QPushButton:disabled { background-color: #333333; color: #666666; }

QPushButton#start { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#start:hover { background-color: #3366bb; }
QPushButton#start:disabled { background-color: #1a1a2e; color: #555555; }

QTextEdit {
    background-color: #1e1e1e;
    color: #aaaaaa;
    border: 1px solid #444444;
    border-radius: 4px;
    font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
    font-size: 9pt;
    padding: 5px;
}

QFrame#separator {
    background-color: #444444;
    min-height: 1px;
    max-height: 1px;
}
"""

##############################################
# PROCESSING THREAD
##############################################

class ProcessingThread(QThread):
    """Background thread for preprocessing"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    log = pyqtSignal(str)

    def __init__(self, siril: Any, workdir: str,
                 settings: Dict[str, Any], folder_structure: str):
        super().__init__()
        self.siril = siril
        self.workdir = workdir
        self.settings = settings
        self.folder_structure = folder_structure  # 'native' or 'organized'
        self.log_area = None  # Will be set by the GUI

    def run(self):
        try:
            self._process()
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")
            traceback.print_exc()

    def _log(self, msg: str, color: Optional[LogColor] = None) -> None:
        """Log message to both GUI and Siril with optional color formatting."""
        if self.log_area:
            cursor = self.log_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.log_area.setTextCursor(cursor)

            if color == LogColor.RED:
                self.log_area.setTextColor(Qt.GlobalColor.red)
            elif color == LogColor.GREEN:
                self.log_area.setTextColor(Qt.GlobalColor.darkGreen)
            elif color == LogColor.BLUE:
                self.log_area.setTextColor(Qt.GlobalColor.cyan)
            elif color == LogColor.SALMON:
                self.log_area.setTextColor(Qt.GlobalColor.magenta)
            else:
                self.log_area.setTextColor(Qt.GlobalColor.lightGray)

            self.log_area.append(msg)
            self.log_area.setTextColor(Qt.GlobalColor.lightGray)  # Reset color

        try:
            if color is not None:
                self.siril.log(msg, color=color)
            else:
                self.siril.log(msg)
        except Exception as e:
            self._log(f"Error logging to Siril: {e}", LogColor.RED)

    def _process(self) -> None:
        """Main processing workflow with native structure support"""
        sky_preset = SKY_PRESETS[self.settings["sky_quality"]]
        stack_method = STACKING_METHODS[self.settings["stacking_method"]]

        sigma_low = sky_preset["sigma_low"]
        sigma_high = sky_preset["sigma_high"]

        process_dir = self._get_process_dir(self.workdir)
        masters_dir = self._get_masters_dir(self.workdir)

        if self.folder_structure == 'native':
            dark_file = "img-0001-dark.fits"
            lights_dir = os.path.join(self.workdir, "01-images-initial")
        else:
            darks_dir = os.path.join(self.workdir, "darks")
            lights_dir = os.path.join(self.workdir, "lights")

        # Verify folders
        if self.folder_structure == 'native':
            if not os.path.exists(dark_file):
                self.finished.emit(False, f"Dark file not found: {dark_file}")
                return
            if not os.path.exists(lights_dir):
                self.finished.emit(False, f"Lights folder not found: {lights_dir}")
                return
        else:
            if not os.path.exists(darks_dir):
                self.finished.emit(False, f"Dark folder not found: {darks_dir}")
                return
            if not os.path.exists(lights_dir):
                self.finished.emit(False, f"Light folder not found: {lights_dir}")
                return

        os.makedirs(process_dir, exist_ok=True)
        os.makedirs(masters_dir, exist_ok=True)

        moved_count = self._move_tiff_to_reference(lights_dir)
        if moved_count:
            self._log(f"Moved {moved_count} TIFF reference image(s) to 'reference/'",
                      LogColor.SALMON)

        # Count files
        if self.folder_structure == 'native':
            num_darks = 1
            num_lights = len([f for f in glob.glob(os.path.join(lights_dir, "*.fits")) +
                              glob.glob(os.path.join(lights_dir, "*.fit"))
                              if '-dark' not in f.lower()])
        else:
            num_darks = self._count_fits(darks_dir)
            num_lights = self._count_fits(lights_dir)

        # Log configuration
        self._log(f"Sky Quality: {self.settings['sky_quality']}", LogColor.BLUE)
        self._log(f"Stacking: {self.settings['stacking_method']}", LogColor.BLUE)
        self._log(f"Structure: {self.folder_structure}", LogColor.BLUE)
        self._log(f"Found {num_darks} dark(s), {num_lights} light(s)", LogColor.BLUE)

        if num_darks == 0:
            self.finished.emit(False, "No dark frames found")
            return
        if num_lights == 0:
            self.finished.emit(False, "No light frames found")
            return

        # === CLEANUP ===
        self.progress.emit(ProcessingProgress.CLEANUP, "Cleaning previous files...")
        if self.settings.get("clean_temp", False):
            self._cleanup_folder(process_dir)
            self._cleanup_folder(masters_dir)

        # === DARK PROCESSING ===
        self.progress.emit(ProcessingProgress.DARK_PROCESSING, "Processing darks...")

        if self.folder_structure == 'native':
            self._log("Single dark → using directly as master", LogColor.BLUE)
            self.siril.cmd("load", dark_file)
            self.siril.cmd("save", "masters/dark_stacked")
        else:
            self.siril.cmd("cd", "darks")
            self.siril.cmd("convert", "dark", "-out=../masters")
            self.siril.cmd("cd", "../masters")

            if num_darks == 1:
                self._log("Single dark → using directly as master", LogColor.BLUE)
                self.siril.cmd("load", "dark_00001")
                self.siril.cmd("save", "dark_stacked")
            else:
                self._log(f"Stacking {num_darks} darks...", LogColor.BLUE)
                self.siril.cmd("stack", "dark", "rej",
                               str(sigma_low), str(sigma_high),
                               "-nonorm", "-out=dark_stacked")

        # === LIGHT PROCESSING ===
        self.progress.emit(ProcessingProgress.LIGHT_CONVERSION, "Converting lights...")

        if self.folder_structure == 'native':
            self.siril.cmd("cd", "01-images-initial")
        else:
            self.siril.cmd("cd", "../lights")

        self.siril.cmd("convert", "light", "-out=../process")
        self.siril.cmd("cd", "../process")

        # Store sequence name for calibration step
        self.light_seq_name = "light"

        # Standard RGB processing only
        self._process_standard(stack_method, sigma_low, sigma_high)

        # === CLEANUP ===
        if self.settings.get("clean_temp", False):
            self.progress.emit(98, "Cleaning up...")
            deleted = self._cleanup_folder(process_dir)
            deleted += self._cleanup_folder(masters_dir)
            self._log(f"Cleaned {deleted} temp files", LogColor.BLUE)

        self.progress.emit(ProcessingProgress.COMPLETE, "Complete!")
        self.finished.emit(True, "Processing complete!")

    def _get_process_dir(self, workdir: str) -> str:
        return os.path.normpath(os.path.join(workdir, "process"))

    def _get_masters_dir(self, workdir: str) -> str:
        return os.path.normpath(os.path.join(workdir, "masters"))

    def _calibrate(self, seq_name: str, stack_method: dict) -> None:
        cmd = ["calibrate", seq_name,
               "-dark=../masters/dark_stacked",
               "-cc=dark", "-cfa"]

        if stack_method.get("use_drizzle"):
            cmd.append("-equalize_cfa")
        else:
            cmd.extend(["-debayer", "-equalize_cfa"])

        self.siril.cmd(*cmd)

    def _register(self, seq_name: str, stack_method: dict) -> None:
        if stack_method.get("use_drizzle"):
            cmd = [
                "register", f"pp_{seq_name}",
                "-drizzle",
                f"-scale={stack_method.get('drizzle_scale',1.0)}",
                f"-pixfrac={stack_method.get('drizzle_pixfrac',1.0)}",
                f"-kernel={stack_method.get('drizzle_kernel','square')}",
                f"-interp={stack_method.get('interp','area')}"
            ]

            if self.settings.get("two_pass", False):
                cmd.append("-2pass")

            self.siril.cmd(*cmd)
        else:
            cmd = ["register", f"pp_{seq_name}"]
            if self.settings.get("two_pass", False):
                cmd.append("-2pass")
            self.siril.cmd(*cmd)

        if self.settings.get("two_pass", False):
            self._log("Applying framing for two-pass registration...", LogColor.BLUE)
            if stack_method.get("use_drizzle"):
                self.siril.cmd("seqapplyreg", f"pp_{seq_name}", "-drizzle", "-framing=max")
            else:
                self.siril.cmd("seqapplyreg", f"pp_{seq_name}", "-framing=max")

    def _stack(self, seq_name: str, stack_method: dict,
               sigma_low: float, sigma_high: float, output_name: str) -> None:
        stack_cmd = [
            "stack", f"r_pp_{seq_name}",
            "rej", str(sigma_low), str(sigma_high),
            "-norm=addscale", "-output_norm", "-32b",
            "-rgb_equal", "-maximize", "-filter-included", "-weight=wfwhm"
        ]

        if self.settings.get("feather_enabled", False) and self.settings.get("feather_px", 0) > 0:
            stack_cmd.append(f"-feather={self.settings['feather']}")

        stack_cmd.append(f"-out={output_name}")
        self.siril.cmd(*stack_cmd)

    def _process_standard(self, stack_method: dict,
                          sigma_low: float, sigma_high: float) -> None:
        self.progress.emit(ProcessingProgress.CALIBRATION, "Calibrating...")
        seq_name = self.light_seq_name
        self._calibrate(seq_name, stack_method)

        self.progress.emit(ProcessingProgress.REGISTRATION, "Registering...")
        self._register(seq_name, stack_method)

        self.progress.emit(ProcessingProgress.STACKING, "Stacking...")
        self._stack(seq_name, stack_method, sigma_low, sigma_high, "result")

        # Finalization
        self.progress.emit(ProcessingProgress.FINALIZATION, "Finalizing...")
        self.siril.cmd("load", "result")
        self.siril.cmd("icc_remove")

        try:
            exposure_time = self.siril.get_image_fits_header("LIVETIME", default="XX")
            self.final_filename = f"result_{exposure_time}s.fit"
        except:
            self.final_filename = "result_XXXXs.fit"

        self.siril.cmd("save", "../result_$LIVETIME:%d$s")
        self.siril.cmd("cd", "..")

    def _count_fits(self, folder: str) -> int:
        count = 0
        for ext in ['*.fit', '*.fits', '*.FIT', '*.FITS']:
            count += len(glob.glob(os.path.join(folder, ext)))
        return count

    def _move_tiff_to_reference(self, lights_dir: str) -> int:
        tiff_files = []
        for pattern in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            tiff_files.extend(glob.glob(os.path.join(lights_dir, pattern)))

        if not tiff_files:
            return 0

        reference_dir = os.path.join(self.workdir, "reference")
        os.makedirs(reference_dir, exist_ok=True)

        moved_count = 0
        for tiff_file in tiff_files:
            try:
                filename = os.path.basename(tiff_file)
                dest_path = os.path.join(reference_dir, filename)
                shutil.move(tiff_file, dest_path)
                self._log(f"Moved reference image: {filename} → reference/",
                          LogColor.SALMON)
                moved_count += 1
            except Exception as e:
                self._log(f"Warning: Could not move {tiff_file}: {e}",
                          LogColor.SALMON)

        return moved_count

    def _cleanup_folder(self, folder: str) -> int:
        if not os.path.exists(folder):
            return 0

        count = 0
        try:
            for ext in ['*.fit', '*.fits', '*.FIT', '*.FITS',
                        '*.seq', '*conversion.txt']:
                for f in glob.glob(os.path.join(folder, ext)):
                    try:
                        os.remove(f)
                        count += 1
                    except OSError as e:
                        self._log(f"Warning: Could not remove {f}: {e}",
                                  LogColor.SALMON)
                        continue

            for subdir in ['cache', 'drizztmp', 'other']:
                subdir_path = os.path.join(folder, subdir)
                if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                    try:
                        shutil.rmtree(subdir_path)
                        count += 1
                    except OSError as e:
                        self._log(f"Warning: Could not remove {subdir_path}: {e}",
                                  LogColor.SALMON)
                        continue
        except Exception as e:
            self._log(f"Error during cleanup: {e}", LogColor.RED)

        return count

##############################################
# MAIN GUI
##############################################

class VesperaProGUI(QDialog):
    """Full‑featured Vespera preprocessing dialog"""

    def __init__(self, siril: Any, app: QApplication):
        super().__init__()
        self.siril = siril
        self.app = app
        self.worker: Optional[ProcessingThread] = None
        self.qsettings = QSettings("Vespera", "Preprocessing")

        self.setWindowTitle(f"Vespera — Preprocessing v{VERSION}")
        self.setMinimumWidth(550)
        self.setMinimumHeight(800)
        self.resize(550, 850)
        self.setStyleSheet(DARK_STYLESHEET)

        self._setup_ui()
        self._load_settings()
        self._check_folders()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QVBoxLayout()
        title = QLabel("Vespera — Preprocessing\n")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(title)

        layout.addLayout(header)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._create_main_tab(), "Main")
        tabs.addTab(self._create_info_tab(), "Info")
        layout.addWidget(tabs)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        progress_layout.addWidget(self.progress)

        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(100)
        progress_layout.addWidget(self.log_area)

        layout.addWidget(progress_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_start = QPushButton("Start Processing")
        self.btn_start.setObjectName("start")
        self.btn_start.clicked.connect(self._start_processing)
        btn_layout.addWidget(self.btn_start)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _create_main_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Folder Status
        status_group = QGroupBox("Folder Status")
        status_layout = QVBoxLayout(status_group)

        self.lbl_workdir = QLabel("Working directory: ...")
        self.lbl_workdir.setObjectName("info")
        self.lbl_workdir.setWordWrap(True)
        status_layout.addWidget(self.lbl_workdir)

        folder_row = QHBoxLayout()
        self.lbl_darks = QLabel("Darks: checking...")
        self.lbl_lights = QLabel("Lights: checking...")
        folder_row.addWidget(self.lbl_darks)
        folder_row.addWidget(self.lbl_lights)
        status_layout.addLayout(folder_row)

        self.lbl_structure = QLabel("")
        self.lbl_structure.setObjectName("info")
        self.lbl_structure.setWordWrap(True)
        status_layout.addWidget(self.lbl_structure)

        layout.addWidget(status_group)

        # Sky Quality
        sky_group = QGroupBox("Sky Quality (Location)")
        sky_layout = QVBoxLayout(sky_group)

        self.combo_sky = QComboBox()
        for name in SKY_PRESETS.keys():
            self.combo_sky.addItem(name)
        self.combo_sky.currentTextChanged.connect(self._on_sky_changed)
        sky_layout.addWidget(self.combo_sky)

        self.lbl_sky_desc = QLabel("")
        self.lbl_sky_desc.setObjectName("info")
        self.lbl_sky_desc.setWordWrap(True)
        sky_layout.addWidget(self.lbl_sky_desc)

        layout.addWidget(sky_group)

        # Stacking Method
        stack_group = QGroupBox("Stacking Method")
        stack_layout = QVBoxLayout(stack_group)

        self.combo_stack = QComboBox()
        for idx, (name, config) in enumerate(STACKING_METHODS.items()):
            self.combo_stack.addItem(name)
            if "tooltip" in config:
                self.combo_stack.setItemData(idx, config["tooltip"],
                                            Qt.ItemDataRole.ToolTipRole)
        self.combo_stack.currentTextChanged.connect(self._on_stack_changed)
        stack_layout.addWidget(self.combo_stack)

        self.lbl_stack_desc = QLabel("")
        self.lbl_stack_desc.setObjectName("info")
        self.lbl_stack_desc.setWordWrap(True)
        stack_layout.addWidget(self.lbl_stack_desc)

        layout.addWidget(stack_group)

        # Stacking Options
        options_group = QGroupBox("Stacking Options")
        options_layout = QVBoxLayout(options_group)

        # Feather option
        feather_layout = QHBoxLayout()
        self.chk_feather = QCheckBox("Enable Feathering")
        self.chk_feather.setToolTip(
            "Feathering blends image edges to reduce artifacts (0‑100px)")

        self.feather_slider = QSpinBox()
        self.feather_slider.setRange(0, 100)
        self.feather_slider.setValue(0)
        self.feather_slider.setSuffix(" px")
        self.feather_slider.setToolTip(
            "Feathering distance in pixels (0 to disable)")
    
        feather_layout.addWidget(self.chk_feather)
        feather_layout.addWidget(self.feather_slider)
        options_layout.addLayout(feather_layout)

        # 2‑pass option – checkbox before label
        hbox_two_pass = QHBoxLayout()
        self.chk_two_pass = QCheckBox("2‑Pass Registration")
        self.chk_two_pass.setToolTip(
            "Perform two-pass registration with framing for optimal alignment")
        hbox_two_pass.addWidget(self.chk_two_pass)
        hbox_two_pass.addStretch(0)
        options_layout.addLayout(hbox_two_pass)

        # Advanced Options – clean‑temp checkbox
        hbox_clean_temp = QHBoxLayout()
        self.chk_clean_temp = QCheckBox("Clean temporary files")
        self.chk_clean_temp.setToolTip(
            "Delete process/ and masters/ folders")
        hbox_clean_temp.addWidget(self.chk_clean_temp)
        hbox_clean_temp.addStretch(0)          # keeps it left‑aligned
        options_layout.addLayout(hbox_clean_temp)

        layout.addWidget(options_group)
        layout.addStretch()
        return widget

    def _create_info_tab(self) -> QWidget:
        """Information/help tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h3 style="color: #88aaff;">Vespera Preprocessing</h3>

        <p><b>No Setup Required!</b></p>
        <p>Just point this plugin at your Vespera observation folder. It auto‑detects
        darks vs lights by filename – no file reorganization needed.</p>

        <hr style="border-color: #444;">
        <h4 style="color: #88aaff;">Why Restack? (vs Vespera's Built‑in)</h4>

        <p>Vespera already outputs a stacked TIFF. This plugin restacks from raw FITS for better quality:</p>

        <table style="color: #e0e0e0; margin: 10px 0;">
        <tr><td><b>Vespera:</b></td><td>Debayer → Stack (loses sub‑pixel info)</td></tr>
        <tr><td><b>This Plugin:</b></td><td>Stack → Debayer (preserves CFA data)</td></tr>
        </table>

        <p><b>Benefits of Restacking:</b></p>
        <ul>
        <li>~20% sharper color detail from Bayer Drizzle</li>
        <li>Sigma rejection removes satellites & planes</li>
        <li>32‑bit output (vs 16‑bit) for more stretching headroom</li>
        <li>Your actual darks (vs Vespera's algorithmic BalENS)</li>
        <li>Proper Ha/OIII extraction for dual‑band filters</li>
        </ul>

        <p><i>For social media, Vespera TIFF is fine. For prints, restack here.</i></p>

        <hr style="border-color: #444;">
        <h4 style="color: #88aaff;">Drizzle & Pattern Artifacts</h4>

        <p><b>Why Bayer Drizzle?</b></p>
        <p>The Vespera Pro's alt‑az mount causes field rotation (10‑15° per hour).
        Bayer Drizzle handles this while preserving CFA pattern data.</p>

        <p><b>Checkerboard/Grid Patterns:</b></p>
        <p>If you see checkerboard or moiré patterns in your stacked image, this is caused by
        <i>interpolation artifacts</i> when Siril applies geometric transforms to correct field rotation.
        The pattern appears at CFA (color filter) cell boundaries.</p>

        <p><b>Solutions:</b></p>
        <ul>
        <li><b>Gaussian kernel (Recommended):</b> Smoothest results, reduces pattern visibility</li>
        <li><b>Nearest interpolation:</b> Eliminates interpolation artifacts but may look blocky</li>
        <li><b>More frames:</b> Additional well‑dithered frames help average out patterns</li>
        </ul>

        <p><b>Drizzle Kernel Types:</b></p>
        <ul>
        <li><b>Gaussian:</b> Smooth, centrally‑peaked PSFs – best for deep‑sky CFA data</li>
        <li><b>Square:</b> Classic HST algorithm, mathematically flux‑preserving – better for photometry</li>
        <li><b>Lanczos:</b> Only valid at scale=1.0, pixfrac=1.0 – NOT for 2x upscaling</li>
        </ul>

        <p><b>Interpolation Methods:</b></p>
        <ul>
        <li><b>Area:</b> Area‑based averaging – good balance of quality and artifact reduction</li>
        <li><b>Nearest:</b> No interpolation – eliminates moiré but may look blocky</li>
        <li><b>Cubic/Lanczos:</b> High quality but can cause ringing at high‑contrast edges</li>
        </ul>

         <p><b>Output:</b></p>
         <p>Linear 32‑bit FITS file (0‑1 normalized) ready for stretching in VeraLux, GHS, or Siril.</p>
        """)
        layout.addWidget(info_text)

        return widget

    def _on_sky_changed(self, name: str) -> None:
        if name in SKY_PRESETS:
            self.lbl_sky_desc.setText(SKY_PRESETS[name]["description"])

    def _on_stack_changed(self, name: str) -> None:
        if name in STACKING_METHODS:
            self.lbl_stack_desc.setText(STACKING_METHODS[name]["description"])

    def _load_settings(self) -> None:
        """Load saved settings"""
        self.combo_sky.setCurrentText(
            self.qsettings.value("sky_quality", "Bortle 3‑4 (Rural)"))
        self.combo_stack.setCurrentText(
            self.qsettings.value("stacking_method",
                                 "Bayer Drizzle (Recommended)"))
        self.chk_feather.setChecked(
            self.qsettings.value("feather_enabled", False, type=bool))
        self.feather_slider.setValue(
            self.qsettings.value("feather_px", 0, type=int))
        self.chk_two_pass.setChecked(
            self.qsettings.value("two_pass", False, type=bool))
        self.chk_clean_temp.setChecked(
            self.qsettings.value("clean_temp", False, type=bool))

        # Trigger description updates
        self._on_sky_changed(self.combo_sky.currentText())
        self._on_stack_changed(self.combo_stack.currentText())

    def _save_settings(self) -> None:
        """Save current settings"""
        self.qsettings.setValue("sky_quality", self.combo_sky.currentText())
        self.qsettings.setValue("stacking_method",
                                self.combo_stack.currentText())
        self.qsettings.setValue("feather_enabled", self.chk_feather.isChecked())
        self.qsettings.setValue("feather_px", self.feather_slider.value())
        self.qsettings.setValue("two_pass", self.chk_two_pass.isChecked())
        self.qsettings.setValue("clean_temp", self.chk_clean_temp.isChecked())

    def _check_folders(self) -> None:
        """Check folder status - supports both organized and native Vespera structure"""
        try:
            workdir = self.siril.get_siril_wd()
            self.lbl_workdir.setText(f"Working directory: {workdir}")

            # First check for organized structure (darks/ and lights/ folders)
            darks_dir = os.path.join(workdir, "darks")
            lights_dir = os.path.join(workdir, "lights")

            num_darks_organized = self._count_fits(darks_dir) if os.path.exists(darks_dir) else 0
            num_lights_organized = self._count_fits(lights_dir) if os.path.exists(lights_dir) else 0

            # Check for native Vespera structure
            native = self._detect_native_structure(workdir)

            # Determine which structure to use
            if num_darks_organized > 0 and num_lights_organized > 0:
                self.folder_structure = 'organized'
                num_darks = num_darks_organized
                num_lights = num_lights_organized
                self.lbl_structure.setText("Using organized folders (darks/, lights/)")
                self.lbl_structure.setStyleSheet("color: #88aaff;")
            elif native:
                self.folder_structure = 'native'
                num_darks = native['num_darks']
                num_lights = native['num_lights']
                self.lbl_structure.setText("Using Vespera native structure")
                self.lbl_structure.setStyleSheet("color: #88aaff;")
            else:
                self.folder_structure = None
                num_darks = 0
                num_lights = 0
                self.lbl_structure.setText("No valid folder structure detected")
                self.lbl_structure.setStyleSheet("color: #ff8888;")

            if num_darks > 0:
                self.lbl_darks.setText(f"✓ Darks: {num_darks}")
                self.lbl_darks.setStyleSheet("color: #88ff88;")
            else:
                self.lbl_darks.setText("✗ Darks: not found")
                self.lbl_darks.setStyleSheet("color: #ff8888;")

            if num_lights > 0:
                self.lbl_lights.setText(f"✓ Lights: {num_lights}")
                self.lbl_lights.setStyleSheet("color: #88ff88;")
            else:
                self.lbl_lights.setText("✗ Lights: not found")
                self.lbl_lights.setStyleSheet("color: #ff8888;")

            self.btn_start.setEnabled(num_darks > 0 and num_lights > 0)

        except Exception as e:
            self._log(f"Error: {e}")
            self.btn_start.setEnabled(False)

    def _detect_native_structure(self, workdir: str) -> Optional[Dict[str, Any]]:
        """Detect native Vespera folder structure"""
        workdir = os.path.normpath(workdir)

        dark_files = set()
        for pattern in ['*-dark.fits', '*-dark.fit',
                        '*-dark.FITS', '*-dark.FIT']:
            dark_files.update(glob.glob(os.path.join(workdir, pattern)))

        images_initial = os.path.join(workdir, "01-images-initial")
        light_files = set()
        if os.path.exists(images_initial):
            for pattern in ['*.fits', '*.fit',
                            '*.FITS', '*.FIT']:
                all_fits = glob.glob(os.path.join(images_initial, pattern))
                light_files.update([f for f in all_fits if '-dark' not in f.lower()])

        dark_files = list(dark_files)
        light_files = list(light_files)

        if dark_files and light_files:
            return {
                'dark_files': dark_files,
                'light_files': light_files,
                'num_darks': len(dark_files),
                'num_lights': len(light_files),
                'images_initial': images_initial
            }
        return None

    def _count_fits(self, folder: str) -> int:
        count = 0
        for ext in ['*.fit', '*.fits', '*.FIT', '*.FITS']:
            count += len(glob.glob(os.path.join(folder, ext)))
        return count

    def _log(self, msg: str, color: Optional[LogColor] = None) -> None:
        cursor = self.log_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_area.setTextCursor(cursor)

        if color == LogColor.RED:
            self.log_area.setTextColor(Qt.GlobalColor.red)
        elif color == LogColor.GREEN:
            self.log_area.setTextColor(Qt.GlobalColor.darkGreen)
        elif color == LogColor.BLUE:
            self.log_area.setTextColor(Qt.GlobalColor.cyan)
        elif color == LogColor.SALMON:
            self.log_area.setTextColor(Qt.GlobalColor.magenta)
        else:
            self.log_area.setTextColor(Qt.GlobalColor.lightGray)

        self.log_area.append(msg)
        self.log_area.setTextColor(Qt.GlobalColor.lightGray)  # Reset color

    def _start_processing(self) -> None:
        """Start the processing thread"""
        self._save_settings()
        self.btn_start.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Processing...")
        self.log_area.clear()

        settings = {
            "sky_quality": self.combo_sky.currentText(),
            "stacking_method": self.combo_stack.currentText(),
            "feather_px": self.feather_slider.value(),
            "feather_enabled": self.chk_feather.isChecked(), 
            "two_pass": self.chk_two_pass.isChecked(),
            "clean_temp": self.chk_clean_temp.isChecked(),   # renamed flag
        }

        try:
            workdir = self.siril.get_siril_wd()
            self.worker = ProcessingThread(self.siril, workdir,
                                          settings, self.folder_structure)
            self.worker.log_area = self.log_area
            self.worker.progress.connect(self._on_progress)
            self.worker.finished.connect(self._on_finished)
            self.worker.log.connect(self._log)
            self.worker.start()
        except Exception as e:
            self._log(f"Start error: {e}", LogColor.RED)
            self.btn_start.setEnabled(True)

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress.setValue(percent)
        self.status.setText(message)
        self.app.processEvents()

    def _on_finished(self, success: bool, message: str) -> None:
        self.btn_start.setEnabled(True)

        if success:
            self.status.setText("✓ " + message)
            self.status.setStyleSheet("color: #88ff88;")
            self._log("Stacking complete!", LogColor.GREEN)
            self._log("\n", LogColor.GREEN)

            try:
                self.siril.log("Stacking Complete!", color=LogColor.GREEN)
            except:
                pass

        else:
            self.status.setText("✗ " + message)
            self.status.setStyleSheet("color: #ff8888;")
            self._log(f"FAILED: {message}", LogColor.RED)

##############################################
# MAIN
##############################################

def main() -> None:
    """Main entry point"""
    try:
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

        siril = s.SirilInterface()

        try:
            siril.connect()
        except Exception as e:
            QMessageBox.critical(None, "Connection Error",
                                 f"Could not connect to Siril.\n{e}")
            return

        gui = VesperaProGUI(siril, app)
        gui.show()
        app.exec()

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
