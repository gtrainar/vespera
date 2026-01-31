# Vespera‑Suite  
**A collection of scripts for the Vaonis Vespera Smart Telescope**

> **Order update:**  
> 1. `sync_vespera.py` – FTP downloader (first).  
> 2. `Vespera_Preprocessing.py` – Raw FITS preprocessing & stacking.  
> 3. `Vespera_Postprocessing.py` – One‑click post‑processing of Vespera TIFF images.  

---

## Table of Contents
1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Usage](#usage)  
   - 4.1 [FTP Downloader](#ftp-downloader)  
   - 4.2 [Preprocessing](#preprocessing)  
   - 4.3 [Post‑Processing](#post-processing)  
5. [Configuration & Settings](#configuration-settings)  
6. [Examples](#examples)  
7. [Troubleshooting](#troubleshooting)  
8. [License](#license)  
9. [Contributing](#contributing)  

---

## Features

| Script | What it does | Key Options |
|--------|--------------|-------------|
| **sync_vespera.py** | FTP client that finds the most recent observation folder, downloads FITS/TIFF files, renames them, moves to an object‑based directory tree and optionally deletes the originals. | • CLI or interactive mode <br>• Progress bar <br>• Cancel with Ctrl‑C |
| **Vespera_Preprocessing.py** | Detects dark/light frames, calibrates, registers (drizzle or standard), stacks, and outputs a 32‑bit FITS ready for stretching. | • `Bayer Drizzle` (recommended) <br>• Feathering 0‑100 px <br>• Two‑pass registration <br>• Clean temporary files |
| **Vespera_Postprocessing.py** | One‑click pipeline: background extraction, plate solving (SIMBAD), SPCC, optional denoising, auto‑launch of VeraLux HMS. | • GraXpert AI / Siril RBF background <br>• Dual‑band Ha/OIII extraction <br>• Silentium / GraXpert / Cosmic Clarity denoise |

---

## Prerequisites

| Component | Minimum Version |
|-----------|-----------------|
| **Python** | 3.10+ |
| **Siril** | 1.4+ (with `sirilpy` plugin) |
| **PyQt6** | 6.x |
| **Other Python Packages** | `numpy` (required by post‑processing) |

> All dependencies are automatically installed when the scripts first run (`sirilpy.ensure_installed("PyQt6", "numpy")`).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/gtrainar/vespera-suite.git
cd vespersa-suite

# (Optional) Create a virtual environment if you run the scripts from a console
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install Python dependencies (PyQt6, numpy)
pip install PyQt6 numpy
```

> The scripts are designed to be dropped into the *Siril* `scripts` folder.  
> For the FTP downloader, run it directly from a terminal.

---

## Usage

### 4.1 FTP Downloader  
Run from the command line:

```bash
python sync_vespera.py          # interactive mode (default)
python sync_vespera.py --cli    # original CLI mode
```

You’ll be prompted for:

* File type(s) to download (TIFF, FITS or both).  
* Destination directory.  
* Whether to delete the files on the server after download.

The script prints a progress bar per file and shows a summary at the end.  
Press **Ctrl‑C** to cancel; partial files are removed automatically.

---

### 4.2 Preprocessing  
Open *Siril* → **Plugins** → **Scripts** → `Vespera_Preprocessing.py`.  

#### Folder Structure – No Re‑organisation Needed!  
The plugin **automatically detects** how your Vespera Pro exported the data. Just point it at your observation folder – no need to reorganise files.

**Supported structures**

```
# Native Vespera export (flat structure) – works automatically!
Vespera_Observation_Folder/
├── img-0001-dark.fits          (Dark frame)
├── 01-images-initial/
│   ├── img-0001.fits           (Light frame 1)
│   ├── img-0002.fits           (Light frame 2)
│   └── …                       (Additional light frames)
└── [Other Vespera files like TIFFs, JSON metadata, etc.]

# Organized structure – also works!
observation_folder/
├── darks/
│   └── dark_000001.fit          (Dark frame)
└── lights/
    ├── light_000001.fit         (Light frame 1)
    └── light_000002.fit         (Light frame 2)
```

The plugin auto‑detects darks vs lights by filename pattern.

#### Running the Plugin

1. Open Siril  
2. Navigate to your Vespera observation folder (as exported)  
3. Go to **Scripts** menu → **Vespera_Preprocessinge**  
4. Configure options in the GUI:  
   - **Filter** – Select your filter type (not shown in the script, but you can set it via the `sky_quality` combo)  
   - **Sky Quality** – Match your Bortle level  
   - **Stacking Method** – Choose drizzle algorithm  
5. Click **Process**

#### Stacking Methods

| Method | Best For | Notes |
|--------|----------|-------|
| **Bayer Drizzle (Recommended)** | Most sessions | Gaussian kernel, reduces moiré patterns |
| **Bayer Drizzle (Square)** | Photometry | Flux‑preserving, classic HST algorithm |
| **Bayer Drizzle (Nearest)** | Pattern issues | Eliminates interpolation artifacts |
| **Standard Registration** | Quick processing | No drizzle, faster but less quality |
| **Drizzle 2x Upscale** | High resolution | Requires 50+ well‑dithered frames |

#### Output

The plugin creates:

* `result_XXXXs.fit` – Final stacked image (32‑bit, linear)  
* `masters/dark_stacked.fit` – Master dark frame  
* `process/` – Intermediate files (deleted unless “Keep temp files” enabled)

---

### 4.3 Post‑Processing  
Open *Siril* → **Plugins** → **Scripts** → `Vespera_Postprocessing.py`.  

#### Usage

1. Load your Vespera 16‑bit TIFF in Siril  
2. Go to **Scripts** menu → **Vespera_Postprocessing**  
3. Configure options (or use defaults)  
4. Click **Process Image**  
5. VeraLux HMS opens with your color‑calibrated, gradient‑free image ready to stretch  

The window offers:

1. Background extraction (GraXpert AI / Siril RBF / Skip).  
2. Plate solve checkbox + optional DSO name field.  
3. SPCC (Spectrophotometric Color Correction) with filter combo.  
4. Denoise selection.  
5. “Launch VeraLux HMS” checkbox.

After clicking **Postprocess Image** a progress bar shows the pipeline.  
If *Launch HMS* is checked, VeraLux will start automatically once finished.

---

## Configuration & Settings

| File | What it configures |
|------|--------------------|
| `Vespera_Preprocessing.py` | Sky presets, stacking methods, feathering distance, two‑pass flag, clean temp |
| `Vespera_Postprocessing.py` | Background method, plate‑solve flag, SPCC filter, denoise method, launch HMS |
| `sync_vespera.py` | FTP server/user/password, remote/local directories, file types, delete‑after‑download flag |

All settings are persisted in *Siril*'s QSettings or via the script’s command‑line prompts.

---

## Examples

```bash
# Download the newest observation set and delete it from the server
python sync_vespera.py --cli

# Preprocess a folder with 1 dark and 50 lights
python Vespera_Preprocessing.py

# Post‑process a loaded TIFF, auto‑launch HMS
python Vespera_Postprocessing.py
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ImportError: No module named 'sirilpy'` | Siril plugin not installed or not running in Siril | Open the script from inside Siril; it will prompt to install `sirilpy`. |
| “No dark frames found” | Dark file missing or not named correctly (`*-dark.fits`) | Place a dark in the `01-images-initial` folder or create a `darks/` directory. |
| Plate solve fails | DSO name extraction failed or SIMBAD lookup timed out | Enter the object manually in the post‑processing GUI. |
| FTP connection refused | Wrong server address or credentials | Verify `FTP_SERVER`, `FTP_USER`, and `FTP_PASSWORD` in `sync_vespera.py`. |
| GUI freezes | Heavy processing on main thread | The scripts use background threads; if you see a freeze, ensure the GUI thread isn’t blocked by a long‑running command. |

---

## License

MIT © 2025 Claude (Anthropic) & G. Trainar  
See [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repo.  
2. Create a feature branch (`git checkout -b feat/…`).  
3. Commit and push.  
4. Open a Pull Request.

All contributions are welcome—especially improvements to the GUI, new stacking methods, or additional plate‑solving backends.