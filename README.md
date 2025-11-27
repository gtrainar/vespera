# FTP FITS / TIFF Downloader for Vespera Smart Telescope

**Author:** G. Trainar  
**Date:** 2025‑11‑27  

This Python script connects to a Vespera Smart Telescope FTP server, locates the most recently modified observation directory, downloads all `.fits` and `.tif` files, renames them by object name and observation date, stores them in a local directory tree, and optionally deletes the originals from the server.

> **Note** – The script is designed as a one‑shot utility.  
> If you need continuous polling, set `CHECK_INTERVAL` and loop the main routine.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **Automatic discovery** | Finds the newest observation folder on the FTP server. |
| **Recursive download** | Downloads all `.fits` / `.tif` files in the selected folder and its sub‑folders. |
| **Renaming & sorting** | Files are renamed to `<OBJECT>_<DATE>.<ext>` and moved into an object‑named subfolder. |
| **Optional cleanup** | Uncomment the `ftp.delete()` line to remove files from the server after download. |
| **Failure counter** | Stops after a configurable number of consecutive “no‑files” runs. |

---

## Prerequisites

* Python 3.8+ (tested on 3.10)
* Standard library only – no external dependencies.

---

## Installation

1. **Clone or download** the repository (or copy the script into a directory of your choice).

   ```bash
   git clone https://github.com/gtrainar/vespera.git
   cd vespera
   ```

2. (Optional) Create a **virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. The script is ready to run – no additional packages are required.

---

## Configuration

Edit the section marked **Configuration** near the top of `downloader.py`:

```python
FTP_SERVER: str = "10.0.0.1"          # FTP server IP / hostname
FTP_USER: str = "anonymous"           # FTP username (empty password for anonymous)
REMOTE_DIR: str = "/user"             # Remote directory to start from
LOCAL_DIR: Path = Path(
    "path to local directory"
)  # Local directory where files will be stored

CHECK_INTERVAL: int = 1800            # Seconds between successive checks (unused in this one‑shot script)
FAILED_CHECKS: int = 0                # Global counter of consecutive “no files” runs
MAX_FAILED_CHECKS: int = 10           # Stop after this many consecutive failures
```

* **FTP_SERVER** – Replace with the IP or hostname of your telescope’s FTP server.  
* **FTP_USER / password** – If the server requires a password, replace the empty string with your credentials.  
* **REMOTE_DIR** – The root directory on the server that contains observation sub‑folders.  
* **LOCAL_DIR** – Absolute or relative path where downloaded files will be stored.  
  Example: `Path("/data/observations")`.  
* **CHECK_INTERVAL** – Not used in this one‑shot script but kept for future extensions.  
* **MAX_FAILED_CHECKS** – Script will terminate after this many consecutive runs that find no files.

---

## Usage

```bash
python sync_vespera.py
```

The script will:

1. Connect to the FTP server.  
2. Find the newest observation directory under `REMOTE_DIR`.  
3. Recursively download all `.fits` and `.tif` files, rename them, and move them into `<LOCAL_DIR>/<OBJECT>/`.  
4. Print progress to stdout.

If you want the script to run continuously (e.g., every 30 minutes), wrap it in a shell loop or use `cron`:

```bash
*/30 * * * * /usr/bin/python3 /path/to/downloader.py >> /var/log/ftp_downloader.log 2>&1
```

---

## How It Works

### Directory Listing

The helper `list_dir_details()` attempts to use the FTP `MLSD` command for machine‑readable listings.  
If `MLSD` is unavailable, it falls back to parsing the plain text output of `dir`.

### Finding the Most Recent Folder

`find_most_recent_dir()` inspects each sub‑directory’s `modify` timestamp (from MLSD) or otherwise picks the first entry.  
The newest folder name is returned for processing.

### Download & Rename

`download_and_process_dir()` walks the chosen directory tree:

* For each file matching `*.fits` or `*.tif`, it:
  * Downloads to a temporary location in `LOCAL_DIR`.
  * Extracts the observation date from the path (`YYYY-MM-DD`).
  * Extracts the object name from a segment matching `observation-<object>`.
  * Creates an object‑named subfolder.
  * Renames the file to `<OBJECT>_<DATE>.<ext>` and moves it into that subfolder.

* Recursion handles nested directories.

### Failure Handling

If a run finds **no** matching files, the global `FAILED_CHECKS` counter increments.  
Once it reaches `MAX_FAILED_CHECKS`, the script exits with a warning.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `Could not connect to <IP>` | Wrong IP, firewall, or credentials | Verify server address and network connectivity. |
| No files downloaded | FTP path incorrect or no new observations | Check `REMOTE_DIR` and ensure that the server contains `.fits/.tif` files. |
| Permission denied on local path | `LOCAL_DIR` not writable or missing | Ensure the directory exists and you have write permissions. |
| Script hangs at `list_dir_details` | Server does not support MLSD and slow `dir` output | The fallback should still work; check server logs for timeouts. |

---

## License

This project is released under the MIT License – see the [LICENSE](LICENSE) file for details.
