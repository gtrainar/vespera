#!/usr/bin/env python3
"""
FTP FITS / TIFF downloader from Vespera Smart Telescope
Author:  G. Trainar
Date:    2025‑11‑27
"""

from __future__ import annotations

import ftplib
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# --------------------------------------------------------------------------- #
#  Configuration – edit these values to match your environment
# --------------------------------------------------------------------------- #

FTP_SERVER: str = "10.0.0.1"          # FTP server IP / hostname
FTP_USER: str = "anonymous"           # FTP username (empty password for anonymous)
REMOTE_DIR: str = "/user"             # Remote directory to start from
LOCAL_DIR: Path = Path(
    "path to local directory"
)  # Local directory where files will be stored

CHECK_INTERVAL: int = 1800            # Seconds between successive checks (unused in this one‑shot script)
FAILED_CHECKS: int = 0                # Global counter of consecutive “no files” runs
MAX_FAILED_CHECKS: int = 10           # Stop after this many consecutive failures

# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #

def list_dir_details(ftp: ftplib.FTP, path: str) -> List[Tuple[str, Dict[str, str]]]:
    """
    Return a list of (name, facts) tuples for the given remote directory.
    Uses MLSD if available; otherwise falls back to parsing `dir` output.

    :param ftp:   Connected ftplib.FTP instance
    :param path:  Remote directory to list
    :return:      List of (name, facts) tuples
    """
    try:
        # MLSD gives us a dictionary of facts (type, modify, size, …)
        return list(ftp.mlsd(path))
    except Exception:
        # Fallback: parse the output of `dir`
        lines: List[str] = []

        def collect(line: str) -> None:
            lines.append(line)

        ftp.dir(path, collect)
        entries: List[Tuple[str, Dict[str, str]]] = []

        for line in lines:
            # Typical format: drwxr-xr-x 1 owner group size month day time name
            parts = line.split(maxsplit=8)
            if len(parts) < 9:
                continue
            perms, _, _, _, size, month, day, time_or_year, name = parts
            entry_type = "dir" if perms[0] == "d" else "file"
            entries.append((name, {"type": entry_type}))
        return entries


def find_most_recent_dir(ftp: ftplib.FTP, remote_root: str) -> str:
    """
    Find the most recently modified sub‑directory under `remote_root`.

    :param ftp:          Connected ftplib.FTP instance
    :param remote_root:  Remote directory to search in
    :return:             Name of the most recent sub‑directory
    """
    entries = list_dir_details(ftp, remote_root)
    dirs: List[Tuple[str, Dict[str, str]]] = [
        (name, facts) for name, facts in entries if facts.get("type") == "dir"
    ]

    if not dirs:
        raise RuntimeError(f"No directories found in {remote_root}")

    # Prefer MLSD modify time; if not available, just pick the first one
    def get_modify_time(name: str, facts: Dict[str, str]) -> datetime | None:
        mod = facts.get("modify")
        if mod:
            # MLSD modify format: YYYYMMDDHHMMSS
            return datetime.strptime(mod, "%Y%m%d%H%M%S")
        return None

    dirs_with_time = [
        (name, get_modify_time(name, facts)) for name, facts in dirs
    ]

    # Filter out entries without a modify time
    dirs_with_time = [(n, t) for n, t in dirs_with_time if t is not None]

    if dirs_with_time:
        # Sort by modify time descending
        most_recent = sorted(dirs_with_time, key=lambda x: x[1], reverse=True)[0][0]
    else:
        # Fallback – just take the first directory in the listing
        most_recent = dirs[0][0]

    return most_recent


def download_and_process_dir(
    ftp: ftplib.FTP, remote_path: str
) -> bool:
    """
    Recursively download all .fits/.tif files from `remote_path` and its
    sub‑directories, rename them according to the object name & date,
    store them locally, and optionally delete them from the server.

    :param ftp:          Connected ftplib.FTP instance
    :param remote_path:  Full path of the directory to process
    :return:             True if at least one file was downloaded, else False
    """
    found_files = False

    entries = list_dir_details(ftp, remote_path)

    # Separate files and sub‑directories
    files_to_download: List[str] = [
        name for name, facts in entries
        if facts.get("type") == "file" and re.search(r"\.(fits|tif)$", name, re.IGNORECASE)
    ]
    subdirs: List[str] = [
        name for name, facts in entries if facts.get("type") == "dir"
    ]

    # --------------------------------------------------------------------- #
    #  Download files
    # --------------------------------------------------------------------- #
    for filename in files_to_download:
        local_file = LOCAL_DIR / filename
        print(f"Downloading {filename} from {remote_path}")

        with open(local_file, "wb") as fp:
            ftp.retrbinary(f"RETR {remote_path}/{filename}", fp.write)

        # ----------------------------------------------------------------- #
        #  Rename & move the file
        # ----------------------------------------------------------------- #
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", remote_path)
        date = date_match.group(0) if date_match else "unknown"

        obj_match = re.search(r"observation-([^/]+)", remote_path)
        objet = obj_match.group(1).upper() if obj_match else "UNKNOWN"

        dest_dir = LOCAL_DIR / objet
        dest_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(filename).suffix.lstrip(".")
        new_name = f"{objet}_{date}.{ext}"
        dest_path = dest_dir / new_name

        shutil.move(str(local_file), str(dest_path))
        print(f"  → stored as {dest_path}")

        # Optional: delete the file from the server
        # ftp.delete(f"{remote_path}/{filename}")

        found_files = True

    # --------------------------------------------------------------------- #
    #  Recurse into sub‑directories
    # --------------------------------------------------------------------- #
    for sub in subdirs:
        sub_path = f"{remote_path}/{sub}"
        if download_and_process_dir(ftp, sub_path):
            found_files = True

    return found_files


# --------------------------------------------------------------------------- #
#  Main routine
# --------------------------------------------------------------------------- #

def main() -> None:
    global FAILED_CHECKS

    # Ensure local base directory exists
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        ftp = ftplib.FTP(FTP_SERVER)
        ftp.login(user=FTP_USER, passwd="")  # anonymous login
    except Exception as exc:
        print(f"❌ Could not connect to {FTP_SERVER}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        most_recent = find_most_recent_dir(ftp, REMOTE_DIR)
        print(f"Exploring most recent directory: {REMOTE_DIR}/{most_recent}")

        found = download_and_process_dir(ftp, f"{REMOTE_DIR}/{most_recent}")

        if not found:
            FAILED_CHECKS += 1
            print(f"⚠️ No .fits/.tif files found. Failed attempt #{FAILED_CHECKS}")
        else:
            FAILED_CHECKS = 0
            print("✅ Files downloaded successfully.")

        if FAILED_CHECKS >= MAX_FAILED_CHECKS:
            print(
                f"❌ Maximum of {MAX_FAILED_CHECKS} consecutive failures reached. "
                "Stopping script."
            )
    finally:
        ftp.quit()


if __name__ == "__main__":
    main()
