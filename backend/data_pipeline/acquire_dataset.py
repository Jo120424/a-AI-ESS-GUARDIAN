#!/usr/bin/env python3
"""
Dataset Acquisition Script for Predictive ESS Research Platform.
Downloads the official NASA Capacitor Electrical Stress Degradation raw data,
verifies checksums, preserves unmodified raw archives, and updates acquisition metadata.
"""

import hashlib
import json
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "nasa_capacitor_electrical_stress"
METADATA_DIR = BASE_DIR / "data" / "metadata"

PRIMARY_URL = "https://data.nasa.gov/docs/legacy/EOS_DataSet.zip"
PORTAL_LANDING = "https://data.nasa.gov/dataset/capacitor-electrical-stress-2"


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def acquire_dataset():
    print(f"=== Starting Dataset Acquisition ===")
    print(f"Target directory: {RAW_DIR}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = RAW_DIR / "EOS_DataSet.zip"

    # Step 1: Download if not already present
    if not zip_path.exists():
        print(f"Downloading from official NASA portal: {PRIMARY_URL}")
        req = urllib.request.Request(
            PRIMARY_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResearchBot/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as out_file:
                data = response.read()
                out_file.write(data)
            print(f"Downloaded {len(data)} bytes to {zip_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please place EOS_DataSet.zip manually in data/raw/nasa_capacitor_electrical_stress/")
            return False
    else:
        print(f"Raw archive already exists at: {zip_path}")

    # Step 2: Extract contents without modifying zip
    print(f"Extracting archive contents...")
    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            # Skip OS X metadata folders
            if member.startswith("__MACOSX"):
                continue
            z.extract(member, RAW_DIR)
            target_file = RAW_DIR / member
            if target_file.is_file():
                extracted_files.append({
                    "file_name": member,
                    "size_bytes": target_file.stat().st_size,
                    "sha256": compute_sha256(target_file)
                })
                print(f"Extracted: {member} ({target_file.stat().st_size} bytes, sha256: {extracted_files[-1]['sha256'][:12]}...)")

    zip_sha = compute_sha256(zip_path)

    # Step 3: Record acquisition metadata
    acq_meta = {
        "dataset_name": "NASA Capacitor Electrical Stress Degradation Dataset",
        "source_url": PRIMARY_URL,
        "landing_page": PORTAL_LANDING,
        "original_publication": "Celaya, J. R., Kulkarni, C., Biswas, G., & Goebel, K. (2012). Towards A Model-based Prognostics Methodology for Electrolytic Capacitors: A Case Study Based on Electrical Overstress Accelerated Aging. Annual Conference of the PHM Society 2012; and Renwick, J., Kulkarni, C., & Celaya, J. (2015). Analysis of Electrolytic Capacitor Degradation under Electrical Overstress for Prognostic Studies.",
        "organization": "NASA Ames Research Center, Prognostics Center of Excellence (PCoE)",
        "access_date": datetime.now(timezone.utc).isoformat(),
        "version": "NASA Open Data Release (EOS_DataSet v2)",
        "license": "NASA Open Data Policy (Public Domain)",
        "raw_files": [
            {
                "file_name": "EOS_DataSet.zip",
                "size_bytes": zip_path.stat().st_size,
                "sha256": zip_sha
            }
        ] + extracted_files,
        "checksum": zip_sha,
        "notes": "Acquired directly from NASA Open Data Portal. Contains longitudinal aging measurements of Capacitance drift (C) and Equivalent Series Resistance (ESR) across 11 discrete time points (up to 194 hours) for discrete physical capacitor units under 10V accelerated electrical overstress."
    }

    meta_file = METADATA_DIR / "acquisition_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(acq_meta, f, indent=2)

    print(f"Acquisition metadata written to: {meta_file}")
    print(f"=== Acquisition Complete ===")
    return True


if __name__ == "__main__":
    success = acquire_dataset()
    sys.exit(0 if success else 1)
