#!/usr/bin/env python3
"""
Script to organize NuPlan dataset according to the required structure.
Moves downloaded files into the proper directory hierarchy.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Optional

def organize_nuplan_data(
    data_root: str,
    target_root: str = "data/nuplan/raw"
) -> None:
    """
    Organize NuPlan data from download directory into proper structure.

    Expected structure after organization:
    {target_root}/
    ├── maps/
    │   ├── nuplan-maps-v1.0.json
    │   ├── sg-one-north/
    │   │   └── 9.17.1964/
    │   │       └── map.gpkg
    │   └── ...
    └── nuplan-v1.1/
        ├── splits/
        │   ├── mini/
        │   │   ├── *.db
        │   │   └── ...
        │   └── ...
        └── sensor_blobs/
            ├── 2021.05.12.22.00.38_veh-35_01008_01518/
            │   ├── CAM_F0/
            │   ├── CAM_B0/
            │   ├── ...
            │   └── MergedPointCloud/
            └── ...

    Args:
        data_root: Path to directory containing downloaded and extracted files
        target_root: Path where organized data should be placed
    """
    data_root = Path(data_root)
    target_root = Path(target_root)

    if not data_root.exists():
        raise ValueError(f"Data root does not exist: {data_root}")

    # Create target directories
    target_root.mkdir(parents=True, exist_ok=True)
    maps_dir = target_root / "maps"
    nuplan_dir = target_root / "nuplan-v1.1"
    splits_dir = nuplan_dir / "splits"
    sensor_blobs_dir = nuplan_dir / "sensor_blobs"

    maps_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    sensor_blobs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target directory: {target_root}")

    # Step 1: Organize maps
    print("\n[1/3] Organizing maps...")
    organize_maps(data_root, maps_dir)

    # Step 2: Organize database files (splits)
    print("\n[2/3] Organizing database files (splits)...")
    organize_splits(data_root, splits_dir)

    # Step 3: Organize sensor blobs
    print("\n[3/3] Organizing sensor blobs...")
    organize_sensor_blobs(data_root, sensor_blobs_dir)

    print(f"\n✓ Data organization complete!")
    print(f"Organized data location: {target_root}")

def organize_maps(data_root: Path, maps_dir: Path) -> None:
    """Organize map files."""
    # Look for maps directory in data_root
    possible_maps = [
        data_root / "maps",
        data_root / "nuplan-maps-v1.0",
    ]

    maps_source = None
    for path in possible_maps:
        if path.exists():
            maps_source = path
            break

    if maps_source is None:
        print("  ⚠ No maps directory found")
        return

    if maps_source != maps_dir:
        if maps_source.is_dir():
            # Copy all contents from maps_source to maps_dir
            for item in maps_source.iterdir():
                dest = maps_dir / item.name
                if dest.exists():
                    print(f"  - Skipping {item.name} (already exists)")
                    continue

                if item.is_dir():
                    shutil.copytree(item, dest)
                    print(f"  ✓ Copied directory: {item.name}")
                else:
                    shutil.copy2(item, dest)
                    print(f"  ✓ Copied file: {item.name}")

def organize_splits(data_root: Path, splits_dir: Path) -> None:
    """Organize database files from different splits."""
    # Look for nuplan-v1.1 directory or split directories
    possible_sources = [
        data_root / "nuplan-v1.1" / "splits",
        data_root / "splits",
        data_root / "data" / "cache",  # Alternative location for .db files
    ]

    # Also check for split folders directly in data_root
    split_names = ["mini", "trainval", "test", "val"]

    found_splits = set()

    # Check explicit splits directories
    for source in possible_sources:
        if source.exists():
            for split_dir in source.iterdir():
                if split_dir.is_dir() and split_dir.name in split_names:
                    organize_split_folder(split_dir, splits_dir / split_dir.name)
                    found_splits.add(split_dir.name)

    # Check for split folders in data_root directly
    for split_name in split_names:
        split_path = data_root / split_name
        if split_path.exists() and split_path.is_dir():
            organize_split_folder(split_path, splits_dir / split_name)
            found_splits.add(split_name)

    if not found_splits:
        print("  ⚠ No split directories found")
    else:
        print(f"  ✓ Organized splits: {', '.join(sorted(found_splits))}")

def organize_split_folder(source: Path, target: Path) -> None:
    """Copy all .db files from source split to target."""
    target.mkdir(parents=True, exist_ok=True)

    db_count = 0
    for item in source.rglob("*.db"):
        dest = target / item.name
        if dest.exists():
            continue
        shutil.copy2(item, dest)
        db_count += 1

    if db_count > 0:
        print(f"  ✓ {source.name}: copied {db_count} database files")

def organize_sensor_blobs(data_root: Path, sensor_blobs_dir: Path) -> None:
    """Organize sensor blob files (camera and lidar data)."""
    # Look for sensor_blobs directory or camera/lidar prefixed directories
    possible_sources = [
        data_root / "nuplan-v1.1" / "sensor_blobs",
        data_root / "sensor_blobs",
    ]

    # Also look for camera/lidar data with prefixes
    camera_patterns = [
        data_root / "nuplan-v1.1_mini_camera_0",
        data_root / "nuplan-v1.1_mini_camera_1",
        data_root / "nuplan-v1.1_trainval_camera_0",
    ]

    lidar_patterns = [
        data_root / "nuplan-v1.1_mini_lidar_0",
        data_root / "nuplan-v1.1_trainval_lidar_0",
    ]

    # First try standard sensor_blobs location
    blobs_source = None
    for path in possible_sources:
        if path.exists():
            blobs_source = path
            break

    sequence_count = 0
    sequences_by_name = {}  # Track sequences to merge camera and lidar

    # Copy from standard sensor_blobs if found
    if blobs_source is not None:
        for item in blobs_source.iterdir():
            if not item.is_dir():
                continue

            dest = sensor_blobs_dir / item.name
            if dest.exists():
                print(f"  - Skipping {item.name} (already exists)")
                continue

            shutil.copytree(item, dest)
            sequence_count += 1

    # Process camera data
    for camera_dir in camera_patterns:
        if not camera_dir.exists():
            continue

        for sequence_dir in camera_dir.iterdir():
            if not sequence_dir.is_dir():
                continue

            sequence_name = sequence_dir.name
            dest = sensor_blobs_dir / sequence_name
            dest.mkdir(parents=True, exist_ok=True)

            # Copy camera subdirectories
            for cam_subdir in sequence_dir.iterdir():
                if cam_subdir.is_dir():
                    cam_dest = dest / cam_subdir.name
                    if not cam_dest.exists():
                        shutil.copytree(cam_subdir, cam_dest)
                        if sequence_name not in sequences_by_name:
                            sequences_by_name[sequence_name] = True
                            sequence_count += 1

    # Process lidar data
    for lidar_dir in lidar_patterns:
        if not lidar_dir.exists():
            continue

        for sequence_dir in lidar_dir.iterdir():
            if not sequence_dir.is_dir():
                continue

            sequence_name = sequence_dir.name
            dest = sensor_blobs_dir / sequence_name
            dest.mkdir(parents=True, exist_ok=True)

            # Copy MergedPointCloud directory
            for lidar_subdir in sequence_dir.iterdir():
                if lidar_subdir.is_dir() and lidar_subdir.name == "MergedPointCloud":
                    lidar_dest = dest / lidar_subdir.name
                    if not lidar_dest.exists():
                        shutil.copytree(lidar_subdir, lidar_dest)
                        if sequence_name not in sequences_by_name:
                            sequences_by_name[sequence_name] = True
                            sequence_count += 1

    if sequence_count > 0:
        print(f"  ✓ Organized {sequence_count} sensor blob sequences")
    else:
        print("  ⚠ No sensor_blobs data found")

def main():
    parser = argparse.ArgumentParser(
        description="Organize NuPlan dataset into required structure"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to directory containing downloaded NuPlan files"
    )
    parser.add_argument(
        "--target_root",
        default="data/nuplan/raw",
        help="Path where organized data should be placed (default: data/nuplan/raw)"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (saves disk space but removes originals)"
    )

    args = parser.parse_args()

    # If move flag is set, use shutil.move instead of copy
    if args.move:
        global shutil_action
        original_copytree = shutil.copytree
        original_copy2 = shutil.copy2

        def move_copytree(src, dst, **kwargs):
            shutil.move(src, dst)

        def move_copy2(src, dst):
            shutil.move(src, dst)

        shutil.copytree = move_copytree
        shutil.copy2 = move_copy2
        print("Using MOVE instead of COPY (files will be removed from source)")

    organize_nuplan_data(args.data_root, args.target_root)

if __name__ == "__main__":
    main()
