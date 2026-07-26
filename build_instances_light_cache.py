#!/usr/bin/env python3
"""Build lightweight instance caches for CornerBench scene retrieval."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instances_light.npz for processed nuPlan scenes.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
        help="Processed nuPlan root containing split directories.",
    )
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        help="Split to process. Repeat this option to process multiple splits.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing caches.")
    return parser.parse_args()


def scene_dirs(processed_root: Path, splits: list[str]) -> list[Path]:
    scenes = []
    for split in splits:
        split_dir = processed_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Processed split not found: {split_dir}")
        scenes.extend(
            sorted(path for path in split_dir.iterdir() if path.is_dir() and path.name[:1].isdigit())
        )
    return scenes


def build_cache(scene_dir: Path, overwrite: bool) -> bool:
    instances_dir = scene_dir / "instances"
    output_path = instances_dir / "instances_light.npz"
    if output_path.exists() and not overwrite:
        return False

    with (instances_dir / "instances_info.json").open() as handle:
        instances_info = json.load(handle)
    with (instances_dir / "frame_instances.json").open() as handle:
        frame_instances = json.load(handle)

    annotation_indices = {
        str(instance_id): {
            int(frame_id): index
            for index, frame_id in enumerate(instance["frame_annotations"]["frame_idx"])
        }
        for instance_id, instance in instances_info.items()
    }
    frame_ids = sorted(int(frame_id) for frame_id in frame_instances)
    frame_starts = [0]
    centers_xy = []
    radii = []

    for frame_id in frame_ids:
        for instance_id in frame_instances.get(str(frame_id), []):
            key = str(instance_id)
            instance = instances_info.get(key)
            annotation_index = annotation_indices.get(key, {}).get(frame_id)
            if instance is None or annotation_index is None:
                continue
            annotations = instance["frame_annotations"]
            pose = np.asarray(annotations["obj_to_world"][annotation_index], dtype=np.float32).reshape(4, 4)
            length, width = annotations["box_size"][annotation_index][:2]
            centers_xy.append([float(pose[0, 3]), float(pose[1, 3])])
            radii.append(0.5 * math.hypot(float(length), float(width)))
        frame_starts.append(len(centers_xy))

    np.savez_compressed(
        output_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        frame_starts=np.asarray(frame_starts, dtype=np.int32),
        centers_xy=np.asarray(centers_xy, dtype=np.float32),
        radii=np.asarray(radii, dtype=np.float32),
    )
    return True


def main() -> int:
    args = parse_args()
    scenes = scene_dirs(args.processed_root.expanduser().resolve(), args.split)
    built = 0
    for index, scene in enumerate(scenes, start=1):
        changed = build_cache(scene, args.overwrite)
        built += int(changed)
        print(f"[{index}/{len(scenes)}] {'built' if changed else 'exists'} {scene.name}")
    print(f"Built {built} cache(s); {len(scenes) - built} already existed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
