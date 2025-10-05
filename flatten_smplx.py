# flatten_smplx_outputs.py
from pathlib import Path
import shutil
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", required=True, help="e.g. output/D0001N/results")
ap.add_argument("--out_dir",     required=True, help="e.g. result/obj/D0001N")
ap.add_argument("--also_pkls",   action="store_true", help="copy 000.pkl too")
args = ap.parse_args()

results_dir = Path(args.results_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

moved = 0
for frame_dir in sorted(results_dir.iterdir()):
    if not frame_dir.is_dir():
        continue
    frame_name = frame_dir.name  # e.g. D0001N_000000000123
    src_obj = frame_dir / "000.obj"
    if src_obj.exists():
        dst_obj = out_dir / f"{frame_name}.obj"
        shutil.copy2(src_obj, dst_obj)
        moved += 1
    if args.also_pkls:
        src_pkl = frame_dir / "000.pkl"
        if src_pkl.exists():
            dst_pkl = out_dir / f"{frame_name}.pkl"
            shutil.copy2(src_pkl, dst_pkl)

print(f"Copied {moved} OBJ(s) to {out_dir}")
