# run_smplx_all.py — batch SMPLify-X over all videos under result/json/*
# Assumes per-video frames live under result/images/<video> (same names).
import argparse, sys, subprocess
import os
from pathlib import Path

def run_and_print(cmd, cwd):
    print("\n[run_smplx] Command:\n ", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print("\n[run_smplx] ---- STDOUT ----\n", res.stdout)
        print("\n[run_smplx] ---- STDERR ----\n", res.stderr)
        raise SystemExit(res.returncode)
    else:
        print(res.stdout)

def main():
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser("Batch SMPLify-X over OpenPose keypoints")
    # roots
    ap.add_argument("--json_root", default="result/json", help="root folder containing <video> subfolders of *_keypoints.json")
    ap.add_argument("--images_root", default="result/images", help="root folder containing per-video frame folders")
    ap.add_argument("--out_root", default="output", help="root where per-video outputs are written")
    # options
    ap.add_argument("--gender", choices=["neutral","male","female"], default="neutral")
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    # filter
    ap.add_argument("--only", nargs="*", help="optional list of video folder names to process (e.g., D0001N D0002N)")
    args = ap.parse_args()

    smplifyx_dir = here / "smplify-x"
    main_py      = smplifyx_dir / "smplifyx" / "main.py"
    cfg_yaml     = smplifyx_dir / "cfg_files" / "fit_smplx.yaml"

    models_dir   = smplifyx_dir / "models"
    smplx_models = models_dir / "smplx"
    vposer_dir   = models_dir / "vposer_v1_0"
    part_segm    = models_dir / "smplx_parts_segm.pkl"

    # sanity checks that do not depend on a specific video
    for p,l in [(main_py,"main.py"),(cfg_yaml,"cfg"),
                (smplx_models,"models/smplx"),
                (vposer_dir,"models/vposer_v1_0"),
                (part_segm,"models/smplx_parts_segm.pkl")]:
        if not p.exists():
            print(f"[run_smplx] Missing {l}: {p}")
            raise SystemExit(1)

    json_root   = (here / args.json_root).resolve()
    images_root = (here / args.images_root).resolve()
    out_root    = (here / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not json_root.exists():
        raise SystemExit(f"[run_smplx] json_root not found: {json_root}")
    if not images_root.exists():
        raise SystemExit(f"[run_smplx] images_root not found: {images_root}")

    # collect per-video folders (those containing *_keypoints.json)
    video_dirs = []
    for sub in sorted(d for d in json_root.iterdir() if d.is_dir()):
        if any(sub.glob("*_keypoints.json")):
            video_dirs.append(sub)

    if args.only:
        only_set = set(args.only)
        video_dirs = [d for d in video_dirs if d.name in only_set]

    if not video_dirs:
        raise SystemExit(f"[run_smplx] No per-video keypoint folders found under {json_root}")

    print(f"[run_smplx] Videos to process: {len(video_dirs)}")
    for keyp_folder in video_dirs:
        vid_name = keyp_folder.name
        img_folder = images_root / vid_name
        out_folder = out_root / vid_name
        out_folder.mkdir(parents=True, exist_ok=True)

        if not img_folder.exists() or not any(img_folder.glob("*.png")):
            print(f"[warn] Skipping {vid_name}: frames not found in {img_folder}")
            continue

        # SMPLify-X expects OpenPose layout when: --dataset openpose
        # That maps BODY_25 body + (optionally) hands/face if present in JSONs.
        main_py      = smplifyx_dir / "smplifyx" / "main.py"
        pkg_dir      = main_py.parent  # .../smplify-x/smplifyx

        cmd = [
            sys.executable, str(main_py),
            "--config",       str(cfg_yaml),
            "--dataset",      "openpose",
            "--keyp_folder",  str(keyp_folder),
            "--img_folder",   str(img_folder),
            "--output_folder",str(out_folder),
            "--result_folder","results",
            "--mesh_folder",  "results",
            "--model_folder", str(models_dir),
            "--vposer_ckpt",  str(models_dir/ "vposer_v1_0"),
            "--part_segm_fn", str(models_dir / "smplx_parts_segm.pkl"),
            "--gender",       args.gender,
            "--use_cuda",     "True" if args.use_cuda else "False",
            "--interactive",  "False",
            "--visualize",    "True" if args.visualize else "False",
            "--use_hands",    "True",
            "--use_face",     "True",
            "--use_face_contour","False",
            "--interpenetration","False",
            "--penalize_outside","False",
            "--max_collisions","0",
            "--use_vposer",   "True",
            "--save_meshes",  "True",
        ]

        print(f"\n[run_smplx] Video: {vid_name}")
        print(f"  keypoints: {keyp_folder}")
        print(f"  images:    {img_folder}")
        print(f"  output:    {out_folder}")

        run_and_print(cmd, cwd=str(pkg_dir))

    print("\n[run_smplx] ✓ All done.")

if __name__ == "__main__":
    main()
