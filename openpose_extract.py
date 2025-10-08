# run_openpose_batch.py  (single-run: JSON+RAW images OR JSON+overlay video)
import argparse
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def parse_args():
    p = argparse.ArgumentParser(description="Batch-run OpenPose on videos")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--exe", type=Path, default=None,
                   help="Path to OpenPoseDemo.exe (default: <root>/openpose/bin/OpenPoseDemo.exe)")
    p.add_argument("--models", type=Path, default=None,
                   help="Path to OpenPose models (default: <root>/openpose/models)")
    p.add_argument("--data", type=Path, default=None,
                   help="Folder of input videos (default: <root>/data/Videos)")
    p.add_argument("--out_json_root", type=Path, default=None,
                   help="Output root for JSON folders (default: <root>/result/json)")
    p.add_argument("--out_images_root", type=Path, default=None,
                   help="Output root for RAW frames (default: <root>/result/images)")
    p.add_argument("--out_video_root", type=Path, default=None,
                   help="Output root for overlay videos (default: <root>/result/videos)")
    p.add_argument("--images_format", type=str, default="png", choices=["png", "jpg"])
    p.add_argument("--video_overlay", action="store_true",
                   help="Write overlayed video instead of RAW frames (still writes JSON)")
    p.add_argument("--gpu_start", type=int, default=0)
    p.add_argument("--num_gpu", type=int, default=1)
    p.add_argument("--hand_res", type=str, default=None, help="e.g. 368x368")
    p.add_argument("--hand_scales", type=float, nargs=2, metavar=("NUM", "RANGE"), default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--show_logs", action="store_true")
    return p.parse_args()

def run(cmd, show_logs):
    if show_logs:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def main():
    args = parse_args()
    root = args.root.resolve()
    exe = (args.exe or (root / "openpose" / "bin" / "OpenPoseDemo.exe")).resolve()
    models = (args.models or (root / "openpose" / "models")).resolve()
    data_dir = (args.data or (root / "data" / "Videos")).resolve()
    json_root = (args.out_json_root or (root / "result" / "json")).resolve()
    img_root  = (args.out_images_root or (root / "result" / "images")).resolve()
    vid_root  = (args.out_video_root or (root / "result" / "videos")).resolve()

    if not exe.exists(): sys.exit(f"[!] OpenPose exe not found: {exe}")
    if not models.exists(): sys.exit(f"[!] Models folder not found: {models}")
    if not data_dir.exists(): sys.exit(f"[!] Data folder not found: {data_dir}")

    json_root.mkdir(parents=True, exist_ok=True)
    img_root.mkdir(parents=True, exist_ok=True)
    vid_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in data_dir.glob("*.mp4") if p.is_file())
    if not videos: sys.exit(f"[!] No .mp4 videos in {data_dir}")

    print(f"[i] Found {len(videos)} video(s).")
    print(f"[i] JSONs  -> {json_root}")
    if args.video_overlay:
        print(f"[i] Mode  -> JSON + overlay video")
        print(f"[i] Video -> {vid_root}")
    else:
        print(f"[i] Mode  -> JSON + RAW frames")
        print(f"[i] Images-> {img_root} (.{args.images_format})")

    iterator = tqdm(videos, desc="Processing videos", unit="vid") if tqdm else videos

    for vid in iterator:
        base = vid.stem
        out_json   = json_root / base
        out_images = img_root / base
        out_video  = vid_root / f"{base}_annotated.avi"

        out_json.mkdir(parents=True, exist_ok=True)
        if not args.video_overlay:
            out_images.mkdir(parents=True, exist_ok=True)

        have_json   = any(out_json.glob("*_keypoints.json"))
        have_images = any(out_images.glob(f"*.{args.images_format}")) if not args.video_overlay else True
        have_video  = out_video.exists() if args.video_overlay else True

        if not args.overwrite and have_json and have_images and have_video:
            (iterator.write if tqdm else print)(f"[skip] {base} (outputs exist)")
            continue

        cmd = [
            str(exe),
            "--video", str(vid),
            "--model_pose", "BODY_25",
            "--hand",
            "--face",
            "--model_folder", str(models),
            "--write_json", str(out_json),
            "--display", "0",
            "--num_gpu", str(args.num_gpu),
            "--num_gpu_start", str(args.gpu_start),

            # Additional configs
            "--net_resolution", "1312x736",
            "--scale_number", "4",
            "--scale_gap", "0.25",
            "--hand_scale_number", "6",
            "--hand_scale_range", "0.4",
        ]

        if args.video_overlay:
            # JSON + overlay VIDEO (draw everything)
            cmd += [
                "--render_pose", "1",
                "--write_video", str(out_video)
            ]
        else:
            # JSON + RAW-LOOKING FRAMES (no visible overlay)
            cmd += [
                "--render_pose", "0",
                "--write_images", str(out_images),
                "--write_images_format", "png",
            ]

        if args.hand_res:
            cmd += ["--hand_net_resolution", args.hand_res]
        if args.hand_scales:
            num, rng = args.hand_scales
            cmd += ["--hand_scale_number", str(int(num)), "--hand_scale_range", str(rng)]

        (iterator.write if tqdm else print)(
            f"[run] {base} ({'overlay video' if args.video_overlay else 'raw frames'})"
        )
        try:
            run(cmd, args.show_logs)
        except subprocess.CalledProcessError as e:
            (iterator.write if tqdm else print)(f"[!] OpenPose failed on {vid.name} (exit {e.returncode})")

    print("\n[✓] Done.")

if __name__ == "__main__":
    main()
