# rename_videos_from_csv.py  (case-sensitive aware)
import csv, re
from pathlib import Path

# --- CONFIG ---
CSV_PATH   = Path("data/label_with_folders.csv")
VIDEO_DIR  = Path("data/Videos")
DO_RENAME  = True
MOVE_INTO_CLASS_FOLDER = False
CASE_SENSITIVE = True  # <- set True to enforce exact case match

# Sanitize a label for Windows filenames (keep accents)
_ILLEGAL = re.compile(r'[\\/:*?"<>|]')
def sanitize(name: str) -> str:
    name = name.strip()
    name = _ILLEGAL.sub(" ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip(" .")

def main():
    assert VIDEO_DIR.is_dir(), f"Folder not found: {VIDEO_DIR.resolve()}"

    # Index directory entries
    files_exact = {p.name: p for p in VIDEO_DIR.iterdir() if p.is_file()}
    files_lower = {p.name.lower(): p for p in VIDEO_DIR.iterdir() if p.is_file()}  # for fallback suggestions

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    planned, used_names = [], {p.name for p in VIDEO_DIR.iterdir() if p.is_file()}

    for row in rows:
        src_name = row["VIDEO"].strip()
        label    = row["LABEL"].strip()
        folder   = (row.get("FOLDER_NAME") or "").strip()
        vid_id   = (row.get("ID") or "").strip()

        # Case-sensitive lookup
        src_path = files_exact.get(src_name)
        if src_path is None:
            if CASE_SENSITIVE:
                # Suggest case-insensitive match if exists
                alt = files_lower.get(src_name.lower())
                if alt is not None:
                    print(f"[SKIP] Case mismatch for '{src_name}'. Actual file is '{alt.name}'.")
                else:
                    print(f"[SKIP] Not found: {src_name}")
                continue
            else:
                # Case-insensitive use
                src_path = files_lower.get(src_name.lower())
                if src_path is None:
                    print(f"[SKIP] Not found: {src_name}")
                    continue

        ext = src_path.suffix
        base = sanitize(label)
        if not base:
            print(f"[SKIP] Empty/invalid label for {src_path.name}")
            continue

        target_dir = VIDEO_DIR / folder if (MOVE_INTO_CLASS_FOLDER and folder) else VIDEO_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        candidate = f"{base}{ext}"
        new_path = target_dir / candidate
        n = 2
        while new_path.name in used_names or new_path.exists():
            if vid_id and f"{base}_{vid_id}{ext}" not in used_names:
                candidate = f"{base}_{vid_id}{ext}"
                new_path = target_dir / candidate
                break
            candidate = f"{base} ({n}){ext}"
            new_path = target_dir / candidate
            n += 1

        planned.append((src_path, new_path))
        used_names.add(new_path.name)

    print("\n=== DRY RUN ===")
    for src, dst in planned:
        try:
            print(f"{src.relative_to(VIDEO_DIR)} -> {dst.relative_to(VIDEO_DIR)}")
        except ValueError:
            print(f"{src} -> {dst}")

    if not DO_RENAME:
        print("\nNothing renamed (DO_RENAME=False). If OK, set DO_RENAME=True and run again.")
        return

    print("\n=== RENAMING ===")
    for src, dst in planned:
        if src.resolve() == dst.resolve():
            print(f"[SKIP] Same name: {src.name}")
            continue
        src.rename(dst)
        print(f"[OK] {src.name} -> {dst.relative_to(VIDEO_DIR)}")

    print("\nDone.")

if __name__ == "__main__":
    main()
