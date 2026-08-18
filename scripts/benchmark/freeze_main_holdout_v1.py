from pathlib import Path
import hashlib
import shutil

root = Path(r"data\final-holdout")
freeze = root / "frozen-main-v1"

freeze.mkdir(parents=True, exist_ok=True)

files = [
    "Promptune_final_holdout_100.csv",
    "Promptune_human_review_all_H001_H100.csv",
    "Promptune_final_holdout_coverage_audit.csv",
]

hash_lines = []

for name in files:
    src = root / name

    if not src.exists():
        raise FileNotFoundError(src)

    dst = freeze / name

    shutil.copy2(src, dst)

    sha256 = hashlib.sha256(
        dst.read_bytes()
    ).hexdigest()

    hash_lines.append(
        f"{sha256}  {name}"
    )

    print(
        f"Frozen: {name}"
    )
    print(
        f"SHA256: {sha256}"
    )

hash_path = freeze / "SHA256SUMS.txt"

hash_path.write_text(
    "\n".join(hash_lines) + "\n",
    encoding="utf-8"
)

print()
print("=== MAIN HOLDOUT FROZEN ===")
print("Folder:", freeze)
print("Files :", len(files))
print("Hashes:", hash_path)
