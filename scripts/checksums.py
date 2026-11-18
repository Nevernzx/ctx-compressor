"""Generate or verify the final delivery hashes; never includes this output itself."""
import argparse
import hashlib
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_files(root):
    files = {p for p in root.iterdir() if p.is_file() and p.name != "SHA256SUMS"}
    for dirname in ("src", "eval", "tests", "scripts", "specs", "data/raw", "data/traj"):
        base = root / dirname
        if base.exists():
            files.update(p for p in base.rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts
                         and ".pytest_cache" not in p.parts)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "SHA256SUMS"
    files = artifact_files(root)
    actual = {p.relative_to(root).as_posix(): digest(p) for p in files}
    if args.verify:
        expected = {line.split("  ", 1)[1]: line.split("  ", 1)[0]
                    for line in output.read_text().splitlines()}
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(k for k in set(actual) & set(expected) if actual[k] != expected[k])
        if missing or extra or changed:
            raise SystemExit(f"Hash verification failed: missing={missing}, extra={extra}, changed={changed}")
        print(f"SHA256SUMS verified: {len(actual)} files (including every data/raw file)")
    else:
        output.write_text("".join(f"{h}  {p}\n" for p, h in actual.items()))
        print(f"SHA256SUMS generated: {len(actual)} files")


if __name__ == "__main__":
    main()
