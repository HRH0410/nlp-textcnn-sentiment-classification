import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


GLOVE_6B_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="embeddings")
    parser.add_argument("--dim", type=int, default=300, choices=[50, 100, 200, 300])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"glove.6B.{args.dim}d.txt"
    if target.exists():
        print(f"exists={target}")
        return

    zip_path = output_dir / "glove.6B.zip"
    if not zip_path.exists():
        print(f"downloading={GLOVE_6B_URL}")
        urlretrieve(GLOVE_6B_URL, zip_path)

    member = f"glove.6B.{args.dim}d.txt"
    print(f"extracting={member}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(member, output_dir)
    print(f"saved={target}")


if __name__ == "__main__":
    main()
