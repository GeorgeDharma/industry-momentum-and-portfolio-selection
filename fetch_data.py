"""Obtain Kenneth R. French Data Library inputs used by the industry study."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile


def digest(contents):
    return sha256(contents).hexdigest()


def ensure_inputs(root, *, source_dir=None, exact=False, refresh=False):
    destination = Path(root) / "data" / "source"
    provenance = json.loads(
        (destination / "provenance.json").read_text(encoding="utf-8")
    )
    if not refresh and not exact and source_dir is None:
        if all((destination / f"{name}.csv").is_file() for name in provenance):
            return
    # Validate the entire set before installing a new vintage.
    with TemporaryDirectory(dir=destination) as temporary:
        staged = Path(temporary)
        observed = {}
        different = []
        for name, record in provenance.items():
            public_name = Path(urlparse(record["url"]).path).name
            if source_dir is not None:
                candidates = [
                    Path(source_dir) / filename
                    for filename in (name + ".zip", public_name)
                ]
                archive = next((path for path in candidates if path.is_file()), None)
                if archive is None:
                    raise FileNotFoundError(
                        f"Missing {name}.zip or {public_name} in {source_dir}."
                    )
                contents = archive.read_bytes()
            elif exact and not refresh and (destination / f"{name}.zip").is_file():
                contents = (destination / f"{name}.zip").read_bytes()
            else:
                request = Request(
                    record["url"],
                    headers={"User-Agent": "George-Dharma-Industry-Research/1.0"},
                )
                try:
                    with urlopen(request, timeout=45) as response:
                        contents = response.read()
                except URLError as error:
                    raise SystemExit(
                        f"Could not download {record['url']}. Supply the four publisher ZIP files with --from-dir."
                    ) from error
            with ZipFile(BytesIO(contents)) as archive:
                members = [
                    entry
                    for entry in archive.namelist()
                    if Path(entry).name.lower() == record["member"].lower()
                ]
                if len(members) != 1:
                    raise ValueError(
                        f"{public_name} does not contain a unique {record['member']}."
                    )
                data = archive.read(members[0])
            hashes = {"sha256_zip": digest(contents), "sha256_csv": digest(data)}
            observed[name] = hashes
            if any(hashes[key] != record[key] for key in hashes):
                different.append(name)
            (staged / f"{name}.zip").write_bytes(contents)
            (staged / f"{name}.csv").write_bytes(data)
        if different and exact:
            raise ValueError(
                "Archives differ from the published snapshot: " + ", ".join(different)
            )
        if different:
            print(
                "Publisher archives differ from the published snapshot for "
                + ", ".join(different)
                + ". Recalculated results may change."
            )
        (staged / "downloaded.json").write_text(json.dumps(observed, indent=2) + "\n")
        for file in staged.iterdir():
            file.replace(destination / file.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-dir",
        type=Path,
        help="Read the four publisher ZIP files from this folder.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Require the hashes used in the published research.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download the current publisher archives again.",
    )
    args = parser.parse_args()
    ensure_inputs(
        Path(__file__).resolve().parent,
        source_dir=args.from_dir,
        exact=args.exact,
        refresh=args.refresh,
    )
