"""Step 1: download the public inputs (no account or API key needed).

- TWDB Groundwater Database (GWDB) bulk download, pipe-delimited text files.
- TWDB major aquifer boundaries (shapefile).
- U.S. Census 2023 cartographic boundary counties (1:500k), filtered to Texas later.

Files go to RAW_DIR (default data/raw/, which is git-ignored). Existing files are kept
unless --force is given. An access log with byte sizes and dates is written next to them.
"""
import sys
import datetime as dt
import urllib.request
import zipfile

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))  # works under python -I
from common import RAW_DIR, SOURCES


def main(force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    for name, url in SOURCES.items():
        out = RAW_DIR / name
        if out.exists() and not force:
            print(f"exists, skipping: {name}")
        else:
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, out)
        with zipfile.ZipFile(out) as z:  # check the archive opens
            n = len(z.namelist())
        log.append(f"{name}\t{url}\t{out.stat().st_size} bytes\t{n} members\t"
                   f"accessed {dt.date.today().isoformat()}")
    (RAW_DIR / "download_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main(force="--force" in sys.argv)
