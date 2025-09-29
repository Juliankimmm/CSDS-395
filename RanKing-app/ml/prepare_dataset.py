import os
import time
import requests
import random
from pathlib import Path

# --- Config ---
DATASET_DIR = "RanKing-app/ml/datasets"

TRAIN_SAFE = os.path.join(DATASET_DIR, "train/safe")
TRAIN_UNSAFE = os.path.join(DATASET_DIR, "train/unsafe")
VAL_SAFE = os.path.join(DATASET_DIR, "val/safe")
VAL_UNSAFE = os.path.join(DATASET_DIR, "val/unsafe")

CATEGORY_LABEL = {
    "neutral": "safe",
    "porn": "unsafe",
    "sexy": "unsafe",
    "hentai": "unsafe",
    "drawings": "unsafe",
}

LIMIT = 200
VAL_SPLIT = 0.2
DELAY = 0.15

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
}

CONTENT_EXT_MAP = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff"
}

URL_FILES = {
    "neutral": "urls_neutral.txt",
    "porn": "urls_porn.txt",
    "sexy": "urls_sexy.txt",
    "hentai": "urls_hentai.txt",
    "drawings": "urls_drawings.txt"
}

# --- Helpers ---
def ensure_dirs():
    for path in [TRAIN_SAFE, TRAIN_UNSAFE, VAL_SAFE, VAL_UNSAFE]:
        os.makedirs(path, exist_ok=True)

def sanitize_url(line: str):
    u = line.strip()
    if not u or not (u.startswith("http://") or u.startswith("https://")):
        return None
    return u

def download_one(url: str, outpath: Path, tries=2, timeout=12):
    for attempt in range(1, tries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
            continue

        if resp.status_code != 200:
            last_exc = f"HTTP {resp.status_code}"
            time.sleep(0.2)
            continue

        ctype = (resp.headers.get("content-type") or "").split(";")[0].lower()
        if not ctype.startswith("image/"):
            return False, f"non-image content-type: {ctype}"

        ext = CONTENT_EXT_MAP.get(ctype, "")
        if not ext:
            ext = Path(url).suffix if Path(url).suffix else ".jpg"

        if outpath.suffix.lower() != ext.lower():
            outpath = outpath.with_suffix(ext)

        try:
            with open(outpath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if outpath.stat().st_size < 1024:
                outpath.unlink(missing_ok=True)
                return False, "file too small"
            return True, "ok"
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
            continue

    return False, f"failed after {tries} tries: {last_exc}"

def process_category(category: str, url_file: str):
    if not os.path.exists(url_file):
        print(f"URL list not found: {url_file}")
        return 0

    with open(url_file, "r", encoding="utf-8", errors="ignore") as f:
        urls = [sanitize_url(u) for u in f.read().splitlines() if sanitize_url(u)]

    urls = urls[:LIMIT]
    random.shuffle(urls)
    split_idx = int(len(urls) * (1 - VAL_SPLIT))
    train_urls, val_urls = urls[:split_idx], urls[split_idx:]

    label = CATEGORY_LABEL[category]
    train_dir = TRAIN_SAFE if label == "safe" else TRAIN_UNSAFE
    val_dir = VAL_SAFE if label == "safe" else VAL_UNSAFE

    count = 0
    # Download train images
    for idx, url in enumerate(train_urls):
        outpath = Path(train_dir) / f"{category}_train_{idx}.jpg"
        ok, msg = download_one(url, outpath)
        if ok:
            count += 1
            print(f"Saved train: {outpath}")
        else:
            print(f"Failed train: {url} -> {msg}")
        time.sleep(DELAY)

    # Download val images
    for idx, url in enumerate(val_urls):
        outpath = Path(val_dir) / f"{category}_val_{idx}.jpg"
        ok, msg = download_one(url, outpath)
        if ok:
            count += 1
            print(f"Saved val: {outpath}")
        else:
            print(f"Failed val: {url} -> {msg}")
        time.sleep(DELAY)

    return count

# --- Main ---
if __name__ == "__main__":
    ensure_dirs()
    BASE_DIR = "RanKing-app/ml/raw_data/nsfw_data_scraper/raw_data"
    total = 0
    for category, url_file in URL_FILES.items():
        # Include the category folder in the path
        url_path = os.path.join(BASE_DIR, category, url_file)
        downloaded = process_category(category, url_path)
        print(f"{category}: downloaded {downloaded} images")
        total += downloaded
    print(f"Total images downloaded: {total}")

