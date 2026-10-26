"""Download one large file with parallel HTTP Range requests (resolves redirects per chunk).

Usage: python3 download_ranged.py <url> <dest> [threads]
"""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

url, dest = sys.argv[1], sys.argv[2]
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CHUNK = 32 * 1024 * 1024

req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
with urllib.request.urlopen(req, timeout=60) as r:
    total = int(r.headers["Content-Range"].split("/")[1])
print(f"{dest}: {total} bytes", flush=True)

tmp = dest + ".part"
with open(tmp, "wb") as f:
    f.truncate(total)

done = 0
start = time.time()


def fetch(offset: int) -> None:
    global done
    end = min(offset + CHUNK, total) - 1
    for attempt in range(8):
        try:
            r = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}"})
            with urllib.request.urlopen(r, timeout=90) as resp:
                data = resp.read()
            if len(data) != end - offset + 1:
                raise IOError(f"short read {len(data)}")
            fd = os.open(tmp, os.O_WRONLY)
            os.pwrite(fd, data, offset)
            os.close(fd)
            done += len(data)
            return
        except Exception as e:  # retry the chunk
            print(f"chunk {offset} attempt {attempt} failed: {e}", flush=True)
            time.sleep(3)
    raise SystemExit(f"chunk {offset} failed permanently")


with ThreadPoolExecutor(threads) as pool:
    futures = [pool.submit(fetch, o) for o in range(0, total, CHUNK)]
    while not all(f.done() for f in futures):
        time.sleep(30)
        print(f"{done / 1e6:.0f}/{total / 1e6:.0f} MB, {done / 1e6 / (time.time() - start):.1f} MB/s", flush=True)
    for f in futures:
        f.result()

assert os.path.getsize(tmp) == total
os.replace(tmp, dest)
print("done", dest, flush=True)
