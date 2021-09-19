import os
import requests
import zipfile
from tqdm.auto import tqdm as tq


def download_data(url: str, path: str = "data/", verbose: bool = False) -> None:
    """Download file with progressbar.
    # Code taken from: https://gist.github.com/ruxi/5d6803c116ec1130d484a4ab8c00c603
    # __author__  = "github.com/ruxi"
    # __license__ = "MIT"
    Usage:
        download_file('http://web4host.net/5MB.zip')
    """
    if url == "NEED_TO_BE_CREATED":
        raise NotImplementedError

    if not os.path.exists(path):
        os.makedirs(path)
    local_filename = os.path.join(path, url.split("/")[-1])
    r = requests.get(url, stream=True, verify=False)
    file_size = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else 0
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)
    if verbose:
        print(dict(file_size=file_size))
        print(dict(num_bars=num_bars))

    if not os.path.exists(local_filename):
        with open(local_filename, "wb") as fp:
            for chunk in tq(
                r.iter_content(chunk_size=chunk_size),
                total=num_bars,
                unit="KB",
                desc=local_filename,
                leave=True,  # progressbar stays
            ):
                fp.write(chunk)  # type: ignore

    if ".zip" in local_filename:
        if os.path.exists(local_filename):
            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extractall(path)
