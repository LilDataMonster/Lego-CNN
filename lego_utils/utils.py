import os
from os import devnull
import tarfile
import requests
from tqdm.auto import tqdm


def extract_tarfile(input_file: str, output_path: str = None):    
    # set output_path to cwd if not defined
    if not output_path:
        output_path = os.getcwd()
    
    # open tar.gz file
    with tarfile.open(name=input_file) as tar:
        # traverse and extract each member
        pbar = tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()))
        for member in pbar:
            pbar.set_description(f"{member.name}")
            tar.extract(member=member, path=output_path)


# download a file from URL
# modified from: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_requests.py
def download_url(url: str, output_path: str = None, output_file: str = None, auto_extract: bool = True, force_download: bool = False):
    filename = url.replace('/', ' ').split()[-1]
    
    # set output_path to cwd if not defined
    if not output_path:
        output_path = os.getcwd()
    
    # make output directory
    os.makedirs(output_path, exist_ok=True)
    
    # if no output_file name defined, use name from url
    if not output_file:
        output_file = filename
    
    full_output_path = os.path.join(output_path, output_file)
    
    # if not forced to download and file already exists, skip
    if not force_download and os.path.exists(full_output_path):
        print(f"'{full_output_path}' already exists. Skipping... Use 'force_download=True' to overwrite.")
        return
    
    # download file chunks
    response = requests.get(url, stream=True)
    with open(full_output_path, "wb") as fout:
        with tqdm(
            # all optional kwargs
            unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
            desc=filename, total=int(response.headers.get('content-length', 0))
        ) as pbar:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)
                pbar.update(len(chunk))
    # following doesn't close the stream so will fail when extracting
    # with tqdm.wrapattr(
    #     open(full_output_path, "wb"), "write",
    #     unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
    #     desc=filename, total=int(response.headers.get('content-length', 0))
    # ) as fout:
    #     for chunk in response.iter_content(chunk_size=4096):
    #         fout.write(chunk)
    
    if auto_extract and (output_file.endswith(".tar.gz") or output_file.endswith(".tgz")):
        extract_tarfile(full_output_path, output_path)