import os
import shutil
from functools import partial
from typing import Callable

import click
import torch

from src.blur import blur_faces_and_save, download_models_if_missing

supported_img_extensions: [str] = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
cli_output_style: {str, str} = {"bg": "blue", "fg": "white"}


def copy_dir(src_dir: str, dst_dir: str, copy_file: Callable[[str, str], None]) -> None:
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        relative_path = os.path.relpath(root, src_dir)
        current_dst_dir = os.path.join(dst_dir, relative_path)
        if not os.path.exists(current_dst_dir):
            os.makedirs(current_dst_dir)

        for file in files:
            copy_file(os.path.join(root, file), os.path.join(current_dst_dir, file))


def process_file(src_file: str, dst_file: str, process_img: Callable[[str, str], None], verbose: float = False) -> None:
    is_img = any(src_file.lower().endswith(ext) for ext in supported_img_extensions)
    if is_img:
        if verbose:
            click.secho(f"PROCESS {src_file}")

        process_img(src_file, dst_file)

        if verbose:
            click.secho(f"PROCESSED {dst_file}", fg="green")
    else:
        shutil.copyfile(src_file, dst_file)


@click.command()
@click.argument("src_dir", type=click.Path(exists=True))
@click.argument("dst_dir", type=click.Path())
@click.option("--max-age", type=int, default=None, help="Upper limit for face blurring.")
@click.option("--device", type=str, default="cuda", help="Device on which tensors will be allocated.")
@click.option("--verbose", is_flag=True, help="Enable verbose mode.")
def cli(src_dir, dst_dir, max_age=None, device="cuda", verbose=False):
    """
    Copy files from the source to the destination directory, maintaining the structure, and blur faces in image files.
    """
    if verbose:
        click.secho(f"COPY from {src_dir} to {dst_dir} and BLUR faces in image files", **cli_output_style)
        if max_age:
            click.secho(f"BLUR faces only if estimated age is less than {max_age} years", **cli_output_style)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    download_models_if_missing()
    blur_faces_max_age = partial(blur_faces_and_save, device=device, max_age=max_age, verbose=verbose)
    copy_dir(src_dir, dst_dir, partial(process_file, process_img=blur_faces_max_age, verbose=verbose))

    if verbose:
        click.secho(f"COPIED from {src_dir} to {dst_dir}", **cli_output_style)


if __name__ == "__main__":
    cli()
