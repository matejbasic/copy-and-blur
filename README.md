# Copy & Blur

Python script for copying directory and all files and subdirectories while blurring faces in
the found image files.

Using [MiVOLO](https://github.com/WildChlamydia/MiVOLO) for face detection and age estimation.

```shell
pip install git+https://github.com/matejbasic/copy-and-blur
```

## Usage

```shell
copy-and-blur ./input/ ./output/ --max-age=20 --verbose --device="cuda:0"
```
