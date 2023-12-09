from pathlib import Path

from setuptools import setup, find_packages

import pkg_resources as pkg

requirements_path = Path("requirements.txt")
requirements = [
    f"{x.name + ' @ ' + x.url if x.url else x.name}{x.specifier}"
    for x in pkg.parse_requirements(requirements_path.read_text())
]

setup(
    name="copy-and-blur",
    url="https://github.com/matejbasic/copy-and-blur",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": "copy-and-blur=src.main:cli"
    },
)
