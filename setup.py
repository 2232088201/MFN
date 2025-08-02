#!/usr/bin/env python
import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="mfn",
    version="0.1",
    description="Syntax-Image Fusion Framework for Handwritten Mathematical Expression Recognition",
    author="Mingyu Fan",
    author_email="fanmingyu@dhu.edu.cn",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    packages=find_packages(),
)
