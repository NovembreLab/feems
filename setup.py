#!/usr/bin/env python

from setuptools import setup

version = "1.0.0"

required = open("requirements.txt").read().split("\n")
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="feems",
    version=version,
    description="Fast Estimation of Effective Migration Surfaces (feems)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="[jhmarcus, haywse]",
    author_email="[jhmarcus@uchicago.edu, haywse@gmail.com]",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/jhmarcus/feems",
    packages=["feems"],
    install_requires=required,
    include_package_data=True,
    package_data={
        "": [
            "data/grid_250.shp",
            "data/grid_250.shx",
            "data/grid_100.shp",
            "data/grid_100.shx",
            "data/wolvesadmix.bed",
            "data/wolvesadmix.coord",
            "data/wolvesadmix.fam",
            "data/wolvesadmix.bim",
            "data/wolvesadmix.outer",
            "data/wolvesadmix.diffs",
        ]
    },
    license="MIT",
)
