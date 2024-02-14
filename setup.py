"""Setup file for qc_syn package."""

from setuptools import find_packages, setup

setup(
    name="qc_syn",
    version="0.1.0",
    author="Michael Kölle",
    author_email="michael.koelle@ifi.lmu.de",
    description="A quantum circuit synthesis environment for reinforcement learning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/michaelkoelle/rl-qc-syn",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt", encoding="utf-8").readlines()
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
