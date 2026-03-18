"""
JARVIS AI OS - Package Setup
"""

import os
from setuptools import setup, find_packages

# README is optional — don't fail the install if the file isn't present yet.
_readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(_readme_path):
    with open(_readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "JARVIS AI Operating System — a production-ready multi-agent AI OS."

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jarvis-ai-os",
    version="1.0.0",
    author="JitenSabharwal",
    description="JARVIS AI Operating System - A production-ready multi-agent AI OS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JitenSabharwal/JARVIS-AI-OS",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "jarvis=jarvis_main:cli_entry",
        ],
    },
)
