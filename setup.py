from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agri-harvest",
    version="2.0.0",
    author="AGRI-HARVEST Team",
    author_email="team@agri-harvest.cm",
    description="Plateforme IA pour agriculture tropicale au Cameroun",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agri-harvest-cm/agri-harvest",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "pytest-asyncio", "black", "flake8", "mypy"],
        "docs": ["sphinx", "myst-parser"],
    },
    entry_points={
        "console_scripts": [
            "agri-harvest=agri_harvest.scripts.cli:main",
        ],
    },
)
