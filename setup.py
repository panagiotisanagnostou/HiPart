import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

__version__ = "0.2.0"

setuptools.setup(
    name="HiPart",
    version=__version__,
    url="https://github.com/panagiotisanagnostou/HiPart",
    author="Panagiotis Anagnostou",
    author_email="panagno@uth.gr",
    description="A hierarchical divisive clustering toolbox",
    keywords=["data structure", "tree", "tools"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT License",
    package_dir={"": "src"},
    packages=["HiPart"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "treelib>=1.6",
        "scipy==1.7",
        "statsmodels>=0.13",
        "kdepy",
        "matplotlib==3.5",
        "plotly",
        "dash>=2.0",
        "numpy==1.21",
        "scikit-learn==1.0",
        "pandas==1.3",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/panagiotisanagnostou/HiPart",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: Web Environment",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Documentation :: Sphinx",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
