import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

__version__ = "0.4.2"

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
        "numpy",
        "treelib>=1.6",
        "scipy",
        "scikit-learn",
        "statsmodels>=0.13",
        "kdepy",
        "matplotlib",
        "plotly",
        "dash>=2.0",
    ],
    project_urls={
        "Documentation": "https://hipart.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/panagiotisanagnostou/HiPart/",
        "Bug Tracker": "https://github.com/panagiotisanagnostou/HiPart/issues",
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
