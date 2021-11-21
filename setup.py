from setuptools import setup

__version__ = '0.1.0'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='HIDIV',
    version=__version__,
    url='https://github.com/panagiotisanagnostou/HIDIV',
    author='Panagiotis Anagnostou',
    author_email='panagno@uth.gr',
    description='A hierarchical divisive clustering toolbox',
    keywords=['data structure', 'tree', 'tools'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT License",
    packages=["HIDIV"],
    package_dir={'': 'src'},
    project_urls={"Bug Tracker": "https://github.com/panagiotisanagnostou/HIDIV"},
    classifiers=[
        'Development Status :: 1 - Beta',
        'Environment :: Console',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Programming Language :: Python',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    python_requires=">=3.6"
)
