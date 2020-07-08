import os
import setuptools

# read the contents of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="peca",
    version="0.1a0",
    py_modules=["peca"],

    install_requires=["numpy", "scipy", "numba"],

    author="Erik Scharwaechter",
    author_email="scharwaechter@bit.uni-bonn.de",
    description="Peak Event Coincidence Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diozaka/pECA",
    license="MIT",
    keywords="event series, time series, statistical association, hypothesis test",
    platforms="OS Independent",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
    ]
)
