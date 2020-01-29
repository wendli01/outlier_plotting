import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="outlier_plotting",
    version="0.2.0",
    author="Lorenz Wendlinger",
    author_email="lorenz.wen@gmx.com",
    description="More graceful handling of outliers in plots",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wendli01/outlier_plotting",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['seaborn>=0.9']
)
