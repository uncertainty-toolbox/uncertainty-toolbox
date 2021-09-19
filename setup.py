import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements/requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="uncertainty_toolbox",
    version="0.1.0",
    author="Willie Neiswanger, Youngseog Chung, Ian Char, Han Guo",
    author_email="willie.neiswanger@gmail.com",
    description=("A python toolbox for predictive uncertainty quantification,"
                 " calibration, metrics, and visualization."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uncertainty-toolbox/uncertainty-toolbox",
    packages=setuptools.find_packages(),
    package_data={"uncertainty_toolbox": ["matplotlibrc"]},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
