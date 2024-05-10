from setuptools import find_packages, setup

VERSION = "0.0.1.dev0"
## TODO: add install_requirements
setup(
    name="quip_sharp",
    version=VERSION,
    url="https://github.com/NSagan271/lplr-q",
    package_dir={"": "."},
    packages=find_packages(".", "."),
    entry_points={},
    python_requires=">=3.8.0"
)

