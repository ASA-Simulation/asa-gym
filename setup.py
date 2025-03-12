"""Sets up the project."""

import pathlib


from setuptools import find_packages, setup

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the asagym current version."""
    path = CWD / "asagym" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


def get_description():
    """Gets the description from README."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return long_description


setup(
    name="asagym",
    version=get_version(),
    description="Gymnasium environments for RL training using ASA as a simulator",
    long_description=get_description(),
    maintainer="Adrisson Samersla",
    maintainer_email="adrissonsamersla@proton.me",
    packages=find_packages(include=["asagym"]),
    package_data={"asagym": ["asagym/assets/*"]},
    install_requires=[
        "pyzmq==25.0.2",
        "grpcio-tools==1.53.0",
        "gymnasium==0.28.1",
        "pygame==2.5.1",
        "numpy==1.26.4",
    ],
    include_package_data=True,
)
