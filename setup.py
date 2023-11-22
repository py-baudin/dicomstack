from setuptools import setup, find_packages

with open("dicomstack/version.py") as version_file:
    exec(version_file.read())

setup(
    name="dicomstack",
    version=__version__,
    packages=find_packages(),
    install_requires=["pydicom>=2.4"],
    entry_points={"console_scripts": ["dicom = dicomstack.cli:cli"]},
    description="A pydicom wrapper for simple loading and handling of DICOM stacks ",
)
