from setuptools import setup, find_packages

with open("dicomstack/version.py") as version_file:
    exec(version_file.read())

setup(
    name="dicomstack",
    version=__version__,
    packages=find_packages(),
    install_requires=["pydicom>=2.0"],
    entry_points={"console_scripts": ["dicom = dicomstack.cli:cli"]},
    description="Convenient wrapper on pydicom for easy searching DICOM files",
)
