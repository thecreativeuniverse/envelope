from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text()

setup(
    name="koipond",
    version="0.0.2",
    description="Final year project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.cs.bham.ac.uk/projects-2023-24/irs084",
    author="Isabella Shrimpton",
    author_email="irs084@student.bham.ac.uk",
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    install_requires=[line.strip() for line in open("requirements.txt").readlines()],
    python_requires=">=3.8",
)
