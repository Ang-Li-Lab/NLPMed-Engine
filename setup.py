import importlib.util
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

about_path = Path("nlpmed_engine/about.py")

spec = importlib.util.spec_from_file_location("about", about_path)
if spec is None:
    message = f"Could not find the module specification for the file: {about_path}"
    raise ImportError(message)

about_module = importlib.util.module_from_spec(spec)
if spec.loader is None:
    message = f"Loader is not available for the module specification: {spec}"
    raise ImportError(message)

spec.loader.exec_module(about_module)

with Path("README.md").open(encoding="utf-8") as f:
    long_desc = f.read()


def read_requirements(file_name: str) -> list:
    """Recursively read and return all requirements from a requirements file, resolving any '-r' references."""
    requirements = []
    with Path(file_name).open(encoding="utf-8") as req_file:
        for line in req_file:
            line_stripped = line.strip()
            if line_stripped.startswith("-r"):
                req_path = line_stripped.split(" ")[1]
                if not Path(req_path).is_absolute():
                    req_path = Path(file_name).parent / req_path
                requirements.extend(read_requirements(req_path))
            elif line_stripped and not line_stripped.startswith("#"):
                requirements.append(line_stripped)
    return requirements


setup(
    name=getattr(about_module, "__title__", ""),
    version=getattr(about_module, "__version__", ""),
    author=getattr(about_module, "__author__", ""),
    url=getattr(about_module, "__url__", ""),
    description=getattr(about_module, "__description__", ""),
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license=getattr(about_module, "__license__", ""),
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements("cpu.txt"),
    extras_require={
        "gpu_apple": read_requirements("gpu_apple.txt"),
        "gpu_cuda": read_requirements("gpu_cuda.txt"),
    },
)
