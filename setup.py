# SPDX-FileCopyrightText: Copyright (C) 2025 Omid Jafari <omidjafari.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
    """
    Recursively read and return all requirements from a requirements file, resolving any '-r' references.

    Args:
        file_name: Path to the base requirements file.

    Returns:
        List of resolved requirement strings.
    """
    requirements = []
    with Path(file_name).open(encoding="utf-8") as req_file:
        for line in req_file:
            line_stripped = line.strip()
            if line_stripped.startswith("-r"):
                req_path = line_stripped.split(" ")[1]
                if not Path(req_path).is_absolute():
                    req_path = str(Path(file_name).parent / req_path)
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
        "License :: OSI Approved :: GNU Affero General Public License v3 (AGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements("cpu.txt"),
    extras_require={
        "gpu_apple": read_requirements("gpu_apple.txt"),
        "gpu_cuda": read_requirements("gpu_cuda.txt"),
    },
)
