# Visual Odometry

-----

## Table of Contents

- [Prerequisites](#prerequisites)
- [Embedded Installation](#embedded-installation)
- [Developer Installation](#developer-installation)
- [Usage](#usage)

## Prerequisites

First install Python 3.12 or greater. Then create a Python virtual environment and update pip if necessary. All 
dependencies will be installed to the virtual environment and will not affect your system install. If needed you can 
remake the virtual environment by deleting the `venv` folder.

```console
python3.12 -m venv venv
. venv/bin/activate
pip install -U pip
```

You will need to activate the virtual environment in every new terminal.

```console
. venv/bin/activate
```

## Embedded Installation

You can install the package directly from the GitHub repository. Installing from ssh is recommended but you can use the
https url instead if needed.

```console
pip install "git+ssh://git@github.com/schebell-q/CS5330-Project.git"
```

or

```console
pip install "git+https://github.com/schebell-q/CS5330-Project.git"
```

## Developer Installation

Clone this repository and then use a developer mode install (editable install). Doing this allows pip to use the source
code directly and you can edit it without reinstalling.

```console
git clone git@github.com:schebell-q/CS5330-Project.git project
cd project
python3.12 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -e .
```

# Usage

example call from ../project:~$
python -m visual_odometry.main -f orb3 -v -o ~/Documents/School/NEU_CS5330/final_project/project/output/orb-trial_01 ~/Documents/School/NEU_CS5330/final_project/data/lab/lab_01_frames_dogleg1

