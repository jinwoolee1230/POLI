<div align="center">
    <h1>ROBIN</h1>
    <a href="https://github.com/MIT-SPARK/ROBIN"><img src="https://img.shields.io/badge/-C++-blue?logo=cplusplus" /></a>
    <a href="https://github.com/MIT-SPARK/ROBIN"><img src="https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/MIT-SPARK/ROBIN"><img src="https://img.shields.io/badge/MATLAB-Supported-g?logo=mathworks" /></a>
    <a href="https://github.com/MIT-SPARK/ROBIN"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://github.com/MIT-SPARK/ROBIN"><img src="https://custom-icon-badges.demolab.com/badge/Windows-0078D6?logo=windows11&logoColor=white" /></a>
    <a href="https://arxiv.org/abs/2011.03659"><img src="https://img.shields.io/badge/arXiv-b33737?logo=arXiv" /></a>
    <a href="https://ieeexplore.ieee.org/document/9562007"><img src="https://img.shields.io/badge/DOI-10.1109/ICRA48506.2021.9562007-004088.svg"/>
  <br />
  <br />
  <p align="center"><img src="https://github.com/user-attachments/assets/072cac24-657b-46b1-a808-354d9f3abc45" alt="ROBIN" width="80%"/></p>
  <p><strong><em>ROBIN is a library for outlier rejection based on compatibility graphs.</em></strong></p>
</div>

--- 

# :gear: Build & Installation

## :package: Dependencies 

ROBIN has the following dependencies:
1. OpenMP
2. Eigen3

Thus, run the following command:

```bash
sudo apt-get install gcc g++ build-essential libeigen3-dev cmake python3-pip python3-dev git ninja-build -y
```

## [![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#) C++ Installation 

Run the following commands to build the library using CMake (inside the repository root directory):

```bash
mkdir build && cd build
cmake .. && make
sudo make install
```

The following CMake options are provided:

```
BUILD_DOCS: Build documentation. Default: OFF
BUILD_TESTS: Enable testing with ctest. Default: ON
BUILD_MATLAB_BINDINGS: Build MATLAB bindings. Default: OFF
USE_ASAN: Enable address sanitizer. Default: OFF
ENABLE_DIAGNOSTIC_PRINT: Enable printing of diagnostic messages. Default: OFF
```

## [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#) Python Installation 

It's simple! To install Python bindings, we need basic packages as follows:

```
pip3 install --upgrade pip setuptools wheel scikit-build-core ninja cmake build
```

And then, just run in out-of-the-box (the `--verbose` option is only for tracking purposes):

```bash
pip3 install "git+https://github.com/MIT-SPARK/ROBIN.git#subdirectory=python" --verbose
```

Using this repository, you can run the following command:

```bash
pip3 install -e python/
```

Please refer to `python/example.py` for usage instructions.

--- 

# Third-party Data

Some test data are from the [Network Repository](http://networkrepository.com/index.php). 
For more information, please refer to: 

Rossi, Ryan, and Nesreen Ahmed. "The network data repository with interactive graph analytics and visualization." Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.

# Third-party Code
- Parallel Maximum Clique (PMC) Library: 
    - License: GNU General Public License (https://github.com/ryanrossi/pmc/blob/master/LICENSE.md) 
- pybind11: https://github.com/pybind/pybind11
    - License: BSD-style (https://github.com/pybind/pybind11/blob/master/LICENSE)
- Catch2: https://github.com/catchorg/Catch2
    - License: Boost (https://github.com/catchorg/Catch2/blob/devel/LICENSE.txt)

# Known Issues

To fix missing CXXABI errors in MATLAB:

```
export LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/7/libstdc++.so
```

---

# Citations

If you find this library helpful or use it in your projects, please cite:
```bibtex
@InProceedings{Shi21icra-robin,
	title={{ROBIN:} a Graph-Theoretic Approach to Reject Outliers in Robust Estimation using Invariants},
	author={J. Shi and H. Yang and L. Carlone},
	booktitle={IEEE Intl. Conf. on Robotics and Automation (ICRA)},
	note = {arXiv preprint: 2011.03659},
	pdf={https://arxiv.org/pdf/2011.03659.pdf},
	year={2021}
}
```
and
```bibtex
@article{Shi22arxiv-PACE,
  author = {J. Shi and H. Yang and L. Carlone},
  title = {Optimal and Robust Category-level Perception: Object Pose and Shape Estimation from {2D and 3D} Semantic Keypoints},
  journal = {arXiv preprint: 2206.12498},
  pdf = {https://arxiv.org/pdf/2206.12498.pdf},
  Year = {2022}
}
```

If you are interested in more works from us, please visit our lab page [here](http://web.mit.edu/sparklab/).
