# Zero-Shot Object Manipulation with LLMs through Geometric Decomposition

This project involves symbolic geometric decomposition for LLM scene understanding, particularly in the domain of grasping.

## Installation

Create the conda environment:
`conda env create -f environment.yml`

Install dependencies:

```
conda install -c conda-forge trimesh
conda install -c conda-forge opencv
pip install coacd
pip install openai==0.27.9
```
## Getting Started

The pipeline depends on a single-view RGB image and binary mask, and a depth image for 3D mode. These files should be named as follows and placed in your specified `data_dir`:

- `{obj}_depth` (not needed in 2D mode)
    - npy or png file, 1 or 3 channels
- `{obj}_mask.npy`
    - npy or png file, 1 or 3 channels, binary or 0-255
- `{obj}_rgb.png`
    - npy or png file
  
## Running the Demo

The `demo.py` script supports `2d` and `3d` mode. You can specify the mode and the object to process using command-line arguments. You can also specify an optional decomposition threshold. Example:

`python demo.py --mode 3d --obj knife --data_dir data/ --threshold 0.2`

## Running the Demo

In case you find our work useful, consider citing:




