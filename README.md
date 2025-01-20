# ShapeGrasp: Zero-Shot Object Manipulation with LLMs through Geometric Decomposition

[Project page](https://shapegrasp.github.io/) | [ShapeGrasp arXiv](https://arxiv.org/abs/2403.18062)

## Abstract

Task-oriented grasping of unfamiliar objects is a necessary skill for robots in dynamic in-home environments. Inspired by the human capability to grasp such objects through intuition about their shape and structure, we present a novel zero-shot task-oriented grasping method leveraging a geometric decomposition of the target object into simple, convex shapes that we represent in a graph structure, including geometric attributes and spatial relationships. Our approach employs minimal essential information—the object's name and the intended task—to facilitate zero-shot task-oriented grasping. We utilize the commonsense reasoning capabilities of large language models to dynamically assign semantic meaning to each decomposed part and subsequently reason over the utility of each part for the intended task. Through extensive experiments on a real-world robotics platform, we demonstrate that our grasping approach's decomposition and reasoning pipeline is capable of selecting the correct part in 92% of the cases and successfully grasping the object in 82% of the tasks we evaluate.

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

- `{obj}_depth.png` (not needed in 2D mode)
    - npy or png file, 1 or 3 channels
- `{obj}_mask.npy`
    - npy or png file, 1 or 3 channels, binary or 0-255
- `{obj}_rgb.png`
    - npy or png file
 
You will need to provide your own OpenAI API key, to be imported from `code/keys.py`.
  
## Running the Demo

The `demo.py` script supports `2d` and `3d` mode. You can specify the mode and the object to process using command-line arguments. You can also specify an optional decomposition threshold. Example:

`python demo.py --mode 2d --obj knife --data_dir data/ --threshold 0.2`

## Citation

In case you find our work useful, consider citing:
```bibtex
@article{Li2024ShapeGraspZT,
  title={ShapeGrasp: Zero-Shot Task-Oriented Grasping with Large Language Models through Geometric Decomposition},
  author={Samuel Li and Sarthak Bhagat and Joseph Campbell and Yaqi Xie and Woojun Kim and Katia P. Sycara and Simon Stepputtis},
  journal={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024},
  pages={10527-10534},
}
```

## License

This project is licensed under the terms of the [MIT License](LICENSE).


