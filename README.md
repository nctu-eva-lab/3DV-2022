# 3DV-2022 Assignment: Single View to 3D

Goals: In this assignment, you will explore the types of loss and decoder functions for regressing to voxels, point clouds, and ~~mesh~~ representation from single view RGB input.

This repo just serve as a template, feel free to adjust most of the code if needed.

## Folder Structure

```bash
├── configs    # Where Hydra reads your configurations.
│   └── config.yml
├── data
│   ├── chair_img_pc_voxel_mesh
│   └── chair_img_pc_voxel_mesh.zip
├── notebooks    # Where your .ipynb files placed.
│   ├── scene.py
│   └── synth.ipynb    # Synthesize datas from ShapeNet mesh.
├── src    # Where your modules placed.
│   ├── dataset.py    # Define your custom PyTorch dataset.
│   ├── losses.py    # Define your custom loss functions.
│   └── model.py    # Define your model structure.
├── README.md
├── requirements.yml    # Create your conda env with this file.
├── eval.py    # Evaluate your model.
└── train.py    # Train your model.
```

## Env Setup

### Clone the template

Click the green `Use this template` button to fork this repo to your github and change it to private repo.

Or `git clone git@github.com:nctu-eva-lab/3DV-2022.git` to only have a local copy.

### Conda Env

Create a conda env:

```bash
conda env create --file requirements.yml
```

### Download dataset

Download dataset from [here](https://drive.google.com/drive/folders/1Me0iJDCQqiF8-I7LTOeasNBs29NzMl1t)

Or use **gdown**[^1] to download the google drive files with command line **in your conda env** you just created.

```bash
conda activate py3d
gdown 1o8QZtd7bBdKVo-XDT3iRHatXpSaxtpRB
```

After downloaded, put the zip file under `data/` and unzip it.

[^1]: [gdwon repo](https://github.com/wkentaro/gdown)

## Exploring loss functions

This section will involve defining a loss function, for fitting voxels, point clouds ~~and meshes~~.

### Fitting a voxel grid

In this subsection, we will define binary cross entropy loss that can help us **fit a 3D binary voxel grid**. Define the loss functions in `src/losses.py` file. For this you can use the pre-defined losses in pytorch library.

### Fitting a point cloud

In this subsection, we will define chamfer loss that can help us **fit a 3D point cloud**. Define the loss functions here in `src/losses.py` file. We encourage you to write your own code for this and not use any pytorch3d[^2] utilities, but you can still use it if you have no idea how to do it.

[^2]: [PyTor3D repo](https://github.com/facebookresearch/pytorch3d/tree/main/pytorch3d)

## Reconstructing 3D from single view

This section will involve training a single view to 3D pipeline for voxels, point clouds ~~and meshes~~.

### Image to voxel grid

In this subsection, we will define a neural network to decode binary voxel grids. Define the decoder network in `src/model.py` file.

Run the file `python train.py type='voxel'`, to train single view to voxel grid pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval.py` file using: `python eval.py type='voxel'`.

You need to add the respective visualization code in `eval.py` to show both the predicted voxel and the mesh side by side.

### Image to point cloud

In this subsection, we will define a neural network to decode point clouds. Similar as above, define the decoder network in `src/model.py` file.

Run the file `python train.py type='point'`, to train single view to pointcloud pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth point cloud and predicted point cloud in `eval.py` file using: `python eval.py type='point'`.

## Analyse effects of hyperparms variations

Analyse the results, by varying an hyperparameter of your choice. For example `n_points` or `vision model` or `lr` etc. Try to be unique and conclusive in your analysis.

## Issues for this template

Feel free to file an issue if you think this template has a major flaw.

[Issues](https://github.com/nctu-eva-lab/3DV-2022/issues)

## Discussions with others

Feel free to start discussion with your classmates about how you can implement your work better!

[Discussions](https://github.com/nctu-eva-lab/3DV-2022/discussions/categories/general)

## Other reference

1. [PyTorch API](https://pytorch.org/docs/stable/index.html)
2. [PyTorch3d API](https://pytorch3d.readthedocs.io/en/latest/)
3. [Trimesh API](https://trimsh.org/trimesh.html)

※ Noted: The API docs might still not be complete, it's essential to trace the source code in their own project github repo.
