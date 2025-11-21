# One Walk is All You Need: Data-Efficient 3D RF Scene Reconstruction with Human Movements
## Keywords:
3DGS; Radiance Field Reconstruction; Human Movements

## Abstract: 
Reconstructing 3D Radiance Field (RF) scenes through opaque obstacles is a long-standing goal, yet it is fundamentally constrained by a laborious data acquisition process requiring thousands of static measurements, which treats human motion as noise to be filtered. This work introduces a new paradigm with a core objective: to perform fast, data-efficient, and high-fidelity RF reconstruction of occluded 3D static scenes, using only a single, brief human walk. We argue that this unstructured motion is not noise, but is in fact an information-rich signal available for reconstruction. To achieve this, we design a factorization framework based on composite 3D Gaussian Splatting (3DGS) that learns to model the dynamic effects of human motion from the persistent static scene geometry within a raw RF stream. Trained on just a single 60-second casual walk, our model reconstructs the full static scene with a Structural Similarity Index (SSIM) of 0.96, remarkably outperforming heavily-sampled state-of-the-art (SOTA) by 12%. By transforming the human movements into its valuable signals, our method eliminates the data acquisition bottleneck and paves the way for on-the-fly 3D RF mapping of unseen environments.


## Dataset 
The dataset structure of a scene in the path location should be as follows:
```
<Static dataset location>
|---spectrum
|   |---<spectrum 00001>
|   |---<spectrum 00002>
|   |---...
|---test_index.txt
|---train_index.txt
|---rx_pos.csv
```
```
<Dynamic dataset location>
|---spectrums
|   |---<spectrum 00001>
|   |---<spectrum 00002>
|   |---...
|---test_index.txt
|---train_index.txt
|---val_index.txt
|---spectrum_info.csv
```
**Note**

Each dataset should include a dynamic dataset and a static dataset for each scene((```./data/static_dataset``` and ```./data/dynamic_dataset```)). The performance of the model may be diminished due to the absence of a complete dataset. The complete dataset of all 3 scenes will be available to the public upon acceptance of this paper.


## Setup
Create and activate conda environment
```python
conda env create --file environment.yml
conda activate one_walk_env
```
Install submodules
```python
cd submodules
pip install ./diff-gaussian-rasterization
pip install ./fused-ssim
pip install ./simple-knn
```


## Training & Testing
A background baseline model is trained on a static dataset from an unoccupied scene. The trained model is then saved and utilized as the background reference for subsequent training stages.
```python
python train_and_save.py
```
Subsequently, the trained model should be placed into the ./checkpoints/ directory to proceed with the second stage of training in dynamic scenes.
```python
python man_train.py
```
The model will be evaluated at epochs of the ```--test_iterations``` argument.
The path of datasets can be changed in python file ```./scene/__init__.py```
The testing results will be saved in ```./logs``` folder and the model checkpoints will be saved in ```./output``` folder.

