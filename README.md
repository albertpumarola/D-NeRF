<img src='https://www.albertpumarola.com/images/2021/D-NeRF/teaser2.gif' align="right" width=400>

# D-NeRF: Neural Radiance Fields for Dynamic Scenes
### [[Project]](https://www.albertpumarola.com/research/D-NeRF/index.html)[ [Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Pumarola_D-NeRF_Neural_Radiance_Fields_for_Dynamic_Scenes_CVPR_2021_paper.pdf) 

[D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html) is a method for synthesizing novel views, at an arbitrary point in time, of dynamic scenes with complex non-rigid geometries. We optimize an underlying deformable volumetric function from a sparse set of input monocular views without the need of ground-truth geometry nor multi-view images.

This project is an extension of [NeRF](http://www.matthewtancik.com/nerf) enabling it to model dynmaic scenes. The code heavily relays on [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). 

![D-NeRF](https://www.albertpumarola.com/images/2021/D-NeRF/model.png)

## Installation
```
git clone https://github.com/albertpumarola/D-NeRF.git
cd D-NeRF
conda create -n dnerf python=3.6
conda activate dnerf
pip install -r requirements.txt
cd torchsearchsorted
pip install .
cd ..
```

## Download Pre-trained Weights
 You can download the pre-trained models from [drive](https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing) or [dropbox](https://www.dropbox.com/s/25sveotbx2x7wap/logs.zip?dl=0). Unzip the downloaded data to the project root dir in order to test it later. See the following directory structure for an example:
```
├── logs 
│   ├── mutant
│   ├── standup 
│   ├── ...
```

## Download Dataset
 You can download the datasets from [drive](https://drive.google.com/file/d/19Na95wk0uikquivC7uKWVqllmTx-mBHt/view?usp=sharing) or [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0). Unzip the downloaded data to the project root dir in order to train. See the following directory structure for an example:
```
├── data 
│   ├── mutant
│   ├── standup 
│   ├── ...
```

## Demo
We provide simple jupyter notebooks to explore the model. To use them first download the pre-trained weights and dataset.

| Description      | Jupyter Notebook |
| ----------- | ----------- |
| Synthesize novel views at an arbitrary point in time. | render.ipynb|
| Reconstruct mesh at an arbitrary point in time. | reconstruct.ipynb|
| Quantitatively evaluate trained model. | metrics.ipynb|

## Test
First download pre-trained weights and dataset. Then, 
```
python run_dnerf.py --config configs/mutant.txt --render_only --render_test
```
This command will run the `mutant` experiment. When finished, results are saved to `./logs/mutant/renderonly_test_799999` To quantitatively evaluate model run `metrics.ipynb` notebook

## Train
First download the dataset. Then,
```
conda activate dnerf
export PYTHONPATH='path/to/D-NeRF'
export CUDA_VISIBLE_DEVICES=0
python run_dnerf.py --config configs/mutant.txt
```

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@article{pumarola2020d,
  title={D-NeRF: Neural Radiance Fields for Dynamic Scenes},
  author={Pumarola, Albert and Corona, Enric and Pons-Moll, Gerard and Moreno-Noguer, Francesc},
  journal={arXiv preprint arXiv:2011.13961},
  year={2020}
}
```
