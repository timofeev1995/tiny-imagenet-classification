# tiny-imagenet-classification

## Installation
1. Make sure thaat you are in clean enviroment (conda/venv/etc)
2. `pip install -r requirements.txt`
3. Make sure that you've got torch which is compatible to your CUDA, if you are going to use GPU.

## Configuration
1. Go to ./configs
2. Use `default_config.yaml` and set your own. Models from torchvision you can use: resnet-family, vgg-family, squeezenet-family, densenet-family, mnasnet-family.
3. If you are going to use GPU-training, set their indices as an yaml-array: `[0, 1]`.

## Training
`python train.py --config=<your_config> --experiments_dir=../<yout_experiments_folder> --experiment_name=<your_experiment_name>`

## Evaluation
Example is in `notebooks/evaluation.ipynb`

## Pretrained models
You can download experiments with resnet18/50, vgg19, mnasnet1.0 here: https://drive.google.com/file/d/1CAseZTl54txzH6TWL1rUf6NAAiKHUhpB/view?usp=sharing. 
You can untar them and use as shown in `notebooks/evaluation.ipynb`. 