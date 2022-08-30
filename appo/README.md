# Asynchronous PPO Baseline for Pogema Environment

## Installation
Install all dependencies using:
```bash
pip install -r docker/requirements.txt
```

## Training APPO
Run ```main.py``` with ```.yaml``` config:
```bash
python main.py --config_path=configs/8x8.yaml
```

## Docker 
We use [crafting](https://pypi.org/project/crafting/) to automate experiments. 
You can find an example of running such a pipeline in ```run.yaml``` file. 
You need to have installed Docker, Nvidia drivers (optionally, for GPU acceleration), and the crafting package. 

The crafting package is available in PyPI:
```bash
pip install crafting
```


To build the image run the command presented below, from the ```docker``` folder:
```bash
sh build.sh
```

You can specify configuration using command field of ```run.yaml``` file, for example ```8x8.yaml``` config experiments:
```yaml
command: 'python main.py --config_path="configs/8x8.yaml"'
```

Finally, to run an experiment just call crafting with that config file:
```
crafting run.yaml
```

For W&B integration you need to set ``WANDB_API_KEY`` environment variable on a host machine (a better way), or set it in ``run.yaml`` file:
```yaml
environment:
- "WANDB_API_KEY=<YOUR KEY>"
```
