# Pogema Baselines

---
**Important Note:**

**The Baselines repository will not be receiving any future updates.** The main baselines have been transferred to the [POGEMA-Benchmark](https://github.com/Cognitive-AI-Systems/pogema-benchmark) repository, which includes more algorithms and tools. 

---

This repository contains a set of baselines for the [Pogema](https://github.com/AIRI-Institute/pogema) environment.

## Multi-Agent RL Baselines

* **QMIX**
* **VDN**
* **IQL** 

Implementations based on the [oxwhirl/pymarl](https://github.com/oxwhirl/pymarl)

## Large-Scale Experiments

* **Asynchronous PPO** based on [alex-petrenko/sample-factory](https://github.com/alex-petrenko/sample-factory)

## Installation

Just install all dependencies using:

```bash
pip install -r docker/requirements.txt
```

## Training Example

Run ```main.py``` with one of the configs from the ``configs`` folder:

```bash
python main.py --config configs/8x8.yaml
```
Detailed instructions are available in the [APPO](https://github.com/Tviskaron/pogema-baselines/tree/main/appo) and [PyMARL](https://github.com/Tviskaron/pogema-baselines/tree/main/pymarl) readme files.

We thank **PyMARL** and **SampleFactory** contributors for their implementations.
