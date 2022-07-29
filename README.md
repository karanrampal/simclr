![SimCLR](https://github.com/karanrampal/simclr/actions/workflows/main.yaml/badge.svg)

# SimCLR
Self-Supervised Contrastive Learning with SimCLR. My implementation of the paper by [Ting Chen et al.](https://arxiv.org/abs/2002.05709).

Multi GPU and multi nodes are supported using DistributedDataParallel. GCP is used for training and development.

Apache Beam is used for dataset creation and DataFlow runner is used as backend.

## Directory structure
Structure of the project
```
.github/
    workflows/
	    main.yaml
configs/
    params.yml
notebooks/
    training_sdk.ipynb
scripts/
    run_training.sh
src/
    dataloader/
        __init__.py
        data_loader.py
    model/
        __init__.py
        net.py
    trainer/
        __init__.py
        train.py
        evaluate.py
    utils/
        __init__.py
        vis_utils.py
        utils.py
    search_hyperparams.py
tests/
    __init__.py
    test_metrics.yp
.gitignore
environments.yaml
LICENSE
Makefile
mypy.ini
.pylintrc
pyproject.toml
README.md
requirements.txt
setup.cfg
```

## Usage
First clone the project as follows,
```
git clone <url> <newprojname>
cd <newprojname>
```
Then build the project by using the following command, (assuming build is already installed in your virtual environment, if not then activate your virtual environment and use `conda install build`)
```
make build
```
Next, install the build wheel file as follows,
```
pip install <path to wheel file>
```
Then to start training on a single node with multiple gpu's we can do the following,
```
python -m trainer.train --args1 --args2
```
For multi-node training, run the following,
```
torchrun --nnodes {num_total_nodes} --nproc_per_node {num_total_gpus} --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT train.py --args1 --args2
```
For hyper-parameter search you can use the following script,
```
python src/search_hyperparams.py --args1 --args2
```

## Requirements
I used Anaconda with python3, but used pip to install the libraries so that they worked with my multi GPU compute environment in GCP

```
make install
conda activate simclr-env
```