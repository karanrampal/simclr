![Attribute Prediction](https://github.com/hm-group/attribute-prediction/actions/workflows/main.yaml/badge.svg)

# Attribute Prediction
Detect attributes of the dataset. This dataset consists of images by designers and our goal is to find attributes of these images.

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
dist/
    AttributePrediction-0.0.1-py3-none-any.whl
    AttributePrediction-0.0.1.tar.gz
notebooks/
    create_dataset.ipynb
    training_sdk.ipynb
    visualize_dataset.ipynb
scripts/
    run_training.sh
    run_create_dataset_dataflow.sh
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
    create_castors.py
    create_dataset_dataflow.py
    create_dataset.py
tests/
    __init__.py
    test_metrics.yp
.gitignore
environments.yaml
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
Then create the dataset by running the following command (this needs to be done only once, and can be done at anytime after cloning this repo),
```
python create_dataset.py -r <path to root dir>
```
Then to start training on a single node with multiple gpu's we can do the following,
```
python -m trainer.train --args1 --args2
```

## Requirements
I used Anaconda with python3, but used pip to install the libraries so that they worked with my multi GPU compute environment in GCP

```
make install
conda activate attrpred-env
```