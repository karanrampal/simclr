SHELL := /bin/bash
CONDAENV := environment.yaml
PROJ := smle-attribution-d237
LOC := europe-west4
REG := smle-docker-registry
IMG := simclr:v0.1

install: $(CONDAENV)
	conda env create -f $(CONDAENV)

install_ci: requirements.txt
	pip install --upgrade pip &&\
		pip install -r requirements.txt

build:
	python -m build

docker_bp: Dockerfile
	docker build -f Dockerfile -t $(LOC)-docker.pkg.dev/$(PROJ)/$(REG)/$(IMG) ./
	docker push $(LOC)-docker.pkg.dev/$(PROJ)/$(REG)/$(IMG)

test:
	pytest -vv --cov --disable-warnings

format:
	black src tests
	isort src tests
	mypy src tests

lint:
	pylint -j 6 src tests

clean:
	rm -r .coverage .mypy_cache .pytest_cache .ipynb_checkpoints dist

all: install lint test

.PHONY: lint format clean all