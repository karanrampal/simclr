FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest

WORKDIR /simclr

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./src .

ENTRYPOINT ["python", "-m", "trainer.train"]