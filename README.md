# Intro
To get everything setup very easily we have created a docker container that has all the libraries installed you need for the labs.
The container also contains local jupyter notebook server, so you can run all the notebooks locally.

## Local setup (without Docker)

Homebrew Python blocks global `pip install`. Use the project virtual environment instead:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

In Cursor/VS Code, select the kernel **"ML Con (.venv)"** or the interpreter at `.venv/bin/python`.

# 1) Build the container
    $ docker build -t aikurs-container .

# 2) Run the container and open interactive bash

    $ docker run -it --rm -v $(pwd):/app -p 8000:8000 -p 8888:8888 -p 5050:5050 -p 5001:5001 -p 8080:8080 aikurs-container bash

- pwd is your current working directory on the host machine
- app is the path in the container where the host directory is mapped to

# 3) Run jupyter notebook server within container

    $ cd /app
    $ jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

Open the URL Jupyter prints in your **host** browser (port 8888 is mapped). For the MLOps lab, MLflow UI, model serving, and Evidently monitoring UI use `http://localhost:5050`, `http://localhost:5001`, and `http://localhost:8080` on the host as well.

# 4) Run uvicorn app within container
    $ uvicorn main:app --host 0.0.0.0 --port 8000

