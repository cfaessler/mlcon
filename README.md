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

    $ docker run -it --rm -v $(pwd):/app -p 8000:8000 -p 8888:8888 aikurs-container bash

- pwd is your current working directory on the host machine
- app is the path in the container where the host directory is mapped to

# 3) Run jupyter notebook server within container

    $ jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 4) Run uvicorn app within container
    $ uvicorn main:app --host 0.0.0.0 --port 8000

