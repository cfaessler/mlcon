# 1 Build the container
    $ docker build -t aikurs-container .

# 2 Run the container and open interactive bash

    $ docker run -it --rm -v $(pwd):/app -p 8000:8000 -p 8888:8888 aikurs-container bash

- pwd is your current working host directory
- app is the path in the container where the host directory is mapped to

# 3 Run jupyter notebook server within container

    $ jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 4 Run uvicorn app within container
    $ uvicorn main:app --host 0.0.0.0 --port 8000

