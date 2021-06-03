# ml-vis

Visualization tools for understanding ML image classifiers

## Conda Environment

To create the `ml-vis` environment, run:
```
conda env create -f conda/environment.yml
```

To enable GPU acceleration, instead run:
```
conda env create -f conda/environment_gpu.yml
```

## Data

Sample input data, models, and predictions can be found in [this Google Drive folder](https://drive.google.com/drive/folders/1-1lyEsSeLfxWBS7Ju8NnA8ES09rRWipN?usp=sharing). Each `.zip` file should be extracted to the top-level `ml-vis` directory.

## Systemd Service

The `service` subdirectory contains an example [systemd](https://freedesktop.org/wiki/Software/systemd/) configuration for `vis_scripts/interactive.py`. To set up the service, first open `service/ml-vis.service` and modify `ExecStart` and `WorkingDirectory` to match your host configuration. Then run
```
cd service
./setup.sh
sudo systemctl start ml-vis.service
```

## Docker

A Docker image for the interactive visualization tool can be downloaded from https://hub.docker.com/repository/docker/mattdutson/ml-vis.

After downloading, run the image using the command
```
docker run -p <PORT>:5006 mattdutson/ml-vis --allow-websocket-origin <HOST>:<PORT>
```
Where `<HOST>` and `<PORT>` are replaced by the hostname and port from which the tool will be accessed (`<PORT>` will likely be 80 for HTTP).

## Code Style

Unless otherwise specified, follow the [PEP8](https://www.python.org/dev/peps/pep-0008) conventions.

Use a line limit of 79 characters.
