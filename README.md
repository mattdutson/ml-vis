# ml-vis

Visualization tools for understanding ML image classifiers

## Conda Environment

To create the `ml-vis` environment, run:
```
conda env create -f conda/environment.yml
```
`environment.yml` lists all required Conda and pip packages.

To enable GPU acceleration, instead run:
```
conda env create -f conda/environment_gpu.yml
```
This requires that NVIDIA drivers and CUDA 10.1 be installed (see the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu)).

After creating one of the above environments, activate it with `conda activate ml-vis`.

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
