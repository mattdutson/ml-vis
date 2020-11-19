# ml-vis

Creating visualization tools for understanding the behavior of ML image classifiers

## Conda Environment

To create the `mobilenet` environment, run:
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
