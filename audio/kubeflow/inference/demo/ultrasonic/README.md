Pipeline for audio's recordings.

## Pipeline description

## Requirements

`probe_config_detection.json`

`probe_config_classification.json`

`col_config.json`

## Data input


## Data output


## Docker image 

`sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-gpu-cert:0.6.1` see [doc](https://github.com/CONABIO/kube_sipecam/blob/master/dockerfiles/audio/tensorflow-yuntu-dask-gpu/0.6.1/Dockerfile)

## Launch

1. Go to [cluster deployment](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#cluster-deployment) for scaling up worker nodes and components.

2. Clone this repo if its not already cloned using a terminal inside jupyterlab, change directory to `kube_sipecam_pipelines/audio/kubeflow/inference/` and execute:

```

```

3. Scale down worker nodes and components: [scale-down-of-components](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#scale-down-of-components).
