Pipeline for habitat suitability index, *aka* HSI.

## Pipeline description

## Data input

`/LUSTRE/MADMEX/tasks/2020/9_data_for_hsi_mariana_munguia/` dirs: `Ponca_DV`, `Ponca_DV_loc`, `forest_jEquihua_mar`.

## Data output 

`/LUSTRE/MADMEX/tasks/2020/9_data_for_hsi_mariana_munguia/results_aws`

## Docker image

`sipecam/hsi-kale:0.6.1` see [doc](https://github.com/CONABIO/kube_sipecam/tree/master/dockerfiles/hsi)


1. Go to [cluster deployment](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#cluster-deployment) for scaling up worker nodes and components.

2. Clone this repo if its not already cloned using a terminal inside jupyterlab, change directory to `kube_sipecam_pipelines/hsi/kubeflow` and execute:

```
python3 hsipipeline.py hsi hsiexperiment
```

3. Scale down worker nodes and components: [scale-down-of-components](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#scale-down-of-components).

