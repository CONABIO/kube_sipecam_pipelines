Pipeline for habitat suitability index, *aka* HSI.

Data input: `hsi-kale` s3 bucket.

Data output: `hsi-kale-results` s3 bucket


1. Go to [cluster deployment](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#cluster-deployment) for scaling up worker nodes and components.

2. Clone this repo if its not already cloned using a terminal inside jupyterlab, change directory to `kube_sipecam_pipelines/hsi/kubeflow` and execute:

```
python3 hsipipeline.py hsi hsiexperiment
```

3. Scale down worker nodes and components: [scale-down-of-components](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#scale-down-of-components).

