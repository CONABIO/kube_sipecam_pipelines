Pipeline for camera's images.

Pipeline description:

Data input:  `ecoinf-snmb-data/kale_aws/` s3 bucket. Source: `/sipecam/ecoinformatica/kale_aws` in sipecamdata server.

Data output: 

Docker image: `sipecam/ecoinf-kale-gpu:0.6.1` see: [doc](https://github.com/CONABIO/kube_sipecam/tree/master/dockerfiles/ecoinf/gpu)


1. Go to [cluster deployment](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#cluster-deployment) for scaling up worker nodes and components.

2. Clone this repo if its not already cloned using a terminal inside jupyterlab, change directory to `kube_sipecam_pipelines/images_camera/kubeflow/1/` and execute:

```
python3 pipeline_1.py
```

3. Scale down worker nodes and components: [scale-down-of-components](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#scale-down-of-components).

