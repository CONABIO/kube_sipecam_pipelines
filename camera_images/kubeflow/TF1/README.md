Pipeline for camera's images.

## Pipeline description

In this pipeline we perform the inference of the [Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md), a generic animal detector in camera trap photos, on the SNMB images of the dataset built in [pipeline_1](https://github.com/CONABIO/kube_sipecam_pipelines/tree/main/camera_images/kubeflow/1), and save the images with the visualizations of the detections in the folder `/shared_volume/ecoinf_tests/kale_aws/data/snmb_megadetector_visualization`.

## Data input

Dataset built in [pipeline_1](https://github.com/CONABIO/kube_sipecam_pipelines/tree/main/camera_images/kubeflow/1)

## Data output

`/sipecam/ecoinformatica/infra_kube_sipecam/ecoinf_tests/results_from_aws/sept-2021/pipeline_1_TF1` in sipecamdata server.

## Docker image

`sipecam/ecoinf-tensorflow1-kale-gpu:0.6.1` see [doc](https://github.com/CONABIO/kube_sipecam/tree/master/dockerfiles/ecoinf/gpu/tensorflow1)

## Launch

1. Go to [cluster deployment](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#cluster-deployment) for scaling up worker nodes and components.

2. Clone this repo if its not already cloned using a terminal inside jupyterlab, change directory to `kube_sipecam_pipelines/images_camera/kubeflow/1/` and execute:

```
python3 pipeline_TF1.py
```

3. Scale down worker nodes and components: [scale-down-of-components](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#scale-down-of-components).
