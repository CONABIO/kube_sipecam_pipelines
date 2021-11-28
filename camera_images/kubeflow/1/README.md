Pipeline for camera's images.

## Pipeline description

In this pipeline, training was performed for 20 epochs of an EfficientNetB3 model with an image resolution of 300x300 px using the Backend tf.Keras.

The dataset was built using SNMB images of species common to those in the Kaggle competition [iWildcam 2021](https://www.kaggle.com/c/iwildcam2021-fgvc8). For these images, the manually labeled species and bounding boxes were taken, and the images were cropped and centered to the size square of the longest side of each bbox. The dataset was partitioned in an 80/20 ratio for training/testing.

During training, basic data augmentation operations (rotation, translation, flip and contrast) were performed, as well as a 20% Droput stage, an Adam optimizer with learning rate of 0.01 and a crossentropy categorical loss function. 
Once the training was completed, model inference was performed on the test partition, which was evaluated for the metrics: precision, recall, accuracy and score-F1 for each class, and a macro-average of each metric was performed to obtain an overall evaluation of the model.

In the folder `/shared_volume/ecoinf_tests/kale_aws/results/pipeline_1/evaluation` the plots of all these evaluations, as well as the normalized confusion matrix, were saved.

## Data input

`/sipecam/ecoinformatica/infra_kube_sipecam/ecoinf_tests` in sipecamdata server.

## Data output

`/sipecam/ecoinformatica/infra_kube_sipecam/ecoinf_tests/results_from_aws/sept-2021/pipeline_1` in sipecamdata server.

## Docker image 

`sipecam/ecoinf-kale-gpu:0.6.1` see: [doc](https://github.com/CONABIO/kube_sipecam/tree/master/dockerfiles/ecoinf/gpu)

## Launch

1. Go to [cluster deployment](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#cluster-deployment) for scaling up worker nodes and components.

2. Clone this repo if its not already cloned using a terminal inside jupyterlab, change directory to `kube_sipecam_pipelines/images_camera/kubeflow/1/` and execute:

```
python3 pipeline_1.py
```

3. Scale down worker nodes and components: [scale-down-of-components](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#scale-down-of-components).

