Pipeline for audio's recordings.

## Pipeline description
The objective of this pipeline is to annotate collections of ultrasonic recordings with models for ultrasonic acoustic event detection/classification of any kind. To do so, the pipeline uses a home made python package called *yuntu* which has classes and methods for creating standardized recording databases, reading and processing acoustic data on the fly from special dask and pandas dataframes with extended capabilities, among other utilities. The detection/classification models are wrapped into a special class called *probe* that emcompasses all the methods that are necessary for annotation formatting and insertion into *yuntu* collections.

## Requirements
To run the pipeline one must specify paths for the following configurations according to input parameters:
**probe_config_detection** : this parameter indicates the configuration for an ultrasonic event detection *probe*. This model is intended to produce unitary class annotations that will be used for classification downstream. In this case, the detection probe is aimed to annotate the order *Chiroptera* (bats) in a wide sense.
**probe_config_classification**: this parameter indicates the configuration for an ultrasonic event classification *probe*. This model is intended to classify preexistig annotations for a class that comprehends its target classes. In this case, detected vocalizations of chiropterans are classified into different bat species.
**col_config**: this parameter specifies the database specs and collection characteristics for the acoustic data to be processed. Within *yuntu* one can specify several kinds of SQL databases for specific use cases. In this case the pipeline uses a remotely accessible postgresql database from an AWS RDS.

## Data input
The backend database produces the data and metadata inputs to be processed so it is itself the main data input. The specific inputs for each run are determined by a query to the specified *yuntu* collection, which correspond to distinct filters for the specified conglomerate id of the SNMB database.


## Data output
Outputs are returned as new  annotations into the collection database that are accessible through an independent connection as soon as they are generated. Annotations consist of a bounding box or time interval, depending on the type of *probe*, and a set of labels for this event.

## Docker image 

`sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-gpu-cert:0.6.1` see [doc](https://github.com/CONABIO/kube_sipecam/blob/master/dockerfiles/audio/tensorflow-yuntu-dask-gpu/0.6.1/Dockerfile)

## Launch

1. Go to [cluster deployment](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#cluster-deployment) for scaling up worker nodes and components.

2. Clone this repo if its not already cloned using a terminal inside jupyterlab, change directory to `kube_sipecam_pipelines/audio/kubeflow/inference/` and execute:

```
python3 batmx-probe.py
```

3. Scale down worker nodes and components: [scale-down-of-components](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#scale-down-of-components).
