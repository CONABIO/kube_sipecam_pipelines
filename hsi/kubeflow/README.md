
1. Go to [cluster usage](https://conabio.github.io/kube_sipecam/1.Deployment-of-Kubernetes-cluster-in-AWS.html#cluster-usage) for scaling up/down worker nodes.

2. Inside `t2.micro` create deployment using [kube_sipecam/deployments/jupyterlab_cert/efs/README.md](https://github.com/CONABIO/kube_sipecam/blob/master/deployments/jupyterlab_cert/efs/README.md)

3. In your browser use next url. This will open a jupyterlab UI.

```
# For testing
https://api.k8s-dummy.dummy.route53-kube-sipecam.net:30001/hsiurl

# For production

https://proc-sys.route53-kube-sipecam.net:30001/hsiurl

```

4. Clone this repo if its not already cloned using a terminal inside jupyterlab, change directory to `kube_sipecam_pipelines/hsi/kubeflow` and execute:

```
python3 <name of pipeline>.py hsi hsiexperiment
```
