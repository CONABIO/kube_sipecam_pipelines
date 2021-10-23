import json

import kfp.dsl as _kfp_dsl
import kfp.components as _kfp_components

from collections import OrderedDict
from kubernetes import client as k8s_client


def apply_detection(BASE_DIR: str, COL_CONFIG: str, CONGLOMERATE_ID: str, MIN_RECORD_DURATION: float, PROBE_CONFIG_DETECTION: str):
    _kale_pipeline_parameters_block = '''
    BASE_DIR = "{}"
    COL_CONFIG = "{}"
    CONGLOMERATE_ID = "{}"
    MIN_RECORD_DURATION = {}
    PROBE_CONFIG_DETECTION = "{}"
    '''.format(BASE_DIR, COL_CONFIG, CONGLOMERATE_ID, MIN_RECORD_DURATION, PROBE_CONFIG_DETECTION)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import os
    import json
    import time
    import pandas as pd
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from yuntu.collection.methods import collection
    from yuntu.soundscape.pipelines.probe_annotate import ProbeAnnotate
    '''

    _kale_block2 = '''
    def probe_conglomerate(conglomerate_id, pipe_name, base_dir, 
                           probe_config_path, col_config_path, col_query,
                           npartitions, client=None):
        with open(probe_config_path) as file:
            probe_config = json.load(file)

        with open(col_config_path) as file:
            col_config = json.load(file)

        work_dir = os.path.join(base_dir, conglomerate_id)
        
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    #     log_path = os.path.join(work_dir, pipe_name, "process.log")
    #     probe_config["kwargs"] = {"log_file": log_path}
        
        col = collection(**col_config)
        nrecordings = col.db_manager.select(col_query, model="recording").count()
        col.db_manager.db.disconnect()

        print(f"Working with conglomerate {conglomerate_id}. Total files: {nrecordings}")
        
        info_path = os.path.join(work_dir, pipe_name, "info.json")
        tpath = os.path.join(work_dir, pipe_name, "persist", "annotation_result.parquet")

        start = time.time()
        if nrecordings > 0:
            pipeline =  ProbeAnnotate(pipe_name, probe_config, col_config, col_query, work_dir=work_dir)

            if not os.path.exists(tpath):
                annotation_result = pipeline["annotation_result"].compute(client=client,
                                                                          feed={"npartitions": npartitions})
            else:
                print("Data already processed. Reading results...")
                annotation_result = pipeline["annotation_result"].read().compute()

                with open(info_path) as json_file:
                    info = json.load(json_file)
                
                return annotation_result, info

        else:
            print(f"No matched ultrasonic recordings in conglomerate {conglomerate_id}.")
            annotation_result = None

        end = time.time()
        elapsed = end - start

        info = {"conglomerate_id": conglomerate_id,
                "total_files": nrecordings,
                "elapsed_time": elapsed}

        if annotation_result is not None:
            with open(info_path, 'w') as outfile:
                json.dump(info, outfile)

        return annotation_result, info
    '''

    _kale_block3 = '''
    cluster = LocalCUDACluster()
    client = Client(cluster)
    npartitions = len(client.ncores())

    det_col_query = eval(F\'\'\'(lambda recording: recording.metadata["conglomerado"]["nombre"] == "{CONGLOMERATE_ID}"
                         and recording.spectrum == "ultrasonic"
                         and recording.media_info["duration"] > {MIN_RECORD_DURATION})\'\'\')
    detection_result, detection_info = probe_conglomerate(CONGLOMERATE_ID,
                                                          "BATMX_probe",
                                                          BASE_DIR,
                                                          PROBE_CONFIG_DETECTION,
                                                          COL_CONFIG,
                                                          det_col_query,
                                                          npartitions,
                                                          client=client)
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/apply_detection.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('apply_detection')

    _kale_mlmdutils.call("mark_execution_complete")


def apply_class(BASE_DIR: str, COL_CONFIG: str, CONGLOMERATE_ID: str, MIN_RECORD_DURATION: float, PROBE_CONFIG_CLASSIFICATION: str):
    _kale_pipeline_parameters_block = '''
    BASE_DIR = "{}"
    COL_CONFIG = "{}"
    CONGLOMERATE_ID = "{}"
    MIN_RECORD_DURATION = {}
    PROBE_CONFIG_CLASSIFICATION = "{}"
    '''.format(BASE_DIR, COL_CONFIG, CONGLOMERATE_ID, MIN_RECORD_DURATION, PROBE_CONFIG_CLASSIFICATION)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import os
    import json
    import time
    import pandas as pd
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from yuntu.collection.methods import collection
    from yuntu.soundscape.pipelines.probe_annotate import ProbeAnnotate
    '''

    _kale_block2 = '''
    def probe_conglomerate(conglomerate_id, pipe_name, base_dir, 
                           probe_config_path, col_config_path, col_query,
                           npartitions, client=None):
        with open(probe_config_path) as file:
            probe_config = json.load(file)

        with open(col_config_path) as file:
            col_config = json.load(file)

        work_dir = os.path.join(base_dir, conglomerate_id)
        
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    #     log_path = os.path.join(work_dir, pipe_name, "process.log")
    #     probe_config["kwargs"] = {"log_file": log_path}
        
        col = collection(**col_config)
        nrecordings = col.db_manager.select(col_query, model="recording").count()
        col.db_manager.db.disconnect()

        print(f"Working with conglomerate {conglomerate_id}. Total files: {nrecordings}")
        
        info_path = os.path.join(work_dir, pipe_name, "info.json")
        tpath = os.path.join(work_dir, pipe_name, "persist", "annotation_result.parquet")

        start = time.time()
        if nrecordings > 0:
            pipeline =  ProbeAnnotate(pipe_name, probe_config, col_config, col_query, work_dir=work_dir)

            if not os.path.exists(tpath):
                annotation_result = pipeline["annotation_result"].compute(client=client,
                                                                          feed={"npartitions": npartitions})
            else:
                print("Data already processed. Reading results...")
                annotation_result = pipeline["annotation_result"].read().compute()

                with open(info_path) as json_file:
                    info = json.load(json_file)
                
                return annotation_result, info

        else:
            print(f"No matched ultrasonic recordings in conglomerate {conglomerate_id}.")
            annotation_result = None

        end = time.time()
        elapsed = end - start

        info = {"conglomerate_id": conglomerate_id,
                "total_files": nrecordings,
                "elapsed_time": elapsed}

        if annotation_result is not None:
            with open(info_path, 'w') as outfile:
                json.dump(info, outfile)

        return annotation_result, info
    '''

    _kale_block3 = '''
    cluster = LocalCUDACluster()
    client = Client(cluster)
    npartitions = len(client.ncores())

    class_col_query = eval(F\'\'\'(lambda recording: recording.metadata["conglomerado"]["nombre"] == "{CONGLOMERATE_ID}"
                           and recording.spectrum == "ultrasonic"
                           and len(recording.annotations)>0
                           and recording.media_info["duration"] > {MIN_RECORD_DURATION})\'\'\')
    class_result, class_info = probe_conglomerate(CONGLOMERATE_ID,
                                                  "BATMX_class_probe",
                                                  BASE_DIR,
                                                  PROBE_CONFIG_CLASSIFICATION,
                                                  COL_CONFIG,
                                                  class_col_query,
                                                  npartitions,
                                                  client=client)
    '''

    _kale_block4 = '''
    
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    _kale_block4,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/apply_class.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('apply_class')

    _kale_mlmdutils.call("mark_execution_complete")


_kale_apply_detection_op = _kfp_components.func_to_container_op(
    apply_detection, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-gpu-cert:0.6.1')


_kale_apply_class_op = _kfp_components.func_to_container_op(
    apply_class, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-gpu-cert:0.6.1')


@_kfp_dsl.pipeline(
    name='batmx-probe-9v3tr',
    description='Apply detection and classification using tensorflow models.'
)
def auto_generated_pipeline(BASE_DIR='/shared_volume/audio-develop-backup-09-06-2021/demo/snmb/results', COL_CONFIG='/shared_volume/audio-develop-backup-09-06-2021/demo/snmb/configs/col_config.json', CONGLOMERATE_ID='117960', MIN_RECORD_DURATION='0.005', PROBE_CONFIG_CLASSIFICATION='/shared_volume/audio-develop-backup-09-06-2021/demo/snmb/configs/probe_config_classification.json', PROBE_CONFIG_DETECTION='/shared_volume/audio-develop-backup-09-06-2021/demo/snmb/configs/probe_config_detection.json', vol_shared_volume='hostpath-pvc'):
    _kale_pvolumes_dict = OrderedDict()
    _kale_volume_step_names = []
    _kale_volume_name_parameters = []

    _kale_annotations = {}

    _kale_volume = _kfp_dsl.PipelineVolume(pvc=vol_shared_volume)

    _kale_pvolumes_dict['/shared_volume'] = _kale_volume

    _kale_volume_step_names.sort()
    _kale_volume_name_parameters.sort()

    _kale_apply_detection_task = _kale_apply_detection_op(BASE_DIR, COL_CONFIG, CONGLOMERATE_ID, MIN_RECORD_DURATION, PROBE_CONFIG_DETECTION)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    _kale_step_limits = {'nvidia.com/gpu': '1'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_apply_detection_task.container.add_resource_limit(
            _kale_k, _kale_v)
    _kale_apply_detection_task.container.working_dir = "//shared_volume/audio-develop-backup-09-06-2021/demo/snmb"
    _kale_apply_detection_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'apply_detection': '/apply_detection.html'})
    _kale_apply_detection_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_apply_detection_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_apply_detection_task.dependent_names +
                       _kale_volume_step_names)
    _kale_apply_detection_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_apply_detection_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_apply_class_task = _kale_apply_class_op(BASE_DIR, COL_CONFIG, CONGLOMERATE_ID, MIN_RECORD_DURATION, PROBE_CONFIG_CLASSIFICATION)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_apply_detection_task)
    _kale_step_limits = {'nvidia.com/gpu': '1'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_apply_class_task.container.add_resource_limit(_kale_k, _kale_v)
    _kale_apply_class_task.container.working_dir = "//shared_volume/audio-develop-backup-09-06-2021/demo/snmb"
    _kale_apply_class_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'apply_class': '/apply_class.html'})
    _kale_apply_class_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_apply_class_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_apply_class_task.dependent_names +
                       _kale_volume_step_names)
    _kale_apply_class_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_apply_class_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('probe-117960')

    # Submit a pipeline run
    from kale.common.kfputils import generate_run_name
    run_name = generate_run_name('batmx-probe-9v3tr')
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
