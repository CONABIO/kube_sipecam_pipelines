import json

import kfp.dsl as _kfp_dsl
import kfp.components as _kfp_components

from collections import OrderedDict
from kubernetes import client as k8s_client


def string2():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    string2 = " Mundo!"
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio-develop-backup-09-06-2021/.simple_pipeline.ipynb.kale.marshal.dir")
    _kale_marshal.save(string2, "string2")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (
        _kale_block1,
        _kale_data_saving_block)
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/string2.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('string2')

    _kale_mlmdutils.call("mark_execution_complete")


def string1():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    string1 = "Hola"
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio-develop-backup-09-06-2021/.simple_pipeline.ipynb.kale.marshal.dir")
    _kale_marshal.save(string1, "string1")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (
        _kale_block1,
        _kale_data_saving_block)
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/string1.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('string1')

    _kale_mlmdutils.call("mark_execution_complete")


def write():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio-develop-backup-09-06-2021/.simple_pipeline.ipynb.kale.marshal.dir")
    string1 = _kale_marshal.load("string1")
    string2 = _kale_marshal.load("string2")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    f = open("/shared_volume/test.txt", 'w')
    f.write("".join([string1, string2]))
    f.close()
    print("listo")
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_data_loading_block,
                    _kale_block1,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/write.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('write')

    _kale_mlmdutils.call("mark_execution_complete")


def print():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio-develop-backup-09-06-2021/.simple_pipeline.ipynb.kale.marshal.dir")
    string1 = _kale_marshal.load("string1")
    string2 = _kale_marshal.load("string2")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    print("".join([string1, string2]))
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_data_loading_block,
                    _kale_block1,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/print.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('print')

    _kale_mlmdutils.call("mark_execution_complete")


_kale_string2_op = _kfp_components.func_to_container_op(
    string2, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-gpu-cert:0.6.1')


_kale_string1_op = _kfp_components.func_to_container_op(
    string1, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-gpu-cert:0.6.1')


_kale_write_op = _kfp_components.func_to_container_op(
    write, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-gpu-cert:0.6.1')


_kale_print_op = _kfp_components.func_to_container_op(
    print, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-gpu-cert:0.6.1')


@_kfp_dsl.pipeline(
    name='mipipeline-d04o4',
    description='simple pipeline'
)
def auto_generated_pipeline(vol_shared_volume='hostpath-pvc'):
    _kale_pvolumes_dict = OrderedDict()
    _kale_volume_step_names = []
    _kale_volume_name_parameters = []

    _kale_annotations = {}

    _kale_volume = _kfp_dsl.PipelineVolume(pvc=vol_shared_volume)

    _kale_pvolumes_dict['/shared_volume'] = _kale_volume

    _kale_volume_step_names.sort()
    _kale_volume_name_parameters.sort()

    _kale_string2_task = _kale_string2_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    _kale_string2_task.container.working_dir = "//shared_volume/audio-develop-backup-09-06-2021"
    _kale_string2_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'string2': '/string2.html'})
    _kale_string2_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_string2_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_string2_task.dependent_names +
                       _kale_volume_step_names)
    _kale_string2_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_string2_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_string1_task = _kale_string1_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    _kale_step_limits = {'nvidia.com/gpu': '1'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_string1_task.container.add_resource_limit(_kale_k, _kale_v)
    _kale_string1_task.container.working_dir = "//shared_volume/audio-develop-backup-09-06-2021"
    _kale_string1_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'string1': '/string1.html'})
    _kale_string1_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_string1_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_string1_task.dependent_names +
                       _kale_volume_step_names)
    _kale_string1_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_string1_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_write_task = _kale_write_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_string1_task, _kale_string2_task)
    _kale_write_task.container.working_dir = "//shared_volume/audio-develop-backup-09-06-2021"
    _kale_write_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'write': '/write.html'})
    _kale_write_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_write_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_write_task.dependent_names +
                       _kale_volume_step_names)
    _kale_write_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_write_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_print_task = _kale_print_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_string1_task, _kale_string2_task)
    _kale_step_limits = {'nvidia.com/gpu': '1'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_print_task.container.add_resource_limit(_kale_k, _kale_v)
    _kale_print_task.container.working_dir = "//shared_volume/audio-develop-backup-09-06-2021"
    _kale_print_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'print': '/print.html'})
    _kale_print_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_print_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_print_task.dependent_names +
                       _kale_volume_step_names)
    _kale_print_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_print_task.add_pod_annotation(
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
    experiment = client.create_experiment('Default')

    # Submit a pipeline run
    from kale.common.kfputils import generate_run_name
    run_name = generate_run_name('mipipeline-d04o4')
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
