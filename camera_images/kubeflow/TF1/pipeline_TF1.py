import json

import kfp.dsl as _kfp_dsl
import kfp.components as _kfp_components

from collections import OrderedDict
from kubernetes import client as k8s_client


def dataset():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import os

    from conabio_ml_vision.datasets.datasets import ImageDataset, ImagePredictionDataset
    from conabio_ml_vision.trainer.model import run_megadetector_inference
    from conabio_ml_vision.utils.scripts import visualize_detections_in_ds

    from conabio_ml.utils.dataset_utils import read_labelmap_file


    BASE_PATH = '/shared_volume/ecoinf_tests/kale_aws/'

    # Results
    results_path = os.path.join(BASE_PATH, "results", "pipeline_1_TF1")
    dataset_csv = os.path.join(results_path, "dataset.csv")
    dets_csv = os.path.join(results_path, "detections_megadet.csv")
    # Data
    snmb_images_dir = os.path.join(BASE_PATH, 'data', "snmb")
    snmb_megadet_vis_dir = os.path.join(BASE_PATH, 'data', "snmb_megadetector_visualization")
    # Files
    snmb_json = os.path.join(BASE_PATH, "files", "snmb_2021_detection-bboxes.json")
    mappings_csv = os.path.join(BASE_PATH, "files", "snmb_to_wcs_compet.csv")
    compet_labelmap_file = os.path.join(BASE_PATH, "files", "compet_labels.txt")
    detector_model_path = os.path.join(BASE_PATH, "files", "megadetector_v4.pb")

    os.makedirs(results_path, exist_ok=True)

    min_score_threshold = 0.3
    '''

    _kale_block2 = '''
    # Dataset creation
    if not os.path.isfile(dataset_csv):
        compet_labelmap = read_labelmap_file(compet_labelmap_file)
        dataset = ImageDataset.from_json(source_path=snmb_json,
                                         images_dir=snmb_images_dir,
                                         categories=list(compet_labelmap.values()),
                                         exclude_categories=['empty'],
                                         mapping_classes=mappings_csv,
                                         not_exist_ok=True)
        dataset.to_csv(dataset_csv)
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (
        _kale_block1,
        _kale_block2,
    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/dataset.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('dataset')

    _kale_mlmdutils.call("mark_execution_complete")


def detection():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import os

    from conabio_ml_vision.datasets.datasets import ImageDataset, ImagePredictionDataset
    from conabio_ml_vision.trainer.model import run_megadetector_inference
    from conabio_ml_vision.utils.scripts import visualize_detections_in_ds

    from conabio_ml.utils.dataset_utils import read_labelmap_file


    BASE_PATH = '/shared_volume/ecoinf_tests/kale_aws/'

    # Results
    results_path = os.path.join(BASE_PATH, "results", "pipeline_1_TF1")
    dataset_csv = os.path.join(results_path, "dataset.csv")
    dets_csv = os.path.join(results_path, "detections_megadet.csv")
    # Data
    snmb_images_dir = os.path.join(BASE_PATH, 'data', "snmb")
    snmb_megadet_vis_dir = os.path.join(BASE_PATH, 'data', "snmb_megadetector_visualization")
    # Files
    snmb_json = os.path.join(BASE_PATH, "files", "snmb_2021_detection-bboxes.json")
    mappings_csv = os.path.join(BASE_PATH, "files", "snmb_to_wcs_compet.csv")
    compet_labelmap_file = os.path.join(BASE_PATH, "files", "compet_labels.txt")
    detector_model_path = os.path.join(BASE_PATH, "files", "megadetector_v4.pb")

    os.makedirs(results_path, exist_ok=True)

    min_score_threshold = 0.3
    '''

    _kale_block2 = '''
    # Megadetector inference
    if not os.path.isfile(dets_csv):
        dataset = ImageDataset.from_csv(dataset_csv, images_dir=snmb_images_dir)
        run_megadetector_inference(dataset=dataset,
                                   out_predictions_csv=dets_csv,
                                   images_dir=snmb_images_dir,
                                   model_path=detector_model_path,
                                   min_score_threshold=min_score_threshold,
                                   include_id=True,
                                   keep_image_id=True,
                                   dataset_partition=None,
                                   num_gpus_per_node=1)    
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (
        _kale_block1,
        _kale_block2,
    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/detection.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('detection')

    _kale_mlmdutils.call("mark_execution_complete")


def visualize_dets():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import os

    from conabio_ml_vision.datasets.datasets import ImageDataset, ImagePredictionDataset
    from conabio_ml_vision.trainer.model import run_megadetector_inference
    from conabio_ml_vision.utils.scripts import visualize_detections_in_ds

    from conabio_ml.utils.dataset_utils import read_labelmap_file


    BASE_PATH = '/shared_volume/ecoinf_tests/kale_aws/'

    # Results
    results_path = os.path.join(BASE_PATH, "results", "pipeline_1_TF1")
    dataset_csv = os.path.join(results_path, "dataset.csv")
    dets_csv = os.path.join(results_path, "detections_megadet.csv")
    # Data
    snmb_images_dir = os.path.join(BASE_PATH, 'data', "snmb")
    snmb_megadet_vis_dir = os.path.join(BASE_PATH, 'data', "snmb_megadetector_visualization")
    # Files
    snmb_json = os.path.join(BASE_PATH, "files", "snmb_2021_detection-bboxes.json")
    mappings_csv = os.path.join(BASE_PATH, "files", "snmb_to_wcs_compet.csv")
    compet_labelmap_file = os.path.join(BASE_PATH, "files", "compet_labels.txt")
    detector_model_path = os.path.join(BASE_PATH, "files", "megadetector_v4.pb")

    os.makedirs(results_path, exist_ok=True)

    min_score_threshold = 0.3
    '''

    _kale_block2 = '''
    # Visualize detections
    visualize_detections_in_ds(detections_csv=dets_csv,
                               images_dir=snmb_images_dir,
                               dest_dir=snmb_megadet_vis_dir)
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (
        _kale_block1,
        _kale_block2,
    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/visualize_dets.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('visualize_dets')

    _kale_mlmdutils.call("mark_execution_complete")


_kale_dataset_op = _kfp_components.func_to_container_op(
    dataset, base_image='sipecam/ecoinf-tensorflow1-kale-gpu:0.6.1')


_kale_detection_op = _kfp_components.func_to_container_op(
    detection, base_image='sipecam/ecoinf-tensorflow1-kale-gpu:0.6.1')


_kale_visualize_dets_op = _kfp_components.func_to_container_op(
    visualize_dets, base_image='sipecam/ecoinf-tensorflow1-kale-gpu:0.6.1')


@_kfp_dsl.pipeline(
    name='test-megadet-2-ltvoo',
    description='Prueba usando el Megadetector sobre un conjunto de fotos del SNMB'
)
def auto_generated_pipeline(vol_shared_volume='efs'):
    _kale_pvolumes_dict = OrderedDict()
    _kale_volume_step_names = []
    _kale_volume_name_parameters = []

    _kale_annotations = {}

    _kale_volume = _kfp_dsl.PipelineVolume(pvc=vol_shared_volume)

    _kale_pvolumes_dict['/shared_volume'] = _kale_volume

    _kale_volume_step_names.sort()
    _kale_volume_name_parameters.sort()

    _kale_dataset_task = _kale_dataset_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    _kale_dataset_task.container.working_dir = "//shared_volume/ecoinf_tests"
    _kale_dataset_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'dataset': '/dataset.html'})
    _kale_dataset_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_dataset_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_dataset_task.dependent_names +
                       _kale_volume_step_names)
    _kale_dataset_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_dataset_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_detection_task = _kale_detection_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_dataset_task)
    _kale_step_limits = {'nvidia.com/gpu': '1'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_detection_task.container.add_resource_limit(_kale_k, _kale_v)
    _kale_detection_task.container.working_dir = "//shared_volume/ecoinf_tests"
    _kale_detection_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'detection': '/detection.html'})
    _kale_detection_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_detection_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_detection_task.dependent_names +
                       _kale_volume_step_names)
    _kale_detection_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_detection_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_visualize_dets_task = _kale_visualize_dets_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_detection_task)
    _kale_visualize_dets_task.container.working_dir = "//shared_volume/ecoinf_tests"
    _kale_visualize_dets_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'visualize_dets': '/visualize_dets.html'})
    _kale_visualize_dets_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_visualize_dets_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_visualize_dets_task.dependent_names +
                       _kale_volume_step_names)
    _kale_visualize_dets_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_visualize_dets_task.add_pod_annotation(
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
    experiment = client.create_experiment('ecoinf-exps-1')

    # Submit a pipeline run
    from kale.common.kfputils import generate_run_name
    run_name = generate_run_name('test-megadet-2-ltvoo')
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
