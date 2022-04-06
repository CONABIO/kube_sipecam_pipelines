import json

import kfp.dsl as _kfp_dsl
import kfp.components as _kfp_components

from collections import OrderedDict
from kubernetes import client as k8s_client


def imps():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import os
    import numpy as np

    from conabio_ml_vision.datasets.datasets import ImageDataset, ImagePredictionDataset
    from conabio_ml_vision.trainer.model import run_megadetector_inference
    from conabio_ml_vision.utils.scripts import create_obj_level_ds_from_dets_and_anns
    from conabio_ml_vision.utils.evaluator_utils import get_image_level_binary_pred_dataset
    from conabio_ml_vision.utils.aux_utils import eval_binary
    from conabio_ml_vision.utils.evaluator_utils import precision_recall_curve

    from utils_aws import classify_dataset_rekog

    base_files_path = '/shared_volume/ecoinf_tests/sipecam-models-images/pipelines_kale/experiment_4'

    results_path = os.path.join(base_files_path, 'results')
    imgs_path = os.path.join(base_files_path, 'data')
    files_path = os.path.join(base_files_path, 'files')

    datasets_path = os.path.join(results_path, 'datasets')
    full_imgs_csv = os.path.join(datasets_path, 'full_imgs_ds.csv')
    detections_csv = os.path.join(datasets_path, 'detections.csv')
    crops_ds_path = os.path.join(datasets_path, 'crops_ds.csv')
    plots_path = os.path.join(results_path, 'plots')
    crops_imgs_path = os.path.join(imgs_path, 'experiment_4_test_crops')
    test_full_imgs_path = os.path.join(imgs_path, 'experiment_4_test_full_imgs')
    classifs_base_path = os.path.join(results_path, 'classifs')
    evals_base_path = os.path.join(results_path, 'evals')
    megadetector_path = os.path.join(files_path, 'megadetector_v4.pb')

    os.makedirs(datasets_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    INIT_THRES = 0.
    END_THRES = 1.
    STEP_THRES = 0.05
    NUM_STEPS = int((END_THRES-INIT_THRES)*(.1/(STEP_THRES))*10)+1
    DEC_DIGITS_RND_SCR = 3
    MIN_SCORE_DETS = 0.01
    MEGADETECTOR_V4_LABELMAP = {
        1: 'Animal',
        2: 'Person',
        3: 'Vehicle',
    }


    # 1. Take the test partition from the dataset described in the previous section to create a true
    #    binary dataset, mapping the empty category to non-fauna and the rest of the
    #    categories (species) to fauna.
    if not os.path.isfile(full_imgs_csv):
        binary_true_ds = ImageDataset.from_folder(test_full_imgs_path, split_by_folder=False)
        binary_true_ds.to_csv(full_imgs_csv)
    else:
        binary_true_ds = ImageDataset.from_csv(
            full_imgs_csv, images_dir=test_full_imgs_path, validate_filenames=False)
    binary_true_ds.map_categories(mapping_classes={'empty': 'non_fauna', '*': 'fauna'})

    # 2. Generate the detections with Megadetector on the dataset generated in 1.
    if not os.path.isfile(detections_csv):
        run_megadetector_inference(
            dataset=binary_true_ds,
            out_predictions_csv=detections_csv,
            model_path=megadetector_path,
            labelmap=MEGADETECTOR_V4_LABELMAP,
            min_score_threshold=MIN_SCORE_DETS)
    dets_ds = ImagePredictionDataset.from_csv(source_path=detections_csv)
    dets_ds.filter_by_categories(['animal'])
    dets_ds.set_data_field_by_expr('score', lambda x: round(x.score, DEC_DIGITS_RND_SCR))

    # 3. Make a cycle varying the threshold from 0 to 1 in steps of 0.05. For each step create a binary
    #    prediction dataset using the detections generated in 2, considering as fauna the photos with
    #    at least one animal detection with score \u2265 threshold and as non-fauna the rest, and make the
    #    binary evaluation of each of these binary prediction datasets with respect to the true binary
    #    dataset of 1.
    evals_results = []
    for score_thres in np.linspace(INIT_THRES, END_THRES, num=NUM_STEPS):
        thres_str = f'{int(round(score_thres, 2)*100)}'

        dets_ds_thres = dets_ds.copy().filter_by_score(min_score=score_thres)
        binary_pred_ds = get_image_level_binary_pred_dataset(
            binary_true_ds, dets_ds_thres, animal_label='fauna', empty_label='non_fauna')

        res_eval = eval_binary(
            dataset_true=binary_true_ds,
            dataset_pred=binary_pred_ds,
            eval_dir=os.path.join(evals_base_path, "megadetector", f"thres_{thres_str}"),
            partition=None,
            pos_label='fauna',
            sample_counts_in_bars=True,
            title=f'Binary image-level evaluation of Megadetector for a threshold {score_thres:.2f}')
        evals_results.append({'thres_str': thres_str, 'res_eval': res_eval})
    precision_recall_curve(
        evals_results, plot_path='precision_recall-megadet.png',
        title=f'Precision-recall curve for binary evaluation at image level of the Megadetector')

    # 4. Clip the bounding boxes of the detections generated in 2 and create a crops dataset
    if not os.path.isfile(crops_ds_path):
        obj_level_ds = create_obj_level_ds_from_dets_and_anns(
            detections=dets_ds,
            annotations=binary_true_ds,
            det_categories=['animal'],
            include_detection_score=True,
            min_score_detections=MIN_SCORE_DETS)
        crops_ds = obj_level_ds.create_classif_ds_from_bboxes_crops(
            dest_path=crops_imgs_path, allow_label_empty=True)
        crops_ds.to_csv(crops_ds_path)
    else:
        crops_ds = ImageDataset.from_csv(
            crops_ds_path, images_dir=crops_imgs_path, validate_filenames=False)

    # 5. Apply the classification of the multiclass model (Rekognition or EfficientNet) to the crops
    #    dataset generated in 4.
    classified_crops_ds = classify_dataset_rekog(
        crops_ds,
        os.path.join(classifs_base_path, 'rekognition-test_part.csv'),
        partition=None,
        max_classifs=1,
        mappings_labels={})

    # 6. Repeat step 3, previously discarding the detections whose crops have been classified
    #    as empty in 5.
    classified_crops_ds.filter_by_categories('empty', mode='exclude')
    evals_results = []
    for score_thres in np.linspace(INIT_THRES, END_THRES, num=NUM_STEPS):
        thres_str = f'{int(round(score_thres, 2)*100)}'

        dets_ds_thres = dets_ds.copy().filter_by_score(min_score=score_thres)
        dets_ids_thres = dets_ds_thres.as_dataframe()['id'].values
        classified_crops_ds_thres = classified_crops_ds.copy().filter_by_column(
            column='id', values=dets_ids_thres)
        binary_pred_ds = get_image_level_binary_pred_dataset(
            binary_true_ds, classified_crops_ds_thres,
            animal_label='fauna', empty_label='non_fauna')

        title = (f'Binary image-level evaluation of Megadetector + Rekognition '
                 f'for a threshold {score_thres:.2f}')
        res_eval = eval_binary(
            dataset_true=binary_true_ds,
            dataset_pred=binary_pred_ds,
            eval_dir=os.path.join(evals_base_path, "megadetector_rekog", f"thres_{thres_str}"),
            partition=None,
            pos_label='fauna',
            sample_counts_in_bars=True,
            title=title)
        evals_results.append({'thres_str': thres_str, 'res_eval': res_eval})
    precision_recall_curve(
        evals_results,
        plot_path=os.path.join(plots_path, 'precision_recall-megadet_rekog.png'),
        title=f'Precision-Recall Curve for binary image-level evaluation of Megadetector + Rekognition')
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (
        _kale_block1,
    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/imps.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('imps')

    _kale_mlmdutils.call("mark_execution_complete")


_kale_imps_op = _kfp_components.func_to_container_op(
    imps, base_image='sipecam/ecoinf-tensorflow1-kale-gpu:0.6.1_2')


@_kfp_dsl.pipeline(
    name='experiment4-zklsa',
    description='Megadetector and Rekognition'
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

    _kale_imps_task = _kale_imps_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    _kale_step_limits = {'nvidia.com/gpu': '1'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_imps_task.container.add_resource_limit(_kale_k, _kale_v)
    _kale_imps_task.container.working_dir = "//shared_volume/ecoinf_tests/sipecam-models-images/pipelines_kale/experiment_4"
    _kale_imps_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'imps': '/imps.html'})
    _kale_imps_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_imps_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_imps_task.dependent_names +
                       _kale_volume_step_names)
    _kale_imps_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_imps_task.add_pod_annotation(
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
    experiment = client.create_experiment('ecoinfemptyfiltering')

    # Submit a pipeline run
    from kale.common.kfputils import generate_run_name
    run_name = generate_run_name('experiment4-zklsa')
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
