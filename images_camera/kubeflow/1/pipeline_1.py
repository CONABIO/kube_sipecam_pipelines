import json

import kfp.dsl as _kfp_dsl
import kfp.components as _kfp_components

from collections import OrderedDict
from kubernetes import client as k8s_client


def dataset_creation():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import os
    import pandas as pd
    from collections import defaultdict

    # Update to TF version compatible with CUDA 10.1
    import subprocess
    bashCommand = "pip3 uninstall tensorflow-gpu -y"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand = "pip3 install tensorflow-gpu==2.3.0"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    import tensorflow as tf

    print(f"Version de tf: { tf.__version__}")
    print(f"Lista de GPUs disponibles: {tf.config.list_physical_devices('GPU')}")

    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers.experimental import preprocessing
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, BatchNormalization, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import EfficientNetB3

    from conabio_ml.datasets.dataset import Partitions
    from conabio_ml.utils.dataset_utils import read_labelmap_file, write_labelmap_file

    from conabio_ml_vision.evaluator.evaluator import ImageClassificationEvaluator
    from conabio_ml_vision.evaluator.evaluator import ImageClassificationMetrics
    from conabio_ml_vision.datasets.datasets import ImageDataset, ImagePredictionDataset

    from conabio_ml.utils.logger import get_logger

    logger = get_logger(__name__)

    def classify_dataset(dataset,
                         classifs_csv,
                         labelmap,
                         model,
                         images_size,
                         batch_size,
                         partition=Partitions.TEST,
                         max_classifs=1):
        if os.path.isfile(classifs_csv):
            return ImagePredictionDataset.from_csv(source_path=classifs_csv)

        test_df = dataset.get_splitted(partitions=partition)
        test_batches = ImageDataGenerator().flow_from_dataframe(
            test_df,
            x_col="item",
            class_mode=None,
            target_size=images_size,
            batch_size=batch_size,
            shuffle=False,
            validate_filenames=False)

        preds = model.predict(test_batches, batch_size=batch_size)
        results = defaultdict(list)
        for i, (_, row) in enumerate(test_df.iterrows()):
            sorted_inds = [y[0] for y in sorted(enumerate(preds[i]), key=lambda x:x[1], reverse=True)]
            for k in range(max_classifs):
                ind = sorted_inds[k]
                results["item"].append(row["item"])
                results["label"].append(labelmap[ind])
                results["score"].append(preds[i][ind])
                results["image_id"].append(row["image_id"])
                results["id"].append(row["id"])
        data = pd.DataFrame(results)
        images_dir = dataset.get_images_dir()
        classifs_on_test_ds = ImagePredictionDataset(data, info={}, images_dir=images_dir)
        classifs_on_test_ds.to_csv(dest_path=classifs_csv)
        return classifs_on_test_ds


    def eval_multi(true_ds, pred_ds, eval_dir, partition=Partitions.TEST):
        res_eval = ImageClassificationEvaluator.eval(
            dataset_true=true_ds,
            dataset_pred=pred_ds,
            eval_config={
                'metrics_set': {
                    ImageClassificationMetrics.Sets.MULTICLASS: {
                        "average": 'macro',
                        "normalize": "true",
                        "zero_division": 1
                    }
                },
                "labels": true_ds.get_categories(),
                'partition': partition,
            })
        os.makedirs(eval_dir, exist_ok=True)
        res_eval.result_plots(dest_path=eval_dir, report=True, sample_counts_in_bars=True)
        res_eval.store_eval_metrics(dest_path=os.path.join(eval_dir, "results.json"))

    def train(model_checkpoint_path,
              dataset,
              labelmap_path,
              model_type,
              model_name,
              images_size,
              epochs,
              batch_size):
        if not os.path.isfile(model_checkpoint_path):
            df_train = dataset.get_splitted(partitions=Partitions.TRAIN)
            n_cats = dataset.get_num_categories()
            classes = dataset.get_classes()
            train_batches = ImageDataGenerator().flow_from_dataframe(df_train,
                                                                     x_col="item",
                                                                     y_col="label",
                                                                     classes=classes,
                                                                     target_size=images_size,
                                                                     batch_size=batch_size,
                                                                     validate_filenames=False)
            labelmap = {v: k for k, v in train_batches.class_indices.items()}
            write_labelmap_file(labelmap=labelmap, dest_path=labelmap_path)
            with tf.distribute.MirroredStrategy().scope():
                model = build_model(num_classes=n_cats, model_type=model_type,
                                    name=model_name, input_shape=(images_size[0], images_size[1], 3))
            logger.info(f"Training model {model_name}")
            model.fit(train_batches, epochs=epochs, verbose=1,
                      callbacks=[ModelCheckpoint(model_checkpoint_path)])
        else:
            pass # model = load_model_from_ckp(model_checkpoint_path)

        # return model

    def load_model_from_ckp(model_checkpoint_path):
        with tf.distribute.MirroredStrategy().scope():
            return load_model(model_checkpoint_path)

    def build_model(num_classes, model_type, name, input_shape):
        img_augmentation = Sequential(
            [
                preprocessing.RandomRotation(factor=0.15),
                preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                preprocessing.RandomFlip(mode="horizontal"),
                preprocessing.RandomContrast(factor=0.1),
            ],
            name="img_augmentation",
        )
        inputs = Input(shape=input_shape)
        x = img_augmentation(inputs)
        model = model_type(include_top=False, input_tensor=x, weights="imagenet")
        model.trainable = False
        x = GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = BatchNormalization()(x)
        x = Dropout(0.2, name="top_dropout")(x)
        outputs = Dense(num_classes, activation="softmax", name="pred")(x)
        model = Model(inputs, outputs, name=name)
        optimizer = Adam(learning_rate=1e-2)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model


    EPOCHS = 20
    RANDOM_STATE = 445
    BATCH_SIZE_EVAL = 16
    BATCH_SIZE_TRAIN = 32
    TRAIN_PERC = 0.8

    BASE_PATH = '/shared_volume/ecoinf_tests/kale_aws/'

    # Results
    results_path = os.path.join(BASE_PATH, "results", "pipeline_1")
    dataset_csv = os.path.join(results_path, "dataset.csv")
    classifs_csv = os.path.join(results_path, "classifs_on_test_part.csv")
    model_checkpoint_path = os.path.join(results_path, "efficientnet_b3_train_1.model.hdf5")
    labelmap_file = os.path.join(results_path, "labels.txt")
    eval_dir = os.path.join(results_path, "evaluation")
    # Data
    snmb_images_dir = os.path.join(BASE_PATH, 'data', "snmb")
    snmb_crops_dir = os.path.join(BASE_PATH, 'data', "snmb_crops")
    # Files
    snmb_json = os.path.join(BASE_PATH, "files", "snmb_2021_detection-bboxes.json")
    mappings_csv = os.path.join(BASE_PATH, "files", "snmb_to_wcs_compet.csv")
    compet_labelmap_file = os.path.join(BASE_PATH, "files", "compet_labels.txt")

    os.makedirs(eval_dir, exist_ok=True)

    '''

    _kale_block2 = '''
    # Dataset creation
    if os.path.isfile(dataset_csv):
        dataset = ImageDataset.from_csv(source_path=dataset_csv, images_dir=snmb_crops_dir)
    else:
        compet_labelmap = read_labelmap_file(compet_labelmap_file)
        dataset = ImageDataset.from_json(
            snmb_json,
            images_dir=snmb_images_dir,
            categories=list(compet_labelmap.values()),
            exclude_categories=['empty'],
            mapping_classes=mappings_csv,
            not_exist_ok=True)
        dataset = dataset.create_classif_ds_from_bboxes_crops(
            dest_path=snmb_crops_dir, include_id=True, inherit_fields=["image_id", 'location'])
        dataset.split(train_perc=TRAIN_PERC, test_perc=1.-TRAIN_PERC,
                      val_perc=0, group_by_field="location")
        dataset.to_csv(dest_path=dataset_csv, columns=["image_id"])
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/ecoinf_tests/.pipeline_1.ipynb.kale.marshal.dir")
    _kale_marshal.save(dataset, "dataset")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (
        _kale_block1,
        _kale_block2,
        _kale_data_saving_block)
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/dataset_creation.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('dataset_creation')

    _kale_mlmdutils.call("mark_execution_complete")


def fit_model():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/ecoinf_tests/.pipeline_1.ipynb.kale.marshal.dir")
    dataset = _kale_marshal.load("dataset")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import os
    import pandas as pd
    from collections import defaultdict

    # Update to TF version compatible with CUDA 10.1
    import subprocess
    bashCommand = "pip3 uninstall tensorflow-gpu -y"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand = "pip3 install tensorflow-gpu==2.3.0"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    import tensorflow as tf

    print(f"Version de tf: { tf.__version__}")
    print(f"Lista de GPUs disponibles: {tf.config.list_physical_devices('GPU')}")

    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers.experimental import preprocessing
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, BatchNormalization, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import EfficientNetB3

    from conabio_ml.datasets.dataset import Partitions
    from conabio_ml.utils.dataset_utils import read_labelmap_file, write_labelmap_file

    from conabio_ml_vision.evaluator.evaluator import ImageClassificationEvaluator
    from conabio_ml_vision.evaluator.evaluator import ImageClassificationMetrics
    from conabio_ml_vision.datasets.datasets import ImageDataset, ImagePredictionDataset

    from conabio_ml.utils.logger import get_logger

    logger = get_logger(__name__)

    def classify_dataset(dataset,
                         classifs_csv,
                         labelmap,
                         model,
                         images_size,
                         batch_size,
                         partition=Partitions.TEST,
                         max_classifs=1):
        if os.path.isfile(classifs_csv):
            return ImagePredictionDataset.from_csv(source_path=classifs_csv)

        test_df = dataset.get_splitted(partitions=partition)
        test_batches = ImageDataGenerator().flow_from_dataframe(
            test_df,
            x_col="item",
            class_mode=None,
            target_size=images_size,
            batch_size=batch_size,
            shuffle=False,
            validate_filenames=False)

        preds = model.predict(test_batches, batch_size=batch_size)
        results = defaultdict(list)
        for i, (_, row) in enumerate(test_df.iterrows()):
            sorted_inds = [y[0] for y in sorted(enumerate(preds[i]), key=lambda x:x[1], reverse=True)]
            for k in range(max_classifs):
                ind = sorted_inds[k]
                results["item"].append(row["item"])
                results["label"].append(labelmap[ind])
                results["score"].append(preds[i][ind])
                results["image_id"].append(row["image_id"])
                results["id"].append(row["id"])
        data = pd.DataFrame(results)
        images_dir = dataset.get_images_dir()
        classifs_on_test_ds = ImagePredictionDataset(data, info={}, images_dir=images_dir)
        classifs_on_test_ds.to_csv(dest_path=classifs_csv)
        return classifs_on_test_ds


    def eval_multi(true_ds, pred_ds, eval_dir, partition=Partitions.TEST):
        res_eval = ImageClassificationEvaluator.eval(
            dataset_true=true_ds,
            dataset_pred=pred_ds,
            eval_config={
                'metrics_set': {
                    ImageClassificationMetrics.Sets.MULTICLASS: {
                        "average": 'macro',
                        "normalize": "true",
                        "zero_division": 1
                    }
                },
                "labels": true_ds.get_categories(),
                'partition': partition,
            })
        os.makedirs(eval_dir, exist_ok=True)
        res_eval.result_plots(dest_path=eval_dir, report=True, sample_counts_in_bars=True)
        res_eval.store_eval_metrics(dest_path=os.path.join(eval_dir, "results.json"))

    def train(model_checkpoint_path,
              dataset,
              labelmap_path,
              model_type,
              model_name,
              images_size,
              epochs,
              batch_size):
        if not os.path.isfile(model_checkpoint_path):
            df_train = dataset.get_splitted(partitions=Partitions.TRAIN)
            n_cats = dataset.get_num_categories()
            classes = dataset.get_classes()
            train_batches = ImageDataGenerator().flow_from_dataframe(df_train,
                                                                     x_col="item",
                                                                     y_col="label",
                                                                     classes=classes,
                                                                     target_size=images_size,
                                                                     batch_size=batch_size,
                                                                     validate_filenames=False)
            labelmap = {v: k for k, v in train_batches.class_indices.items()}
            write_labelmap_file(labelmap=labelmap, dest_path=labelmap_path)
            with tf.distribute.MirroredStrategy().scope():
                model = build_model(num_classes=n_cats, model_type=model_type,
                                    name=model_name, input_shape=(images_size[0], images_size[1], 3))
            logger.info(f"Training model {model_name}")
            model.fit(train_batches, epochs=epochs, verbose=1,
                      callbacks=[ModelCheckpoint(model_checkpoint_path)])
        else:
            pass # model = load_model_from_ckp(model_checkpoint_path)

        # return model

    def load_model_from_ckp(model_checkpoint_path):
        with tf.distribute.MirroredStrategy().scope():
            return load_model(model_checkpoint_path)

    def build_model(num_classes, model_type, name, input_shape):
        img_augmentation = Sequential(
            [
                preprocessing.RandomRotation(factor=0.15),
                preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                preprocessing.RandomFlip(mode="horizontal"),
                preprocessing.RandomContrast(factor=0.1),
            ],
            name="img_augmentation",
        )
        inputs = Input(shape=input_shape)
        x = img_augmentation(inputs)
        model = model_type(include_top=False, input_tensor=x, weights="imagenet")
        model.trainable = False
        x = GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = BatchNormalization()(x)
        x = Dropout(0.2, name="top_dropout")(x)
        outputs = Dense(num_classes, activation="softmax", name="pred")(x)
        model = Model(inputs, outputs, name=name)
        optimizer = Adam(learning_rate=1e-2)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model


    EPOCHS = 20
    RANDOM_STATE = 445
    BATCH_SIZE_EVAL = 16
    BATCH_SIZE_TRAIN = 32
    TRAIN_PERC = 0.8

    BASE_PATH = '/shared_volume/ecoinf_tests/kale_aws/'

    # Results
    results_path = os.path.join(BASE_PATH, "results", "pipeline_1")
    dataset_csv = os.path.join(results_path, "dataset.csv")
    classifs_csv = os.path.join(results_path, "classifs_on_test_part.csv")
    model_checkpoint_path = os.path.join(results_path, "efficientnet_b3_train_1.model.hdf5")
    labelmap_file = os.path.join(results_path, "labels.txt")
    eval_dir = os.path.join(results_path, "evaluation")
    # Data
    snmb_images_dir = os.path.join(BASE_PATH, 'data', "snmb")
    snmb_crops_dir = os.path.join(BASE_PATH, 'data', "snmb_crops")
    # Files
    snmb_json = os.path.join(BASE_PATH, "files", "snmb_2021_detection-bboxes.json")
    mappings_csv = os.path.join(BASE_PATH, "files", "snmb_to_wcs_compet.csv")
    compet_labelmap_file = os.path.join(BASE_PATH, "files", "compet_labels.txt")

    os.makedirs(eval_dir, exist_ok=True)

    '''

    _kale_block2 = '''
    # Fit model
    train(model_checkpoint_path,
          dataset,
          labelmap_file,
          model_type=EfficientNetB3,
          model_name='EfficientNetB3',
          images_size=(300, 300),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE_TRAIN)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/ecoinf_tests/.pipeline_1.ipynb.kale.marshal.dir")
    _kale_marshal.save(dataset, "dataset")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_data_loading_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_data_saving_block)
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/fit_model.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('fit_model')

    _kale_mlmdutils.call("mark_execution_complete")


def classification():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/ecoinf_tests/.pipeline_1.ipynb.kale.marshal.dir")
    dataset = _kale_marshal.load("dataset")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import os
    import pandas as pd
    from collections import defaultdict

    # Update to TF version compatible with CUDA 10.1
    import subprocess
    bashCommand = "pip3 uninstall tensorflow-gpu -y"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand = "pip3 install tensorflow-gpu==2.3.0"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    import tensorflow as tf

    print(f"Version de tf: { tf.__version__}")
    print(f"Lista de GPUs disponibles: {tf.config.list_physical_devices('GPU')}")

    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers.experimental import preprocessing
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, BatchNormalization, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import EfficientNetB3

    from conabio_ml.datasets.dataset import Partitions
    from conabio_ml.utils.dataset_utils import read_labelmap_file, write_labelmap_file

    from conabio_ml_vision.evaluator.evaluator import ImageClassificationEvaluator
    from conabio_ml_vision.evaluator.evaluator import ImageClassificationMetrics
    from conabio_ml_vision.datasets.datasets import ImageDataset, ImagePredictionDataset

    from conabio_ml.utils.logger import get_logger

    logger = get_logger(__name__)

    def classify_dataset(dataset,
                         classifs_csv,
                         labelmap,
                         model,
                         images_size,
                         batch_size,
                         partition=Partitions.TEST,
                         max_classifs=1):
        if os.path.isfile(classifs_csv):
            return ImagePredictionDataset.from_csv(source_path=classifs_csv)

        test_df = dataset.get_splitted(partitions=partition)
        test_batches = ImageDataGenerator().flow_from_dataframe(
            test_df,
            x_col="item",
            class_mode=None,
            target_size=images_size,
            batch_size=batch_size,
            shuffle=False,
            validate_filenames=False)

        preds = model.predict(test_batches, batch_size=batch_size)
        results = defaultdict(list)
        for i, (_, row) in enumerate(test_df.iterrows()):
            sorted_inds = [y[0] for y in sorted(enumerate(preds[i]), key=lambda x:x[1], reverse=True)]
            for k in range(max_classifs):
                ind = sorted_inds[k]
                results["item"].append(row["item"])
                results["label"].append(labelmap[ind])
                results["score"].append(preds[i][ind])
                results["image_id"].append(row["image_id"])
                results["id"].append(row["id"])
        data = pd.DataFrame(results)
        images_dir = dataset.get_images_dir()
        classifs_on_test_ds = ImagePredictionDataset(data, info={}, images_dir=images_dir)
        classifs_on_test_ds.to_csv(dest_path=classifs_csv)
        return classifs_on_test_ds


    def eval_multi(true_ds, pred_ds, eval_dir, partition=Partitions.TEST):
        res_eval = ImageClassificationEvaluator.eval(
            dataset_true=true_ds,
            dataset_pred=pred_ds,
            eval_config={
                'metrics_set': {
                    ImageClassificationMetrics.Sets.MULTICLASS: {
                        "average": 'macro',
                        "normalize": "true",
                        "zero_division": 1
                    }
                },
                "labels": true_ds.get_categories(),
                'partition': partition,
            })
        os.makedirs(eval_dir, exist_ok=True)
        res_eval.result_plots(dest_path=eval_dir, report=True, sample_counts_in_bars=True)
        res_eval.store_eval_metrics(dest_path=os.path.join(eval_dir, "results.json"))

    def train(model_checkpoint_path,
              dataset,
              labelmap_path,
              model_type,
              model_name,
              images_size,
              epochs,
              batch_size):
        if not os.path.isfile(model_checkpoint_path):
            df_train = dataset.get_splitted(partitions=Partitions.TRAIN)
            n_cats = dataset.get_num_categories()
            classes = dataset.get_classes()
            train_batches = ImageDataGenerator().flow_from_dataframe(df_train,
                                                                     x_col="item",
                                                                     y_col="label",
                                                                     classes=classes,
                                                                     target_size=images_size,
                                                                     batch_size=batch_size,
                                                                     validate_filenames=False)
            labelmap = {v: k for k, v in train_batches.class_indices.items()}
            write_labelmap_file(labelmap=labelmap, dest_path=labelmap_path)
            with tf.distribute.MirroredStrategy().scope():
                model = build_model(num_classes=n_cats, model_type=model_type,
                                    name=model_name, input_shape=(images_size[0], images_size[1], 3))
            logger.info(f"Training model {model_name}")
            model.fit(train_batches, epochs=epochs, verbose=1,
                      callbacks=[ModelCheckpoint(model_checkpoint_path)])
        else:
            pass # model = load_model_from_ckp(model_checkpoint_path)

        # return model

    def load_model_from_ckp(model_checkpoint_path):
        with tf.distribute.MirroredStrategy().scope():
            return load_model(model_checkpoint_path)

    def build_model(num_classes, model_type, name, input_shape):
        img_augmentation = Sequential(
            [
                preprocessing.RandomRotation(factor=0.15),
                preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                preprocessing.RandomFlip(mode="horizontal"),
                preprocessing.RandomContrast(factor=0.1),
            ],
            name="img_augmentation",
        )
        inputs = Input(shape=input_shape)
        x = img_augmentation(inputs)
        model = model_type(include_top=False, input_tensor=x, weights="imagenet")
        model.trainable = False
        x = GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = BatchNormalization()(x)
        x = Dropout(0.2, name="top_dropout")(x)
        outputs = Dense(num_classes, activation="softmax", name="pred")(x)
        model = Model(inputs, outputs, name=name)
        optimizer = Adam(learning_rate=1e-2)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model


    EPOCHS = 20
    RANDOM_STATE = 445
    BATCH_SIZE_EVAL = 16
    BATCH_SIZE_TRAIN = 32
    TRAIN_PERC = 0.8

    BASE_PATH = '/shared_volume/ecoinf_tests/kale_aws/'

    # Results
    results_path = os.path.join(BASE_PATH, "results", "pipeline_1")
    dataset_csv = os.path.join(results_path, "dataset.csv")
    classifs_csv = os.path.join(results_path, "classifs_on_test_part.csv")
    model_checkpoint_path = os.path.join(results_path, "efficientnet_b3_train_1.model.hdf5")
    labelmap_file = os.path.join(results_path, "labels.txt")
    eval_dir = os.path.join(results_path, "evaluation")
    # Data
    snmb_images_dir = os.path.join(BASE_PATH, 'data', "snmb")
    snmb_crops_dir = os.path.join(BASE_PATH, 'data', "snmb_crops")
    # Files
    snmb_json = os.path.join(BASE_PATH, "files", "snmb_2021_detection-bboxes.json")
    mappings_csv = os.path.join(BASE_PATH, "files", "snmb_to_wcs_compet.csv")
    compet_labelmap_file = os.path.join(BASE_PATH, "files", "compet_labels.txt")

    os.makedirs(eval_dir, exist_ok=True)

    '''

    _kale_block2 = '''
    # Inference on Test partition
    model = load_model_from_ckp(model_checkpoint_path)
    classifs_ds = classify_dataset(dataset,
                                   classifs_csv,
                                   read_labelmap_file(labelmap_file),
                                   model,
                                   images_size=(300, 300),
                                   batch_size=BATCH_SIZE_EVAL)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/ecoinf_tests/.pipeline_1.ipynb.kale.marshal.dir")
    _kale_marshal.save(classifs_ds, "classifs_ds")
    _kale_marshal.save(dataset, "dataset")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_data_loading_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_data_saving_block)
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/classification.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('classification')

    _kale_mlmdutils.call("mark_execution_complete")


def evaluation():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/ecoinf_tests/.pipeline_1.ipynb.kale.marshal.dir")
    classifs_ds = _kale_marshal.load("classifs_ds")
    dataset = _kale_marshal.load("dataset")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import os
    import pandas as pd
    from collections import defaultdict

    # Update to TF version compatible with CUDA 10.1
    import subprocess
    bashCommand = "pip3 uninstall tensorflow-gpu -y"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand = "pip3 install tensorflow-gpu==2.3.0"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    import tensorflow as tf

    print(f"Version de tf: { tf.__version__}")
    print(f"Lista de GPUs disponibles: {tf.config.list_physical_devices('GPU')}")

    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers.experimental import preprocessing
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, BatchNormalization, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import EfficientNetB3

    from conabio_ml.datasets.dataset import Partitions
    from conabio_ml.utils.dataset_utils import read_labelmap_file, write_labelmap_file

    from conabio_ml_vision.evaluator.evaluator import ImageClassificationEvaluator
    from conabio_ml_vision.evaluator.evaluator import ImageClassificationMetrics
    from conabio_ml_vision.datasets.datasets import ImageDataset, ImagePredictionDataset

    from conabio_ml.utils.logger import get_logger

    logger = get_logger(__name__)

    def classify_dataset(dataset,
                         classifs_csv,
                         labelmap,
                         model,
                         images_size,
                         batch_size,
                         partition=Partitions.TEST,
                         max_classifs=1):
        if os.path.isfile(classifs_csv):
            return ImagePredictionDataset.from_csv(source_path=classifs_csv)

        test_df = dataset.get_splitted(partitions=partition)
        test_batches = ImageDataGenerator().flow_from_dataframe(
            test_df,
            x_col="item",
            class_mode=None,
            target_size=images_size,
            batch_size=batch_size,
            shuffle=False,
            validate_filenames=False)

        preds = model.predict(test_batches, batch_size=batch_size)
        results = defaultdict(list)
        for i, (_, row) in enumerate(test_df.iterrows()):
            sorted_inds = [y[0] for y in sorted(enumerate(preds[i]), key=lambda x:x[1], reverse=True)]
            for k in range(max_classifs):
                ind = sorted_inds[k]
                results["item"].append(row["item"])
                results["label"].append(labelmap[ind])
                results["score"].append(preds[i][ind])
                results["image_id"].append(row["image_id"])
                results["id"].append(row["id"])
        data = pd.DataFrame(results)
        images_dir = dataset.get_images_dir()
        classifs_on_test_ds = ImagePredictionDataset(data, info={}, images_dir=images_dir)
        classifs_on_test_ds.to_csv(dest_path=classifs_csv)
        return classifs_on_test_ds


    def eval_multi(true_ds, pred_ds, eval_dir, partition=Partitions.TEST):
        res_eval = ImageClassificationEvaluator.eval(
            dataset_true=true_ds,
            dataset_pred=pred_ds,
            eval_config={
                'metrics_set': {
                    ImageClassificationMetrics.Sets.MULTICLASS: {
                        "average": 'macro',
                        "normalize": "true",
                        "zero_division": 1
                    }
                },
                "labels": true_ds.get_categories(),
                'partition': partition,
            })
        os.makedirs(eval_dir, exist_ok=True)
        res_eval.result_plots(dest_path=eval_dir, report=True, sample_counts_in_bars=True)
        res_eval.store_eval_metrics(dest_path=os.path.join(eval_dir, "results.json"))

    def train(model_checkpoint_path,
              dataset,
              labelmap_path,
              model_type,
              model_name,
              images_size,
              epochs,
              batch_size):
        if not os.path.isfile(model_checkpoint_path):
            df_train = dataset.get_splitted(partitions=Partitions.TRAIN)
            n_cats = dataset.get_num_categories()
            classes = dataset.get_classes()
            train_batches = ImageDataGenerator().flow_from_dataframe(df_train,
                                                                     x_col="item",
                                                                     y_col="label",
                                                                     classes=classes,
                                                                     target_size=images_size,
                                                                     batch_size=batch_size,
                                                                     validate_filenames=False)
            labelmap = {v: k for k, v in train_batches.class_indices.items()}
            write_labelmap_file(labelmap=labelmap, dest_path=labelmap_path)
            with tf.distribute.MirroredStrategy().scope():
                model = build_model(num_classes=n_cats, model_type=model_type,
                                    name=model_name, input_shape=(images_size[0], images_size[1], 3))
            logger.info(f"Training model {model_name}")
            model.fit(train_batches, epochs=epochs, verbose=1,
                      callbacks=[ModelCheckpoint(model_checkpoint_path)])
        else:
            pass # model = load_model_from_ckp(model_checkpoint_path)

        # return model

    def load_model_from_ckp(model_checkpoint_path):
        with tf.distribute.MirroredStrategy().scope():
            return load_model(model_checkpoint_path)

    def build_model(num_classes, model_type, name, input_shape):
        img_augmentation = Sequential(
            [
                preprocessing.RandomRotation(factor=0.15),
                preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                preprocessing.RandomFlip(mode="horizontal"),
                preprocessing.RandomContrast(factor=0.1),
            ],
            name="img_augmentation",
        )
        inputs = Input(shape=input_shape)
        x = img_augmentation(inputs)
        model = model_type(include_top=False, input_tensor=x, weights="imagenet")
        model.trainable = False
        x = GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = BatchNormalization()(x)
        x = Dropout(0.2, name="top_dropout")(x)
        outputs = Dense(num_classes, activation="softmax", name="pred")(x)
        model = Model(inputs, outputs, name=name)
        optimizer = Adam(learning_rate=1e-2)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model


    EPOCHS = 20
    RANDOM_STATE = 445
    BATCH_SIZE_EVAL = 16
    BATCH_SIZE_TRAIN = 32
    TRAIN_PERC = 0.8

    BASE_PATH = '/shared_volume/ecoinf_tests/kale_aws/'

    # Results
    results_path = os.path.join(BASE_PATH, "results", "pipeline_1")
    dataset_csv = os.path.join(results_path, "dataset.csv")
    classifs_csv = os.path.join(results_path, "classifs_on_test_part.csv")
    model_checkpoint_path = os.path.join(results_path, "efficientnet_b3_train_1.model.hdf5")
    labelmap_file = os.path.join(results_path, "labels.txt")
    eval_dir = os.path.join(results_path, "evaluation")
    # Data
    snmb_images_dir = os.path.join(BASE_PATH, 'data', "snmb")
    snmb_crops_dir = os.path.join(BASE_PATH, 'data', "snmb_crops")
    # Files
    snmb_json = os.path.join(BASE_PATH, "files", "snmb_2021_detection-bboxes.json")
    mappings_csv = os.path.join(BASE_PATH, "files", "snmb_to_wcs_compet.csv")
    compet_labelmap_file = os.path.join(BASE_PATH, "files", "compet_labels.txt")

    os.makedirs(eval_dir, exist_ok=True)

    '''

    _kale_block2 = '''
    # Evaluation
    eval_multi(dataset, classifs_ds, eval_dir)
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_data_loading_block,
                    _kale_block1,
                    _kale_block2,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/evaluation.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('evaluation')

    _kale_mlmdutils.call("mark_execution_complete")


_kale_dataset_creation_op = _kfp_components.func_to_container_op(
    dataset_creation, base_image='sipecam/ecoinf-kale-gpu:0.6.1')


_kale_fit_model_op = _kfp_components.func_to_container_op(
    fit_model, base_image='sipecam/ecoinf-kale-gpu:0.6.1')


_kale_classification_op = _kfp_components.func_to_container_op(
    classification, base_image='sipecam/ecoinf-kale-gpu:0.6.1')


_kale_evaluation_op = _kfp_components.func_to_container_op(
    evaluation, base_image='sipecam/ecoinf-kale-gpu:0.6.1')


@_kfp_dsl.pipeline(
    name='test-2-ab1wx',
    description='Prueba de entrenamiento-clasificacion-evaluacion. Aqui se usan las 10 K fotos'
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

    _kale_dataset_creation_task = _kale_dataset_creation_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    _kale_dataset_creation_task.container.working_dir = "//shared_volume/ecoinf_tests"
    _kale_dataset_creation_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update(
        {'dataset_creation': '/dataset_creation.html'})
    _kale_dataset_creation_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_dataset_creation_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_dataset_creation_task.dependent_names +
                       _kale_volume_step_names)
    _kale_dataset_creation_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_dataset_creation_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_fit_model_task = _kale_fit_model_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_dataset_creation_task)
    _kale_step_limits = {'nvidia.com/gpu': '1'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_fit_model_task.container.add_resource_limit(_kale_k, _kale_v)
    _kale_fit_model_task.container.working_dir = "//shared_volume/ecoinf_tests"
    _kale_fit_model_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'fit_model': '/fit_model.html'})
    _kale_fit_model_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_fit_model_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_fit_model_task.dependent_names +
                       _kale_volume_step_names)
    _kale_fit_model_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_fit_model_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_classification_task = _kale_classification_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_fit_model_task)
    _kale_step_limits = {'nvidia.com/gpu': '1'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_classification_task.container.add_resource_limit(
            _kale_k, _kale_v)
    _kale_classification_task.container.working_dir = "//shared_volume/ecoinf_tests"
    _kale_classification_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'classification': '/classification.html'})
    _kale_classification_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_classification_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_classification_task.dependent_names +
                       _kale_volume_step_names)
    _kale_classification_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_classification_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_evaluation_task = _kale_evaluation_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_classification_task)
    _kale_evaluation_task.container.working_dir = "//shared_volume/ecoinf_tests"
    _kale_evaluation_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'evaluation': '/evaluation.html'})
    _kale_evaluation_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_evaluation_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_evaluation_task.dependent_names +
                       _kale_volume_step_names)
    _kale_evaluation_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_evaluation_task.add_pod_annotation(
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
    run_name = generate_run_name('test-2-ab1wx')
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
