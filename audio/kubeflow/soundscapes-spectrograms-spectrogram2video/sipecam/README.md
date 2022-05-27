<p float="left" align="center">
  <img src="https://i.imgur.com/fUFVFBl.png" width="300" />
  <img src="https://i.imgur.com/qymEZ1n.png" width="120" /> 
</p>

# Objective

This _pipeline_ aims to generate soundscapes using audio data from the Sitios Permanentes de Calibración y Monitoreo de la Biodiversidad (SiPeCaM) project. In particular, we are processing data using the following fields:

- "Cumulus"  (int)

Using cumulus information, we group audio data using the following fields to generate each processing unit:

- Cumulus
- Node
- Recorder
- Deployment date

## Outputs

The following products are generated for each processing unit: 

- `hashed_soundscape.parquet`: File with _dataframe_ format including indices to compute soundscapes.
- `soundscape_metadata.json`: metadata associated to `hashed_soundscape.parquet`
- `sequence.png`: Soundscape sequential plot
- `sequence_metadata.json`:  metadata associated to `sequence.png`
- `mean_soundscape.png`:  Soundscape mean plot
- `mean_soundscape_metadata.json`:  metadata associated to `mean_soundscape.png`
- `std_soundscape.png`: Soundscape standard deviation plot
- `std_soundscape_metadata.json`:  metadata associated to `std_soundscape.png`
- `idx_spectrogram.png`: Spectrogram associated to audio file with `idx` identifier.
- `idx_spectrogram_metadata.json`: metadata associated to `idx_spectrogram.png`
- `idx_spectrogram_video.mp4`: Spectrogram video (including audio) associated to audio file with `idx` identifier.
- `idx_spectrogram_video_metadata.json`: metadata associated to `idx_spectrogram_video.mp4`

## Pipeline Steps:

<p align="center">
  <img src="https://i.imgur.com/YxryqRX.png"/>
</p>



Our pipeline includes the following steps:

- `get-audio-df`:  Load alfresco credentials, get audio `dataframe` from `audio-collection`
- `create-results-dirstruct`: Create folder structure to store results
- `compute-soundscapes` : Process audio data to compute soundscapes at cumulus-node-recording-deployment level
- `spec-n-specvid`: Process spectrograms and spectrograms videos for samples of two audio at at cumulus-node-recording-deployment levels.
- `upload-to-alfresco`: Upload processed files to `alfresco-site` : `sipecam-soundscape`
- `upload-alfresco-model-data`: Upload files' metadata to `alfresco-model`

The pipeline is computed using [kale](https://github.com/kubeflow-kale/kale)-[kubeflow](https://www.kubeflow.org/).

- [`sound-scape-nod-rec-dep-tons1.kale.py`](https://github.com/CONABIO/kube_sipecam_pipelines/blob/main/audio/kubeflow/soundscapes/sipecam/.kale/sound-scape-nod-rec-dep-tons1.kale.py)  uses the following `docker` image (no GPU usage):

[sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-cert:0.6.1_dev](https://github.com/CONABIO/yuntu-private/blob/alfresco/dockerfiles/tensorflow-yuntu-dask/0.6.1_dev/Dockerfile)

## Requirements

1. `.env` file

	To run this pipeline must exist an environment file (`.env`) in the root directory of the project. The file contains the corresponding _alfresco_ credentials, and follows the structure below:

```bash
ALFRESCO_URL="<alfresco-url>"
API_ENDPOINT="<api-endpoint>"
ALFRESCO_USER=<alfresco-user>
ALFRESCO_PASSWORD=<password>
X_API_KEY=<x-api-key>
```

2. The `utils.py` module provided in this directory must be located in the following path: `/shared_volume/audio/utils.py`.


## Metadata examples:

- `soundscape_metadata.json`

```json
{
    "product_id": "c58892f5d93ec1adf0b17325c6c17f4a",
    "product_parent": "Null",
    "product_name": "Soundscape",
    "product_configs": {
        "slice_config": {
            "time_unit": 30,
            "frequency_bins": 96,
            "frequency_limits": [
                0,
                24000
            ],
            "feature_type": "spectrogram",
            "feature_config": {
                "n_fft": 1024,
                "hop_length": 512,
                "window_function": "hann"
            },
            "frequency_hop": 1.0,
            "time_hop": 1.0
        },
        "indices": [
            "TOTAL",
            "CORE",
            "TAIL",
            "INFORMATION",
            "ICOMPLEXITY",
            "EXAG"
        ],
        "hash_name": "crono_hash_30m",
        "hasher_config": {
            "module": {
                "object_name": "yuntu.soundscape.hashers.crono.CronoHasher"
            },
            "kwargs": {
                "time_utc_column": "abs_start_time",
                "time_unit": 1800,
                "time_module": 48,
                "start_tzone": "America/Mexico_City",
                "start_time": "2018-01-01 00:00:00",
                "start_format": "%Y-%m-%d %H:%M:%S",
                "aware_start": null
            }
        },
        "npartitions": 15
    },
    "product_path": "/shared_volume/audio/soundscapes/92/1_95_1_1350/248C7D075B1F7CDB/2021-08-04/hashed_soundscape.parquet",
    "product_spectrum": "Audible",
    "CumulusName": 92,
    "NodeCategoryIntegrity": "Integro",
    "NomenclatureNode": "1_95_1_1350",
    "SerialNumber": "248C7D075B1F7CDB",
    "DateDeployment": "2021-08-04"
}
```



- `sequence_metadata.json`

```json
{
    "product_id": "9f80823bc2f694e48c659be973fa9a0e",
    "product_parent": "c58892f5d93ec1adf0b17325c6c17f4a",
    "product_name": "Soundscape sequential plot",
    "product_configs": {
        "slice_config": {
            "time_unit": 30,
            "frequency_bins": 96,
            "frequency_limits": [
                0,
                24000
            ],
            "feature_type": "spectrogram",
            "feature_config": {
                "n_fft": 1024,
                "hop_length": 512,
                "window_function": "hann"
            },
            "frequency_hop": 1.0,
            "time_hop": 1.0
        },
        "indices": [
            "TOTAL",
            "CORE",
            "TAIL",
            "INFORMATION",
            "ICOMPLEXITY",
            "EXAG"
        ],
        "hash_name": "crono_hash_30m",
        "hasher_config": {
            "module": {
                "object_name": "yuntu.soundscape.hashers.crono.CronoHasher"
            },
            "kwargs": {
                "time_utc_column": "abs_start_time",
                "time_unit": 1800,
                "time_module": 48,
                "start_tzone": "America/Mexico_City",
                "start_time": "2018-01-01 00:00:00",
                "start_format": "%Y-%m-%d %H:%M:%S",
                "aware_start": null
            }
        },
        "npartitions": 15
    },
    "product_path": "/shared_volume/audio/soundscapes/92/1_95_1_1350/248C7D075B1F7CDB/2021-08-04/soundscape_seq.png",
    "product_spectrum": "Audible",
    "CumulusName": 92,
    "NodeCategoryIntegrity": "Integro",
    "NomenclatureNode": "1_95_1_1350",
    "SerialNumber": "248C7D075B1F7CDB",
    "DateDeployment": "2021-08-04"
}
```



