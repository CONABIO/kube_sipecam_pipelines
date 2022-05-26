import json

import kfp.dsl as _kfp_dsl
import kfp.components as _kfp_components

from collections import OrderedDict
from kubernetes import client as k8s_client


def get_audio_df(CUMULO: int, PAGESIZE: int, SAMPLERATE: float):
    _kale_pipeline_parameters_block = '''
    CUMULO = {}
    PAGESIZE = {}
    SAMPLERATE = {}
    '''.format(CUMULO, PAGESIZE, SAMPLERATE)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import datetime
    import hashlib
    import json
    import multiprocessing 
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import psutil
    import shutil
    import subprocess
    import time

    from dask.distributed import Client, LocalCluster
    from datetime import timedelta
    from dotenv import load_dotenv
    from os.path import exists as file_exists

    from yuntu.collection.methods import collection
    from yuntu.soundscape.hashers.crono import DEFAULT_HASHER_CONFIG
    from yuntu.soundscape.processors.indices.direct import ICOMPLEXITY, TAIL
    from yuntu.soundscape.pipelines.build_soundscape import CronoSoundscape, HASHER_CONFIG
    '''

    _kale_block2 = '''
    def create_results_folder_str(results_dir, cumulo, nodes_list, rec_list, dep_list): 
        # results directory
        os.makedirs(results_dir, exist_ok=True)
        # cumulus subdir
        cum_subdir = os.path.join(results_dir, str(cumulo))
        os.makedirs(cum_subdir, exist_ok=True)
        # node subdirs
        for node in nodes_list:
            node_subdir = os.path.join(cum_subdir, node)
            os.makedirs(node_subdir, exist_ok=True)
            # recorder subdirs
            for rec in rec_list:
                rec_subdir = os.path.join(node_subdir, rec)
                os.makedirs(rec_subdir, exist_ok=True)
                # deployment subdirs
                for dep in dep_list:
                    dep_subdir = os.path.join(rec_subdir, dep)
                    os.makedirs(dep_subdir, exist_ok=True)
                    
    def remove_empty_folders(path_abs):
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            if len(os.listdir(path)) == 0:
                os.rmdir(path)            
                
    def save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent="Null"):
        if product_type == "soundscape":
            product_name = "Soundscape"
            file_path = os.path.join(path, "hashed_soundscape.parquet")
            metadata_filename = os.path.join(path, "soundscape_metadata.json")
        elif product_type == "sequence":
            product_name = "Soundscape sequential plot"
            file_path = os.path.join(path, "soundscape_seq.png")
            metadata_filename = os.path.join(path, "soundscape_seq_metadata.json")
        elif product_type == "standard_deviation":
            product_name = "Soundscape standard deviation plot"
            file_path = os.path.join(path, "std_soundscape.png")
            metadata_filename = os.path.join(path, "std_soundscape_metadata.json")
        elif product_type == "mean":
            product_name = "Soundscape mean plot"
            file_path = os.path.join(path, "mean_soundscape.png")
            metadata_filename = os.path.join(path, "mean_soundscape_metadata.json")
        
        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": parent,
            "product_name": product_name,
            "product_configs": sc_config,
            "product_path": file_path,
            "product_spectrum": product_spectrum,
            "CumulusName": cumulus,
            "NodeCategoryIntegrity": node_category,
            "NomenclatureNode": node,
            "SerialNumber": recorder,
            "DateDeployment": deployment
        }
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def plot_soundscape(soundscape, product_type, product_spectrum, sc_config, path, 
                        cumulus, node, recorder, deployment, parent, indices, min_freq=None,
                      figsize=(20,15), plt_style='ggplot'):
        
        if min_freq:
            soundscape = soundscape[soundscape['min_freq']<=min_freq]
            
        if product_type == "sequence":
            file_path = os.path.join(path, "sequence.png")
            product_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
            
            plt.style.use(plt_style)
            fig, ax = plt.subplots(figsize=figsize)
            soundscape.sndscape.plot_sequence(rgb=indices, time_format='%Y-%m %H:%M', ax=ax)
            plt.xticks(rotation = 90)
            ax.grid(False)
            plt.tight_layout()
            plt.savefig(file_path) 
            plt.show()
            # save metadata
            save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent=parent)
             
        elif product_type == "standard_deviation":
            file_path = os.path.join(path, "std_soundscape.png")
            product_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
            
            plt.style.use(plt_style)
            fig, ax = plt.subplots(figsize=figsize)
            soundscape.sndscape.plot_cycle(rgb=indices, aggr="std", time_format='%H:%M', 
                                           xticks=24, ax=ax)
            plt.xticks(rotation = 90)
            ax.grid(False)
            plt.tight_layout() 
            plt.savefig(file_path)
            plt.show()
            
            # save metadata
            save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)     
            
        elif product_type == "mean": 
            file_path = os.path.join(path, "mean_soundscape.png")
            product_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
            
            plt.style.use(plt_style)
            fig, ax = plt.subplots(figsize=figsize)
            soundscape.sndscape.plot_cycle(rgb=indices, aggr="mean", time_format='%H:%M', 
                                           xticks=24, ax=ax)
            plt.xticks(rotation = 90)
            ax.grid(False)
            plt.tight_layout()
            plt.savefig(file_path)
            plt.show()
            
            # save metadata
            save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)    
            
        print(f"File saved at {file_path}")
    '''

    _kale_block3 = '''
    load_dotenv()
    DB_CONFIG = {
        'provider': 'alfresco',
        'config': {
            'api_url': 'https://api.conabio.gob.mx',
            'page_size': PAGESIZE,
            'api_key': os.getenv("X_API_KEY"),
            'base_filter': "+TYPE: \\"sipecamAudio:audiofileSipecam\\"",
            'recording_parser': {"path": "/shared_volume/audio/utils.py",
                                 "object_name": "parser"}
        }
    }

    COL_CONFIG = {
        "col_type": "alfresco",
        "db_config": DB_CONFIG
    }

    col = collection(**COL_CONFIG)
    query = f"(sipecam:CumulusName:\\"{CUMULO}\\") AND (sipecamAudio:SampleRate:{SAMPLERATE})"
    recs = col.get_recording_dataframe(query, limit=10, with_metadata = True, with_geometry = False)

    # include filtering columns for processing units
    recs.loc[:, "node"] = recs.metadata.apply(lambda x: x["entry"]["properties"]["sipecam:NomenclatureNode"])
    recs.loc[:, "recorder"] = recs.metadata.apply(lambda x: x["entry"]["properties"]["sipecamAudio:SerialNumber"]) 
    recs.loc[:, "deployment"] = recs.metadata.apply(lambda x: x["entry"]["path"]["name"].split("/audio")[0].split("/")[-1])
    recs.loc[:,"proc_unit"] = recs.apply(lambda x: (x["node"], x["recorder"], x["deployment"]), axis=1)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.soundscape-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
    _kale_marshal.save(recs, "recs")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    _kale_data_saving_block)
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/get_audio_df.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('get_audio_df')

    _kale_mlmdutils.call("mark_execution_complete")


def create_results_dirstruct(CUMULO: int, RESULTS_DIR: str):
    _kale_pipeline_parameters_block = '''
    CUMULO = {}
    RESULTS_DIR = "{}"
    '''.format(CUMULO, RESULTS_DIR)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.soundscape-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
    recs = _kale_marshal.load("recs")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import datetime
    import hashlib
    import json
    import multiprocessing 
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import psutil
    import shutil
    import subprocess
    import time

    from dask.distributed import Client, LocalCluster
    from datetime import timedelta
    from dotenv import load_dotenv
    from os.path import exists as file_exists

    from yuntu.collection.methods import collection
    from yuntu.soundscape.hashers.crono import DEFAULT_HASHER_CONFIG
    from yuntu.soundscape.processors.indices.direct import ICOMPLEXITY, TAIL
    from yuntu.soundscape.pipelines.build_soundscape import CronoSoundscape, HASHER_CONFIG
    '''

    _kale_block2 = '''
    def create_results_folder_str(results_dir, cumulo, nodes_list, rec_list, dep_list): 
        # results directory
        os.makedirs(results_dir, exist_ok=True)
        # cumulus subdir
        cum_subdir = os.path.join(results_dir, str(cumulo))
        os.makedirs(cum_subdir, exist_ok=True)
        # node subdirs
        for node in nodes_list:
            node_subdir = os.path.join(cum_subdir, node)
            os.makedirs(node_subdir, exist_ok=True)
            # recorder subdirs
            for rec in rec_list:
                rec_subdir = os.path.join(node_subdir, rec)
                os.makedirs(rec_subdir, exist_ok=True)
                # deployment subdirs
                for dep in dep_list:
                    dep_subdir = os.path.join(rec_subdir, dep)
                    os.makedirs(dep_subdir, exist_ok=True)
                    
    def remove_empty_folders(path_abs):
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            if len(os.listdir(path)) == 0:
                os.rmdir(path)            
                
    def save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent="Null"):
        if product_type == "soundscape":
            product_name = "Soundscape"
            file_path = os.path.join(path, "hashed_soundscape.parquet")
            metadata_filename = os.path.join(path, "soundscape_metadata.json")
        elif product_type == "sequence":
            product_name = "Soundscape sequential plot"
            file_path = os.path.join(path, "soundscape_seq.png")
            metadata_filename = os.path.join(path, "soundscape_seq_metadata.json")
        elif product_type == "standard_deviation":
            product_name = "Soundscape standard deviation plot"
            file_path = os.path.join(path, "std_soundscape.png")
            metadata_filename = os.path.join(path, "std_soundscape_metadata.json")
        elif product_type == "mean":
            product_name = "Soundscape mean plot"
            file_path = os.path.join(path, "mean_soundscape.png")
            metadata_filename = os.path.join(path, "mean_soundscape_metadata.json")
        
        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": parent,
            "product_name": product_name,
            "product_configs": sc_config,
            "product_path": file_path,
            "product_spectrum": product_spectrum,
            "CumulusName": cumulus,
            "NodeCategoryIntegrity": node_category,
            "NomenclatureNode": node,
            "SerialNumber": recorder,
            "DateDeployment": deployment
        }
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def plot_soundscape(soundscape, product_type, product_spectrum, sc_config, path, 
                        cumulus, node, recorder, deployment, parent, indices, min_freq=None,
                      figsize=(20,15), plt_style='ggplot'):
        
        if min_freq:
            soundscape = soundscape[soundscape['min_freq']<=min_freq]
            
        if product_type == "sequence":
            file_path = os.path.join(path, "sequence.png")
            product_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
            
            plt.style.use(plt_style)
            fig, ax = plt.subplots(figsize=figsize)
            soundscape.sndscape.plot_sequence(rgb=indices, time_format='%Y-%m %H:%M', ax=ax)
            plt.xticks(rotation = 90)
            ax.grid(False)
            plt.tight_layout()
            plt.savefig(file_path) 
            plt.show()
            # save metadata
            save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent=parent)
             
        elif product_type == "standard_deviation":
            file_path = os.path.join(path, "std_soundscape.png")
            product_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
            
            plt.style.use(plt_style)
            fig, ax = plt.subplots(figsize=figsize)
            soundscape.sndscape.plot_cycle(rgb=indices, aggr="std", time_format='%H:%M', 
                                           xticks=24, ax=ax)
            plt.xticks(rotation = 90)
            ax.grid(False)
            plt.tight_layout() 
            plt.savefig(file_path)
            plt.show()
            
            # save metadata
            save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)     
            
        elif product_type == "mean": 
            file_path = os.path.join(path, "mean_soundscape.png")
            product_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
            
            plt.style.use(plt_style)
            fig, ax = plt.subplots(figsize=figsize)
            soundscape.sndscape.plot_cycle(rgb=indices, aggr="mean", time_format='%H:%M', 
                                           xticks=24, ax=ax)
            plt.xticks(rotation = 90)
            ax.grid(False)
            plt.tight_layout()
            plt.savefig(file_path)
            plt.show()
            
            # save metadata
            save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)    
            
        print(f"File saved at {file_path}")
    '''

    _kale_block3 = '''
    # create results folder structure
    nodes_list = recs.node.unique()
    recorders_list = recs.recorder.unique()
    deployments_list = recs.deployment.unique()
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    create_results_folder_str(RESULTS_DIR, CUMULO, nodes_list, recorders_list, deployments_list)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.soundscape-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
    _kale_marshal.save(recs, "recs")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block, _kale_data_loading_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    _kale_data_saving_block)
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/create_results_dirstruct.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('create_results_dirstruct')

    _kale_mlmdutils.call("mark_execution_complete")


def compute_soundscapes(BLUE_IDX: str, CUMULO: int, FREQUENCY_BINS: int, FREQUENCY_LIMITS_LB: int, FREQUENCY_LIMITS_UB: int, GREEN_IDX: str, HASHER_TIME_MODULE: int, HASHER_TIME_UNIT: int, HASH_NAME: str, MIN_FREQ_SC: int, RED_IDX: str, RESULTS_DIR: str, SPECTRUM: str, THREADS_PER_WORKER: int, TIME_UNIT: int, WORK_DIR_PIPELINE: str):
    _kale_pipeline_parameters_block = '''
    BLUE_IDX = "{}"
    CUMULO = {}
    FREQUENCY_BINS = {}
    FREQUENCY_LIMITS_LB = {}
    FREQUENCY_LIMITS_UB = {}
    GREEN_IDX = "{}"
    HASHER_TIME_MODULE = {}
    HASHER_TIME_UNIT = {}
    HASH_NAME = "{}"
    MIN_FREQ_SC = {}
    RED_IDX = "{}"
    RESULTS_DIR = "{}"
    SPECTRUM = "{}"
    THREADS_PER_WORKER = {}
    TIME_UNIT = {}
    WORK_DIR_PIPELINE = "{}"
    '''.format(BLUE_IDX, CUMULO, FREQUENCY_BINS, FREQUENCY_LIMITS_LB, FREQUENCY_LIMITS_UB, GREEN_IDX, HASHER_TIME_MODULE, HASHER_TIME_UNIT, HASH_NAME, MIN_FREQ_SC, RED_IDX, RESULTS_DIR, SPECTRUM, THREADS_PER_WORKER, TIME_UNIT, WORK_DIR_PIPELINE)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.soundscape-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
    recs = _kale_marshal.load("recs")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import datetime
    import hashlib
    import json
    import multiprocessing 
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import psutil
    import shutil
    import subprocess
    import time

    from dask.distributed import Client, LocalCluster
    from datetime import timedelta
    from dotenv import load_dotenv
    from os.path import exists as file_exists

    from yuntu.collection.methods import collection
    from yuntu.soundscape.hashers.crono import DEFAULT_HASHER_CONFIG
    from yuntu.soundscape.processors.indices.direct import ICOMPLEXITY, TAIL
    from yuntu.soundscape.pipelines.build_soundscape import CronoSoundscape, HASHER_CONFIG
    '''

    _kale_block2 = '''
    def create_results_folder_str(results_dir, cumulo, nodes_list, rec_list, dep_list): 
        # results directory
        os.makedirs(results_dir, exist_ok=True)
        # cumulus subdir
        cum_subdir = os.path.join(results_dir, str(cumulo))
        os.makedirs(cum_subdir, exist_ok=True)
        # node subdirs
        for node in nodes_list:
            node_subdir = os.path.join(cum_subdir, node)
            os.makedirs(node_subdir, exist_ok=True)
            # recorder subdirs
            for rec in rec_list:
                rec_subdir = os.path.join(node_subdir, rec)
                os.makedirs(rec_subdir, exist_ok=True)
                # deployment subdirs
                for dep in dep_list:
                    dep_subdir = os.path.join(rec_subdir, dep)
                    os.makedirs(dep_subdir, exist_ok=True)
                    
    def remove_empty_folders(path_abs):
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            if len(os.listdir(path)) == 0:
                os.rmdir(path)            
                
    def save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent="Null"):
        if product_type == "soundscape":
            product_name = "Soundscape"
            file_path = os.path.join(path, "hashed_soundscape.parquet")
            metadata_filename = os.path.join(path, "soundscape_metadata.json")
        elif product_type == "sequence":
            product_name = "Soundscape sequential plot"
            file_path = os.path.join(path, "soundscape_seq.png")
            metadata_filename = os.path.join(path, "soundscape_seq_metadata.json")
        elif product_type == "standard_deviation":
            product_name = "Soundscape standard deviation plot"
            file_path = os.path.join(path, "std_soundscape.png")
            metadata_filename = os.path.join(path, "std_soundscape_metadata.json")
        elif product_type == "mean":
            product_name = "Soundscape mean plot"
            file_path = os.path.join(path, "mean_soundscape.png")
            metadata_filename = os.path.join(path, "mean_soundscape_metadata.json")
        
        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": parent,
            "product_name": product_name,
            "product_configs": sc_config,
            "product_path": file_path,
            "product_spectrum": product_spectrum,
            "CumulusName": cumulus,
            "NodeCategoryIntegrity": node_category,
            "NomenclatureNode": node,
            "SerialNumber": recorder,
            "DateDeployment": deployment
        }
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def plot_soundscape(soundscape, product_type, product_spectrum, sc_config, path, 
                        cumulus, node, recorder, deployment, parent, indices, min_freq=None,
                      figsize=(20,15), plt_style='ggplot'):
        
        if min_freq:
            soundscape = soundscape[soundscape['min_freq']<=min_freq]
            
        if product_type == "sequence":
            file_path = os.path.join(path, "sequence.png")
            product_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
            
            plt.style.use(plt_style)
            fig, ax = plt.subplots(figsize=figsize)
            soundscape.sndscape.plot_sequence(rgb=indices, time_format='%Y-%m %H:%M', ax=ax)
            plt.xticks(rotation = 90)
            ax.grid(False)
            plt.tight_layout()
            plt.savefig(file_path) 
            plt.show()
            # save metadata
            save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent=parent)
             
        elif product_type == "standard_deviation":
            file_path = os.path.join(path, "std_soundscape.png")
            product_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
            
            plt.style.use(plt_style)
            fig, ax = plt.subplots(figsize=figsize)
            soundscape.sndscape.plot_cycle(rgb=indices, aggr="std", time_format='%H:%M', 
                                           xticks=24, ax=ax)
            plt.xticks(rotation = 90)
            ax.grid(False)
            plt.tight_layout() 
            plt.savefig(file_path)
            plt.show()
            
            # save metadata
            save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)     
            
        elif product_type == "mean": 
            file_path = os.path.join(path, "mean_soundscape.png")
            product_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
            
            plt.style.use(plt_style)
            fig, ax = plt.subplots(figsize=figsize)
            soundscape.sndscape.plot_cycle(rgb=indices, aggr="mean", time_format='%H:%M', 
                                           xticks=24, ax=ax)
            plt.xticks(rotation = 90)
            ax.grid(False)
            plt.tight_layout()
            plt.savefig(file_path)
            plt.show()
            
            # save metadata
            save_metadata(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)    
            
        print(f"File saved at {file_path}")
    '''

    _kale_block3 = '''
    execution_info = {}
    start_time_compute_soundscapes = time.monotonic()

    # hasher config 
    hasher_config = {'module': {'object_name': 'yuntu.soundscape.hashers.crono.CronoHasher'},
                     'kwargs': {'time_utc_column': 'abs_start_time'}}

    hasher_config["kwargs"]["time_unit"] = HASHER_TIME_UNIT
    hasher_config["kwargs"]["time_module"] = HASHER_TIME_MODULE
    hasher_config["kwargs"]["start_tzone"] = "America/Mexico_City"
    hasher_config["kwargs"]["start_time"] = DEFAULT_HASHER_CONFIG["start_time"]
    hasher_config["kwargs"]["start_format"] = DEFAULT_HASHER_CONFIG["start_format"]
    hasher_config["kwargs"]["aware_start"] = None

    # soundscape config 
    slice_config  = dict(CronoSoundscape()["slice_config"].data)
    slice_config["time_unit"] = TIME_UNIT
    slice_config["frequency_bins"] = FREQUENCY_BINS
    slice_config["frequency_limits"] = (FREQUENCY_LIMITS_LB, FREQUENCY_LIMITS_UB)

    # FED configuration ["TOTAL", "CORE", "TAIL", "INFORMATION", "ICOMPLEXITY", "EXAG"]
    indices = CronoSoundscape()["indices"].data + [ICOMPLEXITY()]  + [TAIL()]

    # dask local cluster
    n_workers = int(0.95 * multiprocessing .cpu_count()) 
    cluster = LocalCluster(n_workers = n_workers, 
                           threads_per_worker = THREADS_PER_WORKER)
    client = Client(cluster)
    npartitions = len(client.ncores())

    # FEED
    FEED = {
        "slice_config": slice_config,
        "indices": indices,
        "hash_name": HASH_NAME,
        "hasher_config": hasher_config,
        "npartitions": npartitions
    }

    # adjust for metadata
    indexes_computed = ["TOTAL", "CORE", "TAIL", "INFORMATION", "ICOMPLEXITY", "EXAG"]
    FEED_metadata = FEED.copy()
    FEED_metadata["indices"] = indexes_computed

    plot_indices = [RED_IDX, GREEN_IDX, BLUE_IDX] # rgb order

    # soundscape per unit (cumulus-node-recorder-deployment_date)
    proc_units = recs.proc_unit.unique()

    for proc_unit in proc_units:
        try: 
            start_soundscape = time.monotonic()
            node, recorder, deployment = proc_unit
            print(f"* Processing: node {node} | recorder {recorder} | deployment date {deployment}")
            file_path = os.path.join(RESULTS_DIR, str(CUMULO), str(node), recorder, deployment)
            parent_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
            # soundscape = recs[recs.proc_unit == proc_unit].audio.get_soundscape(client=client, npartitions=n_workers, **soundscape_config)
            soundscape_data = recs[recs.proc_unit == proc_unit]
            pipeline = CronoSoundscape(name = "soundscape", work_dir = WORK_DIR_PIPELINE, recordings = soundscape_data)
            soundscape = pipeline["hashed_soundscape"].compute(client=client, feed=FEED)

            # sequence
            plot_soundscape(soundscape, "sequence", SPECTRUM, FEED_metadata, file_path,
                            CUMULO, node, recorder, deployment, parent_id, plot_indices, MIN_FREQ_SC)    
            # mean
            plot_soundscape(soundscape, "mean", SPECTRUM, FEED_metadata, file_path, 
                            CUMULO, node, recorder, deployment, parent_id, plot_indices, MIN_FREQ_SC)

            # standard deviation
            plot_soundscape(soundscape, "standard_deviation", SPECTRUM, FEED_metadata, file_path, 
                            CUMULO, node, recorder, deployment, parent_id, plot_indices, MIN_FREQ_SC)

            # save soundscape vector
            soundscape_path = os.path.join(file_path, "hashed_soundscape.parquet")
            # soundscape_orig_path = os.path.join(RESULTS_DIR, "get_soundscape/persist/hashed_soundscape.parquet") 
            soundscape_orig_path = '/shared_volume/audio/soundscape/persist/hashed_soundscape.parquet'
            shutil.move(soundscape_orig_path,soundscape_path)
            save_metadata(parent_id, "soundscape", SPECTRUM, FEED_metadata, file_path,
                          CUMULO, node, recorder, deployment)
            shutil.rmtree('/shared_volume/audio/soundscape')

        except:
            pass
        # restart client
        client.restart()

    # total time (soundscapes)
    execution_info["time_compute_soundscapes"] = str(timedelta(seconds=time.monotonic() - start_time_compute_soundscapes))
       
    client.close()
    cluster.close()

    # remove empty subdirectories
    remove_empty_folders(RESULTS_DIR)

    # execution info
    # arch info
    arch_info_dict = {}
    arch_info = subprocess.check_output("lscpu", shell=True).strip().decode().split("\\n")[:-1]
    arch_info = [x.replace(" ", "") for x in arch_info]

    for field in arch_info:
        key, value = field.split(":")
        arch_info_dict[key] = value
    arch_info_dict["RAM Memory (GB)"] = psutil.virtual_memory().total >> 30

    execution_info["arch_info_dict"] = arch_info_dict

    execution_path = os.path.join(RESULTS_DIR, "execution_info.json")

    if os.path.exists(execution_path):
        os.remove(execution_path)
    with open(execution_path, 'w', encoding='utf-8') as f:
        json.dump(execution_info, f, ensure_ascii=False, indent=4)    
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block, _kale_data_loading_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/compute_soundscapes.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('compute_soundscapes')

    _kale_mlmdutils.call("mark_execution_complete")


_kale_get_audio_df_op = _kfp_components.func_to_container_op(
    get_audio_df, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-cert:0.6.1_dev')


_kale_create_results_dirstruct_op = _kfp_components.func_to_container_op(
    create_results_dirstruct, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-cert:0.6.1_dev')


_kale_compute_soundscapes_op = _kfp_components.func_to_container_op(
    compute_soundscapes, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-cert:0.6.1_dev')


@_kfp_dsl.pipeline(
    name='sound-scape-nod-rec-dep-wyqgj',
    description='Computes Sipecam Soundscapes using cumulus, node, recorder and deployment'
)
def auto_generated_pipeline(BLUE_IDX='CORE', CUMULO='92', FREQUENCY_BINS='96', FREQUENCY_LIMITS_LB='0', FREQUENCY_LIMITS_UB='24000', GREEN_IDX='INFORMATION', HASHER_TIME_MODULE='48', HASHER_TIME_UNIT='1800', HASH_NAME='crono_hash_30m', MIN_FREQ_SC='10000', PAGESIZE='1000', RED_IDX='EXAG', RESULTS_DIR='/shared_volume/audio/soundscapes', SAMPLERATE='48000.0', SPECTRUM='Audible', THREADS_PER_WORKER='2', TIME_UNIT='30', WORK_DIR_PIPELINE='.', vol_shared_volume='hostpath-pvc'):
    _kale_pvolumes_dict = OrderedDict()
    _kale_volume_step_names = []
    _kale_volume_name_parameters = []

    _kale_annotations = {}

    _kale_volume = _kfp_dsl.PipelineVolume(pvc=vol_shared_volume)

    _kale_pvolumes_dict['/shared_volume'] = _kale_volume

    _kale_volume_step_names.sort()
    _kale_volume_name_parameters.sort()

    _kale_get_audio_df_task = _kale_get_audio_df_op(CUMULO, PAGESIZE, SAMPLERATE)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    _kale_get_audio_df_task.container.working_dir = "//shared_volume/audio"
    _kale_get_audio_df_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'get_audio_df': '/get_audio_df.html'})
    _kale_get_audio_df_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_get_audio_df_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_get_audio_df_task.dependent_names +
                       _kale_volume_step_names)
    _kale_get_audio_df_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_get_audio_df_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_create_results_dirstruct_task = _kale_create_results_dirstruct_op(CUMULO, RESULTS_DIR)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_get_audio_df_task)
    _kale_create_results_dirstruct_task.container.working_dir = "//shared_volume/audio"
    _kale_create_results_dirstruct_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update(
        {'create_results_dirstruct': '/create_results_dirstruct.html'})
    _kale_create_results_dirstruct_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_create_results_dirstruct_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_create_results_dirstruct_task.dependent_names +
                       _kale_volume_step_names)
    _kale_create_results_dirstruct_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_create_results_dirstruct_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_compute_soundscapes_task = _kale_compute_soundscapes_op(BLUE_IDX, CUMULO, FREQUENCY_BINS, FREQUENCY_LIMITS_LB, FREQUENCY_LIMITS_UB, GREEN_IDX, HASHER_TIME_MODULE, HASHER_TIME_UNIT, HASH_NAME, MIN_FREQ_SC, RED_IDX, RESULTS_DIR, SPECTRUM, THREADS_PER_WORKER, TIME_UNIT, WORK_DIR_PIPELINE)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_create_results_dirstruct_task, _kale_get_audio_df_task)
    _kale_compute_soundscapes_task.container.working_dir = "//shared_volume/audio"
    _kale_compute_soundscapes_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update(
        {'compute_soundscapes': '/compute_soundscapes.html'})
    _kale_compute_soundscapes_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_compute_soundscapes_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_compute_soundscapes_task.dependent_names +
                       _kale_volume_step_names)
    _kale_compute_soundscapes_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_compute_soundscapes_task.add_pod_annotation(
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
    run_name = generate_run_name('sound-scape-nod-rec-dep-tons1')
    pipeline_parameters = {"CUMULO": 92,
                           "SAMPLERATE": 48000.0,
                           "PAGESIZE": 1000,
                           "RED_IDX": "EXAG",
                           "GREEN_IDX": "INFORMATION",
                           "BLUE_IDX": "CORE",
                           "MIN_FREQ_SC": 10000,
                           "WORK_DIR_PIPELINE": ".",
                           "SPECTRUM": "Audible",
                           "TIME_UNIT": 30,
                           "FREQUENCY_BINS": 96,
                           "FREQUENCY_LIMITS_LB": 0,
                           "FREQUENCY_LIMITS_UB": 24000,
                           "HASHER_TIME_UNIT": 1800,
                           "HASHER_TIME_MODULE": 48,
                           "THREADS_PER_WORKER": 1,
                           "RESULTS_DIR": '/shared_volume/audio/soundscapes'}
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, pipeline_parameters)
    import time
    time.sleep(180)
    pipeline_parameters = {"CUMULO": 95,
                           "SAMPLERATE": 48000.0,
                           "PAGESIZE": 1000,
                           "RED_IDX": "EXAG",
                           "GREEN_IDX": "INFORMATION",
                           "BLUE_IDX": "CORE",
                           "MIN_FREQ_SC": 10000,
                           "WORK_DIR_PIPELINE": ".",
                           "SPECTRUM": "Audible",
                           "TIME_UNIT": 30,
                           "FREQUENCY_BINS": 96,
                           "FREQUENCY_LIMITS_LB": 0,
                           "FREQUENCY_LIMITS_UB": 24000,
                           "HASHER_TIME_UNIT": 1800,
                           "HASHER_TIME_MODULE": 48,
                           "THREADS_PER_WORKER": 1,
                           "RESULTS_DIR": '/shared_volume/audio/soundscapes'}
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, pipeline_parameters)
