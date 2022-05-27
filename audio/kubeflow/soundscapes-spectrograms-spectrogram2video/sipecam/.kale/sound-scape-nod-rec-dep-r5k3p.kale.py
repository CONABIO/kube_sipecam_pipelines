import json

import kfp.dsl as _kfp_dsl
import kfp.components as _kfp_components

from collections import OrderedDict
from kubernetes import client as k8s_client


def get_audio_df(AUTH_ENDPOINT: str, BASE_ENDPOINT: str, CUMULO: int, LIMIT: int, PAGESIZE: int, SAMPLERATE: float):
    _kale_pipeline_parameters_block = '''
    AUTH_ENDPOINT = "{}"
    BASE_ENDPOINT = "{}"
    CUMULO = {}
    LIMIT = {}
    PAGESIZE = {}
    SAMPLERATE = {}
    '''.format(AUTH_ENDPOINT, BASE_ENDPOINT, CUMULO, LIMIT, PAGESIZE, SAMPLERATE)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import base64
    import datetime
    import glob
    import hashlib
    import io
    import itertools
    import json
    import matplotlib.pyplot as plt
    import multiprocessing 
    import numpy as np
    import os
    import pandas as pd
    import psutil
    import requests
    import shutil
    import subprocess
    import time
    import warnings

    from dask.distributed import Client, LocalCluster
    from datetime import timedelta
    from dotenv import load_dotenv
    from matplotlib import cm
    from moviepy.editor import concatenate, VideoFileClip, AudioFileClip
    from moviepy.audio.AudioClip import AudioArrayClip
    from moviepy.video.VideoClip import ImageClip
    from os.path import exists as file_exists
    from PIL import Image
    from skimage.transform import resize

    from yuntu import Audio
    from yuntu.soundscape.utils import aware_time
    from yuntu.collection.methods import collection
    from yuntu.soundscape.hashers.crono import DEFAULT_HASHER_CONFIG
    from yuntu.soundscape.processors.indices.direct import ICOMPLEXITY, TAIL
    from yuntu.soundscape.pipelines.build_soundscape import CronoSoundscape, HASHER_CONFIG
    '''

    _kale_block2 = '''
    def audio2video(audio_id,
                    identifier,
                    audio_df,
                    save_path_folder,
                    product_spectrum,
                    cumulus,
                    abs_start=None,
                    fps=60,
                    spec_configs={'hop_length': 512, 'n_fft': 1024, 'window_function': 'hann'},
                    rate=24,
                    frame_duration=3.0,
                    min_freq=0,
                    max_freq=None,
                    cmap="Greys",
                    figsize=(5, 8),
                    dpi=100,
                    bands=None):
        \'\'\'Takes and audio object and produces a mp4 video of the spectrogram with audio\'\'\'

        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        id_audio = sub_audio_df['id'].values[0]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        audio = sub_audio_df.audio[0].cut(0,1)
        
        colormap = cm.get_cmap(cmap)
        duration = audio.duration
        step = 1/rate
        start = -(frame_duration/2.0)
        stop = start + frame_duration
        clips = []
        last_stop = None

        if max_freq is None:
            max_freq = audio.samplerate / 2.0
        
        if min_freq is None:
            min_freq = 0
        

        with audio.features.db_spectrogram(**spec_configs) as spec:
            min_spec = np.amin(spec)
            max_spec = np.amax(spec)
            spec_range = (max_spec-min_spec)
            
            while stop <= duration+(frame_duration/2.0):
                clip = produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start, colormap,
                                    min_spec, spec_range, figsize, dpi, bands=bands)
                clips.append(clip)
                
                if start + step + frame_duration > duration:
                    last_stop = stop

                start = start + step
                stop = start + frame_duration
        
        video = concatenate(clips)
        # edaudio = AudioArrayClip(audio_array, fps=audio.samplerate)
        edaudio = AudioFileClip(audio.path).set_end(audio.duration)
        video = video.set_audio(edaudio)
        file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram_video.mp4")
        video.write_videofile(file_path, fps=fps)
        
        save_metadata_videoclip(id_audio, identifier, product_spectrum,
                      save_path_folder, cumulus, node, recorder, deployment, 0.0, audio.duration)
        video.close()
        edaudio.close()
        
        for c in clips:
            c.close()

    def change_type_sipecam_sc(session, root_folder_id, path, file_type):
        if file_type == "sequence.png":
            metadata_name = "soundscape_seq_metadata.json"
            aggr_type = "None"
        elif file_type == "mean_soundscape.png":
            metadata_name = "mean_soundscape_metadata.json"
            aggr_type = "Mean"
        elif file_type ==  "std_soundscape.png":
            metadata_name = "std_soundscape_metadata.json" 
            aggr_type = "Standard deviation"
        elif file_type == "hashed_soundscape.parquet":
            metadata_name = "soundscape_metadata.json.json" 
            aggr_type = "None"

        try:
            semi_path = path.split("soundscapes/")[-1]
            semi_path_file = os.path.join(semi_path, file_type)
            local_path_file_metadata = os.path.join(path, metadata_name)
            print(f"Changing type for {os.path.join(semi_path_file)}")
            alfresco_path = os.path.join("/Company Home/Sites/sipecam-soundscape/documentLibrary/", semi_path)
            response = session.get(
                os.getenv("ALFRESCO_URL")
                + BASE_ENDPOINT
                + "/nodes/"
                + root_folder_id
                + "/children?relativePath="+semi_path_file+"&include=aspectNames&skipCount=0&maxItems=1"
            )

            # error flag
            is_error = False

            # if request is successful then continue
            if response.status_code == 200:

                data_file = open(local_path_file_metadata)
                data_json = json.load(data_file)
                response_entries = response.json()["list"]["entries"][0]

                if response_entries["entry"]["isFile"]:

                    prop_dict = {}
                    # map properties
                    prop_dict["soundscape:CumulusName"] = str(data_json["CumulusName"])
                    prop_dict["soundscape:DateDeployment"] = data_json["DateDeployment"]
                    prop_dict["soundscape:NodeCategoryIntegrity"] = str(data_json["NodeCategoryIntegrity"])
                    prop_dict["soundscape:NomenclatureNode"] = str(data_json["NomenclatureNode"])
                    prop_dict["soundscape:SerialNumber"] = str(data_json["SerialNumber"])
                    prop_dict["soundscape:aggr"] = str(aggr_type)
                    prop_dict["soundscape:cycle_config_aware_start"] = str(data_json["product_configs"]['hasher_config']['kwargs']['aware_start'])
                    prop_dict["soundscape:cycle_config_start_format"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_format'])
                    prop_dict["soundscape:cycle_config_start_time"] = data_json["product_configs"]['hasher_config']['kwargs']['start_time'] #
                    prop_dict["soundscape:cycle_config_start_tzone"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_tzone'])
                    prop_dict["soundscape:cycle_config_time_module"] = int(data_json["product_configs"]['hasher_config']['kwargs']['time_module'])
                    prop_dict["soundscape:cycle_config_time_unit"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_unit'])
                    prop_dict["soundscape:cycle_config_time_utc_column"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_utc_column'])
                    prop_dict["soundscape:frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:frequency_hop"] = int(data_json["product_configs"]["slice_config"]["frequency_hop"])
                    prop_dict["soundscape:frequency_limits"] = str(data_json["product_configs"]["slice_config"]["frequency_limits"])
                    prop_dict["soundscape:hash_name"] = str(data_json["product_configs"]["hash_name"])
                    prop_dict["soundscape:hop_length"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["hop_length"])
                    prop_dict["soundscape:indices"] = str(data_json['product_configs']['indices'])
                    prop_dict["soundscape:n_fft"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["n_fft"])
                    prop_dict["soundscape:npartitions"] = int(data_json["product_configs"]['npartitions'])
                    prop_dict["soundscape:product_id"] = str(data_json["product_id"])
                    prop_dict["soundscape:product_name"] = str(data_json["product_name"])
                    prop_dict["soundscape:product_parent"] = str(data_json["product_parent"])
                    prop_dict["soundscape:product_path"] = str(alfresco_path)
                    prop_dict["soundscape:product_spectrum"] = str(data_json["product_spectrum"])
                    prop_dict["soundscape:slice_config_feature_type"] = str(data_json["product_configs"]["slice_config"]["feature_type"])
                    prop_dict["soundscape:slice_config_frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:slice_config_time_unit"] = int(data_json["product_configs"]["slice_config"]["time_unit"])
                    prop_dict["soundscape:time_hop"] = int(data_json["product_configs"]["slice_config"]["time_hop"])
                    prop_dict["soundscape:window_function"] = str(data_json["product_configs"]["slice_config"]["feature_config"]["window_function"])


                    aspects = response_entries["entry"]["aspectNames"]

                    data = {"aspectNames": aspects, "nodeType": 'soundscape:product', "properties": prop_dict}

                    # update properties request
                    update = session.put(
                        os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT
                        + "/nodes/"
                        + response_entries["entry"]["id"],
                        data=json.dumps(data),
                    )
                    print(update.json())
                    if update.status_code == 200:
                        print("Updated " + response_entries["entry"]["id"])
        except Exception as e:
            print("Could not add any aspect to this file: ", e)
            
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
                    
    def distance_to_mean(vector, mean):
        \'\'\'Return euclidean distance to mean\'\'\'
        return np.sqrt(np.sum(np.square(mean - vector)))

    def find_subfolders(path_abs):
        subdir_list = []
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            len_path = path.split("/")
            if len(len_path) == 8:
                subdir_list.append(path)  
                
        return subdir_list

    def get_audio_ids(soundscape_path, indices):
        df = pd.read_parquet(os.path.join(soundscape_path, "hashed_soundscape.parquet"))

        with open(os.path.join(soundscape_path, "soundscape_metadata.json")) as f:
            metadata = json.load(f)
            f.close()

        # indices = metadata["product_configs"]["indices"]
        # indices = ["EXAG", "ICOMPLEXITY", "CORE"]
        hash_name = metadata["product_configs"]["hash_name"]
        cycle_config = metadata["product_configs"]["hasher_config"]["kwargs"]
        time_unit = cycle_config["time_unit"]
        zero_t = aware_time( cycle_config["start_time"], cycle_config["start_tzone"], cycle_config["start_format"]) 
        
        # sample
        samples_df = get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=3)
        sub_df = samples_df[samples_df.crono_hash_30m == 8]
        audio_id_list = list(sub_df["id"].unique())
        
        return audio_id_list

    def get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=5):
        \'\'\'Return dataframe of 'nsamples' samples for each tag in 'hash_name' column that are closest to the mean vector by tag\'\'\'
        proj_df = df[(df.max_freq <= 10000)]
        crono_tags = proj_df.crono_hash_30m.unique()
        proj_df.loc[: , f"{hash_name}_time"] = proj_df[hash_name].apply(lambda x: zero_t + datetime.timedelta(seconds=float(x*time_unit)))
        vectors = vectorize_soundscape(proj_df, hash_name, indices)
        min_index_vector = np.amin(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        max_index_vector = np.amax(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        index_range = (max_index_vector - min_index_vector)
        vectors.loc[:, "normalized_index_vector"] = vectors.index_vector.apply(lambda x: (x-min_index_vector)/index_range)
        all_samples = []

        for crono_tag in crono_tags:
            unit_vectors = vectors[vectors[hash_name] == crono_tag]
            mean_unit_vector = unit_vectors.normalized_index_vector.mean()
            unit_vectors.loc[:, "distance"] = unit_vectors.normalized_index_vector.apply(lambda x: distance_to_mean(x, mean_unit_vector))
            all_samples.append(unit_vectors.sort_values(by="distance").head(nsamples))

        return pd.concat(all_samples)

    def get_vectors(group, indices):
        \'\'\'Return array of indices by frequency\'\'\'
        return group.sort_values(by="max_freq")[indices].values

    def login():
        """
        Tries a login to alfresco api and returns a session
        object with credentials 
        Returns: 
            session (Session):  A session object to make 
                                requests to zendro.
        """
        try:
            auth = {
                "userId": os.getenv("ALFRESCO_USER"),
                "password": os.getenv("ALFRESCO_PASSWORD"),
            }

            login = requests.post(os.getenv("ALFRESCO_URL") + AUTH_ENDPOINT + "/tickets",data=json.dumps(auth))

            base64_login = base64.b64encode(bytes(login.json()["entry"]["id"], 'utf-8')).decode()

            # se crea un objeto de Session para hacer requests
            session = requests.Session()
            # se establece bearer token
            session.headers.update({'Authorization': 'Basic ' + base64_login})

            return session
        except Exception as e:
            print("Login failed: ",e)

    def plot_spectrogram(audio_id, identifier, audio_df, save_path_folder, spectrum, cumulus):
        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        # plot
        fig, ax = plt.subplots(2,1,figsize=(20,10), sharex=True)
        sub_audio_df.audio[0].plot(ax=ax[0], color='grey')
        sub_audio_df.audio[0].features.db_spectrogram().plot(ax=ax[1])
        ax[0].set_ylabel('Amplitude')
        ax[0].grid(False)
        ax[1].set_ylabel('F (KHz)')
        ax[1].set_xlabel('Time (seconds)')
        fig.text(0.75, 0.04, f"Cumulus: {cumulus} - Node: {node} - Recorder: {recorder}", va='center')
        plt.show()
        if save_path_folder:
            file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram.png")
            fig.savefig(file_path)
        
        save_metadata_spectrogram(audio_id, identifier, spectrum, save_path_folder, 
                                  cumulus, node, recorder, deployment, parent="Null")
        
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)    
            
        print(f"File saved at {file_path}")
        
    def produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start=None, 
                     colormap=cm.get_cmap("Greys"), min_spec=0, spec_range=1.0, figsize=(5, 4), 
                     dpi=100, bands=None):
        \'\'\'Takes an individual frame and produces an image with references\'\'\'
        frame = spec.cut_array(start_time=start, end_time=stop, min_freq=min_freq, max_freq=max_freq, pad=True)
        plt.style.use('dark_background')
        frame = np.flip((frame - min_spec)/spec_range, axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frame, cmap=colormap, extent=[0, frame_duration, min_freq/1000, max_freq/1000], 
                  aspect="auto", vmin = 0, vmax = 1.0)

        if bands is not None:
            band_arr = np.flip(resize(np.expand_dims(bands, axis=1), (frame.shape[0], frame.shape[1])), axis=0)
            ax.imshow(band_arr, extent=[0, frame_duration, min_freq/1000, max_freq/1000], aspect="auto", vmin = 0, 
                      vmax = 1.0, alpha=0.5)

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        mid = frame_duration/2.0
        ax.axvline(x=mid, color="red")
        ax.set_ylabel('F (kHz)')
        ax.set_xticks([])
        ax.set_xticks([], minor=True)

        if abs_start is not None:
            time_text = (abs_start + datetime.timedelta(seconds=start+mid)).strftime('%H:%M:%S.%f').strip()[:-4]
            ax.text(mid-0.3, -0.6, time_text)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        im = Image.open(buf)
        im.format = "PNG"
        plt.close(fig)

        return ImageClip(np.asarray(im),
                         duration=step)
        
    def remove_empty_folders(path_abs):
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            if len(os.listdir(path)) == 0:
                os.rmdir(path)            
                
    def save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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

    def save_metadata_spectrogram(product_id, identifier, product_spectrum,
                      path, cumulus, node, recorder, deployment, parent="Null"):
        product_name = "Spectrogram"
        file_path = os.path.join(path, f"{identifier}_spectrogram.png")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": parent,
            "product_name": product_name,
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
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def save_metadata_videoclip(product_id, identifier, product_spectrum, path, cumulus, node, recorder, 
                                deployment, clip_start, clip_end, parent="Null"):
        product_name = "spectrogram_video"
        file_path = os.path.join(path, f"{identifier}_spectrogram_video.mp4")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_video_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": "Null",
            "product_name": product_name,
            "product_path": file_path,
            "product_spectrum": product_spectrum,
            "CumulusName": cumulus,
            "NodeCategoryIntegrity": node_category,
            "NomenclatureNode": node,
            "SerialNumber": recorder,
            "DateDeployment": deployment,
            "ClipStart": clip_start,
            "ClipEnd": clip_end
        }
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def upload(session, node_id, data, file):
        """
        Uploads a file to a specific folder.
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            data (dict):                Dict that contains file options
            file (object):              File to upload
        
        Returns:
            (list):     A list containing status code and status data
        """

        try:
            response = session.post(os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT + "/nodes/" + node_id + "/children",
                        data = data,
                        files = file
                        )
                        
            return [response.json(), response.status_code];
        except Exception as e: 
            print("File " + data["name"] + " could not be uploaded: ", e)

    def upload_files(file_patterns ,session, node_id, dir_path, recursive, file_identifier=""):
        """
        Uploads the files stored in a specific dir
        to alfresco
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            dir_path (string):          The name and path of the dir where files are stored
            recursive (boolean):        A boolean to know if upload  must be recursive
                                        in the specifed dir, and should preserve the
                                        structure of dirs inside.
            file_identifier (string):   File identifier for all files inside a dir
        Returns:
            (string):           Returns the info of recent created site.
        """

        if recursive:
            expression = "/**/*"
        else:
            expression = "/*"

        files_in_dir = list(
            itertools.chain.from_iterable(
                glob.iglob(dir_path + expression + pattern, recursive=recursive)
                for pattern in file_patterns
            )
        )
        print("files_in_dir", files_in_dir)
        filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        total_files = len(files_in_dir)
        print(total_files)
        starttime = time.time()

        try:
            files_uploaded = []
            for idx, file_with_path in enumerate(files_in_dir):

                # total time since last login or script start
                total_time = round((time.time() - starttime), 2)

                if total_time > 2400:
                    """
                    if total time is bigger than 2400
                    or 40 minutes relogin to avoid ticket
                    expiration
                    """
                    time.sleep(5)

                    print("Re-logging in to alfresco...")

                    session = login.login()
                    # restart time
                    starttime = time.time()
                    time.sleep(5)
                    print("Login sucessful, continuing upload\\n")

                len_of_path = len(file_with_path.split("/"))
                name_of_file = file_with_path.split("/")[len_of_path - 1]
                root_dir_path = file_with_path.replace(dir_path, "").replace(
                    file_with_path.split("/")[len_of_path - 1], ""
                )

                data = {
                    "name": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    ),
                    "nodeType": "cm:content",
                }

                data["relativePath"] = root_dir_path

                data["properties"] = {
                    "cm:title": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    )
                }

                print("Uploading " + data["name"] + " file...")

                files = {"filedata": open(file_with_path, "rb")}
                upload_response = upload(session, node_id, data, files)
                if upload_response[1] and upload_response[1] == 201:
                    files_uploaded.append(upload_response[0])
                    print("Uploaded " + data["name"])

                    filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
                    with open(filename, 'a') as log_file:
                        log_file.writelines("%s\\n" % file_with_path)

                elif upload_response[1] and upload_response[1] == 409:
                    if "already exists" in upload_response[0]["error"]["errorKey"]:
                        print("File " + data["name"] + " already uploaded")

                else:
                    print("An error ocurred, file " + data["name"] + " cannot be uploaded")

                print("Uploaded file " + str(idx + 1) + " of " + str(total_files))
                print("\\n\\n")

            return files_uploaded
        except Exception as e:
            print("An error ocurred in file upload: ", e)
        
    def vectorize_soundscape(df, hash_name, indices):
        \'\'\'Return dataframe with array column containing indices by frequency\'\'\'
        return (df
                .groupby(by=["id", hash_name, "start_time", "end_time"])
                .apply(get_vectors, indices)
                .reset_index()
                .rename(columns={0:"index_vector"}))
    '''

    _kale_block3 = '''
    load_dotenv()
    DB_CONFIG = {
        'provider': 'alfresco',
        'config': {
            'api_url': 'https://api.conabio.gob.mx/test',
            'page_size': PAGESIZE,
            'api_key': os.getenv("X_API_KEY"),
            'base_filter': "+TYPE: \\"sipecam:audio\\" AND -TYPE: \\"dummyType\\"",
            'recording_parser': {"path": "/shared_volume/audio/utils.py",
                                 "object_name": "parser"}
        }
    }

    COL_CONFIG = {
        "col_type": "alfresco",
        "db_config": DB_CONFIG
    }

    col = collection(**COL_CONFIG)
    query = f"(sipecam:CumulusName:\\"{CUMULO}\\") AND (sipecam:SampleRate:{SAMPLERATE})"

    if LIMIT:
        recs = col.get_recording_dataframe(query, limit=LIMIT, with_metadata = True, with_geometry = False)
    else:
        recs = col.get_recording_dataframe(query, with_metadata = True, with_geometry = False)

    # include filtering columns for processing units
    recs.loc[:, "node"] = recs.metadata.apply(lambda x: x["entry"]["properties"]["sipecam:NomenclatureNode"])
    recs.loc[:, "recorder"] = recs.metadata.apply(lambda x: x["entry"]["properties"]["sipecam:SerialNumber"]) 
    recs.loc[:, "deployment"] = recs.metadata.apply(lambda x: x["entry"]["path"]["name"].split("/audio")[0].split("/")[-1])
    recs.loc[:,"proc_unit"] = recs.apply(lambda x: (x["node"], x["recorder"], x["deployment"]), axis=1)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.sndscs_spec_specvid-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
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


def create_results_dirstruct(AUTH_ENDPOINT: str, BASE_ENDPOINT: str, CUMULO: int, RESULTS_DIR: str):
    _kale_pipeline_parameters_block = '''
    AUTH_ENDPOINT = "{}"
    BASE_ENDPOINT = "{}"
    CUMULO = {}
    RESULTS_DIR = "{}"
    '''.format(AUTH_ENDPOINT, BASE_ENDPOINT, CUMULO, RESULTS_DIR)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.sndscs_spec_specvid-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
    recs = _kale_marshal.load("recs")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import base64
    import datetime
    import glob
    import hashlib
    import io
    import itertools
    import json
    import matplotlib.pyplot as plt
    import multiprocessing 
    import numpy as np
    import os
    import pandas as pd
    import psutil
    import requests
    import shutil
    import subprocess
    import time
    import warnings

    from dask.distributed import Client, LocalCluster
    from datetime import timedelta
    from dotenv import load_dotenv
    from matplotlib import cm
    from moviepy.editor import concatenate, VideoFileClip, AudioFileClip
    from moviepy.audio.AudioClip import AudioArrayClip
    from moviepy.video.VideoClip import ImageClip
    from os.path import exists as file_exists
    from PIL import Image
    from skimage.transform import resize

    from yuntu import Audio
    from yuntu.soundscape.utils import aware_time
    from yuntu.collection.methods import collection
    from yuntu.soundscape.hashers.crono import DEFAULT_HASHER_CONFIG
    from yuntu.soundscape.processors.indices.direct import ICOMPLEXITY, TAIL
    from yuntu.soundscape.pipelines.build_soundscape import CronoSoundscape, HASHER_CONFIG
    '''

    _kale_block2 = '''
    def audio2video(audio_id,
                    identifier,
                    audio_df,
                    save_path_folder,
                    product_spectrum,
                    cumulus,
                    abs_start=None,
                    fps=60,
                    spec_configs={'hop_length': 512, 'n_fft': 1024, 'window_function': 'hann'},
                    rate=24,
                    frame_duration=3.0,
                    min_freq=0,
                    max_freq=None,
                    cmap="Greys",
                    figsize=(5, 8),
                    dpi=100,
                    bands=None):
        \'\'\'Takes and audio object and produces a mp4 video of the spectrogram with audio\'\'\'

        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        id_audio = sub_audio_df['id'].values[0]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        audio = sub_audio_df.audio[0].cut(0,1)
        
        colormap = cm.get_cmap(cmap)
        duration = audio.duration
        step = 1/rate
        start = -(frame_duration/2.0)
        stop = start + frame_duration
        clips = []
        last_stop = None

        if max_freq is None:
            max_freq = audio.samplerate / 2.0
        
        if min_freq is None:
            min_freq = 0
        

        with audio.features.db_spectrogram(**spec_configs) as spec:
            min_spec = np.amin(spec)
            max_spec = np.amax(spec)
            spec_range = (max_spec-min_spec)
            
            while stop <= duration+(frame_duration/2.0):
                clip = produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start, colormap,
                                    min_spec, spec_range, figsize, dpi, bands=bands)
                clips.append(clip)
                
                if start + step + frame_duration > duration:
                    last_stop = stop

                start = start + step
                stop = start + frame_duration
        
        video = concatenate(clips)
        # edaudio = AudioArrayClip(audio_array, fps=audio.samplerate)
        edaudio = AudioFileClip(audio.path).set_end(audio.duration)
        video = video.set_audio(edaudio)
        file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram_video.mp4")
        video.write_videofile(file_path, fps=fps)
        
        save_metadata_videoclip(id_audio, identifier, product_spectrum,
                      save_path_folder, cumulus, node, recorder, deployment, 0.0, audio.duration)
        video.close()
        edaudio.close()
        
        for c in clips:
            c.close()

    def change_type_sipecam_sc(session, root_folder_id, path, file_type):
        if file_type == "sequence.png":
            metadata_name = "soundscape_seq_metadata.json"
            aggr_type = "None"
        elif file_type == "mean_soundscape.png":
            metadata_name = "mean_soundscape_metadata.json"
            aggr_type = "Mean"
        elif file_type ==  "std_soundscape.png":
            metadata_name = "std_soundscape_metadata.json" 
            aggr_type = "Standard deviation"
        elif file_type == "hashed_soundscape.parquet":
            metadata_name = "soundscape_metadata.json.json" 
            aggr_type = "None"

        try:
            semi_path = path.split("soundscapes/")[-1]
            semi_path_file = os.path.join(semi_path, file_type)
            local_path_file_metadata = os.path.join(path, metadata_name)
            print(f"Changing type for {os.path.join(semi_path_file)}")
            alfresco_path = os.path.join("/Company Home/Sites/sipecam-soundscape/documentLibrary/", semi_path)
            response = session.get(
                os.getenv("ALFRESCO_URL")
                + BASE_ENDPOINT
                + "/nodes/"
                + root_folder_id
                + "/children?relativePath="+semi_path_file+"&include=aspectNames&skipCount=0&maxItems=1"
            )

            # error flag
            is_error = False

            # if request is successful then continue
            if response.status_code == 200:

                data_file = open(local_path_file_metadata)
                data_json = json.load(data_file)
                response_entries = response.json()["list"]["entries"][0]

                if response_entries["entry"]["isFile"]:

                    prop_dict = {}
                    # map properties
                    prop_dict["soundscape:CumulusName"] = str(data_json["CumulusName"])
                    prop_dict["soundscape:DateDeployment"] = data_json["DateDeployment"]
                    prop_dict["soundscape:NodeCategoryIntegrity"] = str(data_json["NodeCategoryIntegrity"])
                    prop_dict["soundscape:NomenclatureNode"] = str(data_json["NomenclatureNode"])
                    prop_dict["soundscape:SerialNumber"] = str(data_json["SerialNumber"])
                    prop_dict["soundscape:aggr"] = str(aggr_type)
                    prop_dict["soundscape:cycle_config_aware_start"] = str(data_json["product_configs"]['hasher_config']['kwargs']['aware_start'])
                    prop_dict["soundscape:cycle_config_start_format"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_format'])
                    prop_dict["soundscape:cycle_config_start_time"] = data_json["product_configs"]['hasher_config']['kwargs']['start_time'] #
                    prop_dict["soundscape:cycle_config_start_tzone"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_tzone'])
                    prop_dict["soundscape:cycle_config_time_module"] = int(data_json["product_configs"]['hasher_config']['kwargs']['time_module'])
                    prop_dict["soundscape:cycle_config_time_unit"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_unit'])
                    prop_dict["soundscape:cycle_config_time_utc_column"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_utc_column'])
                    prop_dict["soundscape:frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:frequency_hop"] = int(data_json["product_configs"]["slice_config"]["frequency_hop"])
                    prop_dict["soundscape:frequency_limits"] = str(data_json["product_configs"]["slice_config"]["frequency_limits"])
                    prop_dict["soundscape:hash_name"] = str(data_json["product_configs"]["hash_name"])
                    prop_dict["soundscape:hop_length"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["hop_length"])
                    prop_dict["soundscape:indices"] = str(data_json['product_configs']['indices'])
                    prop_dict["soundscape:n_fft"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["n_fft"])
                    prop_dict["soundscape:npartitions"] = int(data_json["product_configs"]['npartitions'])
                    prop_dict["soundscape:product_id"] = str(data_json["product_id"])
                    prop_dict["soundscape:product_name"] = str(data_json["product_name"])
                    prop_dict["soundscape:product_parent"] = str(data_json["product_parent"])
                    prop_dict["soundscape:product_path"] = str(alfresco_path)
                    prop_dict["soundscape:product_spectrum"] = str(data_json["product_spectrum"])
                    prop_dict["soundscape:slice_config_feature_type"] = str(data_json["product_configs"]["slice_config"]["feature_type"])
                    prop_dict["soundscape:slice_config_frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:slice_config_time_unit"] = int(data_json["product_configs"]["slice_config"]["time_unit"])
                    prop_dict["soundscape:time_hop"] = int(data_json["product_configs"]["slice_config"]["time_hop"])
                    prop_dict["soundscape:window_function"] = str(data_json["product_configs"]["slice_config"]["feature_config"]["window_function"])


                    aspects = response_entries["entry"]["aspectNames"]

                    data = {"aspectNames": aspects, "nodeType": 'soundscape:product', "properties": prop_dict}

                    # update properties request
                    update = session.put(
                        os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT
                        + "/nodes/"
                        + response_entries["entry"]["id"],
                        data=json.dumps(data),
                    )
                    print(update.json())
                    if update.status_code == 200:
                        print("Updated " + response_entries["entry"]["id"])
        except Exception as e:
            print("Could not add any aspect to this file: ", e)
            
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
                    
    def distance_to_mean(vector, mean):
        \'\'\'Return euclidean distance to mean\'\'\'
        return np.sqrt(np.sum(np.square(mean - vector)))

    def find_subfolders(path_abs):
        subdir_list = []
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            len_path = path.split("/")
            if len(len_path) == 8:
                subdir_list.append(path)  
                
        return subdir_list

    def get_audio_ids(soundscape_path, indices):
        df = pd.read_parquet(os.path.join(soundscape_path, "hashed_soundscape.parquet"))

        with open(os.path.join(soundscape_path, "soundscape_metadata.json")) as f:
            metadata = json.load(f)
            f.close()

        # indices = metadata["product_configs"]["indices"]
        # indices = ["EXAG", "ICOMPLEXITY", "CORE"]
        hash_name = metadata["product_configs"]["hash_name"]
        cycle_config = metadata["product_configs"]["hasher_config"]["kwargs"]
        time_unit = cycle_config["time_unit"]
        zero_t = aware_time( cycle_config["start_time"], cycle_config["start_tzone"], cycle_config["start_format"]) 
        
        # sample
        samples_df = get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=3)
        sub_df = samples_df[samples_df.crono_hash_30m == 8]
        audio_id_list = list(sub_df["id"].unique())
        
        return audio_id_list

    def get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=5):
        \'\'\'Return dataframe of 'nsamples' samples for each tag in 'hash_name' column that are closest to the mean vector by tag\'\'\'
        proj_df = df[(df.max_freq <= 10000)]
        crono_tags = proj_df.crono_hash_30m.unique()
        proj_df.loc[: , f"{hash_name}_time"] = proj_df[hash_name].apply(lambda x: zero_t + datetime.timedelta(seconds=float(x*time_unit)))
        vectors = vectorize_soundscape(proj_df, hash_name, indices)
        min_index_vector = np.amin(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        max_index_vector = np.amax(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        index_range = (max_index_vector - min_index_vector)
        vectors.loc[:, "normalized_index_vector"] = vectors.index_vector.apply(lambda x: (x-min_index_vector)/index_range)
        all_samples = []

        for crono_tag in crono_tags:
            unit_vectors = vectors[vectors[hash_name] == crono_tag]
            mean_unit_vector = unit_vectors.normalized_index_vector.mean()
            unit_vectors.loc[:, "distance"] = unit_vectors.normalized_index_vector.apply(lambda x: distance_to_mean(x, mean_unit_vector))
            all_samples.append(unit_vectors.sort_values(by="distance").head(nsamples))

        return pd.concat(all_samples)

    def get_vectors(group, indices):
        \'\'\'Return array of indices by frequency\'\'\'
        return group.sort_values(by="max_freq")[indices].values

    def login():
        """
        Tries a login to alfresco api and returns a session
        object with credentials 
        Returns: 
            session (Session):  A session object to make 
                                requests to zendro.
        """
        try:
            auth = {
                "userId": os.getenv("ALFRESCO_USER"),
                "password": os.getenv("ALFRESCO_PASSWORD"),
            }

            login = requests.post(os.getenv("ALFRESCO_URL") + AUTH_ENDPOINT + "/tickets",data=json.dumps(auth))

            base64_login = base64.b64encode(bytes(login.json()["entry"]["id"], 'utf-8')).decode()

            # se crea un objeto de Session para hacer requests
            session = requests.Session()
            # se establece bearer token
            session.headers.update({'Authorization': 'Basic ' + base64_login})

            return session
        except Exception as e:
            print("Login failed: ",e)

    def plot_spectrogram(audio_id, identifier, audio_df, save_path_folder, spectrum, cumulus):
        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        # plot
        fig, ax = plt.subplots(2,1,figsize=(20,10), sharex=True)
        sub_audio_df.audio[0].plot(ax=ax[0], color='grey')
        sub_audio_df.audio[0].features.db_spectrogram().plot(ax=ax[1])
        ax[0].set_ylabel('Amplitude')
        ax[0].grid(False)
        ax[1].set_ylabel('F (KHz)')
        ax[1].set_xlabel('Time (seconds)')
        fig.text(0.75, 0.04, f"Cumulus: {cumulus} - Node: {node} - Recorder: {recorder}", va='center')
        plt.show()
        if save_path_folder:
            file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram.png")
            fig.savefig(file_path)
        
        save_metadata_spectrogram(audio_id, identifier, spectrum, save_path_folder, 
                                  cumulus, node, recorder, deployment, parent="Null")
        
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)    
            
        print(f"File saved at {file_path}")
        
    def produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start=None, 
                     colormap=cm.get_cmap("Greys"), min_spec=0, spec_range=1.0, figsize=(5, 4), 
                     dpi=100, bands=None):
        \'\'\'Takes an individual frame and produces an image with references\'\'\'
        frame = spec.cut_array(start_time=start, end_time=stop, min_freq=min_freq, max_freq=max_freq, pad=True)
        plt.style.use('dark_background')
        frame = np.flip((frame - min_spec)/spec_range, axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frame, cmap=colormap, extent=[0, frame_duration, min_freq/1000, max_freq/1000], 
                  aspect="auto", vmin = 0, vmax = 1.0)

        if bands is not None:
            band_arr = np.flip(resize(np.expand_dims(bands, axis=1), (frame.shape[0], frame.shape[1])), axis=0)
            ax.imshow(band_arr, extent=[0, frame_duration, min_freq/1000, max_freq/1000], aspect="auto", vmin = 0, 
                      vmax = 1.0, alpha=0.5)

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        mid = frame_duration/2.0
        ax.axvline(x=mid, color="red")
        ax.set_ylabel('F (kHz)')
        ax.set_xticks([])
        ax.set_xticks([], minor=True)

        if abs_start is not None:
            time_text = (abs_start + datetime.timedelta(seconds=start+mid)).strftime('%H:%M:%S.%f').strip()[:-4]
            ax.text(mid-0.3, -0.6, time_text)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        im = Image.open(buf)
        im.format = "PNG"
        plt.close(fig)

        return ImageClip(np.asarray(im),
                         duration=step)
        
    def remove_empty_folders(path_abs):
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            if len(os.listdir(path)) == 0:
                os.rmdir(path)            
                
    def save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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

    def save_metadata_spectrogram(product_id, identifier, product_spectrum,
                      path, cumulus, node, recorder, deployment, parent="Null"):
        product_name = "Spectrogram"
        file_path = os.path.join(path, f"{identifier}_spectrogram.png")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": parent,
            "product_name": product_name,
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
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def save_metadata_videoclip(product_id, identifier, product_spectrum, path, cumulus, node, recorder, 
                                deployment, clip_start, clip_end, parent="Null"):
        product_name = "spectrogram_video"
        file_path = os.path.join(path, f"{identifier}_spectrogram_video.mp4")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_video_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": "Null",
            "product_name": product_name,
            "product_path": file_path,
            "product_spectrum": product_spectrum,
            "CumulusName": cumulus,
            "NodeCategoryIntegrity": node_category,
            "NomenclatureNode": node,
            "SerialNumber": recorder,
            "DateDeployment": deployment,
            "ClipStart": clip_start,
            "ClipEnd": clip_end
        }
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def upload(session, node_id, data, file):
        """
        Uploads a file to a specific folder.
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            data (dict):                Dict that contains file options
            file (object):              File to upload
        
        Returns:
            (list):     A list containing status code and status data
        """

        try:
            response = session.post(os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT + "/nodes/" + node_id + "/children",
                        data = data,
                        files = file
                        )
                        
            return [response.json(), response.status_code];
        except Exception as e: 
            print("File " + data["name"] + " could not be uploaded: ", e)

    def upload_files(file_patterns ,session, node_id, dir_path, recursive, file_identifier=""):
        """
        Uploads the files stored in a specific dir
        to alfresco
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            dir_path (string):          The name and path of the dir where files are stored
            recursive (boolean):        A boolean to know if upload  must be recursive
                                        in the specifed dir, and should preserve the
                                        structure of dirs inside.
            file_identifier (string):   File identifier for all files inside a dir
        Returns:
            (string):           Returns the info of recent created site.
        """

        if recursive:
            expression = "/**/*"
        else:
            expression = "/*"

        files_in_dir = list(
            itertools.chain.from_iterable(
                glob.iglob(dir_path + expression + pattern, recursive=recursive)
                for pattern in file_patterns
            )
        )
        print("files_in_dir", files_in_dir)
        filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        total_files = len(files_in_dir)
        print(total_files)
        starttime = time.time()

        try:
            files_uploaded = []
            for idx, file_with_path in enumerate(files_in_dir):

                # total time since last login or script start
                total_time = round((time.time() - starttime), 2)

                if total_time > 2400:
                    """
                    if total time is bigger than 2400
                    or 40 minutes relogin to avoid ticket
                    expiration
                    """
                    time.sleep(5)

                    print("Re-logging in to alfresco...")

                    session = login.login()
                    # restart time
                    starttime = time.time()
                    time.sleep(5)
                    print("Login sucessful, continuing upload\\n")

                len_of_path = len(file_with_path.split("/"))
                name_of_file = file_with_path.split("/")[len_of_path - 1]
                root_dir_path = file_with_path.replace(dir_path, "").replace(
                    file_with_path.split("/")[len_of_path - 1], ""
                )

                data = {
                    "name": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    ),
                    "nodeType": "cm:content",
                }

                data["relativePath"] = root_dir_path

                data["properties"] = {
                    "cm:title": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    )
                }

                print("Uploading " + data["name"] + " file...")

                files = {"filedata": open(file_with_path, "rb")}
                upload_response = upload(session, node_id, data, files)
                if upload_response[1] and upload_response[1] == 201:
                    files_uploaded.append(upload_response[0])
                    print("Uploaded " + data["name"])

                    filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
                    with open(filename, 'a') as log_file:
                        log_file.writelines("%s\\n" % file_with_path)

                elif upload_response[1] and upload_response[1] == 409:
                    if "already exists" in upload_response[0]["error"]["errorKey"]:
                        print("File " + data["name"] + " already uploaded")

                else:
                    print("An error ocurred, file " + data["name"] + " cannot be uploaded")

                print("Uploaded file " + str(idx + 1) + " of " + str(total_files))
                print("\\n\\n")

            return files_uploaded
        except Exception as e:
            print("An error ocurred in file upload: ", e)
        
    def vectorize_soundscape(df, hash_name, indices):
        \'\'\'Return dataframe with array column containing indices by frequency\'\'\'
        return (df
                .groupby(by=["id", hash_name, "start_time", "end_time"])
                .apply(get_vectors, indices)
                .reset_index()
                .rename(columns={0:"index_vector"}))
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
    _kale_marshal.set_data_dir("/shared_volume/audio/.sndscs_spec_specvid-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
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


def compute_soundscapes(AUTH_ENDPOINT: str, BASE_ENDPOINT: str, BLUE_IDX: str, CUMULO: int, FREQUENCY_BINS: int, FREQUENCY_LIMITS_LB: int, FREQUENCY_LIMITS_UB: int, GREEN_IDX: str, HASHER_TIME_MODULE: int, HASHER_TIME_UNIT: int, HASH_NAME: str, MIN_FREQ_SC: int, RED_IDX: str, RESULTS_DIR: str, SPECTRUM: str, THREADS_PER_WORKER: int, TIME_UNIT: int, WORK_DIR_PIPELINE: str):
    _kale_pipeline_parameters_block = '''
    AUTH_ENDPOINT = "{}"
    BASE_ENDPOINT = "{}"
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
    '''.format(AUTH_ENDPOINT, BASE_ENDPOINT, BLUE_IDX, CUMULO, FREQUENCY_BINS, FREQUENCY_LIMITS_LB, FREQUENCY_LIMITS_UB, GREEN_IDX, HASHER_TIME_MODULE, HASHER_TIME_UNIT, HASH_NAME, MIN_FREQ_SC, RED_IDX, RESULTS_DIR, SPECTRUM, THREADS_PER_WORKER, TIME_UNIT, WORK_DIR_PIPELINE)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.sndscs_spec_specvid-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
    recs = _kale_marshal.load("recs")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import base64
    import datetime
    import glob
    import hashlib
    import io
    import itertools
    import json
    import matplotlib.pyplot as plt
    import multiprocessing 
    import numpy as np
    import os
    import pandas as pd
    import psutil
    import requests
    import shutil
    import subprocess
    import time
    import warnings

    from dask.distributed import Client, LocalCluster
    from datetime import timedelta
    from dotenv import load_dotenv
    from matplotlib import cm
    from moviepy.editor import concatenate, VideoFileClip, AudioFileClip
    from moviepy.audio.AudioClip import AudioArrayClip
    from moviepy.video.VideoClip import ImageClip
    from os.path import exists as file_exists
    from PIL import Image
    from skimage.transform import resize

    from yuntu import Audio
    from yuntu.soundscape.utils import aware_time
    from yuntu.collection.methods import collection
    from yuntu.soundscape.hashers.crono import DEFAULT_HASHER_CONFIG
    from yuntu.soundscape.processors.indices.direct import ICOMPLEXITY, TAIL
    from yuntu.soundscape.pipelines.build_soundscape import CronoSoundscape, HASHER_CONFIG
    '''

    _kale_block2 = '''
    def audio2video(audio_id,
                    identifier,
                    audio_df,
                    save_path_folder,
                    product_spectrum,
                    cumulus,
                    abs_start=None,
                    fps=60,
                    spec_configs={'hop_length': 512, 'n_fft': 1024, 'window_function': 'hann'},
                    rate=24,
                    frame_duration=3.0,
                    min_freq=0,
                    max_freq=None,
                    cmap="Greys",
                    figsize=(5, 8),
                    dpi=100,
                    bands=None):
        \'\'\'Takes and audio object and produces a mp4 video of the spectrogram with audio\'\'\'

        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        id_audio = sub_audio_df['id'].values[0]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        audio = sub_audio_df.audio[0].cut(0,1)
        
        colormap = cm.get_cmap(cmap)
        duration = audio.duration
        step = 1/rate
        start = -(frame_duration/2.0)
        stop = start + frame_duration
        clips = []
        last_stop = None

        if max_freq is None:
            max_freq = audio.samplerate / 2.0
        
        if min_freq is None:
            min_freq = 0
        

        with audio.features.db_spectrogram(**spec_configs) as spec:
            min_spec = np.amin(spec)
            max_spec = np.amax(spec)
            spec_range = (max_spec-min_spec)
            
            while stop <= duration+(frame_duration/2.0):
                clip = produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start, colormap,
                                    min_spec, spec_range, figsize, dpi, bands=bands)
                clips.append(clip)
                
                if start + step + frame_duration > duration:
                    last_stop = stop

                start = start + step
                stop = start + frame_duration
        
        video = concatenate(clips)
        # edaudio = AudioArrayClip(audio_array, fps=audio.samplerate)
        edaudio = AudioFileClip(audio.path).set_end(audio.duration)
        video = video.set_audio(edaudio)
        file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram_video.mp4")
        video.write_videofile(file_path, fps=fps)
        
        save_metadata_videoclip(id_audio, identifier, product_spectrum,
                      save_path_folder, cumulus, node, recorder, deployment, 0.0, audio.duration)
        video.close()
        edaudio.close()
        
        for c in clips:
            c.close()

    def change_type_sipecam_sc(session, root_folder_id, path, file_type):
        if file_type == "sequence.png":
            metadata_name = "soundscape_seq_metadata.json"
            aggr_type = "None"
        elif file_type == "mean_soundscape.png":
            metadata_name = "mean_soundscape_metadata.json"
            aggr_type = "Mean"
        elif file_type ==  "std_soundscape.png":
            metadata_name = "std_soundscape_metadata.json" 
            aggr_type = "Standard deviation"
        elif file_type == "hashed_soundscape.parquet":
            metadata_name = "soundscape_metadata.json.json" 
            aggr_type = "None"

        try:
            semi_path = path.split("soundscapes/")[-1]
            semi_path_file = os.path.join(semi_path, file_type)
            local_path_file_metadata = os.path.join(path, metadata_name)
            print(f"Changing type for {os.path.join(semi_path_file)}")
            alfresco_path = os.path.join("/Company Home/Sites/sipecam-soundscape/documentLibrary/", semi_path)
            response = session.get(
                os.getenv("ALFRESCO_URL")
                + BASE_ENDPOINT
                + "/nodes/"
                + root_folder_id
                + "/children?relativePath="+semi_path_file+"&include=aspectNames&skipCount=0&maxItems=1"
            )

            # error flag
            is_error = False

            # if request is successful then continue
            if response.status_code == 200:

                data_file = open(local_path_file_metadata)
                data_json = json.load(data_file)
                response_entries = response.json()["list"]["entries"][0]

                if response_entries["entry"]["isFile"]:

                    prop_dict = {}
                    # map properties
                    prop_dict["soundscape:CumulusName"] = str(data_json["CumulusName"])
                    prop_dict["soundscape:DateDeployment"] = data_json["DateDeployment"]
                    prop_dict["soundscape:NodeCategoryIntegrity"] = str(data_json["NodeCategoryIntegrity"])
                    prop_dict["soundscape:NomenclatureNode"] = str(data_json["NomenclatureNode"])
                    prop_dict["soundscape:SerialNumber"] = str(data_json["SerialNumber"])
                    prop_dict["soundscape:aggr"] = str(aggr_type)
                    prop_dict["soundscape:cycle_config_aware_start"] = str(data_json["product_configs"]['hasher_config']['kwargs']['aware_start'])
                    prop_dict["soundscape:cycle_config_start_format"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_format'])
                    prop_dict["soundscape:cycle_config_start_time"] = data_json["product_configs"]['hasher_config']['kwargs']['start_time'] #
                    prop_dict["soundscape:cycle_config_start_tzone"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_tzone'])
                    prop_dict["soundscape:cycle_config_time_module"] = int(data_json["product_configs"]['hasher_config']['kwargs']['time_module'])
                    prop_dict["soundscape:cycle_config_time_unit"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_unit'])
                    prop_dict["soundscape:cycle_config_time_utc_column"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_utc_column'])
                    prop_dict["soundscape:frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:frequency_hop"] = int(data_json["product_configs"]["slice_config"]["frequency_hop"])
                    prop_dict["soundscape:frequency_limits"] = str(data_json["product_configs"]["slice_config"]["frequency_limits"])
                    prop_dict["soundscape:hash_name"] = str(data_json["product_configs"]["hash_name"])
                    prop_dict["soundscape:hop_length"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["hop_length"])
                    prop_dict["soundscape:indices"] = str(data_json['product_configs']['indices'])
                    prop_dict["soundscape:n_fft"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["n_fft"])
                    prop_dict["soundscape:npartitions"] = int(data_json["product_configs"]['npartitions'])
                    prop_dict["soundscape:product_id"] = str(data_json["product_id"])
                    prop_dict["soundscape:product_name"] = str(data_json["product_name"])
                    prop_dict["soundscape:product_parent"] = str(data_json["product_parent"])
                    prop_dict["soundscape:product_path"] = str(alfresco_path)
                    prop_dict["soundscape:product_spectrum"] = str(data_json["product_spectrum"])
                    prop_dict["soundscape:slice_config_feature_type"] = str(data_json["product_configs"]["slice_config"]["feature_type"])
                    prop_dict["soundscape:slice_config_frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:slice_config_time_unit"] = int(data_json["product_configs"]["slice_config"]["time_unit"])
                    prop_dict["soundscape:time_hop"] = int(data_json["product_configs"]["slice_config"]["time_hop"])
                    prop_dict["soundscape:window_function"] = str(data_json["product_configs"]["slice_config"]["feature_config"]["window_function"])


                    aspects = response_entries["entry"]["aspectNames"]

                    data = {"aspectNames": aspects, "nodeType": 'soundscape:product', "properties": prop_dict}

                    # update properties request
                    update = session.put(
                        os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT
                        + "/nodes/"
                        + response_entries["entry"]["id"],
                        data=json.dumps(data),
                    )
                    print(update.json())
                    if update.status_code == 200:
                        print("Updated " + response_entries["entry"]["id"])
        except Exception as e:
            print("Could not add any aspect to this file: ", e)
            
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
                    
    def distance_to_mean(vector, mean):
        \'\'\'Return euclidean distance to mean\'\'\'
        return np.sqrt(np.sum(np.square(mean - vector)))

    def find_subfolders(path_abs):
        subdir_list = []
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            len_path = path.split("/")
            if len(len_path) == 8:
                subdir_list.append(path)  
                
        return subdir_list

    def get_audio_ids(soundscape_path, indices):
        df = pd.read_parquet(os.path.join(soundscape_path, "hashed_soundscape.parquet"))

        with open(os.path.join(soundscape_path, "soundscape_metadata.json")) as f:
            metadata = json.load(f)
            f.close()

        # indices = metadata["product_configs"]["indices"]
        # indices = ["EXAG", "ICOMPLEXITY", "CORE"]
        hash_name = metadata["product_configs"]["hash_name"]
        cycle_config = metadata["product_configs"]["hasher_config"]["kwargs"]
        time_unit = cycle_config["time_unit"]
        zero_t = aware_time( cycle_config["start_time"], cycle_config["start_tzone"], cycle_config["start_format"]) 
        
        # sample
        samples_df = get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=3)
        sub_df = samples_df[samples_df.crono_hash_30m == 8]
        audio_id_list = list(sub_df["id"].unique())
        
        return audio_id_list

    def get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=5):
        \'\'\'Return dataframe of 'nsamples' samples for each tag in 'hash_name' column that are closest to the mean vector by tag\'\'\'
        proj_df = df[(df.max_freq <= 10000)]
        crono_tags = proj_df.crono_hash_30m.unique()
        proj_df.loc[: , f"{hash_name}_time"] = proj_df[hash_name].apply(lambda x: zero_t + datetime.timedelta(seconds=float(x*time_unit)))
        vectors = vectorize_soundscape(proj_df, hash_name, indices)
        min_index_vector = np.amin(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        max_index_vector = np.amax(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        index_range = (max_index_vector - min_index_vector)
        vectors.loc[:, "normalized_index_vector"] = vectors.index_vector.apply(lambda x: (x-min_index_vector)/index_range)
        all_samples = []

        for crono_tag in crono_tags:
            unit_vectors = vectors[vectors[hash_name] == crono_tag]
            mean_unit_vector = unit_vectors.normalized_index_vector.mean()
            unit_vectors.loc[:, "distance"] = unit_vectors.normalized_index_vector.apply(lambda x: distance_to_mean(x, mean_unit_vector))
            all_samples.append(unit_vectors.sort_values(by="distance").head(nsamples))

        return pd.concat(all_samples)

    def get_vectors(group, indices):
        \'\'\'Return array of indices by frequency\'\'\'
        return group.sort_values(by="max_freq")[indices].values

    def login():
        """
        Tries a login to alfresco api and returns a session
        object with credentials 
        Returns: 
            session (Session):  A session object to make 
                                requests to zendro.
        """
        try:
            auth = {
                "userId": os.getenv("ALFRESCO_USER"),
                "password": os.getenv("ALFRESCO_PASSWORD"),
            }

            login = requests.post(os.getenv("ALFRESCO_URL") + AUTH_ENDPOINT + "/tickets",data=json.dumps(auth))

            base64_login = base64.b64encode(bytes(login.json()["entry"]["id"], 'utf-8')).decode()

            # se crea un objeto de Session para hacer requests
            session = requests.Session()
            # se establece bearer token
            session.headers.update({'Authorization': 'Basic ' + base64_login})

            return session
        except Exception as e:
            print("Login failed: ",e)

    def plot_spectrogram(audio_id, identifier, audio_df, save_path_folder, spectrum, cumulus):
        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        # plot
        fig, ax = plt.subplots(2,1,figsize=(20,10), sharex=True)
        sub_audio_df.audio[0].plot(ax=ax[0], color='grey')
        sub_audio_df.audio[0].features.db_spectrogram().plot(ax=ax[1])
        ax[0].set_ylabel('Amplitude')
        ax[0].grid(False)
        ax[1].set_ylabel('F (KHz)')
        ax[1].set_xlabel('Time (seconds)')
        fig.text(0.75, 0.04, f"Cumulus: {cumulus} - Node: {node} - Recorder: {recorder}", va='center')
        plt.show()
        if save_path_folder:
            file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram.png")
            fig.savefig(file_path)
        
        save_metadata_spectrogram(audio_id, identifier, spectrum, save_path_folder, 
                                  cumulus, node, recorder, deployment, parent="Null")
        
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)    
            
        print(f"File saved at {file_path}")
        
    def produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start=None, 
                     colormap=cm.get_cmap("Greys"), min_spec=0, spec_range=1.0, figsize=(5, 4), 
                     dpi=100, bands=None):
        \'\'\'Takes an individual frame and produces an image with references\'\'\'
        frame = spec.cut_array(start_time=start, end_time=stop, min_freq=min_freq, max_freq=max_freq, pad=True)
        plt.style.use('dark_background')
        frame = np.flip((frame - min_spec)/spec_range, axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frame, cmap=colormap, extent=[0, frame_duration, min_freq/1000, max_freq/1000], 
                  aspect="auto", vmin = 0, vmax = 1.0)

        if bands is not None:
            band_arr = np.flip(resize(np.expand_dims(bands, axis=1), (frame.shape[0], frame.shape[1])), axis=0)
            ax.imshow(band_arr, extent=[0, frame_duration, min_freq/1000, max_freq/1000], aspect="auto", vmin = 0, 
                      vmax = 1.0, alpha=0.5)

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        mid = frame_duration/2.0
        ax.axvline(x=mid, color="red")
        ax.set_ylabel('F (kHz)')
        ax.set_xticks([])
        ax.set_xticks([], minor=True)

        if abs_start is not None:
            time_text = (abs_start + datetime.timedelta(seconds=start+mid)).strftime('%H:%M:%S.%f').strip()[:-4]
            ax.text(mid-0.3, -0.6, time_text)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        im = Image.open(buf)
        im.format = "PNG"
        plt.close(fig)

        return ImageClip(np.asarray(im),
                         duration=step)
        
    def remove_empty_folders(path_abs):
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            if len(os.listdir(path)) == 0:
                os.rmdir(path)            
                
    def save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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

    def save_metadata_spectrogram(product_id, identifier, product_spectrum,
                      path, cumulus, node, recorder, deployment, parent="Null"):
        product_name = "Spectrogram"
        file_path = os.path.join(path, f"{identifier}_spectrogram.png")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": parent,
            "product_name": product_name,
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
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def save_metadata_videoclip(product_id, identifier, product_spectrum, path, cumulus, node, recorder, 
                                deployment, clip_start, clip_end, parent="Null"):
        product_name = "spectrogram_video"
        file_path = os.path.join(path, f"{identifier}_spectrogram_video.mp4")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_video_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": "Null",
            "product_name": product_name,
            "product_path": file_path,
            "product_spectrum": product_spectrum,
            "CumulusName": cumulus,
            "NodeCategoryIntegrity": node_category,
            "NomenclatureNode": node,
            "SerialNumber": recorder,
            "DateDeployment": deployment,
            "ClipStart": clip_start,
            "ClipEnd": clip_end
        }
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def upload(session, node_id, data, file):
        """
        Uploads a file to a specific folder.
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            data (dict):                Dict that contains file options
            file (object):              File to upload
        
        Returns:
            (list):     A list containing status code and status data
        """

        try:
            response = session.post(os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT + "/nodes/" + node_id + "/children",
                        data = data,
                        files = file
                        )
                        
            return [response.json(), response.status_code];
        except Exception as e: 
            print("File " + data["name"] + " could not be uploaded: ", e)

    def upload_files(file_patterns ,session, node_id, dir_path, recursive, file_identifier=""):
        """
        Uploads the files stored in a specific dir
        to alfresco
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            dir_path (string):          The name and path of the dir where files are stored
            recursive (boolean):        A boolean to know if upload  must be recursive
                                        in the specifed dir, and should preserve the
                                        structure of dirs inside.
            file_identifier (string):   File identifier for all files inside a dir
        Returns:
            (string):           Returns the info of recent created site.
        """

        if recursive:
            expression = "/**/*"
        else:
            expression = "/*"

        files_in_dir = list(
            itertools.chain.from_iterable(
                glob.iglob(dir_path + expression + pattern, recursive=recursive)
                for pattern in file_patterns
            )
        )
        print("files_in_dir", files_in_dir)
        filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        total_files = len(files_in_dir)
        print(total_files)
        starttime = time.time()

        try:
            files_uploaded = []
            for idx, file_with_path in enumerate(files_in_dir):

                # total time since last login or script start
                total_time = round((time.time() - starttime), 2)

                if total_time > 2400:
                    """
                    if total time is bigger than 2400
                    or 40 minutes relogin to avoid ticket
                    expiration
                    """
                    time.sleep(5)

                    print("Re-logging in to alfresco...")

                    session = login.login()
                    # restart time
                    starttime = time.time()
                    time.sleep(5)
                    print("Login sucessful, continuing upload\\n")

                len_of_path = len(file_with_path.split("/"))
                name_of_file = file_with_path.split("/")[len_of_path - 1]
                root_dir_path = file_with_path.replace(dir_path, "").replace(
                    file_with_path.split("/")[len_of_path - 1], ""
                )

                data = {
                    "name": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    ),
                    "nodeType": "cm:content",
                }

                data["relativePath"] = root_dir_path

                data["properties"] = {
                    "cm:title": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    )
                }

                print("Uploading " + data["name"] + " file...")

                files = {"filedata": open(file_with_path, "rb")}
                upload_response = upload(session, node_id, data, files)
                if upload_response[1] and upload_response[1] == 201:
                    files_uploaded.append(upload_response[0])
                    print("Uploaded " + data["name"])

                    filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
                    with open(filename, 'a') as log_file:
                        log_file.writelines("%s\\n" % file_with_path)

                elif upload_response[1] and upload_response[1] == 409:
                    if "already exists" in upload_response[0]["error"]["errorKey"]:
                        print("File " + data["name"] + " already uploaded")

                else:
                    print("An error ocurred, file " + data["name"] + " cannot be uploaded")

                print("Uploaded file " + str(idx + 1) + " of " + str(total_files))
                print("\\n\\n")

            return files_uploaded
        except Exception as e:
            print("An error ocurred in file upload: ", e)
        
    def vectorize_soundscape(df, hash_name, indices):
        \'\'\'Return dataframe with array column containing indices by frequency\'\'\'
        return (df
                .groupby(by=["id", hash_name, "start_time", "end_time"])
                .apply(get_vectors, indices)
                .reset_index()
                .rename(columns={0:"index_vector"}))
    '''

    _kale_block3 = '''
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
            save_metadata_sc(parent_id, "soundscape", SPECTRUM, FEED_metadata, file_path,
                          CUMULO, node, recorder, deployment)
            shutil.rmtree('/shared_volume/audio/soundscape')

        except:
            pass
        # restart client
        client.restart()

        
    client.close()
    cluster.close()

    # remove empty subdirectories
    remove_empty_folders(RESULTS_DIR)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.sndscs_spec_specvid-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
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
    with open("/compute_soundscapes.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('compute_soundscapes')

    _kale_mlmdutils.call("mark_execution_complete")


def spec_n_specvid(AUTH_ENDPOINT: str, BASE_ENDPOINT: str, CUMULO: int, RESULTS_DIR: str, SPECTRUM: str):
    _kale_pipeline_parameters_block = '''
    AUTH_ENDPOINT = "{}"
    BASE_ENDPOINT = "{}"
    CUMULO = {}
    RESULTS_DIR = "{}"
    SPECTRUM = "{}"
    '''.format(AUTH_ENDPOINT, BASE_ENDPOINT, CUMULO, RESULTS_DIR, SPECTRUM)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.sndscs_spec_specvid-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
    recs = _kale_marshal.load("recs")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import base64
    import datetime
    import glob
    import hashlib
    import io
    import itertools
    import json
    import matplotlib.pyplot as plt
    import multiprocessing 
    import numpy as np
    import os
    import pandas as pd
    import psutil
    import requests
    import shutil
    import subprocess
    import time
    import warnings

    from dask.distributed import Client, LocalCluster
    from datetime import timedelta
    from dotenv import load_dotenv
    from matplotlib import cm
    from moviepy.editor import concatenate, VideoFileClip, AudioFileClip
    from moviepy.audio.AudioClip import AudioArrayClip
    from moviepy.video.VideoClip import ImageClip
    from os.path import exists as file_exists
    from PIL import Image
    from skimage.transform import resize

    from yuntu import Audio
    from yuntu.soundscape.utils import aware_time
    from yuntu.collection.methods import collection
    from yuntu.soundscape.hashers.crono import DEFAULT_HASHER_CONFIG
    from yuntu.soundscape.processors.indices.direct import ICOMPLEXITY, TAIL
    from yuntu.soundscape.pipelines.build_soundscape import CronoSoundscape, HASHER_CONFIG
    '''

    _kale_block2 = '''
    def audio2video(audio_id,
                    identifier,
                    audio_df,
                    save_path_folder,
                    product_spectrum,
                    cumulus,
                    abs_start=None,
                    fps=60,
                    spec_configs={'hop_length': 512, 'n_fft': 1024, 'window_function': 'hann'},
                    rate=24,
                    frame_duration=3.0,
                    min_freq=0,
                    max_freq=None,
                    cmap="Greys",
                    figsize=(5, 8),
                    dpi=100,
                    bands=None):
        \'\'\'Takes and audio object and produces a mp4 video of the spectrogram with audio\'\'\'

        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        id_audio = sub_audio_df['id'].values[0]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        audio = sub_audio_df.audio[0].cut(0,1)
        
        colormap = cm.get_cmap(cmap)
        duration = audio.duration
        step = 1/rate
        start = -(frame_duration/2.0)
        stop = start + frame_duration
        clips = []
        last_stop = None

        if max_freq is None:
            max_freq = audio.samplerate / 2.0
        
        if min_freq is None:
            min_freq = 0
        

        with audio.features.db_spectrogram(**spec_configs) as spec:
            min_spec = np.amin(spec)
            max_spec = np.amax(spec)
            spec_range = (max_spec-min_spec)
            
            while stop <= duration+(frame_duration/2.0):
                clip = produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start, colormap,
                                    min_spec, spec_range, figsize, dpi, bands=bands)
                clips.append(clip)
                
                if start + step + frame_duration > duration:
                    last_stop = stop

                start = start + step
                stop = start + frame_duration
        
        video = concatenate(clips)
        # edaudio = AudioArrayClip(audio_array, fps=audio.samplerate)
        edaudio = AudioFileClip(audio.path).set_end(audio.duration)
        video = video.set_audio(edaudio)
        file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram_video.mp4")
        video.write_videofile(file_path, fps=fps)
        
        save_metadata_videoclip(id_audio, identifier, product_spectrum,
                      save_path_folder, cumulus, node, recorder, deployment, 0.0, audio.duration)
        video.close()
        edaudio.close()
        
        for c in clips:
            c.close()

    def change_type_sipecam_sc(session, root_folder_id, path, file_type):
        if file_type == "sequence.png":
            metadata_name = "soundscape_seq_metadata.json"
            aggr_type = "None"
        elif file_type == "mean_soundscape.png":
            metadata_name = "mean_soundscape_metadata.json"
            aggr_type = "Mean"
        elif file_type ==  "std_soundscape.png":
            metadata_name = "std_soundscape_metadata.json" 
            aggr_type = "Standard deviation"
        elif file_type == "hashed_soundscape.parquet":
            metadata_name = "soundscape_metadata.json.json" 
            aggr_type = "None"

        try:
            semi_path = path.split("soundscapes/")[-1]
            semi_path_file = os.path.join(semi_path, file_type)
            local_path_file_metadata = os.path.join(path, metadata_name)
            print(f"Changing type for {os.path.join(semi_path_file)}")
            alfresco_path = os.path.join("/Company Home/Sites/sipecam-soundscape/documentLibrary/", semi_path)
            response = session.get(
                os.getenv("ALFRESCO_URL")
                + BASE_ENDPOINT
                + "/nodes/"
                + root_folder_id
                + "/children?relativePath="+semi_path_file+"&include=aspectNames&skipCount=0&maxItems=1"
            )

            # error flag
            is_error = False

            # if request is successful then continue
            if response.status_code == 200:

                data_file = open(local_path_file_metadata)
                data_json = json.load(data_file)
                response_entries = response.json()["list"]["entries"][0]

                if response_entries["entry"]["isFile"]:

                    prop_dict = {}
                    # map properties
                    prop_dict["soundscape:CumulusName"] = str(data_json["CumulusName"])
                    prop_dict["soundscape:DateDeployment"] = data_json["DateDeployment"]
                    prop_dict["soundscape:NodeCategoryIntegrity"] = str(data_json["NodeCategoryIntegrity"])
                    prop_dict["soundscape:NomenclatureNode"] = str(data_json["NomenclatureNode"])
                    prop_dict["soundscape:SerialNumber"] = str(data_json["SerialNumber"])
                    prop_dict["soundscape:aggr"] = str(aggr_type)
                    prop_dict["soundscape:cycle_config_aware_start"] = str(data_json["product_configs"]['hasher_config']['kwargs']['aware_start'])
                    prop_dict["soundscape:cycle_config_start_format"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_format'])
                    prop_dict["soundscape:cycle_config_start_time"] = data_json["product_configs"]['hasher_config']['kwargs']['start_time'] #
                    prop_dict["soundscape:cycle_config_start_tzone"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_tzone'])
                    prop_dict["soundscape:cycle_config_time_module"] = int(data_json["product_configs"]['hasher_config']['kwargs']['time_module'])
                    prop_dict["soundscape:cycle_config_time_unit"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_unit'])
                    prop_dict["soundscape:cycle_config_time_utc_column"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_utc_column'])
                    prop_dict["soundscape:frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:frequency_hop"] = int(data_json["product_configs"]["slice_config"]["frequency_hop"])
                    prop_dict["soundscape:frequency_limits"] = str(data_json["product_configs"]["slice_config"]["frequency_limits"])
                    prop_dict["soundscape:hash_name"] = str(data_json["product_configs"]["hash_name"])
                    prop_dict["soundscape:hop_length"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["hop_length"])
                    prop_dict["soundscape:indices"] = str(data_json['product_configs']['indices'])
                    prop_dict["soundscape:n_fft"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["n_fft"])
                    prop_dict["soundscape:npartitions"] = int(data_json["product_configs"]['npartitions'])
                    prop_dict["soundscape:product_id"] = str(data_json["product_id"])
                    prop_dict["soundscape:product_name"] = str(data_json["product_name"])
                    prop_dict["soundscape:product_parent"] = str(data_json["product_parent"])
                    prop_dict["soundscape:product_path"] = str(alfresco_path)
                    prop_dict["soundscape:product_spectrum"] = str(data_json["product_spectrum"])
                    prop_dict["soundscape:slice_config_feature_type"] = str(data_json["product_configs"]["slice_config"]["feature_type"])
                    prop_dict["soundscape:slice_config_frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:slice_config_time_unit"] = int(data_json["product_configs"]["slice_config"]["time_unit"])
                    prop_dict["soundscape:time_hop"] = int(data_json["product_configs"]["slice_config"]["time_hop"])
                    prop_dict["soundscape:window_function"] = str(data_json["product_configs"]["slice_config"]["feature_config"]["window_function"])


                    aspects = response_entries["entry"]["aspectNames"]

                    data = {"aspectNames": aspects, "nodeType": 'soundscape:product', "properties": prop_dict}

                    # update properties request
                    update = session.put(
                        os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT
                        + "/nodes/"
                        + response_entries["entry"]["id"],
                        data=json.dumps(data),
                    )
                    print(update.json())
                    if update.status_code == 200:
                        print("Updated " + response_entries["entry"]["id"])
        except Exception as e:
            print("Could not add any aspect to this file: ", e)
            
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
                    
    def distance_to_mean(vector, mean):
        \'\'\'Return euclidean distance to mean\'\'\'
        return np.sqrt(np.sum(np.square(mean - vector)))

    def find_subfolders(path_abs):
        subdir_list = []
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            len_path = path.split("/")
            if len(len_path) == 8:
                subdir_list.append(path)  
                
        return subdir_list

    def get_audio_ids(soundscape_path, indices):
        df = pd.read_parquet(os.path.join(soundscape_path, "hashed_soundscape.parquet"))

        with open(os.path.join(soundscape_path, "soundscape_metadata.json")) as f:
            metadata = json.load(f)
            f.close()

        # indices = metadata["product_configs"]["indices"]
        # indices = ["EXAG", "ICOMPLEXITY", "CORE"]
        hash_name = metadata["product_configs"]["hash_name"]
        cycle_config = metadata["product_configs"]["hasher_config"]["kwargs"]
        time_unit = cycle_config["time_unit"]
        zero_t = aware_time( cycle_config["start_time"], cycle_config["start_tzone"], cycle_config["start_format"]) 
        
        # sample
        samples_df = get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=3)
        sub_df = samples_df[samples_df.crono_hash_30m == 8]
        audio_id_list = list(sub_df["id"].unique())
        
        return audio_id_list

    def get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=5):
        \'\'\'Return dataframe of 'nsamples' samples for each tag in 'hash_name' column that are closest to the mean vector by tag\'\'\'
        proj_df = df[(df.max_freq <= 10000)]
        crono_tags = proj_df.crono_hash_30m.unique()
        proj_df.loc[: , f"{hash_name}_time"] = proj_df[hash_name].apply(lambda x: zero_t + datetime.timedelta(seconds=float(x*time_unit)))
        vectors = vectorize_soundscape(proj_df, hash_name, indices)
        min_index_vector = np.amin(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        max_index_vector = np.amax(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        index_range = (max_index_vector - min_index_vector)
        vectors.loc[:, "normalized_index_vector"] = vectors.index_vector.apply(lambda x: (x-min_index_vector)/index_range)
        all_samples = []

        for crono_tag in crono_tags:
            unit_vectors = vectors[vectors[hash_name] == crono_tag]
            mean_unit_vector = unit_vectors.normalized_index_vector.mean()
            unit_vectors.loc[:, "distance"] = unit_vectors.normalized_index_vector.apply(lambda x: distance_to_mean(x, mean_unit_vector))
            all_samples.append(unit_vectors.sort_values(by="distance").head(nsamples))

        return pd.concat(all_samples)

    def get_vectors(group, indices):
        \'\'\'Return array of indices by frequency\'\'\'
        return group.sort_values(by="max_freq")[indices].values

    def login():
        """
        Tries a login to alfresco api and returns a session
        object with credentials 
        Returns: 
            session (Session):  A session object to make 
                                requests to zendro.
        """
        try:
            auth = {
                "userId": os.getenv("ALFRESCO_USER"),
                "password": os.getenv("ALFRESCO_PASSWORD"),
            }

            login = requests.post(os.getenv("ALFRESCO_URL") + AUTH_ENDPOINT + "/tickets",data=json.dumps(auth))

            base64_login = base64.b64encode(bytes(login.json()["entry"]["id"], 'utf-8')).decode()

            # se crea un objeto de Session para hacer requests
            session = requests.Session()
            # se establece bearer token
            session.headers.update({'Authorization': 'Basic ' + base64_login})

            return session
        except Exception as e:
            print("Login failed: ",e)

    def plot_spectrogram(audio_id, identifier, audio_df, save_path_folder, spectrum, cumulus):
        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        # plot
        fig, ax = plt.subplots(2,1,figsize=(20,10), sharex=True)
        sub_audio_df.audio[0].plot(ax=ax[0], color='grey')
        sub_audio_df.audio[0].features.db_spectrogram().plot(ax=ax[1])
        ax[0].set_ylabel('Amplitude')
        ax[0].grid(False)
        ax[1].set_ylabel('F (KHz)')
        ax[1].set_xlabel('Time (seconds)')
        fig.text(0.75, 0.04, f"Cumulus: {cumulus} - Node: {node} - Recorder: {recorder}", va='center')
        plt.show()
        if save_path_folder:
            file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram.png")
            fig.savefig(file_path)
        
        save_metadata_spectrogram(audio_id, identifier, spectrum, save_path_folder, 
                                  cumulus, node, recorder, deployment, parent="Null")
        
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)    
            
        print(f"File saved at {file_path}")
        
    def produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start=None, 
                     colormap=cm.get_cmap("Greys"), min_spec=0, spec_range=1.0, figsize=(5, 4), 
                     dpi=100, bands=None):
        \'\'\'Takes an individual frame and produces an image with references\'\'\'
        frame = spec.cut_array(start_time=start, end_time=stop, min_freq=min_freq, max_freq=max_freq, pad=True)
        plt.style.use('dark_background')
        frame = np.flip((frame - min_spec)/spec_range, axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frame, cmap=colormap, extent=[0, frame_duration, min_freq/1000, max_freq/1000], 
                  aspect="auto", vmin = 0, vmax = 1.0)

        if bands is not None:
            band_arr = np.flip(resize(np.expand_dims(bands, axis=1), (frame.shape[0], frame.shape[1])), axis=0)
            ax.imshow(band_arr, extent=[0, frame_duration, min_freq/1000, max_freq/1000], aspect="auto", vmin = 0, 
                      vmax = 1.0, alpha=0.5)

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        mid = frame_duration/2.0
        ax.axvline(x=mid, color="red")
        ax.set_ylabel('F (kHz)')
        ax.set_xticks([])
        ax.set_xticks([], minor=True)

        if abs_start is not None:
            time_text = (abs_start + datetime.timedelta(seconds=start+mid)).strftime('%H:%M:%S.%f').strip()[:-4]
            ax.text(mid-0.3, -0.6, time_text)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        im = Image.open(buf)
        im.format = "PNG"
        plt.close(fig)

        return ImageClip(np.asarray(im),
                         duration=step)
        
    def remove_empty_folders(path_abs):
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            if len(os.listdir(path)) == 0:
                os.rmdir(path)            
                
    def save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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

    def save_metadata_spectrogram(product_id, identifier, product_spectrum,
                      path, cumulus, node, recorder, deployment, parent="Null"):
        product_name = "Spectrogram"
        file_path = os.path.join(path, f"{identifier}_spectrogram.png")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": parent,
            "product_name": product_name,
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
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def save_metadata_videoclip(product_id, identifier, product_spectrum, path, cumulus, node, recorder, 
                                deployment, clip_start, clip_end, parent="Null"):
        product_name = "spectrogram_video"
        file_path = os.path.join(path, f"{identifier}_spectrogram_video.mp4")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_video_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": "Null",
            "product_name": product_name,
            "product_path": file_path,
            "product_spectrum": product_spectrum,
            "CumulusName": cumulus,
            "NodeCategoryIntegrity": node_category,
            "NomenclatureNode": node,
            "SerialNumber": recorder,
            "DateDeployment": deployment,
            "ClipStart": clip_start,
            "ClipEnd": clip_end
        }
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def upload(session, node_id, data, file):
        """
        Uploads a file to a specific folder.
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            data (dict):                Dict that contains file options
            file (object):              File to upload
        
        Returns:
            (list):     A list containing status code and status data
        """

        try:
            response = session.post(os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT + "/nodes/" + node_id + "/children",
                        data = data,
                        files = file
                        )
                        
            return [response.json(), response.status_code];
        except Exception as e: 
            print("File " + data["name"] + " could not be uploaded: ", e)

    def upload_files(file_patterns ,session, node_id, dir_path, recursive, file_identifier=""):
        """
        Uploads the files stored in a specific dir
        to alfresco
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            dir_path (string):          The name and path of the dir where files are stored
            recursive (boolean):        A boolean to know if upload  must be recursive
                                        in the specifed dir, and should preserve the
                                        structure of dirs inside.
            file_identifier (string):   File identifier for all files inside a dir
        Returns:
            (string):           Returns the info of recent created site.
        """

        if recursive:
            expression = "/**/*"
        else:
            expression = "/*"

        files_in_dir = list(
            itertools.chain.from_iterable(
                glob.iglob(dir_path + expression + pattern, recursive=recursive)
                for pattern in file_patterns
            )
        )
        print("files_in_dir", files_in_dir)
        filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        total_files = len(files_in_dir)
        print(total_files)
        starttime = time.time()

        try:
            files_uploaded = []
            for idx, file_with_path in enumerate(files_in_dir):

                # total time since last login or script start
                total_time = round((time.time() - starttime), 2)

                if total_time > 2400:
                    """
                    if total time is bigger than 2400
                    or 40 minutes relogin to avoid ticket
                    expiration
                    """
                    time.sleep(5)

                    print("Re-logging in to alfresco...")

                    session = login.login()
                    # restart time
                    starttime = time.time()
                    time.sleep(5)
                    print("Login sucessful, continuing upload\\n")

                len_of_path = len(file_with_path.split("/"))
                name_of_file = file_with_path.split("/")[len_of_path - 1]
                root_dir_path = file_with_path.replace(dir_path, "").replace(
                    file_with_path.split("/")[len_of_path - 1], ""
                )

                data = {
                    "name": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    ),
                    "nodeType": "cm:content",
                }

                data["relativePath"] = root_dir_path

                data["properties"] = {
                    "cm:title": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    )
                }

                print("Uploading " + data["name"] + " file...")

                files = {"filedata": open(file_with_path, "rb")}
                upload_response = upload(session, node_id, data, files)
                if upload_response[1] and upload_response[1] == 201:
                    files_uploaded.append(upload_response[0])
                    print("Uploaded " + data["name"])

                    filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
                    with open(filename, 'a') as log_file:
                        log_file.writelines("%s\\n" % file_with_path)

                elif upload_response[1] and upload_response[1] == 409:
                    if "already exists" in upload_response[0]["error"]["errorKey"]:
                        print("File " + data["name"] + " already uploaded")

                else:
                    print("An error ocurred, file " + data["name"] + " cannot be uploaded")

                print("Uploaded file " + str(idx + 1) + " of " + str(total_files))
                print("\\n\\n")

            return files_uploaded
        except Exception as e:
            print("An error ocurred in file upload: ", e)
        
    def vectorize_soundscape(df, hash_name, indices):
        \'\'\'Return dataframe with array column containing indices by frequency\'\'\'
        return (df
                .groupby(by=["id", hash_name, "start_time", "end_time"])
                .apply(get_vectors, indices)
                .reset_index()
                .rename(columns={0:"index_vector"}))
    '''

    _kale_block3 = '''
    plt.style.use('dark_background')
    warnings.filterwarnings('ignore')

    sub_folder_results = find_subfolders(RESULTS_DIR)

    for sc_path in sub_folder_results:
        ids_audios = get_audio_ids(sc_path, indices = ["EXAG", "ICOMPLEXITY", "CORE"])
        idx_audio = 1
        for id_audio in ids_audios:
            print(f"Processing audio {id_audio}")
            plot_spectrogram(id_audio, f"spl{idx_audio}", recs, sc_path, SPECTRUM, CUMULO)   
            audio2video(id_audio, f"spl{idx_audio}", recs, sc_path, SPECTRUM, CUMULO)
            idx_audio += 1
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.sndscs_spec_specvid-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
    _kale_marshal.save(sub_folder_results, "sub_folder_results")
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
    with open("/spec_n_specvid.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('spec_n_specvid')

    _kale_mlmdutils.call("mark_execution_complete")


def upload_to_alfresco(ALFRESCO_NODE_ID: str, AUTH_ENDPOINT: str, BASE_ENDPOINT: str, RESULTS_DIR: str):
    _kale_pipeline_parameters_block = '''
    ALFRESCO_NODE_ID = "{}"
    AUTH_ENDPOINT = "{}"
    BASE_ENDPOINT = "{}"
    RESULTS_DIR = "{}"
    '''.format(ALFRESCO_NODE_ID, AUTH_ENDPOINT, BASE_ENDPOINT, RESULTS_DIR)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import base64
    import datetime
    import glob
    import hashlib
    import io
    import itertools
    import json
    import matplotlib.pyplot as plt
    import multiprocessing 
    import numpy as np
    import os
    import pandas as pd
    import psutil
    import requests
    import shutil
    import subprocess
    import time
    import warnings

    from dask.distributed import Client, LocalCluster
    from datetime import timedelta
    from dotenv import load_dotenv
    from matplotlib import cm
    from moviepy.editor import concatenate, VideoFileClip, AudioFileClip
    from moviepy.audio.AudioClip import AudioArrayClip
    from moviepy.video.VideoClip import ImageClip
    from os.path import exists as file_exists
    from PIL import Image
    from skimage.transform import resize

    from yuntu import Audio
    from yuntu.soundscape.utils import aware_time
    from yuntu.collection.methods import collection
    from yuntu.soundscape.hashers.crono import DEFAULT_HASHER_CONFIG
    from yuntu.soundscape.processors.indices.direct import ICOMPLEXITY, TAIL
    from yuntu.soundscape.pipelines.build_soundscape import CronoSoundscape, HASHER_CONFIG
    '''

    _kale_block2 = '''
    def audio2video(audio_id,
                    identifier,
                    audio_df,
                    save_path_folder,
                    product_spectrum,
                    cumulus,
                    abs_start=None,
                    fps=60,
                    spec_configs={'hop_length': 512, 'n_fft': 1024, 'window_function': 'hann'},
                    rate=24,
                    frame_duration=3.0,
                    min_freq=0,
                    max_freq=None,
                    cmap="Greys",
                    figsize=(5, 8),
                    dpi=100,
                    bands=None):
        \'\'\'Takes and audio object and produces a mp4 video of the spectrogram with audio\'\'\'

        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        id_audio = sub_audio_df['id'].values[0]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        audio = sub_audio_df.audio[0].cut(0,1)
        
        colormap = cm.get_cmap(cmap)
        duration = audio.duration
        step = 1/rate
        start = -(frame_duration/2.0)
        stop = start + frame_duration
        clips = []
        last_stop = None

        if max_freq is None:
            max_freq = audio.samplerate / 2.0
        
        if min_freq is None:
            min_freq = 0
        

        with audio.features.db_spectrogram(**spec_configs) as spec:
            min_spec = np.amin(spec)
            max_spec = np.amax(spec)
            spec_range = (max_spec-min_spec)
            
            while stop <= duration+(frame_duration/2.0):
                clip = produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start, colormap,
                                    min_spec, spec_range, figsize, dpi, bands=bands)
                clips.append(clip)
                
                if start + step + frame_duration > duration:
                    last_stop = stop

                start = start + step
                stop = start + frame_duration
        
        video = concatenate(clips)
        # edaudio = AudioArrayClip(audio_array, fps=audio.samplerate)
        edaudio = AudioFileClip(audio.path).set_end(audio.duration)
        video = video.set_audio(edaudio)
        file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram_video.mp4")
        video.write_videofile(file_path, fps=fps)
        
        save_metadata_videoclip(id_audio, identifier, product_spectrum,
                      save_path_folder, cumulus, node, recorder, deployment, 0.0, audio.duration)
        video.close()
        edaudio.close()
        
        for c in clips:
            c.close()

    def change_type_sipecam_sc(session, root_folder_id, path, file_type):
        if file_type == "sequence.png":
            metadata_name = "soundscape_seq_metadata.json"
            aggr_type = "None"
        elif file_type == "mean_soundscape.png":
            metadata_name = "mean_soundscape_metadata.json"
            aggr_type = "Mean"
        elif file_type ==  "std_soundscape.png":
            metadata_name = "std_soundscape_metadata.json" 
            aggr_type = "Standard deviation"
        elif file_type == "hashed_soundscape.parquet":
            metadata_name = "soundscape_metadata.json.json" 
            aggr_type = "None"

        try:
            semi_path = path.split("soundscapes/")[-1]
            semi_path_file = os.path.join(semi_path, file_type)
            local_path_file_metadata = os.path.join(path, metadata_name)
            print(f"Changing type for {os.path.join(semi_path_file)}")
            alfresco_path = os.path.join("/Company Home/Sites/sipecam-soundscape/documentLibrary/", semi_path)
            response = session.get(
                os.getenv("ALFRESCO_URL")
                + BASE_ENDPOINT
                + "/nodes/"
                + root_folder_id
                + "/children?relativePath="+semi_path_file+"&include=aspectNames&skipCount=0&maxItems=1"
            )

            # error flag
            is_error = False

            # if request is successful then continue
            if response.status_code == 200:

                data_file = open(local_path_file_metadata)
                data_json = json.load(data_file)
                response_entries = response.json()["list"]["entries"][0]

                if response_entries["entry"]["isFile"]:

                    prop_dict = {}
                    # map properties
                    prop_dict["soundscape:CumulusName"] = str(data_json["CumulusName"])
                    prop_dict["soundscape:DateDeployment"] = data_json["DateDeployment"]
                    prop_dict["soundscape:NodeCategoryIntegrity"] = str(data_json["NodeCategoryIntegrity"])
                    prop_dict["soundscape:NomenclatureNode"] = str(data_json["NomenclatureNode"])
                    prop_dict["soundscape:SerialNumber"] = str(data_json["SerialNumber"])
                    prop_dict["soundscape:aggr"] = str(aggr_type)
                    prop_dict["soundscape:cycle_config_aware_start"] = str(data_json["product_configs"]['hasher_config']['kwargs']['aware_start'])
                    prop_dict["soundscape:cycle_config_start_format"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_format'])
                    prop_dict["soundscape:cycle_config_start_time"] = data_json["product_configs"]['hasher_config']['kwargs']['start_time'] #
                    prop_dict["soundscape:cycle_config_start_tzone"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_tzone'])
                    prop_dict["soundscape:cycle_config_time_module"] = int(data_json["product_configs"]['hasher_config']['kwargs']['time_module'])
                    prop_dict["soundscape:cycle_config_time_unit"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_unit'])
                    prop_dict["soundscape:cycle_config_time_utc_column"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_utc_column'])
                    prop_dict["soundscape:frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:frequency_hop"] = int(data_json["product_configs"]["slice_config"]["frequency_hop"])
                    prop_dict["soundscape:frequency_limits"] = str(data_json["product_configs"]["slice_config"]["frequency_limits"])
                    prop_dict["soundscape:hash_name"] = str(data_json["product_configs"]["hash_name"])
                    prop_dict["soundscape:hop_length"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["hop_length"])
                    prop_dict["soundscape:indices"] = str(data_json['product_configs']['indices'])
                    prop_dict["soundscape:n_fft"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["n_fft"])
                    prop_dict["soundscape:npartitions"] = int(data_json["product_configs"]['npartitions'])
                    prop_dict["soundscape:product_id"] = str(data_json["product_id"])
                    prop_dict["soundscape:product_name"] = str(data_json["product_name"])
                    prop_dict["soundscape:product_parent"] = str(data_json["product_parent"])
                    prop_dict["soundscape:product_path"] = str(alfresco_path)
                    prop_dict["soundscape:product_spectrum"] = str(data_json["product_spectrum"])
                    prop_dict["soundscape:slice_config_feature_type"] = str(data_json["product_configs"]["slice_config"]["feature_type"])
                    prop_dict["soundscape:slice_config_frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:slice_config_time_unit"] = int(data_json["product_configs"]["slice_config"]["time_unit"])
                    prop_dict["soundscape:time_hop"] = int(data_json["product_configs"]["slice_config"]["time_hop"])
                    prop_dict["soundscape:window_function"] = str(data_json["product_configs"]["slice_config"]["feature_config"]["window_function"])


                    aspects = response_entries["entry"]["aspectNames"]

                    data = {"aspectNames": aspects, "nodeType": 'soundscape:product', "properties": prop_dict}

                    # update properties request
                    update = session.put(
                        os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT
                        + "/nodes/"
                        + response_entries["entry"]["id"],
                        data=json.dumps(data),
                    )
                    print(update.json())
                    if update.status_code == 200:
                        print("Updated " + response_entries["entry"]["id"])
        except Exception as e:
            print("Could not add any aspect to this file: ", e)
            
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
                    
    def distance_to_mean(vector, mean):
        \'\'\'Return euclidean distance to mean\'\'\'
        return np.sqrt(np.sum(np.square(mean - vector)))

    def find_subfolders(path_abs):
        subdir_list = []
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            len_path = path.split("/")
            if len(len_path) == 8:
                subdir_list.append(path)  
                
        return subdir_list

    def get_audio_ids(soundscape_path, indices):
        df = pd.read_parquet(os.path.join(soundscape_path, "hashed_soundscape.parquet"))

        with open(os.path.join(soundscape_path, "soundscape_metadata.json")) as f:
            metadata = json.load(f)
            f.close()

        # indices = metadata["product_configs"]["indices"]
        # indices = ["EXAG", "ICOMPLEXITY", "CORE"]
        hash_name = metadata["product_configs"]["hash_name"]
        cycle_config = metadata["product_configs"]["hasher_config"]["kwargs"]
        time_unit = cycle_config["time_unit"]
        zero_t = aware_time( cycle_config["start_time"], cycle_config["start_tzone"], cycle_config["start_format"]) 
        
        # sample
        samples_df = get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=3)
        sub_df = samples_df[samples_df.crono_hash_30m == 8]
        audio_id_list = list(sub_df["id"].unique())
        
        return audio_id_list

    def get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=5):
        \'\'\'Return dataframe of 'nsamples' samples for each tag in 'hash_name' column that are closest to the mean vector by tag\'\'\'
        proj_df = df[(df.max_freq <= 10000)]
        crono_tags = proj_df.crono_hash_30m.unique()
        proj_df.loc[: , f"{hash_name}_time"] = proj_df[hash_name].apply(lambda x: zero_t + datetime.timedelta(seconds=float(x*time_unit)))
        vectors = vectorize_soundscape(proj_df, hash_name, indices)
        min_index_vector = np.amin(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        max_index_vector = np.amax(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        index_range = (max_index_vector - min_index_vector)
        vectors.loc[:, "normalized_index_vector"] = vectors.index_vector.apply(lambda x: (x-min_index_vector)/index_range)
        all_samples = []

        for crono_tag in crono_tags:
            unit_vectors = vectors[vectors[hash_name] == crono_tag]
            mean_unit_vector = unit_vectors.normalized_index_vector.mean()
            unit_vectors.loc[:, "distance"] = unit_vectors.normalized_index_vector.apply(lambda x: distance_to_mean(x, mean_unit_vector))
            all_samples.append(unit_vectors.sort_values(by="distance").head(nsamples))

        return pd.concat(all_samples)

    def get_vectors(group, indices):
        \'\'\'Return array of indices by frequency\'\'\'
        return group.sort_values(by="max_freq")[indices].values

    def login():
        """
        Tries a login to alfresco api and returns a session
        object with credentials 
        Returns: 
            session (Session):  A session object to make 
                                requests to zendro.
        """
        try:
            auth = {
                "userId": os.getenv("ALFRESCO_USER"),
                "password": os.getenv("ALFRESCO_PASSWORD"),
            }

            login = requests.post(os.getenv("ALFRESCO_URL") + AUTH_ENDPOINT + "/tickets",data=json.dumps(auth))

            base64_login = base64.b64encode(bytes(login.json()["entry"]["id"], 'utf-8')).decode()

            # se crea un objeto de Session para hacer requests
            session = requests.Session()
            # se establece bearer token
            session.headers.update({'Authorization': 'Basic ' + base64_login})

            return session
        except Exception as e:
            print("Login failed: ",e)

    def plot_spectrogram(audio_id, identifier, audio_df, save_path_folder, spectrum, cumulus):
        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        # plot
        fig, ax = plt.subplots(2,1,figsize=(20,10), sharex=True)
        sub_audio_df.audio[0].plot(ax=ax[0], color='grey')
        sub_audio_df.audio[0].features.db_spectrogram().plot(ax=ax[1])
        ax[0].set_ylabel('Amplitude')
        ax[0].grid(False)
        ax[1].set_ylabel('F (KHz)')
        ax[1].set_xlabel('Time (seconds)')
        fig.text(0.75, 0.04, f"Cumulus: {cumulus} - Node: {node} - Recorder: {recorder}", va='center')
        plt.show()
        if save_path_folder:
            file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram.png")
            fig.savefig(file_path)
        
        save_metadata_spectrogram(audio_id, identifier, spectrum, save_path_folder, 
                                  cumulus, node, recorder, deployment, parent="Null")
        
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)    
            
        print(f"File saved at {file_path}")
        
    def produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start=None, 
                     colormap=cm.get_cmap("Greys"), min_spec=0, spec_range=1.0, figsize=(5, 4), 
                     dpi=100, bands=None):
        \'\'\'Takes an individual frame and produces an image with references\'\'\'
        frame = spec.cut_array(start_time=start, end_time=stop, min_freq=min_freq, max_freq=max_freq, pad=True)
        plt.style.use('dark_background')
        frame = np.flip((frame - min_spec)/spec_range, axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frame, cmap=colormap, extent=[0, frame_duration, min_freq/1000, max_freq/1000], 
                  aspect="auto", vmin = 0, vmax = 1.0)

        if bands is not None:
            band_arr = np.flip(resize(np.expand_dims(bands, axis=1), (frame.shape[0], frame.shape[1])), axis=0)
            ax.imshow(band_arr, extent=[0, frame_duration, min_freq/1000, max_freq/1000], aspect="auto", vmin = 0, 
                      vmax = 1.0, alpha=0.5)

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        mid = frame_duration/2.0
        ax.axvline(x=mid, color="red")
        ax.set_ylabel('F (kHz)')
        ax.set_xticks([])
        ax.set_xticks([], minor=True)

        if abs_start is not None:
            time_text = (abs_start + datetime.timedelta(seconds=start+mid)).strftime('%H:%M:%S.%f').strip()[:-4]
            ax.text(mid-0.3, -0.6, time_text)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        im = Image.open(buf)
        im.format = "PNG"
        plt.close(fig)

        return ImageClip(np.asarray(im),
                         duration=step)
        
    def remove_empty_folders(path_abs):
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            if len(os.listdir(path)) == 0:
                os.rmdir(path)            
                
    def save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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

    def save_metadata_spectrogram(product_id, identifier, product_spectrum,
                      path, cumulus, node, recorder, deployment, parent="Null"):
        product_name = "Spectrogram"
        file_path = os.path.join(path, f"{identifier}_spectrogram.png")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": parent,
            "product_name": product_name,
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
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def save_metadata_videoclip(product_id, identifier, product_spectrum, path, cumulus, node, recorder, 
                                deployment, clip_start, clip_end, parent="Null"):
        product_name = "spectrogram_video"
        file_path = os.path.join(path, f"{identifier}_spectrogram_video.mp4")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_video_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": "Null",
            "product_name": product_name,
            "product_path": file_path,
            "product_spectrum": product_spectrum,
            "CumulusName": cumulus,
            "NodeCategoryIntegrity": node_category,
            "NomenclatureNode": node,
            "SerialNumber": recorder,
            "DateDeployment": deployment,
            "ClipStart": clip_start,
            "ClipEnd": clip_end
        }
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def upload(session, node_id, data, file):
        """
        Uploads a file to a specific folder.
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            data (dict):                Dict that contains file options
            file (object):              File to upload
        
        Returns:
            (list):     A list containing status code and status data
        """

        try:
            response = session.post(os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT + "/nodes/" + node_id + "/children",
                        data = data,
                        files = file
                        )
                        
            return [response.json(), response.status_code];
        except Exception as e: 
            print("File " + data["name"] + " could not be uploaded: ", e)

    def upload_files(file_patterns ,session, node_id, dir_path, recursive, file_identifier=""):
        """
        Uploads the files stored in a specific dir
        to alfresco
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            dir_path (string):          The name and path of the dir where files are stored
            recursive (boolean):        A boolean to know if upload  must be recursive
                                        in the specifed dir, and should preserve the
                                        structure of dirs inside.
            file_identifier (string):   File identifier for all files inside a dir
        Returns:
            (string):           Returns the info of recent created site.
        """

        if recursive:
            expression = "/**/*"
        else:
            expression = "/*"

        files_in_dir = list(
            itertools.chain.from_iterable(
                glob.iglob(dir_path + expression + pattern, recursive=recursive)
                for pattern in file_patterns
            )
        )
        print("files_in_dir", files_in_dir)
        filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        total_files = len(files_in_dir)
        print(total_files)
        starttime = time.time()

        try:
            files_uploaded = []
            for idx, file_with_path in enumerate(files_in_dir):

                # total time since last login or script start
                total_time = round((time.time() - starttime), 2)

                if total_time > 2400:
                    """
                    if total time is bigger than 2400
                    or 40 minutes relogin to avoid ticket
                    expiration
                    """
                    time.sleep(5)

                    print("Re-logging in to alfresco...")

                    session = login.login()
                    # restart time
                    starttime = time.time()
                    time.sleep(5)
                    print("Login sucessful, continuing upload\\n")

                len_of_path = len(file_with_path.split("/"))
                name_of_file = file_with_path.split("/")[len_of_path - 1]
                root_dir_path = file_with_path.replace(dir_path, "").replace(
                    file_with_path.split("/")[len_of_path - 1], ""
                )

                data = {
                    "name": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    ),
                    "nodeType": "cm:content",
                }

                data["relativePath"] = root_dir_path

                data["properties"] = {
                    "cm:title": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    )
                }

                print("Uploading " + data["name"] + " file...")

                files = {"filedata": open(file_with_path, "rb")}
                upload_response = upload(session, node_id, data, files)
                if upload_response[1] and upload_response[1] == 201:
                    files_uploaded.append(upload_response[0])
                    print("Uploaded " + data["name"])

                    filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
                    with open(filename, 'a') as log_file:
                        log_file.writelines("%s\\n" % file_with_path)

                elif upload_response[1] and upload_response[1] == 409:
                    if "already exists" in upload_response[0]["error"]["errorKey"]:
                        print("File " + data["name"] + " already uploaded")

                else:
                    print("An error ocurred, file " + data["name"] + " cannot be uploaded")

                print("Uploaded file " + str(idx + 1) + " of " + str(total_files))
                print("\\n\\n")

            return files_uploaded
        except Exception as e:
            print("An error ocurred in file upload: ", e)
        
    def vectorize_soundscape(df, hash_name, indices):
        \'\'\'Return dataframe with array column containing indices by frequency\'\'\'
        return (df
                .groupby(by=["id", hash_name, "start_time", "end_time"])
                .apply(get_vectors, indices)
                .reset_index()
                .rename(columns={0:"index_vector"}))
    '''

    _kale_block3 = '''
    FILE_PATTERNS = [".mp4", ".png"] #".parquet"
    session = login()
    upload_files(FILE_PATTERNS, session, ALFRESCO_NODE_ID, RESULTS_DIR, recursive= True, file_identifier="")
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
    with open("/upload_to_alfresco.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('upload_to_alfresco')

    _kale_mlmdutils.call("mark_execution_complete")


def upload_alfresco_model_data(ALFRESCO_NODE_ID: str, AUTH_ENDPOINT: str, BASE_ENDPOINT: str):
    _kale_pipeline_parameters_block = '''
    ALFRESCO_NODE_ID = "{}"
    AUTH_ENDPOINT = "{}"
    BASE_ENDPOINT = "{}"
    '''.format(ALFRESCO_NODE_ID, AUTH_ENDPOINT, BASE_ENDPOINT)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/audio/.sndscs_spec_specvid-sipecam-cumulus-node-recorder-deployment-aws.ipynb.kale.marshal.dir")
    sub_folder_results = _kale_marshal.load("sub_folder_results")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import base64
    import datetime
    import glob
    import hashlib
    import io
    import itertools
    import json
    import matplotlib.pyplot as plt
    import multiprocessing 
    import numpy as np
    import os
    import pandas as pd
    import psutil
    import requests
    import shutil
    import subprocess
    import time
    import warnings

    from dask.distributed import Client, LocalCluster
    from datetime import timedelta
    from dotenv import load_dotenv
    from matplotlib import cm
    from moviepy.editor import concatenate, VideoFileClip, AudioFileClip
    from moviepy.audio.AudioClip import AudioArrayClip
    from moviepy.video.VideoClip import ImageClip
    from os.path import exists as file_exists
    from PIL import Image
    from skimage.transform import resize

    from yuntu import Audio
    from yuntu.soundscape.utils import aware_time
    from yuntu.collection.methods import collection
    from yuntu.soundscape.hashers.crono import DEFAULT_HASHER_CONFIG
    from yuntu.soundscape.processors.indices.direct import ICOMPLEXITY, TAIL
    from yuntu.soundscape.pipelines.build_soundscape import CronoSoundscape, HASHER_CONFIG
    '''

    _kale_block2 = '''
    def audio2video(audio_id,
                    identifier,
                    audio_df,
                    save_path_folder,
                    product_spectrum,
                    cumulus,
                    abs_start=None,
                    fps=60,
                    spec_configs={'hop_length': 512, 'n_fft': 1024, 'window_function': 'hann'},
                    rate=24,
                    frame_duration=3.0,
                    min_freq=0,
                    max_freq=None,
                    cmap="Greys",
                    figsize=(5, 8),
                    dpi=100,
                    bands=None):
        \'\'\'Takes and audio object and produces a mp4 video of the spectrogram with audio\'\'\'

        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        id_audio = sub_audio_df['id'].values[0]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        audio = sub_audio_df.audio[0].cut(0,1)
        
        colormap = cm.get_cmap(cmap)
        duration = audio.duration
        step = 1/rate
        start = -(frame_duration/2.0)
        stop = start + frame_duration
        clips = []
        last_stop = None

        if max_freq is None:
            max_freq = audio.samplerate / 2.0
        
        if min_freq is None:
            min_freq = 0
        

        with audio.features.db_spectrogram(**spec_configs) as spec:
            min_spec = np.amin(spec)
            max_spec = np.amax(spec)
            spec_range = (max_spec-min_spec)
            
            while stop <= duration+(frame_duration/2.0):
                clip = produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start, colormap,
                                    min_spec, spec_range, figsize, dpi, bands=bands)
                clips.append(clip)
                
                if start + step + frame_duration > duration:
                    last_stop = stop

                start = start + step
                stop = start + frame_duration
        
        video = concatenate(clips)
        # edaudio = AudioArrayClip(audio_array, fps=audio.samplerate)
        edaudio = AudioFileClip(audio.path).set_end(audio.duration)
        video = video.set_audio(edaudio)
        file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram_video.mp4")
        video.write_videofile(file_path, fps=fps)
        
        save_metadata_videoclip(id_audio, identifier, product_spectrum,
                      save_path_folder, cumulus, node, recorder, deployment, 0.0, audio.duration)
        video.close()
        edaudio.close()
        
        for c in clips:
            c.close()

    def change_type_sipecam_sc(session, root_folder_id, path, file_type):
        if file_type == "sequence.png":
            metadata_name = "soundscape_seq_metadata.json"
            aggr_type = "None"
        elif file_type == "mean_soundscape.png":
            metadata_name = "mean_soundscape_metadata.json"
            aggr_type = "Mean"
        elif file_type ==  "std_soundscape.png":
            metadata_name = "std_soundscape_metadata.json" 
            aggr_type = "Standard deviation"
        elif file_type == "hashed_soundscape.parquet":
            metadata_name = "soundscape_metadata.json.json" 
            aggr_type = "None"

        try:
            semi_path = path.split("soundscapes/")[-1]
            semi_path_file = os.path.join(semi_path, file_type)
            local_path_file_metadata = os.path.join(path, metadata_name)
            print(f"Changing type for {os.path.join(semi_path_file)}")
            alfresco_path = os.path.join("/Company Home/Sites/sipecam-soundscape/documentLibrary/", semi_path)
            response = session.get(
                os.getenv("ALFRESCO_URL")
                + BASE_ENDPOINT
                + "/nodes/"
                + root_folder_id
                + "/children?relativePath="+semi_path_file+"&include=aspectNames&skipCount=0&maxItems=1"
            )

            # error flag
            is_error = False

            # if request is successful then continue
            if response.status_code == 200:

                data_file = open(local_path_file_metadata)
                data_json = json.load(data_file)
                response_entries = response.json()["list"]["entries"][0]

                if response_entries["entry"]["isFile"]:

                    prop_dict = {}
                    # map properties
                    prop_dict["soundscape:CumulusName"] = str(data_json["CumulusName"])
                    prop_dict["soundscape:DateDeployment"] = data_json["DateDeployment"]
                    prop_dict["soundscape:NodeCategoryIntegrity"] = str(data_json["NodeCategoryIntegrity"])
                    prop_dict["soundscape:NomenclatureNode"] = str(data_json["NomenclatureNode"])
                    prop_dict["soundscape:SerialNumber"] = str(data_json["SerialNumber"])
                    prop_dict["soundscape:aggr"] = str(aggr_type)
                    prop_dict["soundscape:cycle_config_aware_start"] = str(data_json["product_configs"]['hasher_config']['kwargs']['aware_start'])
                    prop_dict["soundscape:cycle_config_start_format"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_format'])
                    prop_dict["soundscape:cycle_config_start_time"] = data_json["product_configs"]['hasher_config']['kwargs']['start_time'] #
                    prop_dict["soundscape:cycle_config_start_tzone"] = str(data_json["product_configs"]['hasher_config']['kwargs']['start_tzone'])
                    prop_dict["soundscape:cycle_config_time_module"] = int(data_json["product_configs"]['hasher_config']['kwargs']['time_module'])
                    prop_dict["soundscape:cycle_config_time_unit"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_unit'])
                    prop_dict["soundscape:cycle_config_time_utc_column"] = str(data_json["product_configs"]['hasher_config']['kwargs']['time_utc_column'])
                    prop_dict["soundscape:frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:frequency_hop"] = int(data_json["product_configs"]["slice_config"]["frequency_hop"])
                    prop_dict["soundscape:frequency_limits"] = str(data_json["product_configs"]["slice_config"]["frequency_limits"])
                    prop_dict["soundscape:hash_name"] = str(data_json["product_configs"]["hash_name"])
                    prop_dict["soundscape:hop_length"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["hop_length"])
                    prop_dict["soundscape:indices"] = str(data_json['product_configs']['indices'])
                    prop_dict["soundscape:n_fft"] = int(data_json["product_configs"]["slice_config"]["feature_config"]["n_fft"])
                    prop_dict["soundscape:npartitions"] = int(data_json["product_configs"]['npartitions'])
                    prop_dict["soundscape:product_id"] = str(data_json["product_id"])
                    prop_dict["soundscape:product_name"] = str(data_json["product_name"])
                    prop_dict["soundscape:product_parent"] = str(data_json["product_parent"])
                    prop_dict["soundscape:product_path"] = str(alfresco_path)
                    prop_dict["soundscape:product_spectrum"] = str(data_json["product_spectrum"])
                    prop_dict["soundscape:slice_config_feature_type"] = str(data_json["product_configs"]["slice_config"]["feature_type"])
                    prop_dict["soundscape:slice_config_frequency_bins"] = int(data_json["product_configs"]["slice_config"]["frequency_bins"])
                    prop_dict["soundscape:slice_config_time_unit"] = int(data_json["product_configs"]["slice_config"]["time_unit"])
                    prop_dict["soundscape:time_hop"] = int(data_json["product_configs"]["slice_config"]["time_hop"])
                    prop_dict["soundscape:window_function"] = str(data_json["product_configs"]["slice_config"]["feature_config"]["window_function"])


                    aspects = response_entries["entry"]["aspectNames"]

                    data = {"aspectNames": aspects, "nodeType": 'soundscape:product', "properties": prop_dict}

                    # update properties request
                    update = session.put(
                        os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT
                        + "/nodes/"
                        + response_entries["entry"]["id"],
                        data=json.dumps(data),
                    )
                    print(update.json())
                    if update.status_code == 200:
                        print("Updated " + response_entries["entry"]["id"])
        except Exception as e:
            print("Could not add any aspect to this file: ", e)
            
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
                    
    def distance_to_mean(vector, mean):
        \'\'\'Return euclidean distance to mean\'\'\'
        return np.sqrt(np.sum(np.square(mean - vector)))

    def find_subfolders(path_abs):
        subdir_list = []
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            len_path = path.split("/")
            if len(len_path) == 8:
                subdir_list.append(path)  
                
        return subdir_list

    def get_audio_ids(soundscape_path, indices):
        df = pd.read_parquet(os.path.join(soundscape_path, "hashed_soundscape.parquet"))

        with open(os.path.join(soundscape_path, "soundscape_metadata.json")) as f:
            metadata = json.load(f)
            f.close()

        # indices = metadata["product_configs"]["indices"]
        # indices = ["EXAG", "ICOMPLEXITY", "CORE"]
        hash_name = metadata["product_configs"]["hash_name"]
        cycle_config = metadata["product_configs"]["hasher_config"]["kwargs"]
        time_unit = cycle_config["time_unit"]
        zero_t = aware_time( cycle_config["start_time"], cycle_config["start_tzone"], cycle_config["start_format"]) 
        
        # sample
        samples_df = get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=3)
        sub_df = samples_df[samples_df.crono_hash_30m == 8]
        audio_id_list = list(sub_df["id"].unique())
        
        return audio_id_list

    def get_recording_samples(df, hash_name, indices, time_unit, zero_t, nsamples=5):
        \'\'\'Return dataframe of 'nsamples' samples for each tag in 'hash_name' column that are closest to the mean vector by tag\'\'\'
        proj_df = df[(df.max_freq <= 10000)]
        crono_tags = proj_df.crono_hash_30m.unique()
        proj_df.loc[: , f"{hash_name}_time"] = proj_df[hash_name].apply(lambda x: zero_t + datetime.timedelta(seconds=float(x*time_unit)))
        vectors = vectorize_soundscape(proj_df, hash_name, indices)
        min_index_vector = np.amin(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        max_index_vector = np.amax(np.stack(list(vectors.index_vector.values)), axis=(0,1))
        index_range = (max_index_vector - min_index_vector)
        vectors.loc[:, "normalized_index_vector"] = vectors.index_vector.apply(lambda x: (x-min_index_vector)/index_range)
        all_samples = []

        for crono_tag in crono_tags:
            unit_vectors = vectors[vectors[hash_name] == crono_tag]
            mean_unit_vector = unit_vectors.normalized_index_vector.mean()
            unit_vectors.loc[:, "distance"] = unit_vectors.normalized_index_vector.apply(lambda x: distance_to_mean(x, mean_unit_vector))
            all_samples.append(unit_vectors.sort_values(by="distance").head(nsamples))

        return pd.concat(all_samples)

    def get_vectors(group, indices):
        \'\'\'Return array of indices by frequency\'\'\'
        return group.sort_values(by="max_freq")[indices].values

    def login():
        """
        Tries a login to alfresco api and returns a session
        object with credentials 
        Returns: 
            session (Session):  A session object to make 
                                requests to zendro.
        """
        try:
            auth = {
                "userId": os.getenv("ALFRESCO_USER"),
                "password": os.getenv("ALFRESCO_PASSWORD"),
            }

            login = requests.post(os.getenv("ALFRESCO_URL") + AUTH_ENDPOINT + "/tickets",data=json.dumps(auth))

            base64_login = base64.b64encode(bytes(login.json()["entry"]["id"], 'utf-8')).decode()

            # se crea un objeto de Session para hacer requests
            session = requests.Session()
            # se establece bearer token
            session.headers.update({'Authorization': 'Basic ' + base64_login})

            return session
        except Exception as e:
            print("Login failed: ",e)

    def plot_spectrogram(audio_id, identifier, audio_df, save_path_folder, spectrum, cumulus):
        sub_audio_df = audio_df[audio_df["id"]==audio_id]
        node = sub_audio_df['node'].values[0]
        recorder = sub_audio_df['recorder'].values[0]
        deployment = sub_audio_df['deployment'].values[0]
        # plot
        fig, ax = plt.subplots(2,1,figsize=(20,10), sharex=True)
        sub_audio_df.audio[0].plot(ax=ax[0], color='grey')
        sub_audio_df.audio[0].features.db_spectrogram().plot(ax=ax[1])
        ax[0].set_ylabel('Amplitude')
        ax[0].grid(False)
        ax[1].set_ylabel('F (KHz)')
        ax[1].set_xlabel('Time (seconds)')
        fig.text(0.75, 0.04, f"Cumulus: {cumulus} - Node: {node} - Recorder: {recorder}", va='center')
        plt.show()
        if save_path_folder:
            file_path = os.path.join(save_path_folder, f"{identifier}_spectrogram.png")
            fig.savefig(file_path)
        
        save_metadata_spectrogram(audio_id, identifier, spectrum, save_path_folder, 
                                  cumulus, node, recorder, deployment, parent="Null")
        
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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
            save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
                      path, cumulus, node, recorder, deployment, parent)    
            
        print(f"File saved at {file_path}")
        
    def produce_clip(spec, frame_duration, min_freq, max_freq, start, stop, step, abs_start=None, 
                     colormap=cm.get_cmap("Greys"), min_spec=0, spec_range=1.0, figsize=(5, 4), 
                     dpi=100, bands=None):
        \'\'\'Takes an individual frame and produces an image with references\'\'\'
        frame = spec.cut_array(start_time=start, end_time=stop, min_freq=min_freq, max_freq=max_freq, pad=True)
        plt.style.use('dark_background')
        frame = np.flip((frame - min_spec)/spec_range, axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frame, cmap=colormap, extent=[0, frame_duration, min_freq/1000, max_freq/1000], 
                  aspect="auto", vmin = 0, vmax = 1.0)

        if bands is not None:
            band_arr = np.flip(resize(np.expand_dims(bands, axis=1), (frame.shape[0], frame.shape[1])), axis=0)
            ax.imshow(band_arr, extent=[0, frame_duration, min_freq/1000, max_freq/1000], aspect="auto", vmin = 0, 
                      vmax = 1.0, alpha=0.5)

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        mid = frame_duration/2.0
        ax.axvline(x=mid, color="red")
        ax.set_ylabel('F (kHz)')
        ax.set_xticks([])
        ax.set_xticks([], minor=True)

        if abs_start is not None:
            time_text = (abs_start + datetime.timedelta(seconds=start+mid)).strftime('%H:%M:%S.%f').strip()[:-4]
            ax.text(mid-0.3, -0.6, time_text)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        im = Image.open(buf)
        im.format = "PNG"
        plt.close(fig)

        return ImageClip(np.asarray(im),
                         duration=step)
        
    def remove_empty_folders(path_abs):
        walk = list(os.walk(path_abs))
        for path, _, _ in walk[::-1]:
            if len(os.listdir(path)) == 0:
                os.rmdir(path)            
                
    def save_metadata_sc(product_id, product_type, product_spectrum, sc_config,
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

    def save_metadata_spectrogram(product_id, identifier, product_spectrum,
                      path, cumulus, node, recorder, deployment, parent="Null"):
        product_name = "Spectrogram"
        file_path = os.path.join(path, f"{identifier}_spectrogram.png")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": parent,
            "product_name": product_name,
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
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def save_metadata_videoclip(product_id, identifier, product_spectrum, path, cumulus, node, recorder, 
                                deployment, clip_start, clip_end, parent="Null"):
        product_name = "spectrogram_video"
        file_path = os.path.join(path, f"{identifier}_spectrogram_video.mp4")
        metadata_filename = os.path.join(path, f"{identifier}_spectrogram_video_metadata.json")

        if int(node.split("_")[2]) == 0:
            node_category = "Degradado"
        elif int(node.split("_")[2]) == 1:
            node_category = "Integro"

        metadata = {
            "product_id": product_id,
            "product_parent": "Null",
            "product_name": product_name,
            "product_path": file_path,
            "product_spectrum": product_spectrum,
            "CumulusName": cumulus,
            "NodeCategoryIntegrity": node_category,
            "NomenclatureNode": node,
            "SerialNumber": recorder,
            "DateDeployment": deployment,
            "ClipStart": clip_start,
            "ClipEnd": clip_end
        }
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        print(f"{file_path} saved.")
        print(f"{metadata_filename} saved.")
        
    def upload(session, node_id, data, file):
        """
        Uploads a file to a specific folder.
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            data (dict):                Dict that contains file options
            file (object):              File to upload
        
        Returns:
            (list):     A list containing status code and status data
        """

        try:
            response = session.post(os.getenv("ALFRESCO_URL")
                        + BASE_ENDPOINT + "/nodes/" + node_id + "/children",
                        data = data,
                        files = file
                        )
                        
            return [response.json(), response.status_code];
        except Exception as e: 
            print("File " + data["name"] + " could not be uploaded: ", e)

    def upload_files(file_patterns ,session, node_id, dir_path, recursive, file_identifier=""):
        """
        Uploads the files stored in a specific dir
        to alfresco
        Parameters:
            session (Session):          A session object to make
                                        requests to alfresco.
            node_id (string):           Node id to which the file is going to be created
            dir_path (string):          The name and path of the dir where files are stored
            recursive (boolean):        A boolean to know if upload  must be recursive
                                        in the specifed dir, and should preserve the
                                        structure of dirs inside.
            file_identifier (string):   File identifier for all files inside a dir
        Returns:
            (string):           Returns the info of recent created site.
        """

        if recursive:
            expression = "/**/*"
        else:
            expression = "/*"

        files_in_dir = list(
            itertools.chain.from_iterable(
                glob.iglob(dir_path + expression + pattern, recursive=recursive)
                for pattern in file_patterns
            )
        )
        print("files_in_dir", files_in_dir)
        filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        total_files = len(files_in_dir)
        print(total_files)
        starttime = time.time()

        try:
            files_uploaded = []
            for idx, file_with_path in enumerate(files_in_dir):

                # total time since last login or script start
                total_time = round((time.time() - starttime), 2)

                if total_time > 2400:
                    """
                    if total time is bigger than 2400
                    or 40 minutes relogin to avoid ticket
                    expiration
                    """
                    time.sleep(5)

                    print("Re-logging in to alfresco...")

                    session = login.login()
                    # restart time
                    starttime = time.time()
                    time.sleep(5)
                    print("Login sucessful, continuing upload\\n")

                len_of_path = len(file_with_path.split("/"))
                name_of_file = file_with_path.split("/")[len_of_path - 1]
                root_dir_path = file_with_path.replace(dir_path, "").replace(
                    file_with_path.split("/")[len_of_path - 1], ""
                )

                data = {
                    "name": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    ),
                    "nodeType": "cm:content",
                }

                data["relativePath"] = root_dir_path

                data["properties"] = {
                    "cm:title": (
                        name_of_file[0 : len(name_of_file) - 4]
                        + file_identifier
                        + name_of_file[len(name_of_file) - 4 : len(name_of_file)]
                    )
                }

                print("Uploading " + data["name"] + " file...")

                files = {"filedata": open(file_with_path, "rb")}
                upload_response = upload(session, node_id, data, files)
                if upload_response[1] and upload_response[1] == 201:
                    files_uploaded.append(upload_response[0])
                    print("Uploaded " + data["name"])

                    filename = "logs/upload_log" + dir_path.replace('/','-') + '.txt'
                    with open(filename, 'a') as log_file:
                        log_file.writelines("%s\\n" % file_with_path)

                elif upload_response[1] and upload_response[1] == 409:
                    if "already exists" in upload_response[0]["error"]["errorKey"]:
                        print("File " + data["name"] + " already uploaded")

                else:
                    print("An error ocurred, file " + data["name"] + " cannot be uploaded")

                print("Uploaded file " + str(idx + 1) + " of " + str(total_files))
                print("\\n\\n")

            return files_uploaded
        except Exception as e:
            print("An error ocurred in file upload: ", e)
        
    def vectorize_soundscape(df, hash_name, indices):
        \'\'\'Return dataframe with array column containing indices by frequency\'\'\'
        return (df
                .groupby(by=["id", hash_name, "start_time", "end_time"])
                .apply(get_vectors, indices)
                .reset_index()
                .rename(columns={0:"index_vector"}))
    '''

    _kale_block3 = '''
    session = login()
    for sc_path in sub_folder_results:
        for file_type in ["sequence.png", "mean_soundscape.png", "std_soundscape.png"]:
            change_type_sipecam_sc(session, ALFRESCO_NODE_ID, sc_path, file_type)
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
    with open("/upload_alfresco_model_data.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('upload_alfresco_model_data')

    _kale_mlmdutils.call("mark_execution_complete")


_kale_get_audio_df_op = _kfp_components.func_to_container_op(
    get_audio_df, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-cert:0.6.1_dev')


_kale_create_results_dirstruct_op = _kfp_components.func_to_container_op(
    create_results_dirstruct, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-cert:0.6.1_dev')


_kale_compute_soundscapes_op = _kfp_components.func_to_container_op(
    compute_soundscapes, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-cert:0.6.1_dev')


_kale_spec_n_specvid_op = _kfp_components.func_to_container_op(
    spec_n_specvid, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-cert:0.6.1_dev')


_kale_upload_to_alfresco_op = _kfp_components.func_to_container_op(
    upload_to_alfresco, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-cert:0.6.1_dev')


_kale_upload_alfresco_model_data_op = _kfp_components.func_to_container_op(
    upload_alfresco_model_data, base_image='sipecam/audio-dgpi-kale-tensorflow-yuntu-dask-cert:0.6.1_dev')


@_kfp_dsl.pipeline(
    name='sound-scape-nod-rec-dep-r5k3p',
    description='Computes Sipecam Soundscapes using cumulus, node, recorder and deployment'
)
def auto_generated_pipeline(ALFRESCO_NODE_ID='cf3a1b97-965d-489f-bfdf-5c8e26c4ac95', AUTH_ENDPOINT='alfresco/api/-default-/public/authentication/versions/1', BASE_ENDPOINT='alfresco/api/-default-/public/alfresco/versions/1', BLUE_IDX='CORE', CUMULO='92', FREQUENCY_BINS='96', FREQUENCY_LIMITS_LB='0', FREQUENCY_LIMITS_UB='24000', GREEN_IDX='INFORMATION', HASHER_TIME_MODULE='48', HASHER_TIME_UNIT='1800', HASH_NAME='crono_hash_30m', LIMIT='10', MIN_FREQ_SC='10000', PAGESIZE='1000', RED_IDX='EXAG', RESULTS_DIR='/shared_volume/audio/soundscapes', SAMPLERATE='48000.0', SPECTRUM='Audible', THREADS_PER_WORKER='2', TIME_UNIT='30', WORK_DIR_PIPELINE='.', vol_shared_volume='hostpath-pvc'):
    _kale_pvolumes_dict = OrderedDict()
    _kale_volume_step_names = []
    _kale_volume_name_parameters = []

    _kale_annotations = {}

    _kale_volume = _kfp_dsl.PipelineVolume(pvc=vol_shared_volume)

    _kale_pvolumes_dict['/shared_volume'] = _kale_volume

    _kale_volume_step_names.sort()
    _kale_volume_name_parameters.sort()

    _kale_get_audio_df_task = _kale_get_audio_df_op(AUTH_ENDPOINT, BASE_ENDPOINT, CUMULO, LIMIT, PAGESIZE, SAMPLERATE)\
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

    _kale_create_results_dirstruct_task = _kale_create_results_dirstruct_op(AUTH_ENDPOINT, BASE_ENDPOINT, CUMULO, RESULTS_DIR)\
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

    _kale_compute_soundscapes_task = _kale_compute_soundscapes_op(AUTH_ENDPOINT, BASE_ENDPOINT, BLUE_IDX, CUMULO, FREQUENCY_BINS, FREQUENCY_LIMITS_LB, FREQUENCY_LIMITS_UB, GREEN_IDX, HASHER_TIME_MODULE, HASHER_TIME_UNIT, HASH_NAME, MIN_FREQ_SC, RED_IDX, RESULTS_DIR, SPECTRUM, THREADS_PER_WORKER, TIME_UNIT, WORK_DIR_PIPELINE)\
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

    _kale_spec_n_specvid_task = _kale_spec_n_specvid_op(AUTH_ENDPOINT, BASE_ENDPOINT, CUMULO, RESULTS_DIR, SPECTRUM)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_compute_soundscapes_task)
    _kale_spec_n_specvid_task.container.working_dir = "//shared_volume/audio"
    _kale_spec_n_specvid_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'spec_n_specvid': '/spec_n_specvid.html'})
    _kale_spec_n_specvid_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_spec_n_specvid_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_spec_n_specvid_task.dependent_names +
                       _kale_volume_step_names)
    _kale_spec_n_specvid_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_spec_n_specvid_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_upload_to_alfresco_task = _kale_upload_to_alfresco_op(ALFRESCO_NODE_ID, AUTH_ENDPOINT, BASE_ENDPOINT, RESULTS_DIR)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_spec_n_specvid_task)
    _kale_upload_to_alfresco_task.container.working_dir = "//shared_volume/audio"
    _kale_upload_to_alfresco_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update(
        {'upload_to_alfresco': '/upload_to_alfresco.html'})
    _kale_upload_to_alfresco_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_upload_to_alfresco_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_upload_to_alfresco_task.dependent_names +
                       _kale_volume_step_names)
    _kale_upload_to_alfresco_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_upload_to_alfresco_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_upload_alfresco_model_data_task = _kale_upload_alfresco_model_data_op(ALFRESCO_NODE_ID, AUTH_ENDPOINT, BASE_ENDPOINT)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_upload_to_alfresco_task)
    _kale_upload_alfresco_model_data_task.container.working_dir = "//shared_volume/audio"
    _kale_upload_alfresco_model_data_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update(
        {'upload_alfresco_model_data': '/upload_alfresco_model_data.html'})
    _kale_upload_alfresco_model_data_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_upload_alfresco_model_data_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_upload_alfresco_model_data_task.dependent_names +
                       _kale_volume_step_names)
    _kale_upload_alfresco_model_data_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_upload_alfresco_model_data_task.add_pod_annotation(
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
                           "LIMIT" : False,
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
                           "HASH_NAME" : "crono_hash_30m",
                           "THREADS_PER_WORKER": 2,
                           "RESULTS_DIR": '/shared_volume/audio/soundscapes',
                           "ALFRESCO_NODE_ID" : "cf3a1b97-965d-489f-bfdf-5c8e26c4ac95",
                           "BASE_ENDPOINT" : "alfresco/api/-default-/public/alfresco/versions/1",
                           "AUTH_ENDPOINT" : "alfresco/api/-default-/public/authentication/versions/1"}
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, pipeline_parameters)
    import time
    time.sleep(180)
    pipeline_parameters = {"CUMULO": 95,
                           "SAMPLERATE": 48000.0,
                           "PAGESIZE": 1000,
                           "LIMIT": False,
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
                           "HASH_NAME": "crono_hash_30m",
                           "THREADS_PER_WORKER": 2,
                           "RESULTS_DIR": '/shared_volume/audio/soundscapes',
                           "ALFRESCO_NODE_ID": "cf3a1b97-965d-489f-bfdf-5c8e26c4ac95",
                           "BASE_ENDPOINT": "alfresco/api/-default-/public/alfresco/versions/1",
                           "AUTH_ENDPOINT": "alfresco/api/-default-/public/authentication/versions/1"}
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, pipeline_parameters)
    time.sleep(180)
    pipeline_parameters = {"CUMULO": 32,
                           "SAMPLERATE": 48000.0,
                           "PAGESIZE": 1000,
                           "LIMIT": False,
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
                           "HASH_NAME": "crono_hash_30m",
                           "THREADS_PER_WORKER": 2,
                           "RESULTS_DIR": '/shared_volume/audio/soundscapes',
                           "ALFRESCO_NODE_ID": "cf3a1b97-965d-489f-bfdf-5c8e26c4ac95",
                           "BASE_ENDPOINT": "alfresco/api/-default-/public/alfresco/versions/1",
                           "AUTH_ENDPOINT": "alfresco/api/-default-/public/authentication/versions/1"}
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, pipeline_parameters)
