import os
import tarfile
import numpy as np
import tensorflow as tf
import math
from skimage.transform import resize

import sys, os
sys.path.append(os.path.dirname(__file__))
from models.keras import fc_dense
from models.keras import fc_detection

LOG10 = tf.math.log(tf.constant(10, dtype=tf.dtypes.float64))
LABELS = np.load(os.path.join(os.path.dirname(__file__), 'labels.npy'))
TARGET_SPEC_TIME_RESOLUTION = 1502.1165425613658
TARGET_SPEC_FREQ_RESOLUTION = 0.002671875
HOP_LENGTH = 256
N_FFT = 1024
TARGET_SR = 384000

@tf.function(experimental_relax_shapes=True)
def slice_by_detection(batch, indices):
    new_batch = tf.gather(batch, indices, axis=0)
    orshape = tf.shape(new_batch)
    
    return tf.reshape(new_batch, [orshape[0]*orshape[1], orshape[2], orshape[3], orshape[4]])

@tf.function(experimental_relax_shapes=True)
def get_safe_indices(detection, threshold=0.5):
    tdetections = tf.squeeze(detection)>threshold
    total_detections = tf.reduce_sum(tf.cast(tdetections, tf.dtypes.float64))
    safe_indices = tf.cond(total_detections > 0, lambda: tf.where(tdetections), lambda: tf.zeros([1,1], dtype=tf.dtypes.int64))
    return safe_indices, total_detections

@tf.function(experimental_relax_shapes=True)
def custom_sigmoid(x, sensitivity):
    mid_point = -tf.math.log((1/sensitivity)-1)
    return 1 / (1.0 + tf.math.exp(-(x-mid_point)))

def fetch_model(path, out_dir):
    out_dir =  os.path.join(os.path.dirname(__file__), out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    outpath = os.path.join(out_dir, "1")

    if os.path.exists(outpath):
        return outpath
    

    base_name = os.path.basename(path)
    download_path = os.path.join(out_dir, base_name)

    
    if not os.path.exists(download_path):
        from s3fs.core import S3FileSystem
        s3 = S3FileSystem()
        s3.get(path, download_path)

    tar = tarfile.open(download_path, "r:gz")
    tar.extractall(path=out_dir)
    tar.close()
        
    print("Fetched "+outpath)
    return outpath

def normalize_spec(spec):
    max_val = tf.math.reduce_max(spec)
    min_val = tf.math.reduce_min(spec)
    return (spec - min_val + 1e-15) / (max_val - min_val + 1e-15)

def decibel_scale(spec):
    spec = tf.math.square(spec)
    numerator = tf.math.log(spec + 1e-10)
    return tf.constant(10, dtype=numerator.dtype) * tf.math.divide(numerator, LOG10)

def format_raw_input(x):
    return normalize_spec(decibel_scale(x))

def format_input(x):
    return normalize_spec(x)

def format_class_input(x, detection_threshold):
    spec = normalize_spec(decibel_scale(x))
    return (spec, detection_threshold)

@tf.function(experimental_relax_shapes=True)
def batch_prediction(batch, model, detection_threshold):
    thresh = tf.ones([tf.shape(batch)[0]])*detection_threshold
    formated_batch = tf.map_fn(fn=lambda x: format_class_input(x[0], x[1]), elems=(batch, thresh))
    return model(formated_batch)

@tf.function(experimental_relax_shapes=True)
def batch_prediction_det(batch, model):
    formated_batch = tf.map_fn(fn=format_raw_input, elems=batch)
    return model(formated_batch)

@tf.function(experimental_relax_shapes=True)
def batch_prediction_class(batch, model):
    formated_batch = tf.map_fn(fn=format_raw_input, elems=batch)
    return model(formated_batch)

def smooth(x, window_len=3):
    w=np.ones(window_len,'d')
    y=np.convolve(w/w.sum(),np.squeeze(x), mode='same')
    return y

def signal_to_pieces(signal, threshold=0.5):
    mask = (signal > threshold)
    segments = mask.astype(float)
    length = signal.shape[0]
    indices = list(np.where(segments[:-1] != segments[1:])[0] + 1)

    if np.amin(mask) == True:
        limits = [[0, length]]
    elif len(indices) == 1:
        limits = [[0, indices[0]]] + [[indices[0], length]]
    elif len(indices) > 0:
        limits = ([[0, indices[0]]] +
                  [[indices[i], indices[i+1]] for i in range(0, len(indices)-1)] +
                  [[indices[-1], length]])
    else:
        limits = []
    
    return limits, segments

def sliding_window_tandem(detector, classifier, audio, hop=45, detection_threshold=0.5, batch_size=None):
    if audio.samplerate != TARGET_SR:
        au = audio.resample(samplerate=TARGET_SR)
    else:
        au = audio
    with (au
          .features.spectrogram(hop_length=HOP_LENGTH, n_fft=N_FFT)
          .resample(time_resolution=TARGET_SPEC_TIME_RESOLUTION, freq_resolution=TARGET_SPEC_FREQ_RESOLUTION)
          .cut(max_freq=TARGET_SR/2, pad=True)) as spec:
        h, w = spec.shape
        n = int(math.floor(float(w-45)/hop))
        
        inputs = []
        tlimits = []
        
        for i in range(n+1):
            tlimits.append([i*(hop), i*(hop)+45])
    
        total_frames = len(tlimits)
    
        if batch_size is None:
            batch_size = 1

        if batch_size > total_frames:
            batch_size = total_frames
        

        nbatches = math.ceil(float(len(tlimits))/float(batch_size))
        batch_limits = [tlimits[i*batch_size:(i+1)*batch_size] for i in range(nbatches)
                        if len(tlimits[i*batch_size:(i+1)*batch_size]) > 0]

        detection_predictions = []
        class_predictions = []
        for limits in batch_limits:
            batch = tf.convert_to_tensor(np.stack([spec.array[:, l[0]:l[1]] for l in limits], axis=0))
            batch_detections = batch_prediction_det(batch, detector)
            detection_predictions.append(batch_detections)
            
            batch_class = np.empty([batch_detections.shape[0], 101])
            do_class_indx = np.where(np.squeeze(batch_detections) > detection_threshold)

            if do_class_indx[0].size > 0:
                det_class = batch_prediction_class(tf.gather(batch, do_class_indx[0], axis=0), classifier)
                batch_class[do_class_indx, :] = det_class.numpy()

            class_predictions.append(batch_class)
            
            del batch
    
    return np.concatenate(detection_predictions, axis=0), np.concatenate(class_predictions, axis=0)


def sliding_window(dmodel, audio, hop=45, detection_threshold=0.5, batch_size=None):
    if audio.samplerate != TARGET_SR:
        au = audio.resample(samplerate=TARGET_SR)
    else:
        au = audio
    with (au
          .features.spectrogram(hop_length=HOP_LENGTH, n_fft=N_FFT)
          .resample(time_resolution=TARGET_SPEC_TIME_RESOLUTION, freq_resolution=TARGET_SPEC_FREQ_RESOLUTION)
          .cut(max_freq=TARGET_SR/2, pad=True)) as spec:
        h, w = spec.shape
        n = int(math.floor(float(w-45)/hop))
        
        inputs = []
        tlimits = []
        
        for i in range(n+1):
            tlimits.append([i*(hop), i*(hop)+45])
    
        total_frames = len(tlimits)
    
        if batch_size is None:
            batch_size = 1

        if batch_size > total_frames:
            batch_size = total_frames
        

        nbatches = math.ceil(float(len(tlimits))/float(batch_size))
        batch_limits = [tlimits[i*batch_size:(i+1)*batch_size] for i in range(nbatches)
                        if len(tlimits[i*batch_size:(i+1)*batch_size]) > 0]

        detection_predictions = []
        class_predictions = []
        for limits in batch_limits:
            batch = tf.convert_to_tensor(np.stack([spec.array[:, l[0]:l[1]] for l in limits], axis=0))
            #El procesamiento de tensorflow se reduce a esta línea para ambos modelos.
            batch_detections, batch_classifications, indices, total = batch_prediction(batch, dmodel, detection_threshold)
            full_classification = np.zeros([batch_detections.shape[0], 101])
            #Para lograr que el grafo funcione, hay que pasar al menos un frame al clasificador, aunque no existan detecciones.
            #El número de detecciones se regresa como un escalar y sirve para retirar de las clasificaciones dicho frame cuando
            #no existen detecciones.
            if total.numpy() > 0:
                full_classification[np.squeeze(indices), :] = batch_classifications
            detection_predictions.append(batch_detections)
            class_predictions.append(full_classification)

            del batch
    
    return np.concatenate(detection_predictions, axis=0), np.concatenate(class_predictions, axis=0)