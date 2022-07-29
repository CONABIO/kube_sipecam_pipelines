import os
import datetime
from dateutil.parser import parse as dateutil_parse


def time2seconds(tstring):
    duration_ = tstring.split(":")

    return float(duration_[0])*3600 + float(duration_[1])*60 + float(duration_[2])


def parser_prev(datum):
    web_dav_path = datum["entry"]["path"]["name"]
    file_name = datum["entry"]["name"]
    metadata_path = os.path.join(web_dav_path, file_name)
    cumulo = metadata_path.split("cumulo")[1].split("/")[0].replace("-", "")
    web_dav_local_path = os.path.join(
        "/shared_volume/webdav_alfresco/Sites/sipecam/documentLibrary", cumulo)
    adj_metadata_path = metadata_path.split("documentLibrary/")[-1]

    path = os.path.join(web_dav_local_path, adj_metadata_path)

    properties = datum["entry"]["properties"]
    file_hash = file_name.split(".")[0]
    samplerate = properties["sipecamAudio:SampleRate"]
    timeexp = 1.0

    # Este es el dato de duración que trae la cabecera pero puede ser distinto del valor real.
    duration = time2seconds(properties['sipecamAudio:Duration'])

    filesize = float(
        properties["sipecamAudio:FileSize"].replace(" MiB", ""))*(2**20)

    media_info = {
        'nchannels': properties["sipecamAudio:NumChannels"],
        'sampwidth': 2,
        'samplerate': samplerate,
        'length': int(samplerate*duration),
        'filesize': filesize,
        'duration': duration
    }

    spectrum = 'ultrasonic' if samplerate > 100000 else 'audible'

    dtime_zone = properties["sipecamAudio:Timezone"]
    dtime = dateutil_parse(properties["sipecamAudio:Datetime"])
    dtime_format = "%H:%M:%S %d/%m/%Y (%z)"
    dtime_raw = datetime.datetime.strftime(dtime, format=dtime_format)

    latitude = properties["sipecamAudio:Latitude"]
    longitude = properties["sipecamAudio:Longitude"]

    metadata = dict(datum)

    return {
        'id': datum["entry"]['id'],
        'path': path,
        'hash': file_hash,
        'timeexp': 1,
        'media_info': media_info,
        'metadata': metadata,
        'spectrum': spectrum,
        'time_raw': dtime_raw,
        'time_format': dtime_format,
        'time_zone': dtime_zone,
        'time_utc': dtime,
        'latitude': latitude,
        'longitude': longitude
    }


def parser(datum):
    web_dav_path = datum["entry"]["path"]["name"]
    file_name = datum["entry"]["name"]
    metadata_path = os.path.join(web_dav_path, file_name)
    cumulo = metadata_path.split("documentLibrary/")[1].split("/")[0]
    web_dav_local_path = os.path.join(
        "/shared_volume/webdav_alfresco/Sites/sipecam/documentLibrary", cumulo)
    adj_metadata_path = metadata_path.split(f"documentLibrary/{cumulo}/")[-1]
    path = os.path.join(web_dav_local_path, adj_metadata_path)
    properties = datum["entry"]["properties"]
    file_hash = file_name.split(".")[0]
    samplerate = properties["sipecam:SampleRate"]
    timeexp = 1.0
    duration = properties['sipecam:Duration']
    filesize = properties["sipecam:FileSize"]
    sampwidth = properties["sipecam:BitsPerSample"]/8.0
    nchannels = properties["sipecam:NumChannels"]
    media_info = {
        'nchannels': nchannels,
        'sampwidth': sampwidth,
        'samplerate': samplerate,
        'length': int(samplerate*duration),
        'filesize': filesize,
        'duration': duration
    }
    spectrum = 'ultrasonic' if samplerate > 100000 else 'audible'
    dtime_zone = properties["sipecam:Timezone"]
    dtime = dateutil_parse(properties["sipecam:DateTimeOriginal"])
    dtime_format = "%H:%M:%S %d/%m/%Y (%z)"
    dtime_raw = datetime.datetime.strftime(dtime, format=dtime_format)

    latitude = properties["sipecam:Latitude"]
    longitude = properties["sipecam:Longitude"]
    metadata = dict(datum)
    return {
        'id': datum["entry"]['id'],
        'path': path,
        'hash': file_hash,
        'timeexp': 1,
        'media_info': media_info,
        'metadata': metadata,
        'spectrum': spectrum,
        'time_raw': dtime_raw,
        'time_format': dtime_format,
        'time_zone': dtime_zone,
        'time_utc': dtime,
        'latitude': latitude,
        'longitude': longitude
    }
