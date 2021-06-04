#execute as python3 hsipipeline.py hsi hsiexp

import json
import sys
import datetime
import random
import string

import kfp.dsl as _kfp_dsl
import kfp.components as _kfp_components

from collections import OrderedDict
from kubernetes import client as k8s_client

name_file, name_project, name_experiment = sys.argv

datetime_pipeline_launch = datetime.datetime.now().strftime('%Y%m%d')

name_experiment = name_experiment + datetime_pipeline_launch

length = 6

random_word = ''.join(random.choice(string.ascii_lowercase) for i in range(length))

name_pipeline_run = ''.join([name_project, datetime_pipeline_launch, '-',random_word])


def downloadfroms3():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import os
    import subprocess
    import glob

    from IPython import get_ipython

    ipython = get_ipython()
    '''

    _kale_block2 = '''
    bucket_with_data = "hsi-kale"

    input_dir_data = "/shared_volume/input_data"

    if not os.path.exists(input_dir_data):
        os.makedirs(input_dir_data)

        
    cmd_subprocess = ["aws", "s3", "cp",
                      "s3://" + bucket_with_data,
                      input_dir_data,
                      "--recursive"]

    subprocess.run(cmd_subprocess)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    _kale_marshal.save(input_dir_data, "input_dir_data")
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
    with open("/downloadfroms3.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('downloadfroms3')

    _kale_mlmdutils.call("mark_execution_complete")


def readdatainput(dir_mask_specie: str, dir_specie: str, file_mask_specie: str, file_specie: str):
    _kale_pipeline_parameters_block = '''
    dir_mask_specie = "{}"
    dir_specie = "{}"
    file_mask_specie = "{}"
    file_specie = "{}"
    '''.format(dir_mask_specie, dir_specie, file_mask_specie, file_specie)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    input_dir_data = _kale_marshal.load("input_dir_data")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import os
    import subprocess
    import glob

    from IPython import get_ipython

    ipython = get_ipython()
    '''

    _kale_block2 = '''
    #
    ipython.magic("load_ext rpy2.ipython")
    #

    string_libraries = """R library(rgdal); library(raster)"""

    ipython.magic(string_libraries)

    ##assignment statements to build string

    variable_specie_loc = "specie_loc"

    variable_mask_specie = "specie_mask"

    string1 = "R " + variable_specie_loc + " <- rgdal::readOGR("

    string2 = os.path.join(input_dir_data, dir_specie)

    string3 = variable_mask_specie + " <- raster::raster("

    string4 = os.path.join(input_dir_data, dir_mask_specie, file_mask_specie)

    string_data_input = "".join([string1, "\\"", string2, "\\",", 
                                 "\\"", file_specie, "\\"",");",
                                 string3, "\\"", string4, "\\"", ")"])

    ##(end) assignment statements to build string

    ipython.magic(string_data_input)

    specie_loc = ipython.magic("Rget " + variable_specie_loc)
    specie_mask = ipython.magic("Rget " + variable_mask_specie)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    _kale_marshal.save(input_dir_data, "input_dir_data")
    _kale_marshal.save(specie_loc, "specie_loc")
    _kale_marshal.save(specie_mask, "specie_mask")
    _kale_marshal.save(variable_mask_specie, "variable_mask_specie")
    _kale_marshal.save(variable_specie_loc, "variable_specie_loc")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block, _kale_data_loading_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_data_saving_block)
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/readdatainput.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('readdatainput')

    _kale_mlmdutils.call("mark_execution_complete")


def reproject():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    specie_loc = _kale_marshal.load("specie_loc")
    variable_specie_loc = _kale_marshal.load("variable_specie_loc")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import os
    import subprocess
    import glob

    from IPython import get_ipython

    ipython = get_ipython()
    '''

    _kale_block2 = '''
    #
    ipython.magic("load_ext rpy2.ipython")
    print(specie_loc)
    ipython.magic("Rpush " + variable_specie_loc)
    #

    string_libraries = """R library(rgdal)"""

    ipython.magic(string_libraries)

    ##assignment statements to build string

    variable_specie_loc_transf = "specie_loc_transf"

    string1 = "R " + variable_specie_loc_transf + " <- sp::spTransform("

    string2 = "CRSobj = \\"+proj=lcc +lat_1=17.5 +lat_2=29.5 +lat_0=12 +lon_0=-102 +x_0=2500000 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0\\")"

    string_transform = "".join([string1, variable_specie_loc, ",",
                                string2])

    ##(end) assignment statements to build string

    ipython.magic(string_transform)

    specie_loc_transf = ipython.magic("Rget " + variable_specie_loc_transf)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    _kale_marshal.save(specie_loc_transf, "specie_loc_transf")
    _kale_marshal.save(variable_specie_loc_transf, "variable_specie_loc_transf")
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
    with open("/reproject.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('reproject')

    _kale_mlmdutils.call("mark_execution_complete")


def createtestdata(dir_years: str):
    _kale_pipeline_parameters_block = '''
    dir_years = "{}"
    '''.format(dir_years)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    input_dir_data = _kale_marshal.load("input_dir_data")
    specie_loc_transf = _kale_marshal.load("specie_loc_transf")
    variable_specie_loc_transf = _kale_marshal.load("variable_specie_loc_transf")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import os
    import subprocess
    import glob

    from IPython import get_ipython

    ipython = get_ipython()
    '''

    _kale_block2 = '''
    #
    ipython.magic("load_ext rpy2.ipython")
    print(specie_loc_transf)

    ipython.magic("Rpush " + variable_specie_loc_transf)
    #
    string_libraries = """R library(hsi)"""

    ipython.magic(string_libraries)

    ##assignment statements to build string

    variable_test_sp = "test_sp"

    string1 = "R " + variable_test_sp + " <- sp_temporal_data(occs="

    string2 = "longitude = \\"coords.x1\\",latitude = \\"coords.x2\\",sp_year_var=\\"Year\\",layers_by_year_dir ="

    string3 = os.path.join(input_dir_data, dir_years)

    string4 = "layers_ext = \\"*.tif$\\",reclass_year_data = T)"

    string_test = "".join([string1, variable_specie_loc_transf, ",",
                           string2, "\\"", string3 , "\\",",
                           string4])

    ##(end) assignment statements to build string

    ipython.magic(string_test)

    test_sp = ipython.magic("Rget " + variable_test_sp)

    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    _kale_marshal.save(test_sp, "test_sp")
    _kale_marshal.save(variable_test_sp, "variable_test_sp")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block, _kale_data_loading_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_data_saving_block)
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/createtestdata.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('createtestdata')

    _kale_mlmdutils.call("mark_execution_complete")


def maskandextract():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    specie_mask = _kale_marshal.load("specie_mask")
    test_sp = _kale_marshal.load("test_sp")
    variable_mask_specie = _kale_marshal.load("variable_mask_specie")
    variable_test_sp = _kale_marshal.load("variable_test_sp")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import os
    import subprocess
    import glob

    from IPython import get_ipython

    ipython = get_ipython()
    '''

    _kale_block2 = '''
    #
    ipython.magic("load_ext rpy2.ipython")


    string_libraries = """R library(hsi);library(raster)"""

    ipython.magic(string_libraries)

    print(test_sp)
    print(specie_mask)
    ipython.magic("Rpush " + variable_test_sp)
    ipython.magic("Rpush " + variable_mask_specie)
    #

    ##assignment statements to build string
    variable_test_sp_mask = "test_sp_mask"

    string1 = "R " + variable_test_sp_mask + " <- occs_filter_by_mask("

    string_filter = "".join([string1, variable_test_sp, ",",
                             variable_mask_specie,
                             ")"])

    ##(end)assignment statements to build string

    ipython.magic(string_filter)

    ##assignment statements to build string

    variable_test_sp_clean = "test_sp_clean"

    string1 = "R " + variable_test_sp_clean + " <- clean_dup_by_year(this_species = "

    string2 = ", threshold = res("

    string3 = ")[1])"

    string_clean_test = "".join([string1, variable_test_sp_mask,
                                 string2, variable_mask_specie,
                                 string3])

    ##(end)assignment statements to build string

    ipython.magic(string_clean_test)

    ##assignment statements to build string
    variable_e_test = "e_test"

    string1 = "R " + variable_e_test + " <- extract_by_year(this_species="

    string2 = ",layers_pattern=\\"_mar\\")"

    string_extract = "".join([string1, variable_test_sp_clean, string2])

    ##(end)assignment statements to build string

    ipython.magic(string_extract)

    e_test = ipython.magic("Rget " + variable_e_test)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    _kale_marshal.save(e_test, "e_test")
    _kale_marshal.save(specie_mask, "specie_mask")
    _kale_marshal.save(variable_e_test, "variable_e_test")
    _kale_marshal.save(variable_mask_specie, "variable_mask_specie")
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
    with open("/maskandextract.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('maskandextract')

    _kale_mlmdutils.call("mark_execution_complete")


def bestmodel():
    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    e_test = _kale_marshal.load("e_test")
    variable_e_test = _kale_marshal.load("variable_e_test")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import os
    import subprocess
    import glob

    from IPython import get_ipython

    ipython = get_ipython()
    '''

    _kale_block2 = '''
    #
    ipython.magic("load_ext rpy2.ipython")
    print(e_test)

    ipython.magic("Rpush " + variable_e_test)
    #
    string_libraries = """R library(hsi)"""

    ipython.magic(string_libraries)


    ##assignment statements to build string

    variable_best_model_2004 = "best_model_2004"

    string1 = "R " + variable_best_model_2004 + " <- find_best_model(this_species ="

    string2 = ", cor_threshold = 0.8, ellipsoid_level = 0.975,nvars_to_fit = 3,E = 0.05,RandomPercent = 70,NoOfIteration = 1000,parallel = TRUE,n_cores = 24,plot3d = FALSE)"

    string_best_model = "".join([string1, variable_e_test, string2])

    ##(end)assignment statements to build string


    ipython.magic(string_best_model)

    best_model_2004 = ipython.magic("Rget " + variable_best_model_2004)
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    _kale_marshal.save(best_model_2004, "best_model_2004")
    _kale_marshal.save(variable_best_model_2004, "variable_best_model_2004")
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
    with open("/bestmodel.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('bestmodel')

    _kale_mlmdutils.call("mark_execution_complete")


def temporalprojection(date_of_processing: str, specie: str):
    _kale_pipeline_parameters_block = '''
    date_of_processing = "{}"
    specie = "{}"
    '''.format(date_of_processing, specie)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    best_model_2004 = _kale_marshal.load("best_model_2004")
    specie_mask = _kale_marshal.load("specie_mask")
    variable_best_model_2004 = _kale_marshal.load("variable_best_model_2004")
    variable_mask_specie = _kale_marshal.load("variable_mask_specie")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import os
    import subprocess
    import glob

    from IPython import get_ipython

    ipython = get_ipython()
    '''

    _kale_block2 = '''
    #
    ipython.magic("load_ext rpy2.ipython")

    string_libraries = """R library(hsi);library(raster)"""

    ipython.magic(string_libraries)

    print(best_model_2004)
    print(specie_mask)
    ipython.magic("Rpush " + variable_best_model_2004)
    ipython.magic("Rpush " + variable_mask_specie)
    #

    dir_results = "/shared_volume/new_model_parallel"

    save_dir = os.path.join(dir_results, date_of_processing)

    ##assignment statements to build string

    string1 = "R temporal_projection(this_species = "

    string2 = ",save_dir = "

    string3 = "sp_mask = "

    string4 = ",crs_model = NULL,sp_name ="

    string5 = ",plot3d = FALSE)"

    string_temporal_proj = "".join([string1, variable_best_model_2004,
                                    string2, "\\"", save_dir, "\\",",
                                    string3, variable_mask_specie,
                                    string4, "\\"", specie, "\\"", string5])


    ##(end)assignment statements to build string


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    ipython.magic(string_temporal_proj)

    #temporal_projection = ipython.magic("Rget temporal_projection")
    '''

    _kale_data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    _kale_marshal.save(save_dir, "save_dir")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block, _kale_data_loading_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_data_saving_block)
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/temporalprojection.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('temporalprojection')

    _kale_mlmdutils.call("mark_execution_complete")


def uploadtos3(date_of_processing: str):
    _kale_pipeline_parameters_block = '''
    date_of_processing = "{}"
    '''.format(date_of_processing)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale import marshal as _kale_marshal
    _kale_marshal.set_data_dir("/shared_volume/kube_sipecam_playground/hsi/notebooks/.hsi_using_r2py.ipynb.kale.marshal.dir")
    save_dir = _kale_marshal.load("save_dir")
    # -----------------------DATA LOADING END----------------------------------
    '''

    _kale_block1 = '''
    import os
    import subprocess
    import glob

    from IPython import get_ipython

    ipython = get_ipython()
    '''

    _kale_block2 = '''
    dir_to_upload = glob.glob(save_dir + '*')[0]

    bucket_results = "s3://hsi-kale-results"


    bucket_path_uploading = os.path.join(bucket_results, date_of_processing)

    cmd_subprocess = ["aws", "s3", "cp",
                      dir_to_upload,
                      bucket_path_uploading,
                      "--recursive"]

    subprocess.run(cmd_subprocess)
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block, _kale_data_loading_block,
                    _kale_block1,
                    _kale_block2,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/uploadtos3.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('uploadtos3')

    _kale_mlmdutils.call("mark_execution_complete")


_kale_downloadfroms3_op = _kfp_components.func_to_container_op(
    downloadfroms3, base_image='sipecam/hsi-kale:0.6.1')


_kale_readdatainput_op = _kfp_components.func_to_container_op(
    readdatainput, base_image='sipecam/hsi-kale:0.6.1')


_kale_reproject_op = _kfp_components.func_to_container_op(
    reproject, base_image='sipecam/hsi-kale:0.6.1')


_kale_createtestdata_op = _kfp_components.func_to_container_op(
    createtestdata, base_image='sipecam/hsi-kale:0.6.1')


_kale_maskandextract_op = _kfp_components.func_to_container_op(
    maskandextract, base_image='sipecam/hsi-kale:0.6.1')


_kale_bestmodel_op = _kfp_components.func_to_container_op(
    bestmodel, base_image='sipecam/hsi-kale:0.6.1')


_kale_temporalprojection_op = _kfp_components.func_to_container_op(
    temporalprojection, base_image='sipecam/hsi-kale:0.6.1')


_kale_uploadtos3_op = _kfp_components.func_to_container_op(
    uploadtos3, base_image='sipecam/hsi-kale:0.6.1')


@_kfp_dsl.pipeline(
    name=name_pipeline_run,
    description='Pipeline hsi'
)
def auto_generated_pipeline(date_of_processing='02_06_2021', dir_mask_specie='Ponca_DV', dir_specie='Ponca_DV_loc', dir_years='forest_jEquihua_mar', file_mask_specie='poncamask.tif', file_specie='poncadav2', specie='pan_onca', vol_shared_volume='efs'):
    _kale_pvolumes_dict = OrderedDict()
    _kale_volume_step_names = []
    _kale_volume_name_parameters = []

    _kale_annotations = {}

    _kale_volume = _kfp_dsl.PipelineVolume(pvc=vol_shared_volume)

    _kale_pvolumes_dict['/shared_volume'] = _kale_volume

    _kale_volume_step_names.sort()
    _kale_volume_name_parameters.sort()

    _kale_downloadfroms3_task = _kale_downloadfroms3_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    
    _kale_downloadfroms3_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    
    _kale_downloadfroms3_task.container.working_dir = "//shared_volume/kubeflow/pipelines"
    _kale_downloadfroms3_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'downloadfroms3': '/downloadfroms3.html'})
    _kale_downloadfroms3_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_downloadfroms3_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_downloadfroms3_task.dependent_names +
                       _kale_volume_step_names)
    _kale_downloadfroms3_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_downloadfroms3_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_readdatainput_task = _kale_readdatainput_op(dir_mask_specie, dir_specie, file_mask_specie, file_specie)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_downloadfroms3_task)
    
    _kale_readdatainput_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    _kale_readdatainput_task.container.working_dir = "//shared_volume/kubeflow/pipelines"
    _kale_readdatainput_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'readdatainput': '/readdatainput.html'})
    _kale_readdatainput_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_readdatainput_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_readdatainput_task.dependent_names +
                       _kale_volume_step_names)
    _kale_readdatainput_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_readdatainput_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_reproject_task = _kale_reproject_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_readdatainput_task)
    
    _kale_reproject_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    _kale_reproject_task.container.working_dir = "//shared_volume/kubeflow/pipelines"
    _kale_reproject_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'reproject': '/reproject.html'})
    _kale_reproject_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_reproject_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_reproject_task.dependent_names +
                       _kale_volume_step_names)
    _kale_reproject_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_reproject_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_createtestdata_task = _kale_createtestdata_op(dir_years)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_reproject_task)
    
    #see: https://github.com/kubeflow/pipelines/issues/4857#issuecomment-740279537
    
    _kale_createtestdata_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    _kale_createtestdata_task.container.working_dir = "//shared_volume/kubeflow/pipelines"
    _kale_createtestdata_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'createtestdata': '/createtestdata.html'})
    _kale_createtestdata_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_createtestdata_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_createtestdata_task.dependent_names +
                       _kale_volume_step_names)
    _kale_createtestdata_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_createtestdata_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_maskandextract_task = _kale_maskandextract_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_createtestdata_task)
    
    _kale_maskandextract_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    _kale_maskandextract_task.container.working_dir = "//shared_volume/kubeflow/pipelines"
    _kale_maskandextract_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'maskandextract': '/maskandextract.html'})
    _kale_maskandextract_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_maskandextract_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_maskandextract_task.dependent_names +
                       _kale_volume_step_names)
    _kale_maskandextract_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_maskandextract_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_bestmodel_task = _kale_bestmodel_op()\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_maskandextract_task)
    
    _kale_bestmodel_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    _kale_bestmodel_task.container.working_dir = "//shared_volume/kubeflow/pipelines"
    _kale_bestmodel_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'bestmodel': '/bestmodel.html'})
    _kale_bestmodel_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_bestmodel_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_bestmodel_task.dependent_names +
                       _kale_volume_step_names)
    _kale_bestmodel_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_bestmodel_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_temporalprojection_task = _kale_temporalprojection_op(date_of_processing, specie)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_bestmodel_task)
    
    _kale_temporalprojection_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    _kale_temporalprojection_task.container.working_dir = "//shared_volume/kubeflow/pipelines"
    _kale_temporalprojection_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update(
        {'temporalprojection': '/temporalprojection.html'})
    _kale_temporalprojection_task.output_artifact_paths.update(
        _kale_output_artifacts)
    _kale_temporalprojection_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_temporalprojection_task.dependent_names +
                       _kale_volume_step_names)
    _kale_temporalprojection_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_temporalprojection_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_uploadtos3_task = _kale_uploadtos3_op(date_of_processing)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after(_kale_temporalprojection_task)
    
    _kale_uploadtos3_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    _kale_uploadtos3_task.container.working_dir = "//shared_volume/kubeflow/pipelines"
    _kale_uploadtos3_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'uploadtos3': '/uploadtos3.html'})
    _kale_uploadtos3_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_uploadtos3_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_uploadtos3_task.dependent_names +
                       _kale_volume_step_names)
    _kale_uploadtos3_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_uploadtos3_task.add_pod_annotation(
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
    experiment = client.create_experiment(name_experiment)

    # Submit a pipeline run
    from kale.common.kfputils import generate_run_name
    run_name = generate_run_name(name_pipeline_run)
    pipeline_parameters = {"date_of_processing" : datetime_pipeline_launch + "_7_from_py", 
                           "dir_mask_specie" : 'Ponca_DV', 
                           "dir_specie" : 'Ponca_DV_loc', 
                           "dir_years" : 'forest_jEquihua_mar', 
                           "file_mask_specie" : 'poncamask.tif', 
                           "file_specie" : 'poncadav2', 
                           "specie" : 'pan_onca'}
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, pipeline_parameters)
