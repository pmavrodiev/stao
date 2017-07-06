### General

#### Installation

Just clone the git repository. Google for instructions.

#### Code structure

 **`input_reader`**

 The input module, which reads-in all necessary data.
 Currently it looks for plain .csv files, which are read into Pandas dataframes.
 In the future it is planned to directly bind with Teradata.

 **`models`**

 Define your models here. Each model is a separate Python package,
 i.e. a subdirectory with its own \_\_init\_\_.py.
 The \_\_init\_\_.py of the package should create an instance of the model
 named **model**.

 Each package should also contain a settings file, which defines the parameter of the model.
 For concrete code details refer to some of the existing implementations.

 **`simple_logging`**

 A custom logging module. No rocket science here.

#### Running

Run without arguments to see the Usage:

`/opt/r/anaconda/bin/python3 main.py`

To run the currenty used model MBI v.1.0:

`/opt/r/anaconda/bin/python3 main.py -m model_MBI_v1_0 -c models/model_MBI_v1_0/mbi1.0_settings.cfg  -l logs/model_mbi_1_0/gm_mbi_v_1_0.log`


#### Model Settings

The model parameters are configured in a settings file supplied as a command-line argument.
Each model can define its own structure for this file, however the most common one is listed below.
For details refer to specific model implementations.

##### General Structure

##### [global]

 - **cache_enabled**

      If True the input data will be read from the cached pickled Python
      objects specified in the [cache_config] section.

      If False the input data will be read from the files specified in
      the [inputdata] section. Once read, the data will be cached into
      the pickled objects specified in [cache_config] if [cache_input_data] is True.

      On a first run set this to **False** to build the cache first.

 - **cache_input_data**

      If True will cache the input data into the pickled objects specified
      in [cache_config].
      This option has obviously no effect if [cache_enabled] is True,
      since all data would be read from the cache anyway.

 - **single_store**

      If the analysis should be run for a single store only (e.g.
      for testing) then specify the store's name, e.g. SM_MIG_49972_11914

      If empty, the analysis is run over all stores.

##### [parallel]

 This section defines parallelization settings in case a model needs that.
 Which functionality is parallelized is then dependent on the actual implementation.
 For code examples see model_MBI_v_1_1.

 - **cpu_count**

    Number of processes to be used.
    This is the parameter for the *multiprocessing.pool.Pool* constructor
    from the Python multiprocessing module.

 - **chunk_size**

    The number of chunks into which the data should be split.
    This is the *chunk_size* parameter for the *multiprocessing.pool.Pool.map*
    function in the Python multiprocessing module.

##### [parameter_basic]

Defines the free parameters of the model.
Refer to particular implementations for examples

##### [parameter_basic_sweep]

Defines the  parameter space to explore.
Parameter values are supplied as python lists.
Refer to particular implementations for examples


##### [inputdata]

Location of the raw input files:

 - **stores_cm**
 - **drivetimes**
 - **haushalt**

 More input data files can be present depending on the model.

##### [cache_config]

 Defines the location where the cached raw input files will be saved.
 The cached files are stored as pickled pandas objects.

 - **cache_dir**
 - **stores_cm_cached**
 - **stores_cm_migros_only_cached**
 - **drivetimes_cached**
 - **haushalt_cached**

##### [output]

The output file where the predicted turnover is to be written.
Make sure that the output directory exists (or submit a merge request which voids this note :-) ).