### General

#### Installation

Just clone the git repository. Google for instructions on how to do that.

#### Code structure

 **`input_reader`**

 The input module, which reads-in all necessary data.
 Currently it looks for plain .csv files, which are read into Pandas dataframes.
 In the future it is planned to bind directly with data sources (e.g. Teradata).

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

To run the currenty used model MBI v.1.2:

`/opt/r/anaconda/bin/python3 /userdata/pmavrodi/Projekte/Stao/src/main.py -m model_MBI_v1_2 -c models/model_MBI_v1_2/mbi1.2_settings.cfg  -l logs/model_mbi_1_2/gm_mbi_v_1_2.log`


#### Model Settings

The model parameters are configured in a settings file supplied as a command-line argument.
Each model can define its own settings and process them accordingly in its own package.
For details refer to specific model implementations.
