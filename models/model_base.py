import abc

class ModelBase(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def whoami(self):
        return 'Should never reach here'

    @abc.abstractmethod
    def entry(self, pandas_dt, logger):
        """

        :param pandas_dt:
            All available data is flattened into this pandas data table. You can always assume the following format:
            |----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            |        |fahrzeit |hektar_id|          ID      |FORMAT|vfl    |RELEVANZ|	type      |Tot_Haushaltausgaben	|Tot_Haushaltausgaben_corrected|
            |----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            |OBJECTID|
            |----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            |    6   |	21  	|61341718|SM_MIG_61607_15939|	M  |878.621|   1.0  |	MIG	      | 7800.0              |	7800.0                     |
            |----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            |    6	 |  21	    |61341719|SM_MIG_61607_15939|	M  |878.621|   1.0  |   MIG	      | 15600.0	            |   15600.0                    |
            |----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

        :param logger:
            Python logging object to which logging info will be sent

        :return:
            Nothing.

         Main entry point of a model.
        """

    @abc.abstractmethod
    def process_settings(self, config):
        """

        :param config:
            Python Configparser object from which model-specific settings will be extracted
        :return:
            Nothing. Populates model-specific data attributes
        """

    @abc.abstractmethod
    def whoami(self):
        """

        :return:
            returns a string representing the implementing class
        """

    @abc.abstractmethod
    def gen_umsatz_prognose(self, enriched_pruned_pd, stores_migros_pd, referenz_pd, logger):
        """

        :param enriched_pruned_pd:
        :param stores_migros_pd:
        :param referenz_pd:
        :param logger:
        :return:
        """