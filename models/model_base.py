import abc

class ModelBase(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def whoami(self):
        return 'Should never reach here'

    @abc.abstractmethod
    def entry(self, tables_dict, config, logger):
        """
        :param table_l:
            A list with all pandas table objects needed for the model.
            At least the following should be defined.

            table_l[0]:
                All available data is flattened into this pandas data table. You can always assume the following header:

                'StoreID', 'StoreName', 'Retailer', 'Format', 'VFL', 'Adresse', 'PLZ',
                'Ort', 'lon', 'lat', 'E_LV03', 'N_LV03', 'HARasterID', 'ProfitKSTID',
                'KostenstelleID', 'JahrID', 'Food', 'Frische', 'Near/Non Food',
                'Fachmaerkte', 'Oeffnungsdatum', 'velo_StartHARasterID',
                'velo_ZielHARasterID', 'auto_distanzminuten', 'velo_distanzminuten',
                'FZ', 'Tot_Haushaltausgaben'

            table_l[1]:
                A Pandas DataFrame containing the SBB data.
        :param config:
            A config object containing the model settings supplied by the user

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
