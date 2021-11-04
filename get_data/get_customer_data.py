import pandas as pd
from loggerapp.logger import App_Logger

class Data_Getter:
    """
    This class shall  be used for obtaining the data from the source for training.

    Written By: Anmol Dubey
    Version: 1.0
    Revisions: None

    """
    def __init__(self):
        self.training_file='EDA/Scaled_Input_File.csv'
        self.log_writer = App_Logger()

    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

        Written By: Anmol Dubey
        Version: 1.0
        Revisions: None

        """

        file = open("General_logs.txt",'a+')
        self.log_writer.log(file, 'Entered the get_data method of the Data_Getter class')
        try:
            self.data = pd.read_csv(self.training_file)  # reading the data file
            self.log_writer.log(file,'Data Load Successful.Exited the get_data method of the Data_Getter class')
            return self.data
        except Exception as e:
            self.log_writer.log(file,'Exception occured in get_data method of the Data_Getter class. Exception message: ' + str(e))
            self.log_writer.log(file,'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()

        file.close()


