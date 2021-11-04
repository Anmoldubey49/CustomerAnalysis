from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
from loggerapp.logger import App_Logger
import os
import pickle
import shutil

class Model_Finder:
    """
                This class shall be used to create an XGBoost classification model
                Written By: Anmol Dubey
                Version: 1.0
                Revisions: None

    """

    def __init__(self):
        self.log_writer = App_Logger()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def save_model(self, model, filename):

        """
                Method Name: save_model
                Description: Save the model file to directory
                Outcome: File gets saved
                On Failure: Raise Exception

                Written By: Anmol Dubey
                Version: 1.0
                Revisions: None
            """

        file = open('General_logs.txt','a+')
        model_directory = 'XGB_model/'
        self.log_writer.log(file, 'Entered the save_model method of the File_Operation class')
        try:
            path = model_directory  # create seperate directory for each cluster
            if os.path.isdir(path):  # remove previously existing models for each clusters
                shutil.rmtree(model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)  #
            with open(path + '/' + filename + '.sav','wb') as f:
                pickle.dump(model, f)  # save the model to file
            self.log_writer.log(file,'Model File ' + filename + ' saved. Exited the save_model method of the Model_Finder class')

        except Exception as e:
            self.log_writer.log(file,'Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.log_writer.log(file,'Model File ' + filename + ' could not be saved. Exited the save_model method of the Model_Finder class')
            raise e
        file.close()






    def get_best_params_for_xgboost(self, train_x, train_y):


        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: Anmol Dubey
                                        Version: 1.0
                                        Revisions: None

            """

        file = open('General_logs.txt','a+')

        self.log_writer.log(file,"Entered the get_best_params_for_xgboost method of the Model_Finder class")

        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth,n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.log_writer.log(file,'XGBoost best params: ' + str(self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.log_writer.log(file,'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(e))
            self.log_writer.log(file,'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

        file.close()

    def get_xgb_model(self,train_x,train_y,test_x,test_y):

        """
                                                Method Name: get_best_model
                                                Description: Get the XGBoost model
                                                On Failure: Raise Exception

                                                Written By: Anmol Dubey
                                                Version: 1.0
                                                Revisions: None

            """

        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            #self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model
            y_pred = self.xgboost.predict(test_x)
            predictions = [round(value) for value in y_pred]
            accuracy = accuracy_score(y_pred, test_y)
            print('accuracy of XGBoost model is',accuracy*100,'%')
            return 'XGBoost', self.xgboost

        except Exception as e:
            raise e






