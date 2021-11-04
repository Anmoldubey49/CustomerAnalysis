import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from loggerapp.logger import App_Logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Clustering:
    """
            This class shall be used to perform cluster analysis

            Written By: Anmol Dubey
            Version: 1.0
            Revisions: None

    """

    def __init__(self):
        self.log_writer = App_Logger()


    def elbow_plot(self,data):
        """
                        Method Name: elbow_plot
                        Description: This method saves the plot to decide the optimum number of clusters to the file.
                        Output: A picture saved to the directory
                        On Failure: Raise Exception

                        Written By: Anmol Dubey
                        Version: 1.0
                        Revisions: None

        """
        file = open('General_logs.txt','a+')
        self.log_writer.log(file, 'Entered the elbow_plot method of the Clustering class')
        wcss=[] # initializing an empty list
        try:
            for i in range (1,11):
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) # initializing the KMeans object
                kmeans.fit(data) # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss) # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            #plt.show()
            plt.savefig('preprocessing_data/K-Means_Elbow.PNG') # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.log_writer.log(file, 'The optimum number of clusters is: '+str(self.kn.knee)+' . Exited the elbow_plot method of the Clustering class')
            return self.kn.knee

        except Exception as e:
            self.log_writer.log(file,'Exception occured in elbow_plot method of the Clustering class. Exception message:  ' + str(e))
            self.log_writer.log(file,'Finding the number of clusters failed. Exited the elbow_plot method of the Clustering class')
            raise Exception()
        file.close()

    def create_clusters(self,data,number_of_clusters):
        """
                                Method Name: create_clusters
                                Description: Create a new dataframe consisting of the cluster information.
                                Output: A datframe with cluster column
                                On Failure: Raise Exception

                                Written By: Anmol Dubey
                                Version: 1.0
                                Revisions: None

                        """
        file = open('General_logs.txt','a+')
        self.log_writer.log(file, 'Entered the create_clusters method of the Clustering class')
        self.data=data
        try:
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            #self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            self.y_kmeans=self.kmeans.fit_predict(data) #  divide data into clusters

            df = pd.read_csv('EDA/Input_File.csv')


            df['Cluster']=self.y_kmeans +1  # create a new column in dataset for storing the cluster information
            if not os.path.isdir('KMeansFile/'):
                os.mkdir('KMeansFile/')
            df.to_csv('KMeansFile/KMeans_Final_File',index_label=None,header = True)
            self.log_writer.log(file, 'succesfully created '+str(self.kn.knee)+ 'clusters. Exited the create_clusters method of the Clustering class')
            return df
        except Exception as e:
            self.log_writer.log(file,'Exception occured in create_clusters method of the Clustering class. Exception message:  ' + str(e))
            self.log_writer.log(file,'Fitting the data to clusters failed. Exited the create_clusters method of the Clustering class')
            raise Exception()

    def KMeansTakeaways(self,df):

        sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black', 'axes.grid' : False, 'font.family': 'Ubuntu'})

        for i in df:
            diag = sns.FacetGrid(df, col = "Cluster", hue = "Cluster", palette = "Set1")
            diag.map(plt.hist, i, bins=6, ec="k")
            diag.set_xticklabels(rotation=25, color = 'white')
            diag.set_yticklabels(color = 'white')
            diag.set_xlabels(size=16, color = 'white')
            diag.set_titles(size=16, color = '#f01132', fontweight="bold")
            diag.fig.set_figheight(6)
            diag.savefig('KMeans_takeaways/K-Means_takeaway' + i+ '.PNG') # saving the elbow plot locally
