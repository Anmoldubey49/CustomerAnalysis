import pandas as pd

from get_data.get_customer_data import Data_Getter
from KMeansclustering import Clustering
from sklearn.model_selection import train_test_split
from XGBoost_Classifier import Model_Finder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


dataframe = Data_Getter()
df = dataframe.get_data()

#KMeans Clustering
cluster = Clustering() #creating instance
number_of_clusters = cluster.elbow_plot(df) #getting optimum number of clusters
X=cluster.create_clusters(df,number_of_clusters) #creating clusters
cluster.KMeansTakeaways(X)


#XGBoost Classification
XGB = Model_Finder()
df_xg = pd.read_csv('EDA/Input_File.csv')

lbl_encode = LabelEncoder()
cate = ['Education','Marital_Status']
for i in cate:
    df_xg[i] = df_xg[[i]].apply(lbl_encode.fit_transform)


df1 = df_xg.copy()
df1.drop(columns = ["Dt_Customer"],axis=1,inplace = True)


scaled_features = StandardScaler().fit_transform(df1.values)
scaled_features_df = pd.DataFrame(scaled_features, index=df1.index, columns=df1.columns)

xg_features = scaled_features_df.drop('Response', axis=1)
xg_label = df_xg['Response']
x_train, x_test, y_train, y_test = train_test_split(xg_features,xg_label, test_size=1 / 3,random_state=355)
best_model_name, model = XGB.get_xgb_model(x_train, y_train, x_test, y_test)
XGB.save_model(model,'XGBoost')

