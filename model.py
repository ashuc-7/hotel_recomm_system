

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import kmodes
import sklearn
import pickle


data=pd.read_csv("test_point.csv")
sca_destdist=pickle.load(open('sca_destdist.pkl','rb'))
sca_pca=pickle.load(open('sca_pca.pkl','rb'))
sca_staydur=pickle.load(open('sca_stay_duration.pkl','rb'))
clt_model_1=pickle.load(open('cluster_1.pkl','rb'))
xb=pickle.load(open('xb.pkl','rb'))
clt_model_0=pickle.load(open('cluster_0.pkl','rb'))
clt_model_1=pickle.load(open('cluster_1.pkl','rb'))
dict_model_0=pickle.load(open('dict_0.pkl','rb'))
dict_model_1=pickle.load(open('dict_1.pkl','rb'))
data["orig_destination_distance"]=sca_destdist.transform(data["orig_destination_distance"].values.reshape(-1,1))
data["stay_duration"]=sca_staydur.transform(data["stay_duration"].values.reshape(-1,1))
data["0"]=sca_destdist.transform(data["0"].values.reshape(-1,1))

def recomm(X):
    pred=xb.predict(X.values)
    if pred[0]==1:
        cl=clt_model_1.predict(X.values,categorical=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        print("Top 10 recommended hotel clusters for user with booking category {} where user belongs to cluster {} are:".format(pred[0],cl[0]))
        print(dict_model_1.get(cl[0])[0:10])
    else:
        cl=clt_model_0.predict(X.values,categorical= [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] )
        print("Top 10 recommended hotel clusters for user with booking category {} where user belongs to cluster {} are:".format(pred[0],cl[0]))
        print(dict_model_0.get(cl[0])[0:10])
recomm(data)
    