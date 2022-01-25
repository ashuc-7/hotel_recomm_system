#Import libraries
import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import kmodes
import sklearn
import pickle
from PIL import Image

#Setting Application title
st.title('HOTEL RECOMMENDATION SYSTEM')

st.markdown("""
     :dart:  This Streamlit app is made to recommend the top ten hotel cluster to the user based on their interaction with the hotel app/broeser \n
    """)
st.markdown("<h3></h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file)
  st.write(data)

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

  def main():
     #Setting Application sidebar default
    image = Image.open('hotel.png')
    

    st.sidebar.image(image)

    st.title('User hotel recommendation')
    st.write('Overview of input after processing is shown below')
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.dataframe(data)

    def recomm(X):
        pred=xb.predict(X.values)
        if pred[0]==1:
            cl=clt_model_1.predict(X.values,categorical=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
            st.success("Top 10 recommended hotel clusters for user with booking category {} where user belongs to cluster {} are:".format(pred[0],cl[0]))
            st.success(dict_model_1.get(cl[0])[0:10])
        else:
            cl=clt_model_0.predict(X.values,categorical= [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] )
            st.success("Top 10 recommended hotel clusters for user with booking category {} where user belongs to cluster {} are:".format(pred[0],cl[0]))
            st.success(dict_model_0.get(cl[0])[0:10])
    if st.button('Predict'):
          recomm(data)
    



                
  if __name__ == '__main__':
      main()