import streamlit as st
from tensorflow.keras.models import load_model
import pickle as pl
import pandas as pd
import numpy as np

model=load_model("model/model.keras")
sc=pl.load(open("model/scaler.pkl",'rb'))
encoder =pl.load(open("model/encoder.pkl",'rb'))

st.title("NY HOUSE PRICING")
### types
house_type=st.selectbox("house_type",('Condo for sale', 'House for sale', 'Townhouse for sale' ,'Co-op for sale',
 'Multi-family home for sale'))
house_sublocality=st.selectbox("house_sublocality",('Manhattan' ,'New York County' ,'Richmond County' ,'Kings County' 'New York',
 'East Bronx' ,'Brooklyn', 'The Bronx' ,'Queens', 'Staten Island',
 'Queens County' ,'Bronx County', 'Coney Island' ,'Brooklyn Heights',
 'Jackson Heights' ,'Riverdale' ,'Rego Park' ,'Fort Hamilton' 'Flushing',
 'Dumbo' ,'Snyder Avenue'))
house_bath = st.number_input('Number of baths', min_value=0, step=1, format="%d")
house_bed = st.number_input('Number of beds', min_value=0, step=1, format="%d")
house_sqft = st.number_input('Square footage', min_value=0, step=1, format="%d")
df=pd.DataFrame([[house_type,house_bed,house_bath,house_sqft,house_sublocality]], columns=["TYPE","BEDS","BATH","PROPERTYSQFT","SUBLOCALITY"])
df_encoded=encoder.transform(df)

df_scale=sc.transform(df_encoded)

prediction=model.predict(df_scale)
print(prediction)
st.text(np.exp(prediction))
