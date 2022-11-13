#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Desc    :   None
'''


import streamlit as st
import pandas as pd
#import cufflinks as cf
import plotly
import plotly.graph_objs as go
import json
import ast
import os
import time
from PIL import Image


import sys
sys.path.append('scripts/')

from visualization import st_data_visualization

from simpleregression import st_regression



image=Image.open('steel-rolls.jpeg')

def main():
    # SideBar Settings
    st.sidebar.title("Control Panel")
    st.sidebar.info( "Select the necessary functions")
    # Image on the main screen + Intruction text
    st.title("Mechanical Property Prediction \n ")
    st.image(image)
    


    # app functionalities
    primary_function = st.sidebar.selectbox(
        'Choose App Functionality', [ "Data Visualization", \
                     "Modeling"])

    data= pd.read_csv("Data_Modified.csv")
    
    st.write("Here are the first ten rows of the File")
    st.table(data.head(10))
    

    if primary_function == "Data Visualization":
        st_data_visualization()

    if primary_function == "Modeling":
        st_regression()

if __name__ == '__main__':
    main()