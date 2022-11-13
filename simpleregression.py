import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import streamlit as st
import os
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics

class regression:
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    
    
    
    def LR(self):
        from sklearn.linear_model import LinearRegression
        lr_classifier = LinearRegression()
        lr_classifier.fit(self.X_train, self.y_train)
        #joblib.dump(lr_classifier, "model/lr.sav")
        y_pred = lr_classifier.predict(self.X_test)

        st.write("\n")
        st.write("--------------------------------------")
        st.write("### Linear Regression ###")
        st.write("--------------------------------------")
        st.write('Evaluation Report: ')
        test_crs_val = np.sqrt(-cross_val_score(lr_classifier, self.X_test, self.y_test,cv = 10,scoring = "neg_mean_squared_error")).mean()
        st.write("The Root Mean Square Error is :{}".format(test_crs_val))   
        st.write('The coefficient table:')
        coef_table = pd.DataFrame(list(self.X_train.columns)).copy()
        coef_table.insert(len(coef_table.columns),"Coefs",lr_classifier.coef_.transpose())
        st.table(coef_table)          
                
                
       
        
        
        
        
    def KNN(self):
        from sklearn.neighbors import KNeighborsRegressor
        knn_regressor = KNeighborsRegressor()
        knn_regressor.fit(self.X_train, self.y_train)
        #joblib.dump(knn_classifier, "model/knn.sav")
        y_pred = knn_regressor.predict(self.X_test)
        
        st.write("\n")
        st.write("-------------------------------")
        st.write("### K-Neighbors Regressor ###")
        st.write("-------------------------------")
        st.write('Evaluation Report: ')
        st.write("The R^2 value is:{}".format(knn_regressor.score(self.X_test, self.y_test)))
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        st.write("The Root Mean Square Error is: {}".format(rmse)) 

        
        
    def DT(self):
        from sklearn.tree import DecisionTreeRegressor
        tree_regressor = DecisionTreeRegressor()
        tree_regressor.fit(self.X_train, self.y_train)
        y_pred = tree_regressor.predict(self.X_test)
        
        st.write("\n")
        st.write("--------------------------------")
        st.write("### Decision Tree Classifier ###")
        st.write("--------------------------------")
        st.write('Evaluation Report: ')
        st.write("R^2 values:{}".format(tree_regressor.score(self.X_test, self.y_test)))
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        st.write("The Root Mean Square Error: {}".format(rmse))


        


    def GBDT(self):
        from sklearn.ensemble import GradientBoostingRegressor
        gbm_model = GradientBoostingRegressor(learning_rate = 0.2,max_depth = 3, n_estimators = 200,subsample = 0.75)
        gbm_model.fit(self.X_train, self.y_train)
        y_pred = gbm_model.predict(self.X_test)
        st.write("\n")
        st.write("--------------------------------")
        st.write("### Gradient Boosting Decision Tree Regressor ###")
        st.write("--------------------------------")
        st.write('Evaluation Report: ')
        st.write("R^2 values:{}".format(gbm_model.score(self.X_test, self.y_test)))
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        st.write("The Root Mean Square Error: {}".format(rmse))

    def CB(self):
        from catboost import CatBoostRegressor

        catb = CatBoostRegressor()
        catb_model = catb.fit(self.X_train, self.y_train)
        y_pred = catb_model.predict(self.X_test)

        st.write("\n")
        st.write("--------------------------------")
        st.write("### Cat Boost Regressor ###")
        st.write("--------------------------------")

        st.write("The R^2 value is:{}".format(catb_model.score(self.X_test, self.y_test)))
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        st.write("The Root Mean Square Error is: {}".format(rmse))    

        
        
    def RF(self):
        from sklearn.ensemble import RandomForestRegressor
        rf_regressor = RandomForestRegressor(n_estimators = 10)
        rf_regressor.fit(self.X_train, self.y_train)
        
        y_pred = rf_regressor.predict(self.X_test)
        
        st.write("\n")
        st.write("--------------------------------")
        st.write("### Random Forest Regressor ###")
        st.write("--------------------------------")
        

        st.write("The R^2 value is:{}".format(rf_regressor.score(self.X_test, self.y_test)))
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        st.write("The Root Mean Square Error is: {}".format(rmse)) 
        


def st_regression():
    df = pd.read_csv("Data_Modified.csv")

    # select features/columns
    col_names = []
    col_names_y=[]
    feature_list = list(df.columns)

    st.sidebar.write("Select Feature Columns- X)")
    for col_name in feature_list:
        check_box1 = st.sidebar.checkbox(col_name,key=None)
        if check_box1:
            col_names.append(col_name)

    

    df_X = df[col_names]
    df_Y = df["mechanical_target_1"]

    testSize=st.sidebar.slider("Enter Test Data Size (default 0.2)", 0.0,0.4,0.2,0.1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_X,df_Y, test_size = testSize, random_state = 0)


    model = st.sidebar.selectbox(
                'Choose Model', ["Linear Regression", "KNN", "Decision Tree", "Random Forest", "Gradient Boost DT", "Cat Boost"])

    regressions = regression(X_train, X_test, y_train, y_test)

    if model == "Linear Regression":
        try:
            regressions.LR()
        except Exception as e:
            st.write(e)

    if model == "KNN":
        try:
            regressions.KNN()
        except Exception as e:
            st.write(e)
    
    if model == "Decision Tree":
        try:
            regressions.DT()
        except Exception as e:
            st.write(e)

   
    
    if model == "Gradient Boost DT":
        try:
            regressions.GBDT()
        except Exception as e:
            st.write(e)

    if model == "Cat Boost":
        try:
            regressions.CB()
        except Exception as e:
            st.write(e)

    if model == "Random Forest":
        try:
            regressions.RF()
        except Exception as e:
            st.write(e)

    