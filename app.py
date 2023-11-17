import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from copy import deepcopy

st.set_page_config(page_title="Inference App")

st.title("Fetch Receipt Count Inference App")

###########################################################
# hide streamlit extras
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

###########################################################
# Load data

st.subheader("original data")

data = pd.read_csv('data_daily.csv')

data['# Date'] = pd.to_datetime(data['# Date'])

plt.plot(data['# Date'],data['Receipt_Count'])
fig = plt.gcf()
st.pyplot(fig)


###########################################################
# Create input and target 

df = deepcopy(data)

df.set_index('# Date', inplace=True)

np_df = df.to_numpy()

# create input values, time standardized for linear regression alg
X = np.arange(1,len(np_df)+1)
X_mean = X.mean()
X_std = X.std()
X = (X-X_mean)/X_std

# create target values, receipt count, also standardized
mean_count = np_df.mean()
std_count = np_df.std()
scaled_df = (np_df-mean_count)/std_count
y = deepcopy(scaled_df[:,0])

X = torch.unsqueeze( torch.tensor(X).float(), 1)
y = torch.unsqueeze( torch.tensor(y).float(), 1)

###########################################################
# load regression model from pth file

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

model = linearRegression(1, 1)

loaded_model = torch.load('regression_model.pth')

with torch.no_grad():
    model.linear.weight.copy_(loaded_model['linear.weight'])
    model.linear.bias.copy_(loaded_model['linear.bias'])

###########################################################

st.subheader("original data with predictions from linear regression model")

option = st.selectbox(
    'Select a month to predict:',
    ('January','February','March','April','May','June','July','August','September','October','November','December'))

st.write('Month Selected: ', option)

if option == "January":
    month_range = range(365,396)
elif option == "February":
    month_range = range(396,424)
elif option == "March":
    month_range = range(424,455)
elif option == "April":
    month_range = range(455,485)
elif option == "May":
    month_range = range(485,516)
elif option == "June":
    month_range = range(516,546)
elif option == "July":
    month_range = range(546,577)
elif option == "August":
    month_range = range(577,608)
elif option == "September":
    month_range = range(608,638)
elif option == "October":
    month_range = range(638,669)
elif option == "November":
    month_range = range(669,699)
elif option == "December":
    month_range = range(699,730)

month_input = torch.unsqueeze( torch.tensor(month_range).float(), 1)

with torch.no_grad():
    month_prediction = model((month_input-X_mean)/X_std).data.numpy()

with torch.no_grad():
    predicted = model(X).data.numpy()

plt.clf()

# reverse standardized data
unscaled_X = (X*X_std)+X_mean
unscaled_y = (y*std_count)+mean_count
unscaled_p = (predicted*std_count)+mean_count
unscaled_mp = (month_prediction*std_count)+mean_count

plt.plot(unscaled_X, unscaled_y, label='True data')
plt.plot(unscaled_X, unscaled_p, '--', label='Predictions')
plt.plot(month_input, unscaled_mp, '--', label='Month Predictions')
plt.legend()
fig = plt.gcf()

predicted_total = unscaled_mp.sum()
st.write('predicted total receipt count for ', option,' 2022:', predicted_total)

predicted_avg = predicted_total/len(unscaled_mp)
st.write('predicted average receipt count per day for ', option,' 2022:', predicted_avg)

st.pyplot(fig)


































