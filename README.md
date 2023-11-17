# fetch-take-home-linear-regression-app
App to predict scanned receipt count for 2022 based on 2021 data for Fetch take home exercise

Code for the app is contained in app.py and the code for the model is in the notebook regression_model.ipynb

to run the streamlit app run the following commands in the directory containing the Dockerfile

docker build -t [DOCKER_IMAGE_NAME] .

docker run -p 8501:8501 [DOCKER_IMAGE_NAME]

replace [DOCKER_IMAGE_NAME] with a non-conflicting docker image name eg:

docker build -t linear_regression_app_balkir .

docker run -p 8501:8501 linear_regression_app_balkir
