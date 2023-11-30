# Run App

    $ pip install -r requirements.txt
    $ uvicorn main:app --reload

# Lab
The aim of this lab is to create a web API which can be used to do inference on a pre trained model. So you get the idea of a lifecycle of a ML project from
From data loading to model training to deployment.

Tasks:
- Create a Python script train that trains a DecisionTreeClassifier for the given dataset and dumps the trained model
  into a file
- Load the trained model into the main.py
- Tryout inference api on http://localhost:8000/docs
