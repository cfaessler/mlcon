from fastapi import FastAPI
from joblib import load
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

# TODO load model from file
loaded_model = None


@app.get("/predict/")
def read_item(width: float, height: float):
    # TODO Ask trained model (predict()) for the class
    fruit_class = 0
    if fruit_class == 0:
        return "Apple"
    else:
        return "Pear"
