from sklearn import tree
import pandas as pd

X = [[7.1, 7.3],
     [7.9, 7.5],
     [7.4, 7.0],
     [8.2, 7.3],
     [7.6, 6.9],
     [7.8, 8.0],
     [7.0, 7.5],
     [7.1, 7.9],
     [6.8, 8.0],
     [7.3, 8.2],
     [7.2, 7.9]]
Y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0: Apple 1: Pear
df = pd.DataFrame(X, columns=["width", "height"])
classifier = tree.DecisionTreeClassifier()  # Choosing the algorithm
model = classifier.fit(X, Y)  # Do the training on the data
# TODO dump the trained model to a file
