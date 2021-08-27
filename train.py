import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import json
import matplotlib.pyplot as plt 

file = "data.csv"
df = pd.read_csv(file, sep = ',') 

X = np.array(df.X).reshape(-1, 1)
y = np.array(df.Y)
lr = LogisticRegression()
lr.fit(X, y)
score = lr.score(X, y)
pred = lr.predict_proba(X[:1])



graphX = np.append(X,pred[0][0])
graphY = np.append(y,pred[0][1])



plt.title("Result")
plt.plot(graphX,color="red") 
plt.plot(graphY,color="green") 

fig1 = plt.gcf()
fig1.savefig('plot.png', dpi=100)

print("Score: %s" % score)

with open("metrics.json", 'w') as outfile:
        json.dump({ "score": score}, outfile)