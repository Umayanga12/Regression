import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("simplelinearregression.csv")
plt.xlabel("Age")
plt.ylabel("Salary")
x = np.array(data.Age.values)
y = np.array(data.Premium.values)

model = LinearRegression()
#need to provide x values as a 2D numpy array
model.fit(x.reshape((-1,1)),y)

plt.scatter(data.Age, data.Premium, color="red")
m,c = np.polyfit(x,y,1)
plt.plot(x,m*x+c)
plt.show()

age = int(input("Enter the Age : "))

if age > 55:
    print("A past Employee")
else:
    #getting the prediction
    prediction = model.predict(np.array([age]).reshape((-1,1)))
    print(prediction)