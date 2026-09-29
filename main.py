import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("house_prices_dataset.csv")

print(df.head())
x=df[["square_feet","num_rooms","age","distance_to_city(km)"]].values
y=df["price"].values
x_train=x[:9500]
x_test=x[9500:]

y_train=y[:9500]
y_test=y[9500:]

w=np.zeros(x_train.shape[1])
b=0

learning_rate=0.07
n=len(x_train)
x_mean=x_train.mean(axis=0)
y_mean=y_train.mean(axis=0)
x_std=x_train.std(axis=0)
y_std=y_train.std(axis=0)
m=x_train.shape[0]
x_scaled=(x_train-x_mean)/x_std
y_scaled=(y_train-y_mean)/y_std




previos_cost=float("inf")
for  i in range(100000):

    y_pre=np.dot(x_scaled,w)+b
    error=y_pre-y_scaled
    cost=np.mean(error**2)/2
  

    djw=np.dot(error,x_scaled)/m
    djb=np.mean(error)
    w=w-learning_rate*djw
    b=b-learning_rate*djb



    if abs(previos_cost-cost)<=0.001:
        print("converged at iteration",i)
       
        break

    previos_cost=cost
print("final weights",w)
print("final bias",b)




y_pre=np.dot(x_scaled,w)+b

plt.scatter(y_pre,y_scaled,c='b',marker='x')
plt.show()

x_scaled_2=(x_test-x_mean)/x_std

y_output=np.dot(x_scaled_2,w)+b




y_output1=y_output*y_std+y_mean

plt.scatter(y_output1,y_test,c='b')
minimum = min(y_output1.min(), y_test.min())
maximum = max(y_output1.max(), y_test.max())

plt.plot([minimum, maximum],
         [minimum, maximum],
         'r--')

plt.title("House-price-prediction")

plt.xlabel("Model calculated price")
plt.ylabel("real-world value")

plt.show()
ss_res = np.sum((y_test - y_output1) ** 2)

ss_tot = np.sum((y_test - y_test.mean()) ** 2)

r2 = 1 - (ss_res / ss_tot)

print("R²:", r2)
print("Accuracy:", r2 * 100, "%")






















