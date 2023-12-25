# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:39:36 2023

@author: MOON LIGHT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('economic_data.csv')
print(data)

print(data.describe())





print(data.head())
plt.scatter(data['Year'],data['GDP'])
plt.show()





print(data.head())
plt.scatter(data['Year'], data['GDP'], marker='s', color='red')
plt.show()







print(data.head())

plt.scatter(data['Year'], data['GDP'], marker='*', color='purple', edgecolors='black', s=100)

plt.xlabel('Year')

plt.ylabel('GDP')

plt.title('GDP by Year')

plt.grid(True)

plt.show()







print(data.head())

plt.scatter(data['Year'], data['GDP'], color='skyblue', alpha=0.7)

plt.xlabel('Year')

plt.ylabel('GDP')

plt.title('GDP by Year')

plt.grid(True)

plt.show()
















print(data.head)

x=data.iloc[:,:1]
y=data.iloc[:,1]

print(x)
print(y)

from sklearn.linear_model import  LinearRegression
model =LinearRegression()
model.fit(x,y)

print(model.coef_)

print(model.intercept_)



plt.scatter(x,y)
plt.plot(x,model.predict(x),'r')






plt.scatter(x, y)
plt.plot(x, model.predict(x), 'ro-')
plt.suptitle(" الارباح السنوية ")
plt.text(0.02, 0.95, " الارباح السنوية", transform=plt.gca().transAxes, verticalalignment='top')
plt.figtext(0.01, 0.5, "الارباح السنوية", fontsize=12, rotation=90, va='center')
plt.figtext(0.5, 0.01, " الارباح السنوية", ha='center', fontsize=12)







plt.scatter(x, y)
plt.plot(x, model.predict(x), linestyle='--')




plt.scatter(x, y)
plt.plot(x, y, 'ro-')




plt.scatter(x, y)
plt.plot(x, model.predict(x), linewidth=2)



plt.scatter(x, y, marker='o')
plt.plot(x, model.predict(x), marker='s')


plt.plot(x, model.predict(x), 'r', linewidth=2.5)






plt.scatter(x, y, color='red', marker='*', s=100)

plt.plot(x, model.predict(x), 'r')

plt.suptitle("الارباح")

















model.predict([[2029]])


model.predict([[2024]])
model.predict([[1999]])
model.predict([[2050]])

model.score(x, y)








