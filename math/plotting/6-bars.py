#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ["Farrah", "Fred", "Felicia"]
[apples, bananas, oranges, peaches] = fruit
plt.bar(
    people,
    apples,
    0.5,
    color='r',
    label="apples"
)
plt.bar(
    people,
    bananas,
    0.5,
    color='y',
    bottom=apples,
    label="bananas"
)
plt.bar(
    people,
    oranges,
    0.5,
    color='#ff8000',
    bottom=bananas+apples,
    label="oranges"
)
plt.bar(
    people,
    peaches,
    0.5,
    color='#ffe5b4',
    bottom=bananas+apples+oranges,
    label="peaches"
)
plt.legend(loc="upper right")
plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.axis([None, None, 0, 80])
plt.show()
