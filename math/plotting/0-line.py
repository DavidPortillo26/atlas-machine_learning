#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.plot(y, 'r-')
plt.axis([0, 10, None, None])
plt.show()

cc = ColorConverter('magenta')

plt.figure(figsize=(10, 8))
plt.scatter(x, y, c=cc.to_rgba(), alpha=0.5)

plt.xlabel("Height (in)")
plt.ylabel("Weight (lbs)")
plt.title("Men's Height vs Weight")

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(['Men'], loc='upper left')

plt.tight_layout()
