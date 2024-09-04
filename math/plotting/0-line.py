#!/usr/bin/env python3
""" plots y as a line graph """

import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(y, 'r-')
plt.axis([0, 10, None, None])
plt.show()
