#!/usr/bin/env python3
""" plots y as a line graph """

import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

line_plot = plt.plot(y, 'r-')

# Save the figure
plt.savefig('0-line.png')

# Close the plot
plt.close()
