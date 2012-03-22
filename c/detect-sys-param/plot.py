#!/usr/bin/env python2
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot(range(5))
plt.savefig("a.png")
