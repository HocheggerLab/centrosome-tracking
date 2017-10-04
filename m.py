import matplotlib.pyplot as plt
import pymc

import pymcelastica

R = pymc.MCMC(pymcelastica)  # build the model
R.sample(10000)  # populate and run it
print 'L   ', R.L.value  # print outputs
print 'E    ', R.E.value
print 'F    ', R.F.value
print 'a1   ', R.a1.stats()
print 'a2   ', R.a2.stats()

# pymc.Matplot.autocorrelation(R.y)
plt.show()
