import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter

interval = (0.5, 4)  # start, end
accuracy = 0.0001
reps = 600  # number of repetitions
numtoplot = 200
lims = np.zeros(reps)  # list of 600 zeros

fig, biax = plt.subplots()
fig.set_size_inches(16, 9)

lims[0] = np.random.rand()  # first no. in lims is a random no.
converges = []
R = []
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps-1):
        lims[i+1] = r*math.sin(lims[i]*math.pi)


    biax.plot([r]*numtoplot, lims[reps-numtoplot:], 'b.', markersize=.02)
    #  plot 200 of r and 200 of lims calculated

    # count the number of solutions and store it in converges
    conv = len(Counter(lims[reps-numtoplot:]))
    converges.append(conv)
    R.append(r)

biax.set(xlabel='r', ylabel='x', title='sine map')
plt.show()

doubling = [1]
bifurcation_parameter = [0]
once = 0

# identify consecutive period doubling and their respective values of R for values or between 1 and 2
for i in range(len(converges)):
    if R[i] >= 1 and R[i] <= 2:
        if converges[i] == doubling[-1]*2 and converges[i] not in doubling:
            doubling.append(converges[i])
            bifurcation_parameter.append(R[i])


# arrange the doubling and values of R
dub_bif = list(zip(doubling,bifurcation_parameter))
res = sorted(dub_bif, key = lambda x: x[0])
res.remove(res[0])

# calculate the ratio
for i in range(len(res)):
    if res[i][0] >=8:
        ratio = (res[i-1][1] - res[i-2][1])/(res[i][1] - res[i-1][1])
        print(ratio)



