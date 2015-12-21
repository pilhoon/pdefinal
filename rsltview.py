import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

dx = np.divide(1., [10, 20, 40, 80, 160, 320, 640])
#dx = [10, 20, 40, 80, 160, 320, 640]
ftcs_norm = [0.243534577626, 0.238925532893, 0.235960870238, 0.234299008815, 0.233421159849, 0.232970234627, 0.232741736763]

#cn lambda=1
cn1_norm = [0.0926772922682, 0.0811738412753, 0.0757000092447, 0.0730718731223, 0.0717894869653, 0.0711568301558, 0.0708427284752]

#cn mu=5
cn2_norm = [0.110616537961, 0.13020192936, 0.149855198491, 0.163319337559, 0.171119666506, 0.175294787129, 0.17745034225]

plt.semilogx(dx, ftcs_norm, dx, cn1_norm, dx, cn2_norm)
plt.figure()
plt.yscale('log')
plt.semilogx(dx, ftcs_norm)
plt.figure()
plt.yscale('log')
plt.semilogx(dx, cn1_norm)
plt.figure()
plt.yscale('log')
plt.semilogx(dx, cn2_norm)
plt.show()

