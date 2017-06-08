import numpy as np
import matplotlib.pyplot as plt
import gprtf.main as gpr
from gprtf.common import get_data
import sys, os

test_case = int(sys.argv[1])
variation = int(sys.argv[2])
ker_name =  sys.argv[3]

#x = np.asarray([0.1, 0.7, 1.2, 1.4, 1.6, 1.9, 2.4, 2.7, 3.1, 3.5, 3.9, 4.3, 4.8, 5.3], dtype=np.float32)
#y = np.asarray([2.1, 1.6, 0.5, 0.1, 0.5, 1.2, 2.4, 3.0, 1.6, 2.3, 2.4, 1.8, 1.2, 2.0], dtype=np.float32)
#testx = np.asarray([0.0, 0.3, 1.5, 1.8, 2.6, 3.0, 3.6, 4.2, 5.0, 5.5, 6.0, 6.5])

x, y, testx, truth = get_data(test_case, variation)

opt_hyp, R = gpr.train(x, y, ker_name)
print(opt_hyp, R)
f_mu, f_var = gpr.predict(x, y, ker_name, testx, opt_hyp, R)


plt.figure()
plt.plot(x, y, 'kx', label='Training Pts')
plt.plot(testx, f_mu, 'go', label='Test Pts')
plt.legend(loc=0)
plt.savefig('GPR_Test'+ker_name)
plt.close()
