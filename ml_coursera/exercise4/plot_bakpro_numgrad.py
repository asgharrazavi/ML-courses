import numpy as np
import matplotlib.pyplot as plt

numgrad = np.loadtxt('num_grad_unrolled_debug.txt')
grad1 = np.loadtxt('grad1_debug.txt')
grad2 = np.loadtxt('grad2_debug.txt')
grad = np.concatenate((grad1.flatten(),grad2.flatten()))
diff = grad - numgrad
print np.linalg.norm(numgrad - grad) / float(np.linalg.norm(numgrad + grad))

plt.plot(diff[::10])
plt.show()
