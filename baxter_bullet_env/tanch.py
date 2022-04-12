import numpy as np
import matplotlib.pyplot as plt

in_array = np.linspace(0, 0.3, 200)
out_array = np.tanh(in_array*20)

# red for numpy.tanh()
plt.plot(in_array, out_array, color='red', marker="o")
plt.title("numpy.tanh()")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()