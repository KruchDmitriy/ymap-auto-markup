import numpy as np
import matplotlib.pyplot as plt
from circle_desc import CircleDescriptor


radius = 32
n_circles = 12

cdesc = CircleDescriptor(radius, n_circles, False)
img = np.zeros(shape=(2 * int(radius) + 1, 2 * int(radius) + 1, 3), dtype='uint8')

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        stride = radius / n_circles

        x = i - radius
        y = j - radius
        ring_idx = int(np.sqrt(x * x + y * y) / stride + 0.5)

        if ring_idx >= n_circles:
            continue

        val = (ring_idx + 1) / n_circles * 255

        img[i, j] = [val, val, val]

result = cdesc.compute(img, int(radius + 0.5), int(radius + 0.5))
print(result.shape[0])

print(result)
plt.imshow(img, cmap=plt.get_cmap('hot'))
plt.show()
