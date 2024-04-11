# ValueError: Expected 2D array, got 1D array instead

import numpy as np
from sklearn.linear_model import LinearRegression

# both arrays are 1-dimensional
x = np.array([1, 5, 3, 2, 1])
y = np.array([2, 4, 6, 8, 10])

x = x.reshape(-1, 1)
# [[1]
#  [5]
#  [3]
#  [2]
#  [1]]
print(x)

model = LinearRegression()

reg = model.fit(x, y)

print(reg.score(x, y))  # 👉️ 0.08035714285714268