import numpy as np
from sko.GA import GA
import pandas as pd
import matplotlib.pyplot as plt

def problem(parameter):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2 = parameter
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)

def other_func(x):
    return (x[0]-1)**2 + x[1]**2
def objective_function(x):
    # 计算目标函数的值
    result = other_func(x)
    return result
#ga = GA(func=objective_function, n_dim=2, size_pop=50, max_iter=800,prob_mut=0.001, lb=[-1, -1, -1], ub=[1, 1, 1], precision=1e-7)

ga = GA(func=objective_function, n_dim=2, size_pop=50, max_iter=100, lb=np.zeros(2,), ub=np.ones(2,),precision=1e-7)
'''
from sko.tool_kit import x2gray
x=np.array([[0,1],[0.5,0.6],[0.2,0.4],[0.8,0.2]])
res = x2gray(x,2,lb=np.zeros(2,), ub=np.ones(2,),precision=1e-7).astype(int)
print(ga.Chrom)
print(res)

ga.Chrom = res
'''
for i in range(10):
    best_x, best_y = ga.run(100)
    print('best_x:', best_x, '\n', 'best_y:', best_y)

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()