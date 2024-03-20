from functools import partial
from timeit import timeit
from jax import vmap, jit, random, numpy as jnp

import utils_func as uf
uf.find_idle_gpu()
a=1
b=2
print(jnp.array([[a],[b]]))
a=jnp.array([1,2,3])
b=jnp.array([4,5,6])
c=jnp.concatenate([a,b])
print(c)
input()
'''
test_array=jnp.array([1,2]) # (2,)
test2_array=jnp.array([test_array]) # (1,2)
print(test_array,test2_array,test_array.shape,test2_array.shape)
print(jnp.dot(jnp.transpose(test_array),test_array)) # ()
print(jnp.dot(jnp.transpose(test2_array),test2_array)) # (2,2)
input()
'''
'''
array = jnp.arange(1000)
print(array[-2],array[-1])'''

'''
n, d = 512, 640
a = random.normal(random.PRNGKey(0), (n, d))
b = random.normal(random.PRNGKey(0), (d, d))

mm = jit(jnp.matmul)

aa=jnp.array([a for i in range(1000)])
print(aa.shape)
bb=jnp.array([b for i in range(1000)])


def f1():
    vmap(mm, in_axes=[0,0])(aa, bb)
def f2():
    jit(vmap(mm, in_axes=[0,0]))(aa, bb)
print(timeit(f1, number=1))
print(timeit(f2, number=1))

# DO USE jit to wrap vmap, otherwise the performance will be very bad'''

'''
import numpy as np
from scipy.sparse import coo_matrix
_row  = jnp.array([0, 3, 1, 0])
_col  = jnp.array([0, 3, 1, 0])
_data = jnp.array([4, 5, 7, 9])
coo = coo_matrix((_data, (_row, _col)), shape=(4, 4), dtype=jnp.int64)
coo=coo.toarray()[0:2,:]
print(coo)

# can calculate pos list to a matrix
'''
import jax
matrix = jnp.array([[1, 2], [3, 4]])
norm = jax.numpy.linalg.norm(matrix, ord=2)
print(norm)

from jax.experimental.sparse import BCOO
_row  = jnp.array([0, 3, 1, 0])
_col  = jnp.array([0, 3, 1, 2])
pos=jnp.stack([_row,_col])
pos=pos.transpose()
print(len(pos))
_data = jnp.array([4, 5, 7, 9])
coo = BCOO((_data, pos), shape=(4, 4))
import jax.debug
jax.debug.print("coo:{}",coo.todense())
# can use jax to calculate pos list to a matrix
