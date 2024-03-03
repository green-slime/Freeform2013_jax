from functools import partial
from timeit import timeit
from jax import vmap, jit, random, numpy as jnp

import utils_func as uf
uf.find_idle_gpu()
array = jnp.arange(1000)
print(array[-2],array[-1])

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

import numpy as np
from scipy.sparse import coo_matrix
_row  = jnp.array([0, 3, 1, 0])
_col  = jnp.array([0, 3, 1, 0])
_data = jnp.array([4, 5, 7, 9])
coo = coo_matrix((_data, (_row, _col)), shape=(4, 4), dtype=jnp.int64)
coo=coo.toarray()[0:2,:]
print(coo)