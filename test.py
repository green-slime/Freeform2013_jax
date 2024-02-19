import jax
import jax.numpy as jnp
import numpy as np
from jax import device_put,jit
import time
import matplotlib.pyplot as plt
from BSurface import BSurface
import BSurface
import density_func
from jax import lax
import config as cfg
from jax import vmap
import os
import sys

a=jnp.empty((0,3))
print(a)
print(jnp.concatenate([a,jnp.array([[1,2,3]])]))

@jit
def test_func(x):
    return x*x

xs=jnp.array([1,2,3,4,5,6,7])
chunk_size=2
print(jnp.concatenate([vmap(lambda x:x*x,in_axes=[0])(xs[i:i+chunk_size]) for i in range(0, 7, chunk_size)]))

@jit
def vmap_chunked(f, xs:jnp.array, chunk_size=1):
    return jnp.concatenate([vmap(f,in_axes=[0])(xs[i:i+chunk_size]) for i in range(0, 7, chunk_size)])

print(vmap_chunked(test_func,jnp.array([1,2,3,4,5,6,7]),2))
    
    
def make_indices():
    indices = jnp.array([(j,i) for j in range(3) for i in range(2)])    
    print(indices[:,1])
    return indices
@jit
def test_sum(i,j):
    res=1
    res=res*i/j
    return res

z=vmap(lambda i,j:test_sum(i,j),in_axes=[0,0])(jnp.array([1,2,3]),jnp.array([3,2,1]))
print(z[1])