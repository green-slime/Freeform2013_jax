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
from jax.lax import batch_vmap
import os
import sys
import utils_func as uf
from math import floor
from timeit import timeit

a=np.array([277])
print(a.astype(np.uint8))




uf.find_idle_gpu()

uf.compareTwoImg("/data/wzr/Freeform2013_jax/result_new/blbl_46_1280_gamma1.0/128_1280_img_test7.png","/data/wzr/Freeform2013_jax/result_new/blbl_46_1280_gamma1.0/128_1280_img_test9.png")
input()

def testFunc(a,b):
    return a+b 
def useFunc(x,f):
    print(f(x,1))
useFunc(10,testFunc)
os.system("pause")
# 矩阵乘向量
'''
Pij=jnp.array([[1,2,3],[4,5,6],[7,8,9]])
U=Pij[:,0];print(U)
V=Pij[:,2];print(V)
def f_test():
    res=0
    for i in range(3):
        for j in range(3):
            res+=U[i]*Pij[i,j]*V[j]
def f_2():
    return jnp.dot(U,jnp.dot(Pij,V))
for f in [f_test,f_2]:
    t=timeit.timeit(f,number=1000)
    print(f'{t:.3f}')
#结果：29.660
#      0.484
#     因此向量乘法要好很多
sys.exit()
'''
# 字典传引用，数组传值？
'''
import timeit
dict={"a":jnp.ones((1000,1000)), "b":jnp.zeros((1000,1000))}
a=dict["a"];b=dict["b"]
@jit
def f1(x,dict1):
    a1=dict1["a"];b1=dict1["b"]
    return a1+b1
@jit
def f2(x,a2,b2):
    return a2+b2
jax.config.update('jax_platform_name', 'gpu')
f1r=lambda :jit(vmap(f1,in_axes=[0,None]))(jnp.arange(1000),dict)
f2r=lambda :jit(vmap(f2,in_axes=[0,None,None]))(jnp.arange(1000),a,b)
for f in [f1r,f2r]:    
    t=timeit.timeit(f,number=1000)
    print(f'{t:.3f}')'''
#结果：101.524
#96.429
#因此二者相近，不需要特殊对待

#stack二维数组
'''
emp=jnp.empty((0,2,3))
arr23=jnp.array([[1,2,3],[4,5,6]])
emp=jnp.append(emp,jnp.expand_dims(arr23,axis=0),axis=0)
emp=jnp.append(emp,jnp.expand_dims(arr23,axis=0),axis=0)
print(emp)'''

#XLA_PYTHON_CLIENT_MEM_FRACTION=.50
XLA_PYTHON_CLIENT_PREALLOCATE=False
@jit
def f(x):
    return jnp.array([1,2,3])
#from netket.jax import vmap_chunked

#result2=vmap(f)(jnp.arange(floor(0.5*1e9)))
#jax.clear_backends()
#jax.clear_caches()

#del result2
print("success.")

#vmap_chunked(f,chunk_size=floor(0.25*1e9))(jnp.arange(floor(1e10)))
# Example 1:
#print(vmap(f)(jnp.arange(3)))
#print(batch_vmap(f,batch_size=2)(jnp.arange(3)))
# Example 2:
#vmap(f)(jnp.arange(1e10))
result=jnp.empty((0,floor(0.25*1e9),3))
#result2=vmap(f)(jnp.arange(floor(1e9)))
#del result2
#print("success.")

#试验说明for的确是串行的，与手动复制多次一样，只是内存无法释放
#试验说明直接循环十次没有问题，但是concatenate会造成问题
#tmd我知道了，单纯是数组太大溢出了，跟vmap没关系
a=jnp.ones((1,2,3))

for i in range(10):
    temp_array=vmap(f)(jnp.arange(floor(0.25*1e9)))
    result=jnp.append(result,jnp.expand_dims(a,axis=0))
    # wouldn't OOM
    # but result=jnp.append(temp_array,jnp.expand_dims(a,axis=0)) would.


'''
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
print(z[1])'''