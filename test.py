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
os.environ['CUDA_VISIBLE_DEVICES']='3'
@jit 
def try_zero(array):
  size=jnp.size(array)
  return jnp.zeros((1,size))
@jit 
def try_negate_boolean(array):
  return ~array
# 示例使用
array = jnp.array([-11, 2, 37, -4, 5, 6, 7, -8, 9])  # 原始数组
print(try_zero(array))
condition = jnp.array([False, True, True, False, False, True, False, False, True])  # 布尔数组
condition2 = try_negate_boolean(condition)
zero_array=jnp.zeros((1,jnp.size(array)))
true_result=jnp.where(condition,array,zero_array)
false_result=jnp.where(condition,zero_array,array)
max1=jnp.max(jnp.abs(true_result));max2=jnp.max(jnp.abs(false_result))
weight=max1/max2
print("max2,max1,weight=",max2,max1,weight)
res_final=true_result+false_result*weight
print(array)
print(true_result)
print(false_result)
print(res_final)
print(res_final[0])

a=jnp.array([-1])
b=jnp.sqrt(a)
c=jnp.array([[jnp.nan,jnp.nan],[jnp.nan,jnp.nan]])
d=jnp.array(([jnp.nan,jnp.nan]))
print(jnp.linalg.norm(b))
print(jnp.linalg.solve(c,d))
a=jnp.array([1,2,3])
@jit 
def myFloor(u,a):
  k=jnp.floor(u).astype(jnp.int32)
  res=0
  for i in range(k+1):
    res+=a[i]
  return res
print(myFloor(1.2,a))

print(jnp.dot(a,a))
print(jnp.dot(jnp.transpose(a),a))

time_start=time.time()
@jit 
def outer_B(res):
  return res+1
def B():
  res=0
  for i in range(10):
    res=outer_B(res)
  return res
B()
end_time=time.time()
print('time cost2',end_time-time_start,'s')

time_start=time.time()
def A():
  res=0
  @jit 
  def inner_A(res):
    return res+1
  for i in range(10000):    
    res=inner_A(res)
  return res
A()
end_time=time.time()
print('time cost1',end_time-time_start,'s')

time_start=time.time()
@jit 
def outer_C(res):
  return res+1
def C():
  res=0
  for i in range(10000):
    res=outer_C(res)
  return res
C()
end_time=time.time()
print('time cost3',end_time-time_start,'s')


@jit 
def one_and_one():
  found=1&1
  return lax.cond(found,lambda x:x+1,lambda x:x-1,1)

print(one_and_one())

A = jnp.array([[1, 2], [3, 4]])
print(jnp.dot(jnp.transpose(A), A))

test_mat=jnp.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
indices = jnp.array([(j,i) for j in range(3) for i in range(5)])
bool_mask = (1 <= indices[:, 1]) & (indices[:, 1] <= 3) & (1 <= indices[:, 0]) & (indices[:, 0] <= 3)
#print(bool_mask)
# Expand the original array with the boolean mask
expanded_indices = jnp.concatenate([indices, bool_mask[:, jnp.newaxis]], axis=1)

def newArray(i,j,Pij):
    return Pij[j][i]
  
def newArray2(Pij):
    return vmap(newArray,in_axes=[0,0,None])(indices[:,1],indices[:,0],Pij)
  
print(newArray2(test_mat))

n = 5  # 内部循环次数, 指标j
m = 3  # 外部循环次数，指标i
def sum_double_loop(_i,carry):
    j, acc = carry[1:3]
    i=_i
    return (i, (j+1)%n, acc + test_mat[i,j])

init_val = (0, 0, 0)  # 初始状态

result = lax.fori_loop(0, n, lambda j, carry: lax.fori_loop(0, m, sum_double_loop,carry), init_val)

print(result[0],result[1],result[2])  # 打印求和结果

@jit
def outerfunc(x):
  @jit 
  def innerfunc():
    print("inner")
    print(x)
  innerfunc()
  innerfunc()
  print("outer")
  
outerfunc(1)
outerfunc(2)
#inner
#Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>
#outer

def test_pos(x,y,a,b,c):
    return x+y+a+b+c
import jax

@jax.jit
def A():
  print('compiling a')
  return

@jax.jit
def B():
  A()
  A()

B()
# compiling a

print(A._cache_size())
# 1
b=np.array([1,2,3,4,5,6])
print(b.reshape((2,3)))
a0=jnp.empty((0,3))
a1=jnp.array([1,2,3])
a2=jnp.array([4,5,6])
print(jnp.vstack((a0,a1,a2)))
a=jnp.ones((3,4)).at[1,2].add(3)
print(a)
print(jnp.transpose(a))
print(jnp.linalg.norm(a))

print(indices)
print(expanded_indices)
print(indices[:,0])
print(jax.vmap(test_pos,in_axes=[0,0,None,None,None])(indices[:,0],indices[:,1],1,1,1))
operand = jnp.array([0.])
print(lax.cond(True, lambda x: x+1, lambda x: x-1, operand))
print(operand)
key=jax.random.PRNGKey(0)

size=5

@jit
def mul_test(x,y):
    return x,y,x*y

print(mul_test(3,4))

time_start=time.time()
s=BSurface.BSurface(9,9,np.ones((9,9)))
time_end=time.time()
print('time cost',time_end-time_start,'s')

test_dict=s.queryDict()
#print(test_dict)

@jit
def query_S_2(i,j,gNui3,gdNui3,gNvi3,Pij):
    res=BSurface.query_S(i,j,gNui3,gNvi3,Pij)
    res+=BSurface.query_Su(i,j,gdNui3,gNvi3,Pij)
    return res

time_start=time.time()
print(query_S_2(3,4,test_dict["gNui3"],test_dict["gdNui3"],test_dict["gNvi3"],test_dict["Pij"]))
time_end=time.time()
print('time cost',time_end-time_start,'s')

time_start=time.time()
print(query_S_2(3,5,test_dict["gNui3"],test_dict["gdNui3"],test_dict["gNvi3"],test_dict["Pij"]))
time_end=time.time()
print('time cost',time_end-time_start,'s')




'''
a=np.random.rand(2500,2500)
b=np.random.rand(2500,1)
time_start=time.time()
x = np.linalg.solve(a, b)
time_end=time.time()
print('time cost',time_end-time_start,'s')'''