import numpy as np
import math
from jax import jit
import jax
import jax.numpy as jnp
from functools import partial
import config as cfg
import time

def newdiv(x,y):
    if(abs(y)<1e-10):
        return 0
    else:
        return x*1.0/y

class BSurface:
    def __init__(self,M:int,N:int) -> None:
        self.M=M
        self.N=N
        print("Start establishing B-Spline surface...")
        self.calculate_us_and_vs()
        self.calculate_Ns()
        self.calculateAllNsOnGrid()
        self.calculateAllNsOnGrid_forObj()
        print("End establishing B-Spline surface.")
        
    def calculate_us_and_vs(self):
        self.u=np.array([0,0,0])
        self.u=np.concatenate((self.u,np.arange(self.M+1)/self.M))
        self.u=np.concatenate((self.u,np.array([1,1,1])))
        self.v=np.array([0,0,0])
        self.v=np.concatenate((self.u,np.arange(self.N+1)/self.N))
        self.v=np.concatenate((self.u,np.array([1,1,1])))
        
    def calculate_Ns(self):
        u=self.u
        v=self.v
        M=self.M
        N=self.N
        self.Nui0 = []
        for i in range(M + 6):
            def _Nui0(_u, i=i):
                if i == M + 2:
                    return (_u <= u[i + 1] and _u >= u[i])
                else:
                    return (_u < u[i + 1] and _u >= u[i])
            self.Nui0.append(_Nui0)

        self.Nui1 = []
        for i in range(M + 5):
            def _Nui1(_u, i=i):
                return newdiv(_u - u[i], u[i + 1] - u[i]) * self.Nui0[i](_u) + newdiv(u[i + 2] - _u, u[i + 2] - u[i + 1]) * self.Nui0[i + 1](_u)
            self.Nui1.append(_Nui1)

        self.Nui2 = []
        for i in range(M + 4):
            def _Nui2(_u, i=i):
                return newdiv(_u - u[i], u[i + 2] - u[i]) * self.Nui1[i](_u) + newdiv(u[i + 3] - _u, u[i + 3] - u[i + 1]) * self.Nui1[i + 1](_u)
            self.Nui2.append(_Nui2)

        self.Nui3 = []
        for i in range(M + 3):
            def _Nui3(_u, i=i):
                return newdiv(_u - u[i], u[i + 3] - u[i]) * self.Nui2[i](_u) + newdiv(u[i + 4] - _u, u[i + 4] - u[i + 1]) * self.Nui2[i + 1](_u)
            self.Nui3.append(_Nui3)

        self.dNui3 = []
        for i in range(M + 3):
            def _dNui3(_u, i=i):
                return newdiv(3, u[i + 3] - u[i]) * self.Nui2[i](_u) - newdiv(3, u[i + 4] - u[i + 1]) * self.Nui2[i + 1](_u)
            self.dNui3.append(_dNui3)

        self.ddNui3 = []
        for i in range(M + 3):
            def _ddNui3(_u, i=i):
                return newdiv(3, u[i + 3] - u[i]) * (newdiv(2, u[i + 2] - u[i]) * self.Nui1[i](_u) - newdiv(2, u[i + 3] - u[i + 1]) * self.Nui1[i + 1](_u)) \
                    - newdiv(3, u[i + 4] - u[i + 1]) * (newdiv(2, u[i + 3] - u[i + 1]) * self.Nui1[i + 1](_u) - newdiv(2, u[i + 4] - u[i + 2]) * self.Nui1[i + 2](_u))
            self.ddNui3.append(_ddNui3)
            
        self.Nvi0 = []
        for i in range(N + 6):
            def _Nvi0(_v, i=i):
                if i == N + 2:
                    return (_v <= v[i + 1] and _v >= v[i])
                else:
                    return (_v < v[i + 1] and _v >= v[i])
            self.Nvi0.append(_Nvi0)

        self.Nvi1 = []
        for i in range(N + 5):
            def _Nvi1(_v, i=i):
                return newdiv(_v - v[i], v[i + 1] - v[i]) * self.Nvi0[i](_v) + newdiv(v[i + 2] - _v, v[i + 2] - v[i + 1]) * self.Nvi0[i + 1](_v)
            self.Nvi1.append(_Nvi1)

        self.Nvi2 = []
        for i in range(N + 4):
            def _Nvi2(_v, i=i):
                return newdiv(_v - v[i], v[i + 2] - v[i]) * self.Nvi1[i](_v) + newdiv(v[i + 3] - _v, v[i + 3] - v[i + 1]) * self.Nvi1[i + 1](_v)
            self.Nvi2.append(_Nvi2)

        self.Nvi3 = []
        for i in range(N + 3):
            def _Nvi3(_v, i=i):
                return newdiv(_v - v[i], v[i + 3] - v[i]) * self.Nvi2[i](_v) + newdiv(v[i + 4] - _v, v[i + 4] - v[i + 1]) * self.Nvi2[i + 1](_v)
            self.Nvi3.append(_Nvi3)

        self.dNvi3 = []
        for i in range(N + 3):
            def _dNvi3(_v, i=i):
                return newdiv(3, v[i + 3] - v[i]) * self.Nvi2[i](_v) - newdiv(3, v[i + 4] - v[i + 1]) * self.Nvi2[i + 1](_v)
            self.dNvi3.append(_dNvi3)

        self.ddNvi3 = []
        for i in range(N + 3):
            def _ddNvi3(_v, i=i):
                return newdiv(3, v[i + 3] - v[i]) * (newdiv(2, v[i + 2] - v[i]) * self.Nvi1[i](_v) - newdiv(2, v[i + 3] - v[i + 1]) * self.Nvi1[i + 1](_v)) \
                    - newdiv(3, v[i + 4] - v[i + 1]) * (newdiv(2, v[i + 3] - v[i + 1]) * self.Nvi1[i + 1](_v) - newdiv(2, v[i + 4] - v[i + 2]) * self.Nvi1[i + 2](_v))
            self.ddNvi3.append(_ddNvi3)
           
    def calculateAllNsOnGrid(self):
        # here we use M_sample and N_sample to calculate the grid value of Nui3 and Nvi3
        Ms=cfg.M_sample;Ns=cfg.N_sample
        grid_u=np.arange(Ms+1)/(Ms)
        grid_v=np.arange(Ns+1)/(Ns)
        self.gNui3 = []
        self.gdNui3 = []
        self.gddNui3 = []

        for i in range(len(self.Nui3)):
            _gNui3 = []
            _gdNui3 = []
            _gddNui3 = []
            for k in range(Ms+1):
                _gNui3.append(self.Nui3[i](grid_u[k]))
                _gdNui3.append(self.dNui3[i](grid_u[k]))
                _gddNui3.append(self.ddNui3[i](grid_u[k]))
            self.gNui3.append(_gNui3)
            self.gdNui3.append(_gdNui3)
            self.gddNui3.append(_gddNui3)

        self.gNvi3 = []
        self.gdNvi3 = []
        self.gddNvi3 = []

        for j in range(len(self.Nvi3)):
            _gNvi3 = []
            _gdNvi3 = []
            _gddNvi3 = []
            for k in range(Ns+1):
                _gNvi3.append(self.Nvi3[j](grid_v[k]))
                _gdNvi3.append(self.dNvi3[j](grid_v[k]))
                _gddNvi3.append(self.ddNvi3[j](grid_v[k]))
            self.gNvi3.append(_gNvi3)
            self.gdNvi3.append(_gdNvi3)
            self.gddNvi3.append(_gddNvi3)
            
        self.gNui3=jnp.array(self.gNui3)
        self.gNvi3=jnp.array(self.gNvi3)
        self.gdNui3=jnp.array(self.gdNui3)
        self.gdNvi3=jnp.array(self.gdNvi3)
        self.gddNui3=jnp.array(self.gddNui3)
        self.gddNvi3=jnp.array(self.gddNvi3)
            
    def check(self):
        for k in range(cfg.M_sample+1):
            for i in range(self.M+3):
                print("v={}/{} gNui3[{}][{}]={}".format(k, self.M+3 - 1, i, k, self.gNui3[i][k]))

        input("Press Enter to continue...")  
    
    #@partial(jit,static_argnums=[2,1])   
    
    def queryDict(self):
        res={"gNui3":self.gNui3,"gNvi3":self.gNvi3,"gdNui3":self.gdNui3,"gdNvi3":self.gdNvi3,"gddNui3":self.gddNui3,"gddNvi3":self.gddNvi3}
        return res
    
    def calculateAllNsOnGrid_forObj(self):
        #print("Now calculate grid value for obj...")
        m=cfg.m 
        n=cfg.n
        #start_time=time.time()
        grid_u=np.arange(m)/(m-1)
        grid_v=np.arange(n)/(n-1)
        self.gNui3_for_obj = []
        self.gNvi3_for_obj = []
        for i in range(len(self.Nui3)):
            _gNui3 = []
            for k in range(m):
                _gNui3.append(self.Nui3[i](grid_u[k]))
            self.gNui3_for_obj.append(_gNui3)
        for j in range(len(self.Nvi3)):
            _gNvi3 = []
            for k in range(n):
                _gNvi3.append(self.Nvi3[j](grid_v[k]))
            self.gNvi3_for_obj.append(_gNvi3)
        self.gNui3_for_obj=jnp.array(self.gNui3_for_obj)
        self.gNvi3_for_obj=jnp.array(self.gNvi3_for_obj)
        #end_time=time.time()
        #print("calculate grid value for obj time cost:",end_time-start_time,"s")

    def query_dict_for_obj(self):
        res={"gNui3_for_obj":self.gNui3_for_obj,"gNvi3_for_obj":self.gNvi3_for_obj}
        return res
    
    def calculateAllNsOnGrid_forRender(self):
        #print("Now calculate grid value for obj...")
        rm=cfg.rm 
        rn=cfg.rn
        #start_time=time.time()
        grid_u=np.arange(rm+1)/(rm)
        grid_v=np.arange(rn+1)/(rn)
        self.gNui3_for_render = []
        self.gdNui3_for_render = []
        self.gddNui3_for_render = []

        for i in range(len(self.Nui3)):
            _gNui3_for_render = []
            _gdNui3_for_render = []
            _gddNui3_for_render = []
            for k in range(rm+1):
                _gNui3_for_render.append(self.Nui3[i](grid_u[k]))
                _gdNui3_for_render.append(self.dNui3[i](grid_u[k]))
                _gddNui3_for_render.append(self.ddNui3[i](grid_u[k]))
            self.gNui3_for_render.append(_gNui3_for_render)
            self.gdNui3_for_render.append(_gdNui3_for_render)
            self.gddNui3_for_render.append(_gddNui3_for_render)

        self.gNvi3_for_render = []
        self.gdNvi3_for_render = []
        self.gddNvi3_for_render = []

        for j in range(len(self.Nvi3)):
            _gNvi3_for_render = []
            _gdNvi3_for_render = []
            _gddNvi3_for_render = []
            for k in range(rn+1):
                _gNvi3_for_render.append(self.Nvi3[j](grid_v[k]))
                _gdNvi3_for_render.append(self.dNvi3[j](grid_v[k]))
                _gddNvi3_for_render.append(self.ddNvi3[j](grid_v[k]))
            self.gNvi3_for_render.append(_gNvi3_for_render)
            self.gdNvi3_for_render.append(_gdNvi3_for_render)
            self.gddNvi3_for_render.append(_gddNvi3_for_render)
            
        self.gNui3_for_render=jnp.array(self.gNui3_for_render)
        self.gNvi3_for_render=jnp.array(self.gNvi3_for_render)
        self.gdNui3_for_render=jnp.array(self.gdNui3_for_render)
        self.gdNvi3_for_render=jnp.array(self.gdNvi3_for_render)
        self.gddNui3_for_render=jnp.array(self.gddNui3_for_render)
        self.gddNvi3_for_render=jnp.array(self.gddNvi3_for_render)

    def query_dict_for_render(self):
        res={"gNui3_for_render":self.gNui3_for_render,"gNvi3_for_render":self.gNvi3_for_render,"gdNui3_for_render":self.gdNui3_for_render,"gdNvi3_for_render":self.gdNvi3_for_render,"gddNui3_for_render":self.gddNui3_for_render,"gddNvi3_for_render":self.gddNvi3_for_render}
        return res

@jit 
def query_S_one_pos(i,j,uk,vk,gNui3,gNvi3,Pij):
    return gNui3[i][uk]*gNvi3[j][vk]*Pij[j][i]
@jit
def query_S(uk,vk,gNui3:jnp.ndarray,gNvi3:jnp.ndarray,Pij:jnp.ndarray):
    start_time=time.time()
    '''res=0
    #res=jax.vmap(query_S_one_pos,in_axes=[0,0,None,None,None,None,None])(indices[:,1],indices[:,0],uk,vk,gNui3,gNvi3,Pij)
    
    for i in range(cfg.M+3):
        for j in range(cfg.N+3):
            res+=gNui3[i][uk]*gNvi3[j][vk]*Pij[j][i]'''
    res=jnp.dot(gNui3[:,uk],jnp.dot(Pij,gNvi3[:,vk]))

    end_time=time.time()
    print('query_S time cost',end_time-start_time,'s')
    #return sum(res)
    return res
@jit
def query_Su(uk,vk,gdNui3:jnp.ndarray,gNvi3:jnp.ndarray,Pij:jnp.ndarray):
    '''res=0
    for i in range(cfg.M+3):
        for j in range(cfg.N+3):
            res+=gdNui3[i][uk]*gNvi3[j][vk]*Pij[j][i]'''
    res=jnp.dot(gdNui3[:,uk],jnp.dot(Pij,gNvi3[:,vk]))
    return res
    #return sum(res)
@jit
def query_Sv(uk,vk,gNui3:jnp.ndarray,gdNvi3:jnp.ndarray,Pij:jnp.ndarray):
    '''res=0
    for i in range(cfg.M+3):
        for j in range(cfg.N+3):
            res+=gNui3[i][uk]*gdNvi3[j][vk]*Pij[j][i]'''
    res=jnp.dot(gNui3[:,uk],jnp.dot(Pij,gdNvi3[:,vk]))
    return res

@jit
def query_Suu(uk,vk,gddNui3:jnp.ndarray,gNvi3:jnp.ndarray,Pij:jnp.ndarray):
    '''res=0
    for i in range(cfg.M+3):
        for j in range(cfg.N+3):
            res+=gddNui3[i][uk]*gNvi3[j][vk]*Pij[j][i]'''
    res=jnp.dot(gddNui3[:,uk],jnp.dot(Pij,gNvi3[:,vk]))
    return res
@jit
def query_Suv(uk,vk,gdNui3:jnp.ndarray,gdNvi3:jnp.ndarray,Pij:jnp.ndarray):
    '''res=0
    for i in range(cfg.M+3):
        for j in range(cfg.N+3):
            res+=gdNui3[i][uk]*gdNvi3[j][vk]*Pij[j][i]'''
    res=jnp.dot(gdNui3[:,uk],jnp.dot(Pij,gdNvi3[:,vk]))
    return res
@jit
def query_Svv(uk,vk,gNui3:jnp.ndarray,gddNvi3:jnp.ndarray,Pij:jnp.ndarray):
    '''res=0
    for i in range(cfg.M+3):
        for j in range(cfg.N+3):
            res+=gNui3[i][uk]*gddNvi3[j][vk]*Pij[j][i]'''
    res=jnp.dot(gNui3[:,uk],jnp.dot(Pij,gddNvi3[:,vk]))
    return res

def test():
    pass   
if __name__ == "__main__":
    surface=BSurface(3,3,np.ones((6,6)))
    #print(surface.check())
    dict=surface.queryDict()
    gNui3=dict["gNui3"]
    gNvi3=dict["gNvi3"]
    Pij=dict["Pij"]
    for i in range(cfg.M+3):
        for j in range(cfg.N+3):
            print("i,j:",i,j,query_S(i,j,gNui3,gNvi3,Pij))