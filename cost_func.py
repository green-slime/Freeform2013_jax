import BSurface
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import grad, jit, vmap,lax
import config as cfg
import time
import density_func as df

@jit
def args_calculation(i,j,gNui3,gNvi3,Pij,gdNui3,gdNvi3,gddNui3,gddNvi3,ni,no):
    start_time=time.time()
    x=cfg.xmin+i*cfg.h1
    y=cfg.ymin+j*cfg.h2
    curIndex=j*(cfg.M+3)+i
    z=BSurface.query_S(i,j,gNui3,gNvi3,Pij)
    zx=BSurface.query_Su(i,j,gdNui3,gNvi3,Pij)/(cfg.xmax-cfg.xmin)
    zy=BSurface.query_Sv(i,j,gNui3,gdNvi3,Pij)/(cfg.ymax-cfg.ymin)
    zxx=BSurface.query_Suu(i,j,gddNui3,gNvi3,Pij)/pow(cfg.xmax-cfg.xmin,2)
    zxy=BSurface.query_Suv(i,j,gdNui3,gdNvi3,Pij)/((cfg.xmax-cfg.xmin)*(cfg.ymax-cfg.ymin))
    zyy=BSurface.query_Svv(i,j,gNui3,gddNvi3,Pij)/pow(cfg.ymax-cfg.ymin,2)
    b=jnp.sqrt(cfg.a*(pow(zx,2)+pow(zy,2))+1)
    Ox=-zx*(no*b-ni)
    Oy=-zy*(no*b-ni)
    Oz=ni*(zx*zx+zy*zy)+no*b
    tx=x-(z-cfg.tz)*Ox/Oz
    ty=y-(z-cfg.tz)*Oy/Oz
    end_time=time.time()
    print('args_calculation time cost',end_time-start_time,'s')
    return x,y,curIndex,z,zx,zy,zxx,zxy,zyy,b,Ox,Oy,Oz,tx,ty

@jit
def inner_cost_func(no,ni,a,b,z,zx,zy,tx,ty,tz,zxx,zyy,zxy,img_dict): 
    # cost function for inner points
    c=no*b+ni*(zx*zx+zy*zy)
    A1 = (z - tz) * (z - tz) * no / b * (1 + zx * zx + zy * zy) * (no * b - ni) * (no * b - ni) / (c * c * c)
    A2 = (z - tz) * ((no * b - ni) * (no * b * (zy * zy + 1) - ni * zx * zx) + no * ni * a / b * zx * zx * (zx * zx + zy * zy + 1)) / (c * c)
    A3 = (z - tz) * ((no * b - ni) * (no * b * (zx * zx + 1) - ni * zy * zy) + no * ni * a / b * zy * zy * (zx * zx + zy * zy + 1)) / (c * c)
    A4 = 2 * (z - tz) * zx * zy * (no * ni * a / b * (zx * zx + zy * zy + 1) - (no * no * b * b - ni * ni)) / (c * c)
    #A5 = no * b * (zx * zx + zy * zy + 1) / c-df.I(tx,ty)/df.img_E(tx,ty,img_dict)
    temp_term=lax.cond(jnp.abs(df.E_test(tx,ty))<1e-14,lambda x:0.0,lambda x:df.I(tx,ty)/df.E_test(tx,ty),0.0)
    A5 = no * b * (zx * zx + zy * zy + 1) / c-temp_term
    #A5 = no * b * (zx * zx + zy * zy + 1) / c-df.I(tx,ty)/df.E_test(tx,ty)
    res=A1 * (zxx * zyy - zxy * zxy) + A2 * zxx + A3 * zyy + A4 * zxy + A5
    return res

@jit 
def boundary_cost_func(tx,ty):
    return df.boundary_test(tx,ty)

