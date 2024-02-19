import BSurface
import config as cfg
import time
import pynvml
from jax import vmap
import jax.numpy as jnp
def writeToObj(surface: BSurface,Pij,objname=cfg.objname) -> None:
    print("开始写入obj文件...")
    start_time=time.time()
    m=cfg.m;n=cfg.n
    xmax=cfg.xmax; xmin=cfg.xmin; ymax=cfg.ymax; ymin=cfg.ymin
    size = m * n
    objfile = open(objname, 'w')
    dict=surface.query_dict_for_obj()    
    gNui3_for_obj=dict["gNui3_for_obj"];gNvi3_for_obj=dict["gNvi3_for_obj"]
    
    x = [(xmax - xmin) / (m - 1) * i + xmin for i in range(m)]
    y = [(ymax - ymin) / (n - 1) * i + ymin for i in range(n)]

    if objfile:
        # 顶点
        for j in range(n):
            for i in range(m):
                objfile.write(f"v {x[i]} {y[j]} 0\n")
                
        indices_for_obj=jnp.array([(j,i) for j in range(n) for i in range(m)])
        z_result=vmap(lambda i,j:BSurface.query_S(i,j,gNui3_for_obj,gNvi3_for_obj,Pij))(indices_for_obj[:,1],indices_for_obj[:,0])
        for j in range(n):
            for i in range(m):
                objfile.write(f"v {x[i]} {y[j]} {z_result[j*m+i]}\n")
        '''
        for j in range(n):
            for i in range(m):
                objfile.write(f"v {x[i]} {y[j]} {BSurface.query_S(i,j,gNui3_for_obj,gNvi3_for_obj,Pij)}\n")
        '''
        # 面
        for j in range(n - 1):
            for i in range(m - 1):
                index = j * m + i + 1
                objfile.write(f"f {index} {index + m} {index + m + 1} {index + 1}\n")
                index += size
                objfile.write(f"f {index} {index + 1} {index + m + 1} {index + m}\n")
        
        # 四个面
        # 前
        objfile.write("f 1 " + str(m) + " " + " ".join(str(size + i) for i in range(m, 0, -1)) + "\n")
        # 后
        objfile.write("f " + str(size) + " " + str(size - m + 1) + " " + " ".join(str(2 * size - m + i) for i in range(1, m + 1)) + "\n")
        # 左
        objfile.write("f " + str(size - m + 1) + " 1 " + " ".join(str(size + 1 + j * m) for j in range(n)) + "\n")
        # 右
        objfile.write("f " + str(m) + " " + str(size) + " " + " ".join(str(size + j * m + m) for j in range(n - 1, -1, -1)) + "\n")
        
        objfile.close()
        print("数据已写入文件:", objname)
    else:
        print("无法打开文件。")
    end_time=time.time()
    print("写入obj文件用时:",end_time-start_time,"s")
    
    
class NoIdleGPUError(Exception):
    def __init__(self):
        super().__init__("没有找到空闲的GPU")                
def find_idle_gpu():
    pynvml.nvmlInit()
    
    device_count = pynvml.nvmlDeviceGetCount()
    
    for i in range(device_count-1,-1,-1):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        if info.used <= 4000 * 1024 * 1024:
            return i
        else:
            #print(f"gpu {i} using memory {info.used/1024/1024} mb")
            pass
        
    raise NoIdleGPUError()  # 抛出自定义异常