import BSurface
import config as cfg
import time
import pynvml
from jax import vmap
import os
import sys
import jax.numpy as jnp
def writeToObj(surface: BSurface,Pij,objname=cfg.objname,m=cfg.m,n=cfg.n) -> None:
    print("开始写入obj文件...")
    start_time=time.time()
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
    

# 查找空闲的GPU   
class NoIdleGPUError(Exception):
    def __init__(self):
        super().__init__("没有找到空闲的GPU")                
        
def return_idle_gpu_index():
    pynvml.nvmlInit()
    
    device_count = pynvml.nvmlDeviceGetCount()
    
    for i in range(device_count-1,-1,-1):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        if info.used <= 6000 * 1024 * 1024:
            return i
        else:
            #print(f"gpu {i} using memory {info.used/1024/1024} mb")
            pass
        
    raise NoIdleGPUError()  # 抛出自定义异常

def find_idle_gpu():
    try:
        idle_gpu_index = return_idle_gpu_index()
        print("即将使用GPU：", idle_gpu_index)
        os.environ['CUDA_VISIBLE_DEVICES']=str(idle_gpu_index)
    except NoIdleGPUError:
        print("没有找到空闲的GPU，发生错误")
        sys.exit()
        
import numpy as np
def saveToDict(dict,dict_name=cfg.OT_dict_name):
    try:
        np.save(dict_name,dict)
        print("dict has been saved to "+dict_name)
    except (IOError, OSError) as e:
        print("Failed to save data:", e)
        
def readFromDict(dict_name=cfg.OT_dict_name):
    try:
        loaded_data = np.load(dict_name,allow_pickle=True).item() # need to clarify allow_pickle here if the numpy version high
        return loaded_data
    except (IOError, OSError) as e:
        print("Failed to load data:", e)

import json
def writeToJsonList(filename, data, foldername=cfg.test_folder_name):
    os.makedirs(foldername, exist_ok=True)
    with open(foldername+filename, "w") as f:
        json.dump(data.tolist(), f)
        f.close()
        
def showIterInfo(k,loss,logfile,start_time):
    if(k<=100):
        print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s")
        logfile.write(f"iter:{k} loss:{loss} time cost:{time.time()-start_time}s \n")
        # print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s")
    elif(k%200==0):
        print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s")
        logfile.write(f"iter:{k} loss:{loss} time cost:{time.time()-start_time}s\n")
        
def judgeAndShowEndInfo(k,loss,last_loss,stop,logfile,start_time):
    str=""
    if(abs(loss-last_loss)<1e-16):
        str="relative error reached."
    elif(stop):
        str="stop cond reached."
    print("converged because of "+str)
    print("end at iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s")
    logfile.write(f"iter:{k} loss:{loss} time cost:{time.time()-start_time}s\n")

import cv2,os
def compareTwoImg(path1,path2):
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1.shape != img2.shape:
        print("两张图片大小不一致")
        return
    else:
        diff = cv2.absdiff(img1, img2)
        save_path=os.path.join(os.path.dirname(path1),"diff.png")
        print(save_path)
        cv2.imwrite(save_path, diff)
        print("两张图片已经比较完毕，差异的像素最大值为",np.max(diff))
def put4picturesTogether(mode=1):
    if(mode==1):
        pic1=cv2.imread(cfg.render_picname)
        #pic1=cv2.imread("/home/wzr/Freeform2013_jax/result_new/blbl_57_256_gamma1.0/reRender1280.png")
        pic2=cv2.imread(cfg.render_picname_afterOpt)
        pic3=cv2.imread(cfg.render_picname_afterOptAlter)
        pic4=cv2.imread(cfg.render_picname_allTogether)
    else:
        pic1=cv2.imread(cfg.render_picname_afterOpt)
    # 计算画布大小
    canvas_width = pic1.shape[1] * 2
    canvas_height = pic1.shape[0] * 2
    # 创建空白画布
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    # 将四张图片拼接到画布上
    canvas[0:pic1.shape[0], 0:pic1.shape[1]] = pic1
    canvas[0:pic1.shape[0], pic1.shape[1]:] = pic2
    canvas[pic1.shape[0]:, 0:pic1.shape[1]] = pic3
    canvas[pic1.shape[0]:, pic1.shape[1]:] = pic4
    # 保存拼接好的图片
    path=os.path.join(os.path.dirname(cfg.render_picname),"4pics.png")
    cv2.imwrite(path, canvas)

import image_process as imgp
def integrateReadFromDict(path=cfg.createDictName("train_Init")):
    dict = readFromDict(path)
    Pij=dict["Pij"]   
    M=dict["M"];N=dict["N"]
    print(f"M={M},N={N}")
    assert(M==cfg.M and N==cfg.N)
    #Pij=jnp.ones((cfg.M+3,cfg.N+3))
    surface=BSurface.BSurface(M,N)
    img = imgp.Image(cfg.target_img_path)
    img_dict = img.queryDict()
    print(f"successfully read data from {path}.")
    return Pij,surface,img_dict
    

    
    
if __name__ == "__main__":
    path = "/data/wzr/Freeform2013_jax/result_final/mao_59_512_gamma1.0/"
    Pij, surface, img_dict = integrateReadFromDict(
        path+"OT_dict_test_train_Init_forOnlyOT.npy")
    writeToObj(surface, Pij, path+"init.obj", 400, 400)
    import render as rd
    #rd.render_withColor(Pij, surface, img_dict, path+"colored_render_test2.png")
    
    