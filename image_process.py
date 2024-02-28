import cv2
import math
import os
import numpy as np
import jax.numpy as jnp
from jax import lax
import config as cfg

class Image:
    def __init__(self, filename, gamma=1.6):
        self.gamma = gamma
        # 读取图像
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        # 检查图像是否成功加载
        if image is None:
            raise Exception("图像加载失败, 请检查文件路径是否正确")
        # 获取图像大小
        self.width = image.shape[1]
        self.height = image.shape[0]
        # 转换为灰度图像
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 保存灰度图像, 以及稍后保存gamma处理过后的灰度图像
        if filename.endswith(('.png','.jpg')):
            name, extension = os.path.splitext(filename)       
            self.outputfilename = f"{name}_grey{extension}"
            cv2.imwrite(self.outputfilename, grayImage)
            self.outputfilename_aftergamma = f"{name}_grey_aftergamma{extension}"
        else:
            raise Exception("文件格式不正确, 请使用png或jpg格式的图像")
        self.greyvalue = np.array(grayImage)
                
        # 计算gamma矫正表
        self.gammavalueTable = np.arange(256)
        self.gammavalueTable = np.power((self.gammavalueTable / 255.0+0.055)/1.055, 1./self.gamma)
            
        # 保存aftergamma的灰度图像
        self.grayValueAfterGamma = 255*self.gammavalueTable[self.greyvalue.astype(np.uint8)]
        cv2.imwrite(self.outputfilename_aftergamma, self.grayValueAfterGamma)
        
        # 归一化数组, 找最小值
        self.normalized_intensity = self.grayValueAfterGamma / np.sum(self.grayValueAfterGamma)
        self.normalized_intensity = jnp.array(self.normalized_intensity)
        self.min_intensity=jnp.min(self.normalized_intensity)
        self.minGrayValue = jnp.min(self.grayValueAfterGamma)
        self.maxGrayValue = jnp.max(self.grayValueAfterGamma)
    def queryDict(self):
        self.pixelNum=self.width*self.height
        self.S_pixel=4*cfg.half_height*cfg.half_width/self.pixelNum
        return {"normalized_intensity":self.normalized_intensity, "min_intensity":self.min_intensity, "width":self.width, "height":self.height,"pixelNum":self.pixelNum,"S_pixel":self.S_pixel,"minGrayValue":self.minGrayValue,"maxGrayValue":self.maxGrayValue}
    
    
def queryIntensity(normalized_intensity,u,v,width,height):
    x=jnp.floor(u*width).astype(jnp.int32)
    y=jnp.floor(v*height).astype(jnp.int32)
    x=jnp.minimum(x,width-1)
    y=jnp.minimum(y,height-1)
    return normalized_intensity[y,x]



if __name__ == "__main__":
    filename='./image/sorcery.jpg'
    #filename='./image/4,2.png'
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    gamma=1.6
    image = Image(filename, gamma)
    dict=image.queryDict()
    print(queryIntensity(dict["normalized_intensity"],0.5,0.5,dict["width"],dict["height"]))
        