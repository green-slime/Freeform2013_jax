import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def renderColoredIntensity(matrix,path):
    typeNum=10

    def create_gradient_image(matrix):
        gradient_image = cm.get_cmap('jet', typeNum)(matrix)
        return gradient_image[:, :, :3]

    maxValue=np.max(matrix)
    if(maxValue==0):
        matrix=np.ones_like(matrix)
    else:
        matrix=matrix/maxValue   
        
    def showTwoImg(img1,img2):
        # 创建一个包含两个子图的图形
        fig, axes = plt.subplots(2,1)
        # 在第一个子图中显示第一张图片
        axes[0].imshow(img1)
        plt.axis('off')
        # 在第二个子图中显示第二张图片
        axes[1].imshow(img2)
        axes[1].set_title('color sample')
        plt.axis('off')
        # 调整子图之间的间距
        plt.tight_layout()
        # 显示图形
        #plt.show()          
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"plt save to {path}.")
        plt.close()        

    gradient_image = create_gradient_image(matrix)
    ref_image = create_gradient_image(np.array([np.arange(typeNum)/(typeNum-1)]))
    
    showTwoImg(gradient_image,ref_image)
    '''
    plt.imshow(gradient_image)
    plt.imshow(ref_image)
    plt.axis('off')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"plt save to {path}.")
    plt.close()'''
    
import cv2,os
import image_process as imgp
def readFromImg(path,savePath=None):
    if savePath is None:
        savePath=os.path.join(os.path.dirname(path),"targetColoredIntensity.png")
    image = imgp.Image(path)
    matrix=image.queryDict()["normalized_intensity"]
    maxValue=np.max(matrix)
    minValue=np.min(matrix)
    print(f"maxValue={maxValue},minValue={minValue}")
    matrix=(matrix-minValue)/(maxValue-minValue)
    renderColoredIntensity(matrix,savePath)
        
if __name__ == "__main__":
    readFromImg("/data/wzr/Freeform2013_jax/result_final/zju_57_800_gamma1.0/targetImg.png")
    