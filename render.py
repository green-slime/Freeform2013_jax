from jax.experimental.sparse import BCOO
import utils_func as uf
from jax import vmap, jit, lax
from jax.lax import batch_vmap
import jax.numpy as jnp
import cost_func as cf
import config as cfg
import jax.config
import BSurface
import numpy as np
# from jax.lax import batch_vmap
import cv2
import image_process as imgp
import time
import jax.experimental.sparse.coo as coo


@jit
def outOfRange(tx, ty):
    return (tx < -cfg.half_width) | (tx >= cfg.half_width) | (ty < -cfg.half_height) | (ty >= cfg.half_height)


@jit
def decideOnWhichGrid(tx, ty):
    ind_x = jnp.floor((tx+cfg.half_width)/cfg.dx).astype(jnp.int64)
    # notice that (0,0) is the lefttop of the img
    ind_y = jnp.floor((cfg.half_height-ty)/cfg.dy).astype(jnp.int64)
    return lax.cond(outOfRange(tx, ty), lambda x: jnp.array([cfg.ry, cfg.rx]), lambda x: jnp.array([ind_y, ind_x]), 0.0)
    # if out of range, then return (cfg.ry,cfg.rx), which is not exist


@jit
def makeMatrix(add_index):

    data = jnp.ones(len(add_index))
    grid_array = BCOO((data, add_index), shape=(cfg.ry+1, cfg.rx+1))
    ray_loss = grid_array.todense()[cfg.ry, cfg.rx]
    grid_array = grid_array.todense()[0:cfg.ry, 0:cfg.rx]  # [0,ry)

    # print("ray loss:"+str(round(ray_loss*100/(cfg.rm+1)/(cfg.rn+1),3))+"%.")
    return grid_array
@jit 
def makeMatrix_withLoss(add_index):
    data = jnp.ones(len(add_index))
    grid_array = BCOO((data, add_index), shape=(cfg.ry+1, cfg.rx+1))
    ray_loss = grid_array.todense()[cfg.ry, cfg.rx]
    grid_array = grid_array.todense()[0:cfg.ry, 0:cfg.rx]  # [0,ry)

    # print("ray loss:"+str(round(ray_loss*100/(cfg.rm+1)/(cfg.rn+1),3))+"%.")
    return grid_array,ray_loss

@jit
def render_loss_with_inputIndices(Pij, indices, renderMeshdict, target_intensity: jnp.ndarray):
    '''
    return grid_array; we will add them later
    '''
    gNui3_for_render = renderMeshdict["gNui3_for_render"]
    gNvi3_for_render = renderMeshdict["gNvi3_for_render"]
    gdNui3_for_render = renderMeshdict["gdNui3_for_render"]
    gdNvi3_for_render = renderMeshdict["gdNvi3_for_render"]
    gddNui3_for_render = renderMeshdict["gddNui3_for_render"]
    gddNvi3_for_render = renderMeshdict["gddNvi3_for_render"]

    result = jit(batch_vmap(cf.args_calculation, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None], batch_size=int(cfg.sample_chunk_size)))(indices[:, 1], indices[:, 0], gNui3_for_render, gNvi3_for_render, Pij, gdNui3_for_render, gdNvi3_for_render, gddNui3_for_render, gddNvi3_for_render, cfg.ni, cfg.no, cfg.rm, cfg.rn)

    result = jnp.transpose(jnp.array([result[-2], result[-1]]))
    add_index = jit(batch_vmap(decideOnWhichGrid, in_axes=[
                    0, 0], batch_size=int(cfg.sample_chunk_size)))(result[:, 0], result[:, 1])
    
    grid_array = makeMatrix(add_index)
    return grid_array

@jit
def render_loss_Alter_withRayLoss(Pij, renderMeshdict, target_intensity: jnp.ndarray):
    # when returnType=2, return result_intensity
    gNui3_for_render = renderMeshdict["gNui3_for_render"]
    gNvi3_for_render = renderMeshdict["gNvi3_for_render"]
    gdNui3_for_render = renderMeshdict["gdNui3_for_render"]
    gdNvi3_for_render = renderMeshdict["gdNvi3_for_render"]
    gddNui3_for_render = renderMeshdict["gddNui3_for_render"]
    gddNvi3_for_render = renderMeshdict["gddNvi3_for_render"]
    indices = jnp.array([(j, i) for j in range(cfg.rn+1)
                        for i in range(cfg.rm+1)])

    result = jit(batch_vmap(cf.args_calculation, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None], batch_size=int(cfg.sample_chunk_size)))(indices[:, 1], indices[:, 0], gNui3_for_render, gNvi3_for_render, Pij, gdNui3_for_render, gdNvi3_for_render, gddNui3_for_render, gddNvi3_for_render, cfg.ni, cfg.no, cfg.rm, cfg.rn)

    result = jnp.transpose(jnp.array([result[-2], result[-1]]))
    add_index = jit(batch_vmap(decideOnWhichGrid, in_axes=[
                    0, 0], batch_size=int(cfg.sample_chunk_size)))(result[:, 0], result[:, 1])

    #grid_array = makeMatrix(add_index)
    
    grid_array, rayloss = makeMatrix_withLoss(add_index)
    # when out of range, add_index[i] will be [-1,-1], so nothing will be added to grid_array due to jnp's lazy evaluation

    allNum = jnp.sum(grid_array)
    print(f"all_shouldbe={(cfg.rm+1)*(cfg.rn+1)},all_detected={allNum+rayloss}")
    print("ray loss:"+str(round(rayloss*100/(allNum+rayloss),3))+"%.")
    #jax.debug.print("rayLoss rate={}%",round(rayloss*100/allNum,3))
    #jax.debug.print("allNum={}",allNum)
    #print("m*n=",(cfg.rm+1)*(cfg.rn+1))
    result_intensity = grid_array/allNum

    # return jax.lax.cond(returnType==1,lambda x:jnp.linalg.norm(result_intensity-target_intensity, ord=2),lambda x:result_intensity,0.0)
    #return jnp.reshape(jnp.divide(result_intensity-target_intensity,target_intensity), (cfg.ry*cfg.rx,)) # become a 1D array
    return jnp.reshape(result_intensity-target_intensity,(cfg.ry*cfg.rx,)),rayloss

@jit 
def render_loss_Alter(Pij, renderMeshdict, target_intensity: jnp.ndarray):
    return render_loss_Alter_withRayLoss(Pij, renderMeshdict, target_intensity)[0]


@jit
def render_loss(Pij, renderMeshdict, target_intensity: jnp.ndarray):
    # when returnType=2, return result_intensity
    gNui3_for_render = renderMeshdict["gNui3_for_render"]
    gNvi3_for_render = renderMeshdict["gNvi3_for_render"]
    gdNui3_for_render = renderMeshdict["gdNui3_for_render"]
    gdNvi3_for_render = renderMeshdict["gdNvi3_for_render"]
    gddNui3_for_render = renderMeshdict["gddNui3_for_render"]
    gddNvi3_for_render = renderMeshdict["gddNvi3_for_render"]
    indices = jnp.array([(j, i) for j in range(cfg.rn+1)
                        for i in range(cfg.rm+1)])

    result = jit(batch_vmap(cf.args_calculation, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None], batch_size=cfg.sample_chunk_size))(
        indices[:, 1], indices[:, 0], gNui3_for_render, gNvi3_for_render, Pij, gdNui3_for_render, gdNvi3_for_render, gddNui3_for_render, gddNvi3_for_render, cfg.ni, cfg.no, cfg.rm, cfg.rn)

    tx_ty_list = jnp.transpose(jnp.array([result[-2], result[-1]]))
    add_index = jit(batch_vmap(decideOnWhichGrid, in_axes=[
                    0, 0], batch_size=cfg.sample_chunk_size))(tx_ty_list[:, 0], tx_ty_list[:, 1])

    grid_array = makeMatrix(add_index)
    # when out of range, add_index[i] will be [-1,-1], so nothing will be added to grid_array due to jnp's lazy evaluation

    allNum = jnp.sum(grid_array)
    result_intensity = grid_array/allNum

    # return jax.lax.cond(returnType==1,lambda x:jnp.linalg.norm(result_intensity-target_intensity, ord=2),lambda x:result_intensity,0.0)
    return jax.numpy.linalg.norm(result_intensity-target_intensity, ord=2)


def renderIntensityToImg(imgdict, intensity, picname):
    totalGrayValue_shouldbe = imgdict["totalGrayValue"] / \
        imgdict["pixelNum"]*cfg.rx*cfg.ry
    final_grid_array = totalGrayValue_shouldbe*intensity
    # print(grid_array)
    final_grid_array = np.array(final_grid_array)
    final_grid_array = np.clip(final_grid_array,0,255) # uint8 will mod 256
    final_grid_array = final_grid_array.astype(np.uint8)
    # print(type(grid_array))

    uf.writeToJsonList("final_grid_array.json", final_grid_array)
    # uf.writeToJsonList("grid_array.json",grid_array)

    cv2.imwrite(picname, final_grid_array)

    print("picture saved as", picname)

import showColoredIntensity as sci
def render(Pij, surface: BSurface, imgdict: dict, picname=cfg.render_picname,rm=cfg.rm,rn=cfg.rn,colored_picPath=None):
    start_time = time.time()
    print("rendering...")
    surface.calculateAllNsOnGrid_forRender(rm,rn)
    dict = surface.query_dict_for_render()

    gNui3_for_render = dict["gNui3_for_render"]
    gNvi3_for_render = dict["gNvi3_for_render"]
    gdNui3_for_render = dict["gdNui3_for_render"]
    gdNvi3_for_render = dict["gdNvi3_for_render"]
    gddNui3_for_render = dict["gddNui3_for_render"]
    gddNvi3_for_render = dict["gddNvi3_for_render"]
    # print(gNui3_for_render)

    indices = jnp.array([(j, i) for j in range(rn+1)
                        for i in range(rm+1)])
    # print(indices)
    # uf.writeToJsonList("indices.json",indices)
    result = jit(batch_vmap(cf.args_calculation, in_axes=[0, 0, None, None, None, None, None, None, None, None, None, None, None], batch_size=2000**2))(
        indices[:, 1], indices[:, 0], gNui3_for_render, gNvi3_for_render, Pij, gdNui3_for_render, gdNvi3_for_render, gddNui3_for_render, gddNvi3_for_render, cfg.ni, cfg.no, rm, rn)
    # print("result:",result)
    # print("z:",result[3])
    end_time = time.time()
    # print("tx_ty_list calculating finished. Computation time cost", end_time-start_time, "s.")
    # uf.saveToDict({"result":result},cfg.folder_name+"temp_result.npy")
    # result =result[-2:]
    tx_ty_list = jnp.transpose(jnp.array([result[-2], result[-1]]))
    # print(tx_ty_list) # [[tx1,ty1],[tx2,ty2],...]
    # grid_array=jnp.zeros((cfg.ry,cfg.rx))

    add_index = jit(batch_vmap(decideOnWhichGrid, in_axes=[
                    0, 0], batch_size=2000**2))(tx_ty_list[:, 0], tx_ty_list[:, 1])

    # uf.writeToJsonList("tx_ty_list.json",tx_ty_list)
    # uf.writeToJsonList("add_index.json",add_index)
    # print(need_to_add)
    #grid_array = makeMatrix(add_index)
    grid_array, rayloss = makeMatrix_withLoss(add_index)
    print("ray loss:"+str(round(rayloss*100/(cfg.rm+1)/(cfg.rn+1),3))+"%.")
    # Here we introduce coloredRenderer
    if not colored_picPath==None:     
        sci.renderColoredIntensity(np.array(grid_array),colored_picPath)
    # when out of range, add_index[i] will be [-1,-1], so nothing will be added to grid_array due to jnp's lazy evaluation
    # print(grid_array)
    # maxNum=jnp.max(grid_array);minNum=jnp.min(grid_array);
    allNum = jnp.sum(grid_array)
    result_intensity = grid_array/allNum
    # print(imgdict["maxGrayValue"],imgdict["minGrayValue"])
    # final_grid_array=imgdict["minGrayValue"]+(imgdict["maxGrayValue"]-imgdict["minGrayValue"])*(grid_array-minNum)/(maxNum-minNum)
    totalGrayValue_shouldbe = imgdict["totalGrayValue"] / \
        imgdict["pixelNum"]*cfg.rx*cfg.ry
    final_grid_array = totalGrayValue_shouldbe*result_intensity
    # print(grid_array)
    final_grid_array = np.array(final_grid_array)
    final_grid_array = np.clip(final_grid_array,0,255) # uint8 will mod 256
    final_grid_array = final_grid_array.astype(np.uint8)
    # print(type(grid_array))

    uf.writeToJsonList("final_grid_array.json", final_grid_array)
    # uf.writeToJsonList("grid_array.json",grid_array)

    cv2.imwrite(picname, final_grid_array)
    end_time = time.time()
    print("rendering finished. Computation time cost", end_time-start_time, "s.")
    print("picture saved as", picname)


import os
import LM_algo_Alter as lma
import LM_algo_for_renderLoss as lmr
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')
    uf.find_idle_gpu()
    #dict = uf.readFromDict(cfg.OT_dict_name)
    #dict = uf.readFromDict(cfg.OT_dict_test_name)
    dict = uf.readFromDict("./result_final/zju_67_800_gamma1.0/OT_dict_test_train_onlyOT_2nd.npy")
    Pij = dict["Pij"]
    M = dict["M"]
    N = dict["N"]
    print(M, N)
    assert (M == cfg.M and N == cfg.N)
    # Pij=jnp.ones((cfg.M+3,cfg.N+3))
    surface = BSurface.BSurface(M, N)

    img = imgp.Image(cfg.target_img_path)
    imgdict = img.queryDict()
    #Pij = lma.solve_using_LM(Pij, surface, imgdict)
    #Pij = lmr.solve_using_LM(Pij, surface, imgdict)
    # rd.renderIntensityToImg(img_dict,final_intensity,cfg.render_picname_afterOpt)
    #render(Pij, surface, imgdict, cfg.render_picname_afterOptAlter)
    #render(Pij, surface, imgdict, cfg.render_picname_afterOpt)
    path = "./result_final/zju_67_800_gamma1.0/"
    render(Pij, surface, imgdict, os.path.join(path, "reRender5120.png"),colored_picPath=os.path.join(path,"reRender5120_colored.png"))    

    # jax.config.update("jax_enable_x64", True)
    # uf.find_idle_gpu()

    # result=uf.readFromDict(cfg.folder_name+"temp_result.npy")["result"][-2:]
    # print(jnp.transpose(jnp.array([result[0],result[1]])))
