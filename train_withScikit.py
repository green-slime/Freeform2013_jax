import numpy as np
from sko.GA import GA
import pandas as pd
import matplotlib.pyplot as plt
from train_onlyOT import loss_func
import config as cfg
import BSurface
import jax.numpy as jnp
import jax
from jax import jit, lax
from jax.lax import batch_vmap
import cost_func as cf
import time

@jit
def loss_func_for_one_pos(i, j, bool, gNui3, gNvi3, Pij, gdNui3, gdNvi3, gddNui3, gddNvi3, ni, no, a, tz, img_dict):
    # loss function for one position
    weight = 1.0
    x, y, curIndex, z, zx, zy, zxx, zxy, zyy, b, Ox, Oy, Oz, tx, ty = cf.args_calculation(
        i, j, gNui3, gNvi3, Pij, gdNui3, gdNvi3, gddNui3, gddNvi3, ni, no)
    # print("(j,i)=,",j,i,"z=",z,"tx,ty=",tx,ty)
    res = lax.cond(bool, lambda x: cf.inner_cost_func(no, ni, a, b, z, zx, zy, tx, ty, tz, zxx, zyy, zxy, img_dict), lambda x: cf.boundary_cost_func(tx, ty)*weight, 0)
    #res=cf.cost_func_forInit(i,j,tx,ty,cols=cfg.Init_sample,rows=cfg.Init_sample)
    #return jnp.log(jnp.abs(res)+1)
    return res

@jit
def loss_func(dict, Pij, indices, img_dict, using_varyweight_flag=True, flag_printout=False) -> jnp.ndarray:
    # loss function
    # start_time=time.time()
    # res=jnp.zeros(cfg.totalNum)
    Pij = jnp.reshape(Pij,(cfg.N+3,cfg.M+3))
    gNui3 = dict["gNui3"]
    gNvi3 = dict["gNvi3"]
    gdNui3 = dict["gdNui3"]
    gdNvi3 = dict["gdNvi3"]
    gddNui3 = dict["gddNui3"]
    gddNvi3 = dict["gddNvi3"]
    no = cfg.no
    ni = cfg.ni
    a = cfg.a
    tz = cfg.tz
    res = jit(batch_vmap(loss_func_for_one_pos, in_axes=[0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None], batch_size=cfg.sample_chunk_size))(
        indices[:, 1], indices[:, 0], indices[:, 2], gNui3, gNvi3, Pij, gdNui3, gdNvi3, gddNui3, gddNvi3, ni, no, a, tz, img_dict)

    # let inner loss and edge loss has the same maximum
    condition = indices[:, 2]
    zero_array = jnp.zeros((1, len(indices)))
    true_result = jnp.where(condition, res, zero_array)  # true means inner
    # false means on boundary
    false_result = jnp.where(condition, zero_array, res)
    #inner_max = jnp.max(jnp.abs(true_result))
    #boundary_max = jnp.max(jnp.abs(false_result))
    inner_norm = jnp.linalg.norm(true_result)
    boundary_norm = jnp.linalg.norm(false_result)
    varyweight = lax.cond(boundary_norm < 1e-12, lambda x: 0.0, lambda x: 10**jnp.floor(
        (jnp.log10(inner_norm/boundary_norm))), 0)  # in case that boundary_norm=0
    # varyweight=lax.cond(boundary_max<1e-12,lambda x:0.0,lambda x:inner_max/boundary_max,0) # in case that boundary_norm=0
    # termsweight=jnp.sqrt(cfg.M_sample+1)/2
    termsweight = 1.0
    # only use varyweight when flag_using_varyweight=True, else consider terms
    weight = lax.cond(using_varyweight_flag,
                      lambda x: varyweight, lambda x: termsweight, 0)
    #lax.cond(flag_printout, lambda x: jax.debug.print("inner_result={}, boundary_result={}, weight={}, varyweight={}",inner_norm, boundary_norm, weight, varyweight), lambda x: None, 0)
    res_final = true_result+false_result*weight
    # jax.debug.print("res_final:{}",res_final)
    #jax.debug.print("res_final:{}", res_final)
    return jnp.linalg.norm(res_final)
    return res_final[0]  # since res=[a,b,...] --> res_final=[[a',b',...]]


def make_indices():
    # don't change with the same cfg.M_sample and cfg.N_sample
    start_time = time.time()
    Ms = cfg.M_sample
    Ns = cfg.N_sample
    indices = jnp.array([(j, i) for j in range(Ns+1) for i in range(Ms+1)])
    bool_mask = (1 <= indices[:, 1]) & (
        indices[:, 1] <= Ms-1) & (1 <= indices[:, 0]) & (indices[:, 0] <= Ns-1)
    # Expand the original array with the boolean mask
    indices = jnp.concatenate([indices, bool_mask[:, jnp.newaxis]], axis=1)
    # now indices looks like (j,i,bool)
    end_time = time.time()
    print('indices establish time cost', end_time-start_time, 's')
    return indices

import os
from sko.tool_kit import x2gray
def train_with_scikit(Pij, surface, img_dict):
    start_time = time.time()
    os.makedirs(cfg.folder_name, exist_ok=True)
    os.makedirs(cfg.prefix_name, exist_ok=True)
    print("Now writing files to:", cfg.prefix_name)
    logfile = open(cfg.log_filename5, 'w')
    if not logfile:
        print("无法打开log文件。")
        return
    
    assert ((cfg.rx == img_dict["width"]) & (cfg.ry == img_dict["height"]))
    # uf.writeToObj(s,Pij,cfg.init_objname)
    surface_dict = surface.queryDict()
    indices = make_indices()
    
    Pij = np.reshape(Pij,(cfg.totalBasisNum,))
    
    def target_function(Pij):
        # assume that Pij.shape=(cfg.totalBasisNum,)
        return loss_func(surface_dict, Pij, indices, img_dict, flag_printout=False).astype(float)
               
    # arguments:
    n_dim=cfg.totalBasisNum
    tolerance=0.0001
    lb=cfg.init_h-np.ones(cfg.totalBasisNum,)*tolerance
    ub=cfg.init_h+1.0*np.ones(cfg.totalBasisNum,)*tolerance
    precision=1e-12;size_pop=100
    
    ga = GA(func=target_function, n_dim=n_dim, size_pop=size_pop, max_iter=1000,prob_mut=0.001, lb=lb,ub=ub,precision=precision)    
    
    '''
    x=np.array([0.5+0.000001/size_pop*np.arange(size_pop+2)[1:-1]]).T@np.ones((1,cfg.totalBasisNum)) # 0.5~0.5001
    res = x2gray(x,n_dim=n_dim,lb=lb, ub=ub,precision=precision).astype(int)

    ga.Chrom = res
    '''
    for i in range(100):
        best_x, best_y = ga.run(10)
        print('best_x:', best_x, '\n', 'best_y:', best_y)
        #print('best_y:', best_y)
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.show()
    
    return jnp.reshape(Pij,(cfg.N+3, cfg.M+3))
    

import train_initRect as tri
import utils_func as uf
import image_process as imgp
import render as rd
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')
    # XLA_PYTHON_CLIENT_PREALLOCATE=False
    # 查找空闲的GPU
    uf.find_idle_gpu()

    Pij = cfg.init_h*np.ones((cfg.N+3, cfg.M+3))
    surface = BSurface.BSurface(cfg.M, cfg.N)
    img = imgp.Image(cfg.target_img_path)
    img_dict = img.queryDict()
    
    Pij = train_with_scikit(Pij, surface, img_dict)
    rd.render(Pij, surface, img_dict, cfg.createPicName("train_withScikit"),colored_picPath=cfg.createPicName("train_train_withScikit_colored"))
    '''
    print("=============================================\n Begin initialize as rectangle")
    Pij, surface, img_dict = tri.train_using_LM(Pij, surface, img_dict)
    rd.render(Pij, surface, img_dict, cfg.createPicName("train_Init_forOnlyOT"),colored_picPath=cfg.createPicName("train_Init_forOnlyOT_colored"))
    uf.saveToDict({"Pij": Pij, "M": cfg.M, "N": cfg.N},cfg.createDictName("train_Init_forOnlyOT"))
    
    print("=============================================\n Only use OT Optimization")
    # train()
    Pij, surface, img_dict = train_using_LM(Pij,surface,img_dict)
    rd.render(Pij, surface, img_dict, cfg.createPicName("train_onlyOT_2nd"),colored_picPath=cfg.createPicName("train_onlyOT_2nd_colored"))

    # surface only depends on M and N : s=BSurface.BSurface(cfg.M,cfg.N)
    uf.saveToDict({"Pij": Pij, "M": cfg.M, "N": cfg.N},cfg.createDictName("train_onlyOT_2nd"))
    uf.writeToObj(surface, Pij, cfg.prefix_name+f"{cfg.name}{cfg.M}.obj")'''
    


