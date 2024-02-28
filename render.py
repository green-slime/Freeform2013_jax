import utils_func as uf
from jax import vmap,jit,lax
import jax.numpy as jnp
import cost_func as cf
import config as cfg
import jax.config
import BSurface
#from jax.lax import batch_vmap

@jit
def decideOnWhichGrid(tx,ty,grid_array: jnp.ndarray):
    ind_x=jnp.floor((tx-cfg.xmin)/cfg.dx).astype(jnp.int32)
    ind_y=jnp.floor((cfg.ymax-ty)/cfg.dy).astype(jnp.int32) # notice that (0,0) is the lefttop of the img
    ind_x=jnp.minimum(ind_x,cfg.rx-1)
    ind_y=jnp.minimum(ind_y,cfg.ry-1)
    # grid(ind_y,ind_x)
    #grid_array = grid_array.at[ind_y,ind_x].add(1)
    return jnp.array([ind_y,ind_x])

    

def render(Pij,surface:BSurface,picname=cfg.render_picname):
    surface.calculateAllNsOnGrid_forRender()
    dict = surface.query_dict_for_render()
    
    gNui3_for_render=dict["gNui3_for_render"];gNvi3_for_render=dict["gNvi3_for_render"];gdNui3_for_render=dict["gdNui3_for_render"];gdNvi3_for_render=dict["gdNvi3_for_render"];gddNui3_for_render=dict["gddNui3_for_render"];gddNvi3_for_render=dict["gddNvi3_for_render"]
    #print(gNui3_for_render)
    
    indices=jnp.array([(j,i) for j in range(cfg.rn+1) for i in range(cfg.rm+1)])
    #print(indices)
    result = vmap(cf.args_calculation,in_axes=[0,0,None,None,None,None,None,None,None,None,None,None,None])(indices[:,1], indices[:,0], gNui3_for_render,gNvi3_for_render, Pij, gdNui3_for_render,gdNvi3_for_render,gddNui3_for_render,gddNvi3_for_render, cfg.ni, cfg.no, cfg.rm,cfg.rn)
    #print("result:",result)
    #print("z:",result[3])
    #uf.saveToDict({"result":result},cfg.folder_name+"temp_result.npy")
    #result =result[-2:]
    tx_ty_list=jnp.transpose(jnp.array([result[0],result[1]]))
    #print(tx_ty_list) # [[tx1,ty1],[tx2,ty2],...]
    grid_array=jnp.zeros((cfg.ry,cfg.rx))
    add_index=vmap(decideOnWhichGrid,in_axes=[0,0,None])(tx_ty_list[:,0],tx_ty_list[:,1],grid_array)
    #print(need_to_add)
    for i in range(len(add_index)):
        grid_array=grid_array.at[add_index[i][0],add_index[i][1]].add(1)
    print(grid_array)


def test_func(i,j):
    return i+1,i+2,j+1,j+2


import numpy as np
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')
    uf.find_idle_gpu()
    dict = uf.readFromDict("result_test/blbl_2_5_dict.npy")
    #Pij=dict["Pij"]   
    M=dict["M"];N=dict["N"]
    print(M,N)
    assert(M==cfg.M and N==cfg.N)
    Pij=jnp.ones((cfg.M+3,cfg.N+3))
    surface=BSurface.BSurface(M,N)
    render(Pij,surface,"test_picname.png")
    
    #jax.config.update("jax_enable_x64", True)
    #uf.find_idle_gpu()
    
    #result=uf.readFromDict(cfg.folder_name+"temp_result.npy")["result"][-2:]
    #print(jnp.transpose(jnp.array([result[0],result[1]])))
    
    
                
    