import utils_func as uf
from jax import vmap
import jax.numpy as jnp
import cost_func as cf
import config as cfg
import jax.config
import BSurface
from jax.lax import batch_vmap

def render(Pij,surface:BSurface,picname=cfg.render_picname):
    surface.calculateAllNsOnGrid_forRender()
    dict = surface.query_dict_for_render()

    indices=jnp.array([(j,i) for j in range(cfg.rn+1) for i in range(cfg.rm+1)])
    result = batch_vmap(cf.args_calculation,batch_size=10000,in_axes=[0,0,None,None,None,None,None,None,None,None,None,None])(indices[:,1], indices[:,0], dict["gNui3_for_render"], dict["gNvi3_for_render"], Pij, dict["gdNui3_for_render"], dict["gdNvi3_for_render"], dict["gddNui3_for_render"], dict["gddNvi3_for_render"], cfg.ni, cfg.no, cfg.rx)
    print(result)
    #uf.saveToDict({"result":result},cfg.folder_name+"temp_result.npy")
    result =result[-2:]
    tx_ty_list=jnp.transpose(jnp.array([result[0],result[1]]))
    print(tx_ty_list)
    


def test_func(i,j):
    return i+1,i+2,j+1,j+2


import numpy as np
if __name__ == "__main__":
    dict = uf.readFromDict("result_test/blbl_2_5_dict.npy")
    print(dict)
    
    #jax.config.update("jax_enable_x64", True)
    #uf.find_idle_gpu()
    
    #result=uf.readFromDict(cfg.folder_name+"temp_result.npy")["result"][-2:]
    #print(jnp.transpose(jnp.array([result[0],result[1]])))
    
    
                
    