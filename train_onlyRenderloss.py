import LM_algo_Alter as lma
import train6
import LM_algo_for_renderLoss as lmr
import render as rd
import cost_func
import BSurface
import time
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import grad, jit, vmap, lax
from jax.lax import batch_vmap
import config as cfg
import density_func as df
import jax.config
import utils_func as uf
import os
import image_process as imgp
import sys

@jit
def loss_func(dict, Pij, indices, img_dict, using_varyweight_flag, flag_printout=False) -> jnp.ndarray:
    '''
    Here we introduce render loss.
    '''
    render_loss = rd.render_loss_with_inputIndices(
        Pij, indices, dict, img_dict["normalized_intensity"])  # vector
    return render_loss
    return res_final[0]  # since res=[a,b,...] --> res_final=[[a',b',...]]
    return res

def make_indices():
    indices = jnp.array([(j, i) for j in range(cfg.rn+1) for i in range(cfg.rm+1)])
    return indices

@jit
def Jacobi_for_one_pos(i, j, dict, Pij, indices, avg_len, img_dict, using_varyweight_flag):
    P_temp1 = Pij.at[j, i].add(avg_len)
    P_temp2 = Pij.at[j, i].add(-avg_len)
    res_temp = (loss_func(dict, P_temp1, indices, img_dict, using_varyweight_flag) -
                loss_func(dict, P_temp2, indices, img_dict, using_varyweight_flag))/(2*avg_len)
    return res_temp


@jit
def calculate_Jacobi(dict, Pij, indices, img_dict, using_varyweight_flag):
    # res_mat=jnp.empty((0,cfg.totalNum))
    # calculate Jacobi matrix
    epsilon = 10e-6
    # see norm/sqrt(size) as the average length of each element of Pij
    avg_len = epsilon*jnp.linalg.norm(Pij)/jnp.sqrt(cfg.totalBasisNum)

    indices_for_Pij = jnp.array(
        [(j, i) for j in range(cfg.N+3) for i in range(cfg.M+3)])
    res = jit(batch_vmap(Jacobi_for_one_pos, in_axes=[0, 0, None, None, None, None, None, None], batch_size=cfg.variable_chunk_size))(
        indices_for_Pij[:, 1], indices_for_Pij[:, 0], dict, Pij, indices, avg_len, img_dict, using_varyweight_flag)

    res = jnp.transpose(res)
    return res

# funcs for train()


@jit
def solve_delta_p(A, mu, g):
    return jnp.linalg.solve(A+mu*jnp.eye(cfg.totalBasisNum), g)


@jit
def update_rho_p_f(Pij, delta_p, mu, g, surface_dict, indices, epsilon_p, img_dict, using_varyweight_flag):
    p_new = Pij+delta_p.reshape((cfg.N+3, cfg.M+3))
    f_new = loss_func(surface_dict, p_new, indices,
                      img_dict, using_varyweight_flag, True)
    rho = (jnp.dot(epsilon_p, epsilon_p)-jnp.dot(f_new, f_new)) / \
        (jnp.dot(delta_p, (mu*delta_p+g)))
    return rho, p_new, f_new


@jit
def condition1(delta_p, epsilon_deltap_norm, Pij):
    return jnp.linalg.norm(delta_p) <= epsilon_deltap_norm*(jnp.linalg.norm(Pij)+epsilon_deltap_norm)


def if_positive_rho(epsilon_p, f_new, rho, mu, p_new, surface_dict, indices, epsilon_g_norm, epsilon_relative, img_dict, using_varyweight_flag):
    stop = (jnp.linalg.norm(epsilon_p)-jnp.linalg.norm(f_new)
            ) < epsilon_relative*(jnp.linalg.norm(epsilon_p))
    p = p_new
    A_new, g_new = calculate_A_and_g(
        surface_dict, p, indices, img_dict, using_varyweight_flag)
    # J_new = calculate_Jacobi(surface_dict, p, indices,img_dict, using_varyweight_flag)
    # A_new = jnp.dot(jnp.transpose(J_new), J_new)
    epsilon_p_new = -loss_func(surface_dict, p, indices,
                               img_dict, using_varyweight_flag)
    # g_new = jnp.dot(jnp.transpose(J_new), epsilon_p_new)
    stop2 = (jnp.max(g_new) <= epsilon_g_norm)
    mu = mu*jnp.maximum(1/3, 1-pow(2*rho-1, 3))
    v = 2
    stop = (stop | stop2)
    return stop, mu, p, A_new, epsilon_p_new, g_new, v


def calculate_A_and_g(surface_dict, Pij, indices, img_dict, using_varyweight_flag):
    '''
    Compute the Jacobian matrix piecewise.
    '''
    epsilon = 10e-6
    # see norm/sqrt(size) as the average length of each element of Pij
    avg_len = epsilon*jnp.linalg.norm(Pij)/jnp.sqrt(cfg.totalBasisNum)

    """
    input parameter [indices] controls how much equations we 
    process at one time.
    """
    MaxEqutionNum = int(cfg.maxEqNum/1)
    forNum = int(cfg.totalSampleNum/MaxEqutionNum)+1
    A = jnp.zeros((cfg.totalBasisNum, cfg.totalBasisNum))
    g = jnp.zeros(cfg.totalBasisNum)
    for i in range(forNum):
        indices_need = indices[i *
                               MaxEqutionNum:min((i+1)*MaxEqutionNum, cfg.totalSampleNum), :]
        # print(indices_need.shape)
        epsilon_p = -loss_func(surface_dict, Pij, indices_need,
                               img_dict, using_varyweight_flag)
        J = calculate_Jacobi(surface_dict, Pij, indices_need,
                             img_dict, using_varyweight_flag)
        A = A+jnp.dot(jnp.transpose(J), J)
        g = g+jnp.dot(jnp.transpose(J), epsilon_p)
        # print(A.shape,g.shape)
    return A, g


def train_using_LM(Pij,s,img_dict):
    # train using Levenberg-Marquardt algorithm, pseudocode from https://www.researchgate.net/figure/Pseudocode-for-the-Levenberg-Marquardt-nonlinear-least-squares-algorithm-see-text-for_fig2_220492985
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
    surface_dict = s.query_all_dict()
    indices = make_indices()
    min_loss = 1e16
    last_loss = 0
    # using_varyweight_flag=False
    using_varyweight_flag = True
    has_changed_method_flag = False
    end_varyweight_flag = False  # when iter end, using_varyweight_flag=False
    method_loss = -1
    # initialize parameters
    k = 0
    v = 2
    # J = calculate_Jacobi(surface_dict, Pij, indices,
    #                     img_dict, using_varyweight_flag)
    epsilon_p = -loss_func(surface_dict, Pij, indices,
                           img_dict, using_varyweight_flag, True)
    epsilon_g_norm = 1e-8
    epsilon_deltap_norm = 1e-12
    target_loss = 1e-8
    epsilon_relative = 1e-8
    max_iter = 2000
    # A = jnp.dot(jnp.transpose(J), J)
    # g = jnp.dot(jnp.transpose(J), epsilon_p)
    A, g = calculate_A_and_g(surface_dict, Pij, indices,
                             img_dict, using_varyweight_flag)
    tao = 1e-6
    mu = tao*jnp.max(jnp.diag(A))
    stop = (jnp.max(g) <= epsilon_g_norm)

    while ((not stop) and (k < max_iter)):
        k = k+1
        first_time_flag = True
        while (True):
            delta_p = solve_delta_p(A, mu, g)
            if (condition1(delta_p, epsilon_deltap_norm, Pij) & (~first_time_flag)):
                stop = True
                print("meet stop condition")
                break
            else:
                rho, p_new, f_new = update_rho_p_f(
                    Pij, delta_p, mu, g, surface_dict, indices, epsilon_p, img_dict, using_varyweight_flag)
                if (rho > 0):
                    stop, mu, Pij, A, epsilon_p, g, v = if_positive_rho(
                        epsilon_p, f_new, rho, mu, p_new, surface_dict, indices, epsilon_g_norm, epsilon_relative, img_dict, using_varyweight_flag)
                else:
                    mu = mu*v
                    v = 2*v
                first_time_flag = False
            if (rho > 0 or stop):
                break
        loss = jnp.linalg.norm(epsilon_p)
        stop = (loss <= target_loss)

        # print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s","mu,v,rho:",mu,v,rho)

        if (k <= 100):
            print("iter:", k, "loss:", loss, "time cost:",
                  time.time()-start_time, "s")
            logfile.write(
                f"iter:{k} loss:{loss} time cost:{time.time()-start_time}s \n")
           # print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s")
        elif (k % 200 == 0):
            print("iter:", k, "loss:", loss, "time cost:",
                  time.time()-start_time, "s")
            logfile.write(
                f"iter:{k} loss:{loss} time cost:{time.time()-start_time}s\n")

        if (k == 5):
            rd.render(Pij, s, img_dict, cfg.createPicName("5_onlyRenderloss"))
        elif (k == 30):
            rd.render(Pij, s, img_dict, cfg.createPicName("30_onlyRenderloss"))

        if (k > 1 and (abs(loss-last_loss) < 1e-16 or stop or k == max_iter-1)):
            str = ""
            if (abs(loss-last_loss) < 1e-16):
                str = "relative error reached."
            elif (stop):
                str = "stop cond reached."
            print("converged because of "+str)
            print("end at iter:", k, "loss:", loss,
                  "time cost:", time.time()-start_time, "s")
            logfile.write(
                f"iter:{k} loss:{loss} time cost:{time.time()-start_time}s\n")
            break
            
        last_loss = loss
    logfile.close()
    return Pij, s, img_dict

import train_initRect as tri
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
    
    
    print("=============================================\n Begin initialize as rectangle")
    Pij, surface, img_dict = tri.train_using_LM(Pij, surface, img_dict)
    rd.render(Pij, surface, img_dict, cfg.createPicName("train_Init_forOnlyRenderloss"))
    uf.saveToDict({"Pij": Pij, "M": cfg.M, "N": cfg.N},cfg.createDictName("train_Init_forOnlyRenderloss"))
    
    print("=============================================\n Directly begin renderLoss Optimization")
    # train()
    Pij, surface, img_dict = train_using_LM(Pij,surface,img_dict)
    rd.render(Pij, surface, img_dict, cfg.createPicName("train_onlyRenderloss"))

    # surface only depends on M and N : s=BSurface.BSurface(cfg.M,cfg.N)
    uf.saveToDict({"Pij": Pij, "M": cfg.M, "N": cfg.N}, cfg.createDictName("train_onlyRenderloss"))
    uf.writeToObj(surface, Pij, cfg.objname)