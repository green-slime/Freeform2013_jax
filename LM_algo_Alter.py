from jax import jit
from jax.lax import batch_vmap
import jax.numpy as jnp
import render as rd
import config as cfg
import time,os
from utils_func import showIterInfo, judgeAndShowEndInfo

@jit
def render_loss(Pij,renderMeshdict,target_intensity):
    return rd.render_loss_Alter(Pij,renderMeshdict,target_intensity)
@jit
def Jacobi_for_one_pos(i,j,Pij,avg_len,renderMeshdict,target_intensity):
    P_temp1=Pij.at[j,i].add(avg_len)
    P_temp2=Pij.at[j,i].add(-avg_len)
    #rd.render_loss(Pij,renderMeshdict,target_intensity)
    res_temp=(render_loss(P_temp1,renderMeshdict,target_intensity)-render_loss(P_temp2,renderMeshdict,target_intensity))/(2*avg_len)
    return res_temp

@jit
def calculate_Jacobi(Pij,renderMeshdict,target_intensity):
    #res_mat=jnp.empty((0,cfg.totalNum))
    # calculate Jacobi matrix
    epsilon=10e-6    
    # see norm/sqrt(size) as the average length of each element of Pij
    avg_len=epsilon*jnp.linalg.norm(Pij)/jnp.sqrt(cfg.totalBasisNum)   
    
    indices_for_Pij=jnp.array([(j,i) for j in range(cfg.N+3) for i in range(cfg.M+3)])
    res=jit(batch_vmap(Jacobi_for_one_pos,in_axes=[0,0,None,None,None,None],batch_size=cfg.variable_chunk_size))(indices_for_Pij[:,1],indices_for_Pij[:,0],Pij,avg_len,renderMeshdict,target_intensity)

    res=jnp.transpose(res)
    #res=jnp.array([res]) # shape (totalBasisNum,) to (1,totalBasisNum)
    return res 
@jit 
def solve_delta_p(A,mu,g):
    return jnp.linalg.solve(A+mu*jnp.eye(cfg.totalBasisNum),g)
@jit 
def update_rho_p_f(Pij,delta_p,mu,g,epsilon_p,renderMeshdict,target_intensity):
    p_new=Pij+delta_p.reshape((cfg.N+3,cfg.M+3))  
    f_new=render_loss(p_new,renderMeshdict,target_intensity)              
    #f_new=loss_func(surface_dict,p_new,indices,img_dict,using_varyweight_flag,True)
    #print(epsilon_p.shape,f_new.shape,delta_p.shape,g.shape)
    rho=(jnp.dot(jnp.transpose(epsilon_p),epsilon_p)-jnp.dot(f_new,f_new))/(jnp.dot(jnp.transpose(delta_p),(mu*delta_p+g)))
    #jax.debug.print("rho={}",rho)
    return rho,p_new,f_new
@jit 
def condition1(delta_p,epsilon_deltap_norm,Pij):
    jax.debug.print("delta_p.norm:{},RHS={}",jnp.linalg.norm(delta_p),epsilon_deltap_norm*(jnp.linalg.norm(Pij)+epsilon_deltap_norm))
    return jnp.linalg.norm(delta_p)<=epsilon_deltap_norm*(jnp.linalg.norm(Pij)+epsilon_deltap_norm)
@jit 
def if_positive_rho(epsilon_p,f_new,rho,mu,p_new,epsilon_g_norm,epsilon_relative,Pij,renderMeshdict,target_intensity):
    stop=(jnp.linalg.norm(epsilon_p)-jnp.linalg.norm(f_new))<epsilon_relative*(jnp.linalg.norm(epsilon_p))
    p=p_new
    J_new=calculate_Jacobi(Pij,renderMeshdict,target_intensity)
    A_new=jnp.dot(jnp.transpose(J_new),J_new)
    print(A_new.shape)
    epsilon_p_new=-render_loss(Pij,renderMeshdict,target_intensity)
    #epsilon_p_new=-loss_func(surface_dict,p,indices,img_dict,using_varyweight_flag)
    g_new=jnp.dot(jnp.transpose(J_new),epsilon_p_new)
    stop2=(jnp.max(g_new)<=epsilon_g_norm)
    mu=mu*jnp.maximum(1/3,1-pow(2*rho-1,3))
    v=2
    stop = (stop | stop2)
    return stop,mu,p,J_new,A_new,epsilon_p_new,g_new,v

def solve_using_LM(Pij,surface,imgdict):
    target_intensity=imgdict["normalized_intensity"]
    surface.calculateAllNsOnGrid_forRender()
    renderMeshdict= surface.query_dict_for_render()
    
    start_time=time.time()
    os.makedirs(cfg.folder_name, exist_ok=True)
    os.makedirs(cfg.prefix_name, exist_ok=True)
    logfile=open(cfg.log_renderOptFilename,'w')
    if not logfile:
        print("无法打开log文件。")
        return
    
    k=0;v=2;J=calculate_Jacobi(Pij,renderMeshdict,target_intensity);epsilon_p=-render_loss(Pij,renderMeshdict,target_intensity)
    epsilon_g_norm=1e-8;epsilon_deltap_norm=1e-12;target_loss=1e-8
    epsilon_relative=1e-8
    #print(J)
    max_iter=2000
    A=jnp.dot(jnp.transpose(J),J);g=jnp.dot(jnp.transpose(J),epsilon_p);tao=1e-6
    #print(A.shape)
    mu=tao*jnp.max(jnp.diag(A))
    
    stop=(jnp.max(g)<=epsilon_g_norm)
       
    while((not stop)and(k<max_iter)):
        k=k+1
        first_time_flag=True
        while(True):           
            delta_p=solve_delta_p(A,mu,g)
            if(condition1(delta_p,epsilon_deltap_norm,Pij)&(~first_time_flag)):
                stop=True
                print("meet stop condition")
                break
            else:               
                rho,p_new,f_new=update_rho_p_f(Pij,delta_p,mu,g,epsilon_p,renderMeshdict,target_intensity)
                if(rho>0):
                    print("Now rho>0.",rho)
                    stop,mu,Pij,J,A,epsilon_p,g,v=if_positive_rho(epsilon_p,f_new,rho,mu,p_new,epsilon_g_norm,epsilon_relative,Pij,renderMeshdict,target_intensity)
                else:
                    print("Now rho<0.",rho)
                    mu=mu*v
                    v=2*v
                first_time_flag=False    
            if(rho>0 or stop):
                break
        loss=jnp.linalg.norm(epsilon_p)
        stop=(loss<=target_loss)
        
        #print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s","mu,v,rho:",mu,v,rho)
        
        showIterInfo(k,loss,logfile,start_time)
                        
        if(k>1 and (abs(loss-last_loss)<1e-16 or stop or k==max_iter-1)):
            judgeAndShowEndInfo(k,loss,last_loss,stop,logfile,start_time)
            #break
            break
        last_loss=loss
    logfile.close()
    
    #final_intensity=rd.render_loss(Pij,renderMeshdict,target_intensity,returnType=2)
    
    return Pij

import jax
import utils_func as uf
import BSurface
import image_process as imgp
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')
    uf.find_idle_gpu()
    dict = uf.readFromDict(cfg.dict_name)
    Pij=dict["Pij"]   
    M=dict["M"];N=dict["N"]
    print(M,N)
    assert(M==cfg.M and N==cfg.N)
    #Pij=jnp.ones((cfg.M+3,cfg.N+3))
    surface=BSurface.BSurface(M,N)
    
    img=imgp.Image(cfg.target_img_path)
    imgdict=img.queryDict()
    Pij=solve_using_LM(Pij,surface,imgdict)
    #rd.renderIntensityToImg(img_dict,final_intensity,cfg.render_picname_afterOpt)
    rd.render(Pij,surface,imgdict,cfg.render_picname_afterOptAlter)
    #uf.writeToObj(surface,Pij,cfg.objname_afterOpt)
    
    uf.compareTwoImg(cfg.render_picname,cfg.render_picname_afterOptAlter)
