import cost_func
import BSurface
import time
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import grad, jit, vmap,lax
from jax.lax import batch_vmap
import config as cfg
import density_func as df
import jax.config
import utils_func as uf
import os
import image_process as imgp
import sys

@jit
def loss_func_for_one_pos(i,j,bool,gNui3,gNvi3,Pij,gdNui3,gdNvi3,gddNui3,gddNvi3,ni,no,a,tz,img_dict):
    # loss function for one position
    weight=1.0
    x,y,curIndex,z,zx,zy,zxx,zxy,zyy,b,Ox,Oy,Oz,tx,ty =cost_func.args_calculation(i,j,gNui3,gNvi3,Pij,gdNui3,gdNvi3,gddNui3,gddNvi3,ni,no)
    #print("(j,i)=,",j,i,"z=",z,"tx,ty=",tx,ty)
    res=lax.cond(bool,lambda x:cost_func.inner_cost_func(no,ni,a,b,z,zx,zy,tx,ty,tz,zxx,zyy,zxy,img_dict),lambda x:cost_func.boundary_cost_func(tx,ty)*weight,0)

    return res
@jit
def loss_func(dict,Pij,indices,img_dict,using_varyweight_flag,flag_printout=False)->jnp.ndarray:
    # loss function
    #start_time=time.time()
    #res=jnp.zeros(cfg.totalNum)
    gNui3=dict["gNui3"]
    gNvi3=dict["gNvi3"]
    gdNui3=dict["gdNui3"]
    gdNvi3=dict["gdNvi3"]
    gddNui3=dict["gddNui3"]
    gddNvi3=dict["gddNvi3"]
    no=cfg.no;ni=cfg.ni;a=cfg.a;tz=cfg.tz
    #end_time=time.time()
    #print('dict_find time cost',end_time-start_time,'s')
    res=batch_vmap(loss_func_for_one_pos,batch_size=cfg.sample_chunk_size,in_axes=[0,0,0,None,None,None,None,None,None,None,None,None,None,None,None])(indices[:,1],indices[:,0],indices[:,2],gNui3,gNvi3,Pij,gdNui3,gdNvi3,gddNui3,gddNvi3,ni,no,a,tz,img_dict)
    '''
    res=vmap(loss_func_for_one_pos,in_axes=[0,0,0,None,None,None,None,None,None,None,None,None,None,None,None])(indices[:,1],indices[:,0],indices[:,2],gNui3,gNvi3,Pij,gdNui3,gdNvi3,gddNui3,gddNvi3,ni,no,a,tz,img_dict)
       
    chunk_size=cfg.sample_chunk_size
    res=jnp.array([])
    for i in range(0, cfg.totalSampleNum, chunk_size):
        temp_array=vmap(loss_func_for_one_pos,in_axes=[0,0,0,None,None,None,None,None,None,None,None,None,None,None,None])(indices[i:i+chunk_size,1],indices[i:i+chunk_size,0],indices[i:i+chunk_size,2],gNui3,gNvi3,Pij,gdNui3,gdNvi3,gddNui3,gddNvi3,ni,no,a,tz,img_dict)
        res=jnp.concatenate([res,temp_array])
    #res=jnp.concatenate([vmap(loss_func_for_one_pos,in_axes=[0,0,0,None,None,None,None,None,None,None,None,None,None,None,None])(indices[i:i+chunk_size,1],indices[i:i+chunk_size,0],indices[i:i+chunk_size,2],gNui3,gNvi3,Pij,gdNui3,gdNvi3,gddNui3,gddNvi3,ni,no,a,tz,img_dict) for i in range(0, cfg.totalSampleNum, chunk_size)])
    '''
    #jax.debug.print("res:{}",res)
    
    # let inner loss and edge loss has the same maximum
    condition=indices[:,2]
    zero_array=jnp.zeros((1,(cfg.M_sample+1)*(cfg.N_sample+1)))
    true_result=jnp.where(condition,res,zero_array) # true means inner
    false_result=jnp.where(condition,zero_array,res) # false means on boundary
    inner_max=jnp.max(jnp.abs(true_result));boundary_max=jnp.max(jnp.abs(false_result))
    inner_norm=jnp.linalg.norm(true_result);boundary_norm=jnp.linalg.norm(false_result)
    varyweight=lax.cond(boundary_norm<1e-12,lambda x:0.0,lambda x:10**jnp.floor((jnp.log10(inner_norm/boundary_norm))),0) # in case that boundary_norm=0
    #varyweight=lax.cond(boundary_max<1e-12,lambda x:0.0,lambda x:inner_max/boundary_max,0) # in case that boundary_norm=0
    #termsweight=jnp.sqrt(cfg.M_sample+1)/2
    termsweight=1.0
    weight=lax.cond(using_varyweight_flag,lambda x:varyweight,lambda x:termsweight,0) # only use varyweight when flag_using_varyweight=True, else consider terms
    lax.cond(flag_printout,lambda x:jax.debug.print("inner_result={}, boundary_result={}, weight={}, varyweight={}",inner_norm,boundary_norm,weight,varyweight),lambda x:None,0)
    res_final=true_result+false_result*weight
    #jax.debug.print("res_final:{}",res_final)
    return res_final[0] # since res=[a,b,...] --> res_final=[[a',b',...]]
    return res

def make_indices():
    # don't change with the same cfg.M_sample and cfg.N_sample
    start_time=time.time()
    Ms=cfg.M_sample;Ns=cfg.N_sample
    indices = jnp.array([(j,i) for j in range(Ns+1) for i in range(Ms+1)])    
    bool_mask = (1 <= indices[:, 1]) & (indices[:, 1] <= Ms-1) & (1 <= indices[:, 0]) & (indices[:, 0] <= Ns-1)
    # Expand the original array with the boolean mask
    indices = jnp.concatenate([indices, bool_mask[:, jnp.newaxis]], axis=1)
    # now indices looks like (j,i,bool)
    end_time=time.time()
    print('indices establish time cost',end_time-start_time,'s')
    return indices

@jit
def Jacobi_for_one_pos(i,j,dict,Pij,indices,avg_len,img_dict,using_varyweight_flag):
    P_temp1=Pij.at[j,i].add(avg_len)
    P_temp2=Pij.at[j,i].add(-avg_len)
    res_temp=(loss_func(dict,P_temp1,indices,img_dict,using_varyweight_flag)-loss_func(dict,P_temp2,indices,img_dict,using_varyweight_flag))/(2*avg_len)
    return res_temp

@jit
def calculate_Jacobi(dict,Pij,indices,img_dict,using_varyweight_flag):
    #res_mat=jnp.empty((0,cfg.totalNum))
    # calculate Jacobi matrix
    epsilon=10e-6    
    # see norm/sqrt(size) as the average length of each element of Pij
    avg_len=epsilon*jnp.linalg.norm(Pij)/jnp.sqrt(cfg.totalBasisNum)   
    
    indices_for_Pij=jnp.array([(j,i) for j in range(cfg.N+3) for i in range(cfg.M+3)])
    res=batch_vmap(Jacobi_for_one_pos,batch_size=cfg.variable_chunk_size,in_axes=[0,0,None,None,None,None,None,None])(indices_for_Pij[:,1],indices_for_Pij[:,0],dict,Pij,indices,avg_len,img_dict,using_varyweight_flag)
    '''
    indices_for_Pij=jnp.array([(j,i) for j in range(cfg.N+3) for i in range(cfg.M+3)])
    chunk_size=cfg.variable_chunk_size
    
    #res=jnp.empty((0,cfg.totalSampleNum)) # res_init is a 0*totalSampleNum 2-D array, which would be concatenated with temp_array, which is totalBasisNum*totalSampleNum, in axis=0
    #for i in range(0, cfg.totalBasisNum, chunk_size):
        #temp_array=vmap(Jacobi_for_one_pos,in_axes=[0,0,None,None,None,None,None,None])(indices_for_Pij[i:i+chunk_size,1],indices_for_Pij[i:i+chunk_size,0],dict,Pij,indices,avg_len,img_dict,using_varyweight_flag)
        #res=jnp.concatenate([res,temp_array])
    #res=jnp.concatenate([vmap(Jacobi_for_one_pos,in_axes=[0,0,None,None,None,None,None,None])(indices_for_Pij[i:i+chunk_size,1],indices_for_Pij[i:i+chunk_size,0],dict,Pij,indices,avg_len,img_dict,using_varyweight_flag) for i in range(0, cfg.totalBasisNum, chunk_size)])
    #1111
    '''
    res=jnp.transpose(res)
    return res 

### funcs for train()
@jit 
def solve_delta_p(A,mu,g):
    return jnp.linalg.solve(A+mu*jnp.eye(cfg.totalBasisNum),g)
@jit 
def update_rho_p_f(Pij,delta_p,mu,g,surface_dict,indices,epsilon_p,img_dict,using_varyweight_flag):
    p_new=Pij+delta_p.reshape((cfg.N+3,cfg.M+3))                
    f_new=loss_func(surface_dict,p_new,indices,img_dict,using_varyweight_flag,True)
    rho=(jnp.dot(epsilon_p,epsilon_p)-jnp.dot(f_new,f_new))/(jnp.dot(delta_p,(mu*delta_p+g)))
    return rho,p_new,f_new
@jit 
def condition1(delta_p,epsilon_deltap_norm,Pij):
    return jnp.linalg.norm(delta_p)<=epsilon_deltap_norm*(jnp.linalg.norm(Pij)+epsilon_deltap_norm)
@jit 
def if_positive_rho(epsilon_p,f_new,rho,mu,p_new,surface_dict,indices,epsilon_g_norm,epsilon_relative,img_dict,using_varyweight_flag):
    stop=(jnp.linalg.norm(epsilon_p)-jnp.linalg.norm(f_new))<epsilon_relative*(jnp.linalg.norm(epsilon_p))
    p=p_new
    J_new=calculate_Jacobi(surface_dict,p,indices,img_dict,using_varyweight_flag)
    A_new=jnp.dot(jnp.transpose(J_new),J_new)
    epsilon_p_new=-loss_func(surface_dict,p,indices,img_dict,using_varyweight_flag)
    g_new=jnp.dot(jnp.transpose(J_new),epsilon_p_new)
    stop2=(jnp.max(g_new)<=epsilon_g_norm)
    mu=mu*jnp.maximum(1/3,1-pow(2*rho-1,3))
    v=2
    stop = (stop | stop2)
    return stop,mu,p,J_new,A_new,epsilon_p_new,g_new,v
    
def train_using_LM():
    # train using Levenberg-Marquardt algorithm, pseudocode from https://www.researchgate.net/figure/Pseudocode-for-the-Levenberg-Marquardt-nonlinear-least-squares-algorithm-see-text-for_fig2_220492985
    start_time=time.time()
    os.makedirs(cfg.folder_name, exist_ok=True)
    logfile=open(cfg.log_filename,'w')
    if not logfile:
        print("无法打开log文件。")
        return
    Pij=cfg.init_h*np.ones((cfg.N+3,cfg.M+3))
    s=BSurface.BSurface(cfg.M,cfg.N)
    img=imgp.Image(cfg.target_img_path)
    img_dict=img.queryDict()
    #uf.writeToObj(s,Pij,cfg.init_objname)
    surface_dict=s.queryDict()
    indices=make_indices()
    min_loss=1e16
    last_loss=0
    #using_varyweight_flag=False
    using_varyweight_flag=True
    has_changed_method_flag=False
    end_varyweight_flag=False # when iter end, using_varyweight_flag=False
    method_loss=-1
    # initialize parameters
    k=0;v=2;J=calculate_Jacobi(surface_dict,Pij,indices,img_dict,using_varyweight_flag);epsilon_p=-loss_func(surface_dict,Pij,indices,img_dict,using_varyweight_flag,True)
    epsilon_g_norm=1e-8;epsilon_deltap_norm=1e-12;target_loss=1e-8
    epsilon_relative=1e-8
    max_iter=2000
    A=jnp.dot(jnp.transpose(J),J);g=jnp.dot(jnp.transpose(J),epsilon_p);tao=1e-6
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
                rho,p_new,f_new=update_rho_p_f(Pij,delta_p,mu,g,surface_dict,indices,epsilon_p,img_dict,using_varyweight_flag)
                if(rho>0):
                    stop,mu,Pij,J,A,epsilon_p,g,v=if_positive_rho(epsilon_p,f_new,rho,mu,p_new,surface_dict,indices,epsilon_g_norm,epsilon_relative,img_dict,using_varyweight_flag)
                else:
                    mu=mu*v
                    v=2*v
                first_time_flag=False    
            if(rho>0 or stop):
                break
        loss=jnp.linalg.norm(epsilon_p)
        stop=(loss<=target_loss)
        
        #print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s","mu,v,rho:",mu,v,rho)
        
        if(k<=100):
            print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s")
            logfile.write(f"iter:{k} loss:{loss} time cost:{time.time()-start_time}s \n")
           # print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s")
        elif(k%200==0):
            print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s")
            logfile.write(f"iter:{k} loss:{loss} time cost:{time.time()-start_time}s\n")
            
            
        if(k>1 and (abs(loss-last_loss)<1e-16 or stop or k==max_iter-1)):
            str=""
            if(abs(loss-last_loss)<1e-16):
                str="relative error reached."
            elif(stop):
                str="stop cond reached."
            print("converged because of "+str)
            print("end at iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s")
            logfile.write(f"iter:{k} loss:{loss} time cost:{time.time()-start_time}s\n")
            #break
            #let's do unvaryweight once, from line193-line204
            if(using_varyweight_flag==True):
                print("\nNow using_varyweight_flag=False\n")
                logfile.write(f"Now using_varyweight_flag=False\n")
                using_varyweight_flag=False
                #update parameters
                v=2;J=calculate_Jacobi(surface_dict,Pij,indices,img_dict,using_varyweight_flag);epsilon_p=-loss_func(surface_dict,Pij,indices,img_dict,using_varyweight_flag,True)
                A=jnp.dot(jnp.transpose(J),J);g=jnp.dot(jnp.transpose(J),epsilon_p)
                mu=tao*jnp.max(jnp.diag(A))
                stop=False
                max_iter*=2
                continue
            else:
                break
        last_loss=loss
        '''
        if(k>1 and (abs(loss-last_loss)<1e-16 or stop)):
            
            if(using_varyweight_flag==end_varyweight_flag):
                if(abs(loss-method_loss)/loss<1e-16):
                    print("converged.")
                    break
                else:
                    method_loss=loss
                    print("\nNow method_loss=",method_loss,"\n")
                    logfile.write(f"\nNow method_loss={method_loss}\n")
                
            if(not has_changed_method_flag):
                using_varyweight_flag=not using_varyweight_flag
                print("Now using_varyweight_flag=",using_varyweight_flag)
                logfile.write(f"Now using_varyweight_flag={using_varyweight_flag}\n")
                last_loss = loss
                # update parameters
                v=2;J=calculate_Jacobi(surface_dict,Pij,indices,img_dict,using_varyweight_flag);epsilon_p=-loss_func(surface_dict,Pij,indices,img_dict,using_varyweight_flag,True)
                A=jnp.dot(jnp.transpose(J),J);g=jnp.dot(jnp.transpose(J),epsilon_p)
                mu=tao*jnp.max(jnp.diag(A))
                
                has_changed_method_flag=True
                stop=False
                
            else:
                # 若连续两次loss变化小于1e-16，或者连续两次stop，说明已经收敛
                print("converged.")
                break
        
        else:
            has_changed_method_flag=False
            #using_varyweight_flag=False
            last_loss=loss       
            
    print("iter:",k,"loss:",loss,"time cost:",time.time()-start_time,"s\n")           
    logfile.write(f"iter:{k} loss:{loss} time cost:{time.time()-start_time}s relative loss:{abs(loss-last_loss)/loss} \n")''' 
    logfile.close()
    return Pij,s
          

import render
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')
    #XLA_PYTHON_CLIENT_PREALLOCATE=False
    # 查找空闲的GPU
    uf.find_idle_gpu()
                
    #train()
    Pij,surface=train_using_LM()
    uf.saveToDict({"Pij":Pij,"M":cfg.M,"N":cfg.N}) # surface only depends on M and N : s=BSurface.BSurface(cfg.M,cfg.N)
    
    print(render.render(Pij,surface))
    
    #uf.writeToObj(surface,Pij,cfg.objname)
    
    