import jax.numpy as jnp
from jax import jit, lax
import config as cfg
import os
import image_process as imgp


@jit
def I(tx, ty):
    return 1.0


@jit
def E_test(tx, ty):
    # for ball

    r0 = 1
    '''
    E=4./(r0*r0*jnp.pi)
    r=boundary_test(tx,ty)
    return lax.cond(r<=0,lambda x:E,lambda x:0.,0.)
    '''
    weight = 32./((jnp.sqrt(2)-1)*r0)**2  # notice that r<=(sqrt(2)-1)*r0
    r = boundary_test(tx, ty)
    E = 4./((r0**2+1./weight)*jnp.pi)
    return lax.cond(r <= 0, lambda x: E, lambda x: E*jnp.exp(-weight*r*r), E)


@jit
def boundary_test(tx, ty):
    r0 = 1
    return jnp.sqrt((tx-0)**2+(ty-0)**2)-r0


@jit
def rect_boundary(tx, ty):
    w = cfg.half_width
    h = cfg.half_height
    condlist = jnp.array([(tx < -w) & (ty < -h), (tx < -w) & (ty >= -h) & (ty <= h), (tx < -w) & (ty > h), (tx >= -w) & (tx <= w) & (ty > h), (tx > w) & (
        ty > h), (tx > w) & (ty >= -h) & (ty <= h), (tx > w) & (ty < -h), (tx >= -w) & (tx <= w) & (ty < -h), (tx >= -w) & (tx <= w) & (ty >= -h) & (ty <= h)])
    # print("condlist:",condlist)
    # 3   4    5
    # 2   9    6
    # 1   8    7
    inner_array = jnp.array(
        [jnp.abs(tx+w), jnp.abs(ty+h), jnp.abs(tx-w), jnp.abs(ty-h)])
    choicelist = jnp.array([jnp.sqrt((tx+w)**2+(ty+h)**2), jnp.abs(tx+w), jnp.sqrt((tx+w)**2+(ty-h)**2), jnp.abs(ty-h),
                           jnp.sqrt((tx-w)**2+(ty-h)**2), jnp.abs(tx-w), jnp.sqrt((tx-w)**2+(ty+h)**2), jnp.abs(ty+h), jnp.min(inner_array)])
    # print("choicelist:",choicelist)
    # return dist((tx,ty),rect_boundary) with no square
    return jnp.select(condlist, choicelist, 0.0)


@jit
def inner_img_E(tx, ty, S_pixel, E0, width, height, normalized_intensity):
    w = cfg.half_width
    h = cfg.half_height
    u = (tx+w)/(2*w)
    v = (h-ty)/(2*h) # (u,v)=(0,0) represents the lefttop of the img, i.e. postition (tx,ty)=(-w,h)
    return imgp.queryIntensity(normalized_intensity, u, v, width, height)*E0/S_pixel


@jit
def img_E(tx, ty, img_dict):
    w = cfg.half_width
    h = cfg.half_height
    S_pixel = img_dict["S_pixel"]
    min_intensity = img_dict["min_intensity"]
    width = img_dict["width"]
    height = img_dict["height"]
    normalized_intensity = img_dict["normalized_intensity"]
    weight = 20.0
    E1 = cfg.glassArea*I(tx,ty)/(S_pixel/min_intensity+4*w/weight+4*h/weight+2*jnp.pi/(weight**2))
    E0 = E1*S_pixel/min_intensity
    return lax.cond((tx <= w) & (tx >= -w) & (ty <= h) & (ty >= -h), lambda x: inner_img_E(tx, ty, S_pixel, E0, width, height, normalized_intensity), lambda x: E1*jnp.exp(-weight*rect_boundary(tx, ty)), 0.0)


@jit 
def E_forInit(tx, ty,img_dict):
    w = cfg.half_width
    h = cfg.half_height
    S_pixel = img_dict["S_pixel"]
    width = img_dict["width"]
    height = img_dict["height"]
    weight = 20.0
    # notice that min_intensity = 1/(width*height) = S_pixel/cfg.domainArea
    E1 = cfg.glassArea*I(tx,ty)/(cfg.domainArea+4*w/weight+4*h/weight+2*jnp.pi/(weight**2))
    E0 = E1*cfg.domainArea
    return lax.cond((tx <= w) & (tx >= -w) & (ty <= h) & (ty >= -h), lambda x: E0/cfg.domainArea, lambda x: E1*jnp.exp(-weight*rect_boundary(tx, ty)), 0.0)