from math import floor
# M,N controls the B-Spline, the number of control points is M+3 * N+3
M = 66
N = M
variable_chunk_size = 15*15
# M_sample,N_sample controls the sample points in the target plane, the number of sample points is M_sample*N_sample
M_sample = 512
N_sample = M_sample
sample_chunk_size = 100**2
# half_width, half_height are the half width and half height of the target domain
half_width = 1
half_height = 1

#target_img_path = './image/sorcery.jpg'
#target_img_path = './image/einstein.jpg'
target_img_path = './image/blbl.jpg'
gamma = 1.0

# the following parameters remain relatively constant
# xmin,xmax,ymin,ymax are the boundary of the glass
xmin = -1
ymin = -1
xmax = 1
ymax = 1
init_h = 0.5
# ni,no are the refractive index of incident medium and outside medium
ni = 1.5
no = 1
# tz is the distance between the target and the glass
tz = 150

# the following parameters are the calculated parameters
totalBasisNum = (M+3)*(N+3)
totalSampleNum = (M_sample+1)*(N_sample+1)
h1 = (xmax-xmin)/M_sample
h2 = (ymax-ymin)/N_sample
a = 1-pow(ni/no, 2)
sample_batch_size = floor((6000**2)/totalBasisNum)
variable_batch_size = floor((6000**2)/totalBasisNum)

# file
name = target_img_path.split(
    '/')[-1].split('.')[0]
# +"_once_vary_weight_decay32"  # like "sorcery"
# name = "circle_vary_weight" # one iter vary and one nonvary, when low loss, using nonvary
# using_varyweight_flag is always true
# name = "circle_once_vary_weight_decay32"
folder_name = "./result_new/"
prefix_name = folder_name + f"{name}_{M}_{M_sample}_gamma{gamma}/"
test_folder_name = "./render_test/"
log_filename = prefix_name + "output.txt"
log_renderOptFilename = prefix_name + "renderOpt_output.txt"
init_objname = prefix_name + "init.obj"
# need to use blender
objname = prefix_name + f"{name}_{M}_{M_sample}_gamma{gamma}_" + "result.obj"
objname_afterOpt = prefix_name + \
    f"{name}_afterOpt_{M}_{M_sample}_gamma{gamma}_" + "result.obj"

OT_dict_name = prefix_name + "OT_dict.npy"
Opt_dict_name = prefix_name + "Opt_dict.npy"
OptAlter_dict_name = prefix_name + "OptAlter_dict.npy"
train2_dict_name = prefix_name + "train2_dict.npy"
memory_profile_name = prefix_name + "memory.prof"
img_dict_name = prefix_name + "img_dict.npy"
render_folder_path = prefix_name + "render_results/"
save_target_img_path = prefix_name +"target_img.png"

OT_dict_test_name = prefix_name + "OT_dict_test.npy"
OT_dict_test2_name = prefix_name + "OT_dict_test2.npy"


# objfile sample
m = 200
n = m

# render sample
rm = 1280
rn = rm
dm = (xmax-xmin)/rm  # on glass
dn = (ymax-ymin)/rn
# render resolution
rx = 128
ry = rx
dx = 2*half_width/rx  # on target domain
dy = 2*half_height/ry
render_picname = prefix_name + f"{rx}_{rm}_" + "img.png"
render_picname_test = prefix_name + f"{rx}_{rm}_" + "img_test.png"
render_picname_test2 = prefix_name + f"{rx}_{rm}_" + "img_test2.png"
render_picname_afterOpt = prefix_name + f"{rx}_{rm}_afterOpt_" + "img.png"
render_picname_afterOptAlter = prefix_name + f"{rx}_{rm}_afterOpt_Alter_" + "img.png"
render_picname_allTogether = prefix_name + f"{rx}_{rm}_allTogether_" + "img.png"
