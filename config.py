# M,N controls the B-Spline, the number of control points is M+3 * N+3
M = 17
N = M
variable_chunk_size = 20*20
# M_sample,N_sample controls the sample points in the target plane, the number of sample points is M_sample*N_sample
M_sample = 1024
N_sample = M_sample
sample_chunk_size = 300**2
# half_width, half_height are the half width and half height of the target domain
half_width = 1
half_height = 1

# target_img_path = './image/sorcery.jpg'
# target_img_path = './image/einstein.jpg'
target_img_path = './image/blbl.jpg'

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

# file
name = target_img_path.split(
    '/')[-1].split('.')[0]+"_once_vary_weight_decay32"  # like "sorcery"
# name = "circle_vary_weight" # one iter vary and one nonvary, when low loss, using nonvary
# using_varyweight_flag is always true
# name = "circle_once_vary_weight_decay32"
log_filename = f"./result/{name}_{M}_{M_sample}_output.txt"
init_objname = f"./result/{name}_{M}_{M_sample}_init.obj"
objname = f"./result/{name}_{M}_{M_sample}.obj"

# objfile sample
m = 300
n = m
