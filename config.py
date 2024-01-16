# M,N are the partition numbers, the number of control points is M+3 * N+3
M=27
N=M
# half_width, half_height are the half width and half height of the target domain
half_width=1
half_height=1

target_img_path='./image/sorcery.jpg'

### the following parameters remain relatively constant
# xmin,xmax,ymin,ymax are the boundary of the glass
xmin=-1
ymin=-1
xmax=1
ymax=1
init_h=0.5
# ni,no are the refractive index of incident medium and outside medium
ni=1.5
no=1
# tz is the distance between the target and the glass
tz=150

### the following parameters are the calculated parameters 
totalNum=(M+3)*(N+3)
h1=(xmax-xmin)/(M+2)
h2=(ymax-ymin)/(N+2)
a=1-pow(ni/no,2)

# file
name = target_img_path.split('/')[-1].split('.')[0]+"_vary_weight_mean" # like "sorcery"
# name = "circle_vary_weight" # one iter vary and one nonvary, when low loss, using nonvary
name = "circle_never_vary_weight" # using_varyweight_flag is always true
log_filename = f"./result/{name}_{M}_{N}_output.txt"
init_objname = f"./result/{name}_{M}_{N}_init.obj"
objname = f"./result/{name}_{M}_{N}.obj"

# objfile sample
m=100
n=m