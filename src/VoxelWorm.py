import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import frame3dd
import subprocess
import pfea
import pfeautil
import cProfile
import cuboct
from math import *







#Physical Voxel Properties
#Current voxel pitch is 3in
vox_pitch = 0.0762 #m

####### ######     #    #     # ####### 
#       #     #   # #   ##   ## #       
#       #     #  #   #  # # # # #       
#####   ######  #     # #  #  # #####   
#       #   #   ####### #     # #       
#       #    #  #     # #     # #       
#       #     # #     # #     # ####### 
                                                    
######  ####### ######  #     # #          #    ####### ### ####### #     # 
#     # #     # #     # #     # #         # #      #     #  #     # ##    # 
#     # #     # #     # #     # #        #   #     #     #  #     # # #   # 
######  #     # ######  #     # #       #     #    #     #  #     # #  #  # 
#       #     # #       #     # #       #######    #     #  #     # #   # # 
#       #     # #       #     # #       #     #    #     #  #     # #    ## 
#       ####### #        #####  ####### #     #    #    ### ####### #     # 
                                                                                
#Setting up a 2 by 2 by 5
size_x = 2;
size_y = 2;
size_z = 5;

#Temporary Material Matrix - NxNxN cubic grid (corresponding to cubic-octahedra)
# at the moment:
# 1's correspond to material being there
# 0's correspond to no material
mat_matrix = []
    for i in range(0,size_x+2):
   	tempcol = []
   	for j in range(0,size_y+2):
  		tempdep = [1]*(size_z+1)
  		tempdep.append(0)
  		tempdep[0] = 0
  		if(i*j*(i-(size_x+1))*(j-(size_y+1)) == 0):
 			tempdep = [0]*(size_z+2)
  		tempcol.append(tempdep)
   	mat_matrix.append(tempcol)

# Material Properties
        
node_radius = 0

#STRUT PROPERTIES
    #Physical Properties
    #Assuming a ABS tube with material properties taken from:
    #http://www.plasticsintl.com/abs.htm
    #Density taken from:
    #http://www.teststandard.com/data_sheets/ABS_Data_sheet.pdf
    #and multiplied by about 2 for the glass filled
    #Poisson's Ratio from:
    #http://www.engineersedge.com/plastic/materials_common_plastic.htm
    #Using Newtons, meters, and kilograms as the units
    frame_props = {"nu"  : 0.35, #poisson's ratio +
  			   "d1"	 : 0.00127, #m
  			   "d2"	 : 0.00127, #m
  			   "E"   : 4000000000, #N/m^2 +
  			   "G"   : 2250000000,  #N/m^2 +
  			   "rho" :  1348, #kg/m^3 + density was made by assuming 20% glass # 2500 kg/m^3
  			   "beam_divisions" : 0,
  			   "cross_section"  : 'rectangular',
  			   "roll": 0,
  			   "Le":vox_pitch/sqrt(2.0)}

#Node Map Population
#Referencing the geometry-specific kelvin.py file. 
#Future versions might have different files?
    
node_frame_map = np.zeros((size_x,size_y,size_z,6))
nodes,frames,node_frame_map,dims = kelvin.from_material(mat_matrix,vox_pitch)
frame_props["Le"] = kelvin.frame_length(vox_pitch)

force = 40

#Constraint and load population
constraints = []
loads = []

loads.append({'node':node_frame_map[1][1][2][5],'DOF':3, 'value':force})
loads.append({'node':node_frame_map[1][2][2][5],'DOF':3, 'value':force})
loads.append({'node':node_frame_map[2][1][2][5],'DOF':3, 'value':force})
loads.append({'node':node_frame_map[2][2][2][5],'DOF':3, 'value':force})
loads.append({'node':node_frame_map[1][1][3][5],'DOF':3, 'value':-force})
loads.append({'node':node_frame_map[1][2][3][5],'DOF':3, 'value':-force})
loads.append({'node':node_frame_map[2][1][3][5],'DOF':3, 'value':-force})
loads.append({'node':node_frame_map[2][2][3][5],'DOF':3, 'value':-force})

#####  ### #     #    ####### #     # ####### ######  #     # ####### 
#       #  ##   ##    #     # #     #    #    #     # #     #    #    
#       #  # # # #    #     # #     #    #    #     # #     #    #    
#####   #  #  #  #    #     # #     #    #    ######  #     #    #
    #   #  #     #    #     # #     #    #    #       #     #    #    
    #   #  #     #    #     # #     #    #    #       #     #    #        
#####  ### #     #    #######  #####     #    #        #####     #

#Group frames with their characteristic properties.
out_frames = [(np.array(frames),{'E'   : frame_props["E"],
                'rho' : frame_props["rho"],
                'nu'  : frame_props["nu"],
                'd1'  : frame_props["d1"],
                'd2'  : frame_props["d2"],
                'beam_divisions' : frame_props["beam_divisions"],
                'cross_section'  : frame_props["cross_section"],
                'roll': frame_props["roll"],
                'loads':{'element':0},
                'prestresses':{'element':0},
                'Le': frame_props["Le"]})]
    
#Format node positions
out_nodes = np.array(nodes)
    
#Global Arguments 
global_args = {'frame3dd_filename': os.path.join('experiments','Results','test'),"lump": False, 'length_scaling':1,"using_Frame3dd":False,"debug_plot":True, "gravity" : [0,0,0],"save_matrices":True}

if global_args["using_Frame3dd"]:
        frame3dd.write_frame3dd_file(out_nodes, global_args, out_frames, constraints,loads)
        subprocess.call("frame3dd -i {0}.csv -o {0}.out -q".format(global_args["frame3dd_filename"]), shell=True)
        res_nodes, res_reactions = frame3dd.read_frame3dd_results(global_args["frame3dd_filename"])
        res_displace = frame3dd.read_frame3dd_displacements(global_args["frame3dd_filename"])
else:
	res_displace = pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)

pfeautil.plotLattice(nodes,frames,res_displace,500)
pfeautil.writeMatrices(nodes,beam_sets,Q,args)
