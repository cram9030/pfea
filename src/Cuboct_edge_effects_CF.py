import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import frame3dd
import subprocess
import pfea
import cProfile
import cuboct
import kelvin
from math import *
import os
from scipy import stats
import csv



def plotLattice(nodes,frames,res_displace,scale):
    xs = []
    ys = []
    zs = []
    
    rxs = []
    rys = []
    rzs = []
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    frame_coords = []

    print(matplotlib.projections.get_projection_names())
    for i,node in enumerate(nodes):
	xs.append(node[0])
	ys.append(node[1])
	zs.append(node[2])
	rxs.append(node[0]+res_displace[i][0]*scale)
	rys.append(node[1]+res_displace[i][1]*scale)
	rzs.append(node[2]+res_displace[i][2]*scale)

    for i,frame in enumerate(frames):
	nid1 = int(frame[0])
	nid2 = int(frame[1])
	start = [xs[nid1],ys[nid1],zs[nid1]]
	end   = [xs[nid2],ys[nid2],zs[nid2]]
	rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
	rend   = [rxs[nid2],rys[nid2],rzs[nid2]]
	ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)
	ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=0.3)

    ax.scatter(xs,ys,zs, color='r',alpha=0.1)
    ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
    plt.show()

def calcDisplacement(size_x,size_y,size_z,vox_pitch,force):
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
    #Assuming a Carbon Fiber tube with material properties taken from:
    #http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
    #using Std CF UD parameters
    #deminsions are taken from:
    #http://www.mcmaster.com/#2153t35/=105pcic
    #Using Newtons, meters, and kilograms as the units
    frame_props = {"nu"  : 0.30, #poisson's ratio
  			   "Ro"	 : 0.0064516, #m
  			   "th"	 : 0.0012192, #m
  			   "E"   :  150000000000, #N/m^2
  			   "G"   :  5000000000,  #N/m^2
  			   "rho" :  1600, #kg/m^3
  			   "beam_divisions" : 0,
  			   "cross_section"  : 'circular',
  			   "roll": 0,
  			   "Le":vox_pitch/sqrt(2.0)} 
    
    
    #Node Map Population
    #Referencing the geometry-specific kelvin.py file. 
    #Future versions might have different files?
    
    node_frame_map = np.zeros((size_x,size_y,size_z,6))
    nodes,frames,node_frame_map,dims = cuboct.from_material(mat_matrix,vox_pitch)
    frame_props["Le"] = cuboct.frame_length(vox_pitch)
    
    #Constraint and load population
    constraints = []
    loads = []
    #Constraints are added based on simple requirements right now
    for x in range(1,size_x+1):
	for y in range(1,size_y+1):
		#The bottom-most nodes are constrained to neither translate nor
		#rotate
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':2, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][0],'DOF':5, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':2, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][1][1],'DOF':5, 'value':0})


		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':2, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][0],'DOF':5, 'value':0})

		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':0, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':1, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':2, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':3, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':4, 'value':0})
		constraints.append({'node':node_frame_map[x][y][size_z][1],'DOF':5, 'value':0})
		loads.append(      {'type':'disp','node':node_frame_map[x][y][size_z][3],'DOF':2, 'value':force})
    

    
    #####  ### #     #    ####### #     # ####### ######  #     # ####### 
    #     #  #  ##   ##    #     # #     #    #    #     # #     #    #    
    #        #  # # # #    #     # #     #    #    #     # #     #    #    
    #####   #  #  #  #    #     # #     #    #    ######  #     #    #    
        #  #  #     #    #     # #     #    #    #       #     #    #    
    #     #  #  #     #    #     # #     #    #    #       #     #    #    
    #####  ### #     #    #######  #####     #    #        #####     #    
                                                                        
    
    
    #Group frames with their characteristic properties.
    out_frames = [(np.array(frames),{'E'   : frame_props["E"],
   								 'rho' : frame_props["rho"],
   								 'nu'  : frame_props["nu"],
   								 'd1'  : frame_props["Ro"],
   								 'th'  : frame_props["th"],
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
        #plotLattice(nodes,frames,res_displace,500)
        #avgDisp = np.max(res_displace[:,2])
        avgDisp =  np.average(res_displace[np.where(np.asarray(nodes)[:,2] == np.max(np.asarray(nodes)[:,2])),2])
        #print res_displace[np.where(np.asarray(nodes)[:,2] == np.max(np.asarray(nodes)[:,2])),2]
    else:
	res_displace = pfea.analyze_System(out_nodes, global_args, out_frames, constraints,loads)
        #plotLattice(nodes,frames,res_displace[0],500)
	avgDisp =  np.average(res_displace[0][np.where(np.asarray(nodes)[:,2] == np.max(np.asarray(nodes)[:,2]))[0],2])
	
    return avgDisp
    
#Physical Voxel Properties
#Calculated assuming strut length of .2 m and calculating the
#circumradius from R = sqrt(2)/2a
vox_pitch = 0.282843 #m

forceRange = np.arange(-5.0,2.0,2.0)
E = np.zeros((9,1), dtype=np.float);

for k in range(1,2):
    for j in range(1,10):
        stress = np.zeros_like(forceRange)
        strain = np.zeros_like(forceRange)
        
        for i in range(0,forceRange.size):
            strain[i] = calcDisplacement(j,j,j*k,vox_pitch,forceRange[i])/(j*vox_pitch)
            stress[i] = 4*forceRange[i]*pow(j,2)/(pow(j*vox_pitch,2)) #need to add the 4 multiplier because of the kelvin structure
        #print strain
        #print stress
        #print stats.linregress(np.absolute(strain),stress)
        #plt.plot(strain,stress,'ro')
        #plt.show()
        E[j-1,k-1], intercept, r_value, p_value, std_err = stats.linregress(strain,stress)
        print E[j-1,k-1]

print E
with open('edge_effects_CF.csv','wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    for i in range(0,len(E)):
        spamwriter.writerow(E[i][:])
#X = np.arange(1, 10, 1)
#Y = np.arange(1, 5, 1)
#X, Y = np.meshgrid(X, Y)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(X, Y, E, rstride=1, cstride=1, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)

#ax.set_xlabel('Square Base')
#ax.set_ylabel('Aspect Ratio')
#ax.set_zlabel('E (Pa)')

#fig.colorbar(surf, shrink=0.5, aspect=5)

#plt.show()
plt.plot(range(1,10),E,'ro')
plt.xlabel('Normalized Number of Cells')
plt.ylabel('E (Pa)')
plt.show()
