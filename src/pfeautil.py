#!/usr/bin/env python
from __future__ import division
from numpy import *
import numpy as np
import pfea
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import networkx as nx
import itertools
import cvxopt as co
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

'''
def V3(x,y,z):
    return asarray([x,y,z])
def dots(A,V):
    return inner(A,V).T
def magnitudes(v):
	return sqrt(sum(v**2,axis=-1))
def close(a,b):
  return absolute(a-b)<1e-5
def mid(a):
    return .5*(amax(a)+amin(a))
def avg(a,axis=0):
    return sum(a,axis=axis)/shape(a)[axis]
def extremes(a):
    return array([amin(a),amax(a)])
def span(a):
    return amax(a)-amin(a)
def sqr_magnitude(x):
    return sum(x*x,axis=-1)
def combine_topologies(node_sets,seg_sets):
    assert(len(node_sets)==len(seg_sets))
    all_nodes = vstack(tuple(node_sets))
    offsets = cumsum( [0] + map(lambda x: shape(x)[0], node_sets)[:-1] )
    seg_lists = tuple([os+bs for bs,os in zip(seg_sets,offsets)])
    return all_nodes,seg_lists
def subdivide_topology(nodes,segs):
    n_nodes = shape(nodes)[0]; n_seg = shape(segs)[0]
    nodes = vstack((nodes,.5*sum(nodes[segs],axis=1)))
    segs_1 = hstack((segs[:,0,None],arange(n_seg)[...,None]+n_nodes))
    segs_2 = hstack((arange(n_seg)[...,None]+n_nodes,segs[:,1,None]))
    return nodes,vstack((segs_1,segs_2))

def unique_points(a,tol=1e-5,leafsize=10):
    #Use KDTree to do uniqueness check within tolerance.
    pairs = cKDTree(a,leafsize=leafsize).query_pairs(tol)  #pairs of (i,j) where i<j and d(a[i]-a[j])<tol    
    components = map( sorted, nx.connected_components(nx.Graph(data=list(pairs))) ) #sorted connected components of the proximity graph
    idx = delete( arange(shape(a)[0]),  list(itertools.chain(*[c[1:] for c in components])) ) #all indices of a, except nodes past first in each component
    inv = arange(shape(a)[0])
    for c in components: inv[c[1:]]=c[0]
    inv = searchsorted(idx,inv) 
    return idx,inv
def unique_reduce(nodes,*beamsets):
    #reduced_idx,reduced_inv = unique_rows(nodes)
    reduced_idx,reduced_inv = unique_points(nodes)
    return nodes[reduced_idx],tuple([reduced_inv[bs] for bs in beamsets])
def rotation_matrix(axis, theta):
    axis = asarray(axis)
    theta = asarray(theta)
    axis = axis/sqrt(dot(axis, axis))
    a = cos(theta/2)
    b, c, d = -axis*sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return asarray([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
RX90=rotation_matrix([1,0,0],pi/2.)
RY90=rotation_matrix([0,1,0],pi/2.)
RZ90=rotation_matrix([0,0,1],pi/2.)

def line_plane_intersection(P0,N,l,l0=array([0,0,0])):
    #plane through p0 normal to N, line is l0 + t*l
    #return distance from l0
    return dot(P0-l0,N)/dot(l,N)

'''
def plotLattice(nodes,frames,res_displace,scale):
    # Function to plot the intial lattice configuration
    # and the final version of the lattice configuration
    #
    # Input:    nodes - Initial node location
    #           frames - node frames
    #           res_displace - displacement of nodes
    #           scale - scaling parameter
    
    #intialize arrays
    xs = []
    ys = []
    zs = []
    
    rxs = []
    rys = []
    rzs = []
    
    #create plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    frame_coords = []

    #poplate x, y, and z start and displacement arrays
    for i,node in enumerate(nodes):
	xs.append(node[0])
	ys.append(node[1])
	zs.append(node[2])
	rxs.append(node[0]+res_displace[i][0]*scale)
	rys.append(node[1]+res_displace[i][1]*scale)
	rzs.append(node[2]+res_displace[i][2]*scale)

    # Add frame
    for i,frame in enumerate(frames):
	nid1 = int(frame[0])
	nid2 = int(frame[1])
	start = [xs[nid1],ys[nid1],zs[nid1]]
	end   = [xs[nid2],ys[nid2],zs[nid2]]
	rstart = [rxs[nid1],rys[nid1],rzs[nid1]]
	rend   = [rxs[nid2],rys[nid2],rzs[nid2]]
	ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)
	ax.plot([rstart[0],rend[0]],[rstart[1],rend[1]],[rstart[2],rend[2]],color='b', alpha=0.3)

    #plot
    ax.scatter(xs,ys,zs, color='r',alpha=0.1)
    ax.scatter(rxs,rys,rzs, color='b',alpha=0.3)
    plt.show()

def elastic_K(beam_props):
	# beam_props is a dictionary with the following values
	# xn1   : position vector for start node
	# xn2	: position vector for end node
	# Le    : Effective beam length (taking into account node diameter)
	# Asy   : Effective area for shear effects, y direction
	# Asz   : Effective area for shear effects, z direction
	# G		: Shear modulus
	# E 	: Elastic modulus
	# J 	: Polar moment of inertia
	# Iy 	: Bending moment of inertia, y direction
	# Iz 	: bending moment of inertia, z direction
	# p 	: The roll angle (radians)
	# T 	: internal element end force
	# shear : Do we consider shear effects

	#Start by importing the beam properties
	xn1 	= beam_props["xn1"]
	xn2 	= beam_props["xn2"]
	Le  	= beam_props["Le"]
	Ax		= beam_props["Ax"]
	Asy 	= beam_props["Asy"]
	Asz 	= beam_props["Asz"]
	G   	= beam_props["G"]
	E   	= beam_props["E"]
	J 		= beam_props["J"]
	Iy 		= beam_props["Iy"]
	Iz 		= beam_props["Iz"]
	p 		= beam_props["p"]
	shear 	= beam_props["shear"]

	#initialize the output
	k = np.zeros((12,12))
	#k = co.matrix(0.0,(12,12))
	#define the transform between local and global coordinate frames
	t = coord_trans(xn1,xn2,Le,p)

	#calculate Shear deformation effects
	Ksy = 0
	Ksz = 0

	#begin populating that elastic stiffness matrix
	if shear:
		Ksy = 12.0*E*Iz / (G*Asy*Le*Le)
		Ksz = 12.0*E*Iy / (G*Asz*Le*Le)
	else:
		Ksy = Ksz = 0.0
	
	k[0,0]  = k[6,6]   = 1.0*E*Ax / Le
	k[1,1]  = k[7,7]   = 12.*E*Iz / ( Le*Le*Le*(1.+Ksy) )
	k[2,2]  = k[8,8]   = 12.*E*Iy / ( Le*Le*Le*(1.+Ksz) )
	k[3,3]  = k[9,9]   = 1.0*G*J / Le
	k[4,4]  = k[10,10] = (4.+Ksz)*E*Iy / ( Le*(1.+Ksz) )
	k[5,5]  = k[11,11] = (4.+Ksy)*E*Iz / ( Le*(1.+Ksy) )

	k[4,2]  = k[2,4]   = -6.*E*Iy / ( Le*Le*(1.+Ksz) )
	k[5,1]  = k[1,5]   =  6.*E*Iz / ( Le*Le*(1.+Ksy) )
	k[6,0]  = k[0,6]   = -k[0,0]

	k[11,7] = k[7,11]  =  k[7,5] = k[5,7] = -k[5,1]
	k[10,8] = k[8,10]  =  k[8,4] = k[4,8] = -k[4,2]
	k[9,3]  = k[3,9]   = -k[3,3]
	k[10,2] = k[2,10]  =  k[4,2]
	k[11,1] = k[1,11]  =  k[5,1]

	k[7,1]  = k[1,7]   = -k[1,1]
	k[8,2]  = k[2,8]   = -k[2,2]
	k[10,4] = k[4,10]  = (2.-Ksz)*E*Iy / ( Le*(1.+Ksz) )
	k[11,5] = k[5,11]  = (2.-Ksy)*E*Iz / ( Le*(1.+Ksy) )


	#now we transform k to the global coordinates
	k = atma(t,k)

	# Check and enforce symmetry of the elastic stiffness matrix for the element
	k = 0.5*(k+k.T)

	return k

#GEOMETRIC_K - space frame geometric stiffness matrix, global coordnates

def geometric_K(beam_props):
	# beam_props is a dictionary with the following values
	# xn1   : position vector for start node
	# xn2	: position vector for end node
	# Le    : Effective beam length (taking into account node diameter)
	# Asy   : Effective area for shear effects, y direction
	# Asz   : Effective area for shear effects, z direction
	# G		: Shear modulus
	# E 	: Elastic modulus
	# J 	: Polar moment of inertia
	# Iy 	: Bending moment of inertia, y direction
	# Iz 	: bending moment of inertia, z direction
	# p 	: The roll angle (radians)
	# T 	: internal element end force
	# shear : whether shear effects are considered. 
	
	xn1 	= beam_props["xn1"]
	xn2 	= beam_props["xn2"]
	L   	= beam_props["Le"]
	Le  	= beam_props["Le"]
	Ax		= beam_props["Ax"]
	Asy 	= beam_props["Asy"]
	Asz 	= beam_props["Asz"]
	G   	= beam_props["G"]
	E   	= beam_props["E"]
	J 		= beam_props["J"]
	Iy 		= beam_props["Iy"]
	Iz 		= beam_props["Iz"]
	p 		= beam_props["p"]
	T 		= beam_props["T"]
	shear 	= beam_props["shear"]

	#initialize the geometric stiffness matrix
	kg = np.zeros((12,12))
	t = coord_trans(xn1,xn2,Le,p)

	if shear:
		Ksy = 12.0*E*Iz / (G*Asy*Le*Le);
		Ksz = 12.0*E*Iy / (G*Asz*Le*Le);
		Dsy = (1+Ksy)*(1+Ksy);
		Dsz = (1+Ksz)*(1+Ksz);
	else:
		Ksy = Ksz = 0.0;
		Dsy = Dsz = 1.0;

	#print(T)
	kg[0][0]  = kg[6][6]   =  0.0 # T/L
	 
	kg[1][1]  = kg[7][7]   =  T/L*(1.2+2.0*Ksy+Ksy*Ksy)/Dsy
	kg[2][2]  = kg[8][8]   =  T/L*(1.2+2.0*Ksz+Ksz*Ksz)/Dsz
	kg[3][3]  = kg[9][9]   =  T/L*J/Ax
	kg[4][4]  = kg[10][10] =  T*L*(2.0/15.0+Ksz/6.0+Ksz*Ksz/12.0)/Dsz
	kg[5][5]  = kg[11][11] =  T*L*(2.0/15.0+Ksy/6.0+Ksy*Ksy/12.0)/Dsy
	 
	kg[0][6]  = kg[6][0]   =  0.0 # -T/L
	
	kg[4][2]  = kg[2][4]   =  kg[10][2] = kg[2][10] = -T/10.0/Dsz
	kg[8][4]  = kg[4][8]   =  kg[10][8] = kg[8][10] =  T/10.0/Dsz
	kg[5][1]  = kg[1][5]   =  kg[11][1] = kg[1][11] =  T/10.0/Dsy
	kg[7][5]  = kg[5][7]   =  kg[11][7] = kg[7][11] = -T/10.0/Dsy
	
	kg[3][9]  = kg[9][3]   = -kg[3][3]
	
	kg[7][1]  = kg[1][7]   = -T/L*(1.2+2.0*Ksy+Ksy*Ksy)/Dsy
	kg[8][2]  = kg[2][8]   = -T/L*(1.2+2.0*Ksz+Ksz*Ksz)/Dsz

	kg[10][4] = kg[4][10]  = -T*L*(1.0/30.0+Ksz/6.0+Ksz*Ksz/12.0)/Dsz
	kg[11][5] = kg[5][11]  = -T*L*(1.0/30.0+Ksy/6.0+Ksy*Ksy/12.0)/Dsy

	#now we transform kg to the global coordinates
	kg = atma(t,kg)

	# Check and enforce symmetry of the elastic stiffness matrix for the element
	kg = 0.5*(kg+kg.T)
	
	return kg


def writeMatrices(nodes,beam_sets,Q,args):
	#Initialize K to zeros
	K = np.zeros((args["dof"],args["dof"]))
	M = np.zeros((args["dof"],args["dof"]))

	q_index = 0
	for beamset,bargs in beam_sets:
		#every beam set lists the physical properties
		#associated with that beam
		
		#transfer those properties over
		beam_props = {"Ax"		:bargs["Ax"],
					  "Asy"		: bargs["Asy"],
					  "Asz"		: bargs["Asz"],
					  "G"		: bargs["G"],
					  "E"		: bargs["E"],
					  "J"		: bargs["J"],
					  "Iy"		: bargs["Iy"],
					  "Iz"		: bargs["Iz"],
					  "p"		: bargs["roll"],
					  "Le"		: bargs["Le"],
					  "shear"	: True,
					"rho"		: bargs["rho"]}

		for beam in beamset:
			#Positions of the endpoint nodes for this beam
			xn1 = nodes[beam[0]]
			xn2 = nodes[beam[1]]
			beam_props["xn1"] = xn1
			beam_props["xn2"] = xn2
			
			beam_props["T"]	  = -Q[q_index][0]
			q_index = q_index+1
			ke = elastic_K(beam_props)
			kg = geometric_K(beam_props)
			
			ktot = ke+kg

			K[6*beam[0]:6*beam[0]+6,6*beam[0]:6*beam[0]+6] += ktot[0:6,0:6]
			K[6*beam[1]:6*beam[1]+6,6*beam[0]:6*beam[0]+6] += ktot[6:12,0:6]
			K[6*beam[0]:6*beam[0]+6,6*beam[1]:6*beam[1]+6] += ktot[0:6,6:12]
			K[6*beam[1]:6*beam[1]+6,6*beam[1]:6*beam[1]+6] += ktot[6:12,6:12]

			if args['lump']:
			    m = pfea.lumped_M(beam_props)
			else:
			    m = pfea.consistent_M(beam_props)

			M[6*beam[0]:6*beam[0]+6,6*beam[0]:6*beam[0]+6] += m[0:6,0:6]
			M[6*beam[1]:6*beam[1]+6,6*beam[0]:6*beam[0]+6] += m[6:12,0:6]
			M[6*beam[0]:6*beam[0]+6,6*beam[1]:6*beam[1]+6] += m[0:6,6:12]
			M[6*beam[1]:6*beam[1]+6,6*beam[1]:6*beam[1]+6] += m[6:12,6:12]
	
	np.savetxt('K.txt',K, delimiter=',')
	np.savetxt('M.txt',M, delimiter=',')

def coord_trans(x_n1,x_n2,L,p):
    # Find the coordinate transform from local beam coords
    # to the global coordinate frame
    # x_n1  : x,y,z position of the first node n1
    # x_n2  : x,y,z position of the second node n2
    # L     : length of beam (without accounting for node radius)
    # p     : the roll angle in radians
    L = np.linalg.norm(x_n1-x_n2)
    Cx = (x_n2[0]-x_n1[0])/L
    Cy = (x_n2[1]-x_n1[1])/L
    Cz = (x_n2[2]-x_n1[2])/L

    t = np.zeros(9)

    Cp = cos(p)
    Sp = sin(p)

    # We assume here that the GLOBAL Z AXIS IS VERTICAL.

    if ( fabs(Cz) == 1.0 ):
        t[2] =  Cz;
        t[3] = -Cz*Sp;
        t[4] =  Cp;
        t[6] = -Cz*Cp;
        t[7] = -Sp;
    else:
        den = sqrt ( 1.0 - Cz*Cz );

        t[0] = Cx;
        t[1] = Cy;
        t[2] = Cz;

        t[3] = 1.0*(-Cx*Cz*Sp - Cy*Cp)/den;    
        t[4] = 1.0*(-Cy*Cz*Sp + Cx*Cp)/den;
        t[5] = Sp*den;

        t[6] = 1.0*(-Cx*Cz*Cp + Cy*Sp)/den;
        t[7] = 1.0*(-Cy*Cz*Cp - Cx*Sp)/den;
        t[8] = Cp*den;

    return t

def atma(t,m):
    a = co.matrix(0.0,(12,12))

    #More efficient assignment possible
    for i in range(0,4):
        a[3*i,3*i] = t[0];
        a[3*i,3*i+1] = t[1];
        a[3*i,3*i+2] = t[2];
        a[3*i+1,3*i] = t[3];
        a[3*i+1,3*i+1] = t[4];
        a[3*i+1,3*i+2] = t[5];
        a[3*i+2,3*i] = t[6];
        a[3*i+2,3*i+1] = t[7];
        a[3*i+2,3*i+2] = t[8];
    
    m = co.matrix(np.dot(np.dot(a.T,m),a))

    return m

def swap_Matrix_Rows(M,r1,r2):
    r1 = int(r1)
    r2 = int(r2)
    M[[r1,r2],:] = M[[r2,r1],:]
    
def swap_Matrix_Cols(M,c1,c2):
    c1 = int(c1)
    c2 = int(c2)
    M[:,[c1,c2]] = M[:,[c2,c1]]
    
def swap_Vector_Vals(V,i1,i2):
    V[[i1,i2]] = V[[i2,i1]]

def gen_Node_map(nodes,constraints):
    # we want to generate the map between the input K 
    # and the easily-solved K
    ndof = len(nodes)*6
    index = ndof-len(constraints)
    
    indptr = np.array(range(ndof))
    data = np.array([1.0]*ndof)
    row = np.array(range(ndof))
    col = np.array(range(ndof))
    
    cdof_list = []
    
    for constraint in constraints:
        cdof_list.append(constraint["node"]*6+constraint["DOF"])
        
    for c_id in cdof_list:
        if c_id < ndof-len(constraints):
            not_found = True
            while not_found:
                if index in cdof_list:
                    index = index+1
                else:
                    col[c_id] = index
                    col[index] = c_id
                    not_found = False
                    index=index+1


    return co.spmatrix(data,row,col) 
