#Import the required dependencies
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import networkx as nx
import freud

#define the relevant variables

pb1mrS = 0.0082
pb1mrT = 0.51
average_velocity=0.01
timestep=0.2
RT=0.035
pb1riS = 0.015 
pb1riT = 0.36 
pb1riV = 0.04 
d1 = 0.02 
d3 = 0.02 
alpha = 0.062 
betaW = -0.021
betaA= -0.055 
alpha=0.9
gamma=0.2 
kappa=10 
mu=1
Rrep=0.01
Req=0.03 
Ralign=0.135 
Ratt=0.2 
Kalign=6
field_mean_density= 50
field_min_density= 20 
field_max_density= 120
field_std=field_max_density-field_min_density
Critical_density=1000
N=100
L=np.sqrt(N/Critical_density)
m_S=np.array([(1-pb1mrS),0,1-(1-pb1mrS),0])
m_T=np.array([(1-pb1mrT)*(1-pb1mrS),0,1-((1-pb1mrT)*(1-pb1mrS)),0])
s_S=np.array([0,1-(1-pb1riS),(1-pb1riS),0])
s_T=np.array([0,1-(1-pb1riT)*(1-pb1riS),(1-pb1riT)*(1-pb1riS),0])
s_V=np.array([0,1-(1-pb1riV)*(1-pb1riS),(1-pb1riV)*(1-pb1riS),0])
s_TV=np.array([0,1-(1-pb1riT)*(1-pb1riS)*(1-pb1riV),(1-pb1riV)*(1-pb1riT)*(1-pb1riS),0])
t_M=np.array([1,0,0,0])
STATES=[0,1,2,3]
eta=1

#initialise the agents
pos = np.random.uniform(0,L,size=(N,2))
orient = np.random.uniform(-np.pi, np.pi,size=N)
states=np.random.choice([0,1,2,3],size=N)

#initialise the figure, subplots and quiver
fig, ax= plt.subplots(figsize=(6,6))
qv = ax.quiver(pos[:,0], pos[:,1], np.cos(orient), np.sin(orient), orient, clim=[-np.pi, np.pi])

#define the animation function
def animate(i):

 
    global orient, states, pos, onezero, L,N,average_velocity,STATES
    s=time.time()
    #There are two stages to the function. First we update the positions and orientations of the relevant particles. Then we update the states of the particles. Both are semi stochastic processes.

    #create a vector of new positions for all particles and the movement MASK_get_rid_of_self_contributions_to_local_sums_for_Ralign
    new_states_0=np.copy(states) 
    new_states_3=np.copy(states) 



    cos, sin= np.cos(orient), np.sin(orient)

    posN=np.copy(pos)
    posN[:,0]=posN[:,0] + cos*average_velocity*timestep
    posN[:,1]=posN[:,1] + sin*average_velocity*timestep

    #update orientations of particles in the turning state
    #The first step in doing this is to calculate the order parameter for the particles in the alignment radius

    #velocity array pre orient update
    velocity_array_N2=np.zeros((N,2))
    velocity_array_N2[:,0]=cos*average_velocity
    velocity_array_N2[:,1]=sin*average_velocity

    #kd tree
    tree=cKDTree(pos,boxsize=[L,L])
    dist_Ralign=tree.sparse_distance_matrix(tree, max_distance=Ralign,output_type='coo_matrix')

    #vx_expanded_over_cols_dist_Ralign

    vx_expanded_over_cols_dist_Ralign=velocity_array_N2[dist_Ralign.col,0]
    vy_expanded_over_cols_dist_Ralign=velocity_array_N2[dist_Ralign.col,1]
    states_expanded_over_cols_dist_Ralign=states[dist_Ralign.col]
    MASK_get_rid_of_self_contributions_to_local_sums_for_Ralign = dist_Ralign.row != dist_Ralign.col
    data_togoin_adj_Ralign_without_self_connections=np.ones(len(dist_Ralign.col[MASK_get_rid_of_self_contributions_to_local_sums_for_Ralign]))
    adj_Ralign_without_self_connections = sparse.coo_matrix((data_togoin_adj_Ralign_without_self_connections,(dist_Ralign.row[MASK_get_rid_of_self_contributions_to_local_sums_for_Ralign],dist_Ralign.col[MASK_get_rid_of_self_contributions_to_local_sums_for_Ralign])), shape=dist_Ralign.get_shape())

    local_sum_NO_self_contributions_number_of_neighbours_within_Ralign = np.squeeze(np.asarray(adj_Ralign_without_self_connections.tocsr().sum(axis=1)))

    MASK_from_local_sets_select_only_particles_in_motion_AND_ignore_self_contributions=np.logical_and(states_expanded_over_cols_dist_Ralign==0,dist_Ralign.row != dist_Ralign.col)
    data_togoin_adj_Ralign_rows_are_local_sets_including_only_moving=np.ones(len(dist_Ralign.col[MASK_from_local_sets_select_only_particles_in_motion_AND_ignore_self_contributions]))
    adj_Ralign_rows_are_local_sets_including_only_moving = sparse.coo_matrix((data_togoin_adj_Ralign_rows_are_local_sets_including_only_moving,(dist_Ralign.row[MASK_from_local_sets_select_only_particles_in_motion_AND_ignore_self_contributions],dist_Ralign.col[MASK_from_local_sets_select_only_particles_in_motion_AND_ignore_self_contributions])), shape=dist_Ralign.get_shape())

    vx_Ralign_rows_are_local_sets_including_only_moving = sparse.coo_matrix((vx_expanded_over_cols_dist_Ralign[MASK_from_local_sets_select_only_particles_in_motion_AND_ignore_self_contributions],(dist_Ralign.row[MASK_from_local_sets_select_only_particles_in_motion_AND_ignore_self_contributions],dist_Ralign.col[MASK_from_local_sets_select_only_particles_in_motion_AND_ignore_self_contributions])), shape=dist_Ralign.get_shape())
    vy_Ralign_rows_are_local_sets_including_only_moving = sparse.coo_matrix((vy_expanded_over_cols_dist_Ralign[MASK_from_local_sets_select_only_particles_in_motion_AND_ignore_self_contributions],(dist_Ralign.row[MASK_from_local_sets_select_only_particles_in_motion_AND_ignore_self_contributions],dist_Ralign.col[MASK_from_local_sets_select_only_particles_in_motion_AND_ignore_self_contributions])), shape=dist_Ralign.get_shape())
    local_sum_NO_self_contributions_number_of_neighbours_within_Ralign_including_only_moving = np.squeeze(np.asarray(adj_Ralign_rows_are_local_sets_including_only_moving.tocsr().sum(axis=1)))
    local_sum_NO_self_contributions_total_x_velocity_in_vicinity_of_each_particle_within_Ralign_including_only_moving=vx_Ralign_rows_are_local_sets_including_only_moving.tocsr().sum(axis=1).A1
    local_sum_NO_self_contributions_total_y_velocity_in_vicinity_of_each_particle_within_Ralign_including_only_moving=vy_Ralign_rows_are_local_sets_including_only_moving.tocsr().sum(axis=1).A1
    local_sum_NO_self_contributions_local_order_parameter_within_Ralign_TOTAL=np.sqrt(local_sum_NO_self_contributions_total_x_velocity_in_vicinity_of_each_particle_within_Ralign_including_only_moving**2+local_sum_NO_self_contributions_total_y_velocity_in_vicinity_of_each_particle_within_Ralign_including_only_moving**2)
    MASK_prevent_division_by_0_selecting_number_of_neighbours_within_Ralign_including_only_moving_is0=local_sum_NO_self_contributions_number_of_neighbours_within_Ralign_including_only_moving==0
    number_of_nearest_neighbours_denominator=np.where(MASK_prevent_division_by_0_selecting_number_of_neighbours_within_Ralign_including_only_moving_is0,1,local_sum_NO_self_contributions_number_of_neighbours_within_Ralign_including_only_moving)
    NO_self_contributions_local_order_parameter_within_Ralign_TOTAL_numerator=np.where(MASK_prevent_division_by_0_selecting_number_of_neighbours_within_Ralign_including_only_moving_is0,0,local_sum_NO_self_contributions_local_order_parameter_within_Ralign_TOTAL)
    NO_self_contributions_local_order_parameter_within_Ralign_normalised=NO_self_contributions_local_order_parameter_within_Ralign_TOTAL_numerator/(number_of_nearest_neighbours_denominator*average_velocity)
    NO_self_contributions_VARIANCE_local_orientation_within_Ralign=1-(NO_self_contributions_local_order_parameter_within_Ralign_normalised**2)
    MASK_source_of_the_fatal_NaN_bug=NO_self_contributions_VARIANCE_local_orientation_within_Ralign<=0
    NO_self_contributions_VARIANCE_local_orientation_within_Ralign=np.where(MASK_source_of_the_fatal_NaN_bug,0.01,NO_self_contributions_VARIANCE_local_orientation_within_Ralign)

    #Now generate the new angles
    complex_orient_expanded_over_cols_adj_Ralign_rows_are_local_sets_including_only_moving = np.exp(orient[adj_Ralign_rows_are_local_sets_including_only_moving.col]*1j)
    complex_orient_Ralign_rows_are_local_sets_including_only_moving = sparse.coo_matrix((complex_orient_expanded_over_cols_adj_Ralign_rows_are_local_sets_including_only_moving,(adj_Ralign_rows_are_local_sets_including_only_moving.row,adj_Ralign_rows_are_local_sets_including_only_moving.col)), shape=dist_Ralign.get_shape())
    local_sum_NO_self_contributions_complex_orient_in_vicinity_of_each_particle_within_Ralign_including_only_moving = np.squeeze(np.asarray(complex_orient_Ralign_rows_are_local_sets_including_only_moving.tocsr().sum(axis=1)))
    local_sum_NO_self_contributions_TRUE_MEAN_ANGLE_in_vicinity_of_each_particle_within_Ralign_including_only_moving=np.angle(local_sum_NO_self_contributions_complex_orient_in_vicinity_of_each_particle_within_Ralign_including_only_moving)
    local_sum_NO_self_contributions_buhl_angle_for_each_particle_from_own_orient_and_local_mean_angle=np.angle((gamma)*average_velocity*np.exp(1j*orient)+(1-gamma)*(np.exp(1j*local_sum_NO_self_contributions_TRUE_MEAN_ANGLE_in_vicinity_of_each_particle_within_Ralign_including_only_moving)))
    orientN=local_sum_NO_self_contributions_buhl_angle_for_each_particle_from_own_orient_and_local_mean_angle+(eta)*np.sqrt(NO_self_contributions_VARIANCE_local_orientation_within_Ralign)*np.random.randn(N)
    


    #Now update the positions of the particles in motion state, and the orientations of the particles in the turning state


    MASK_states_is_in_state_0=states==STATES[0]     
    pos[:,0]=np.where(MASK_states_is_in_state_0,posN[:,0],pos[:,0])
    pos[:,1]=np.where(MASK_states_is_in_state_0,posN[:,1],pos[:,1])
    pos[pos>L] -= L
    pos[pos<0] += L
    qv.set_offsets(pos)
    MASK_states_is_in_state_3=states==STATES[3]
    orient=np.where(MASK_states_is_in_state_3,orientN,orient)
    qv.set_UVC(cos, sin,orient)    
    
    
    



    ####Up till this point we have calculated####################################
 #   tree
  #  dist_Ralign
   # local_order parameter_including contributions only from moving particles within Ralign
 #   local_nearest neighbour counts including only neighbours which are moving within Ralign
  #  vx_Ralign_matrix_where rows are local sets including only_moving
   # vy_Ralign_matrix_where rows are local sets including only_moving

   #therefore we shouldnt calculate them again obviously

  ####Now we actually update the states

  #Update the states in motion
    #update the states
    #print("states before new_states_0",states)
    new_states_2=np.copy(states) 



    dist_T=tree.sparse_distance_matrix(tree, max_distance=RT,output_type='coo_matrix')

    MASK_get_rid_of_self_contributions_to_local_sums_for_RT = dist_T.row != dist_T.col

    data_togoin_adj_RT_without_self_connections=np.ones(len(dist_T.col[MASK_get_rid_of_self_contributions_to_local_sums_for_RT]))

    adj_RT_without_self_connections = sparse.coo_matrix((data_togoin_adj_RT_without_self_connections,(dist_T.row[MASK_get_rid_of_self_contributions_to_local_sums_for_RT],dist_T.col[MASK_get_rid_of_self_contributions_to_local_sums_for_RT])), shape=dist_T.get_shape())

    local_sum_NO_self_contributions_number_of_neighbours_within_RT = np.squeeze(np.asarray(adj_RT_without_self_connections.tocsr().sum(axis=1)))

    tactile_adjacency=np.zeros(N)

    tactile_adjacency=np.where(local_sum_NO_self_contributions_number_of_neighbours_within_RT,1,0)
    #stat=STATES[0]
    MASK_particle_is_in_state_0_AND_HAS_INDEED_neighbours_in_tactile_radius=np.logical_and(new_states_0 == STATES[0],tactile_adjacency==1)
    MASK_particle_is_in_state_0_AND_HAS_NOT_neighbours_in_tactile_radius=np.logical_and(new_states_0 == STATES[0], tactile_adjacency==0)
    
    
    states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state0_that_is_in_tactile_contact_with_a_neighbour=np.random.choice(STATES,size=MASK_particle_is_in_state_0_AND_HAS_INDEED_neighbours_in_tactile_radius.sum(),p=m_T)
    states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state0_that_is_NOT_in_tactile_contact_with_a_neighbour=np.random.choice(STATES,size=MASK_particle_is_in_state_0_AND_HAS_NOT_neighbours_in_tactile_radius.sum(),p=m_S)

    ##Note this is a workaround the fact that np.where only works if the two arguments are arrays with the same dimesnions. So we have to distribute the values generated for the states in motion among arrays of 0s in their corresponding indices
    npwhere_array_with_states_dims_to_place_generated_states_for_0andT_particles_at_their_indices=np.zeros_like(new_states_0, dtype=float)
    npwhere_array_with_states_dims_to_place_generated_states_for_0andT_particles_at_their_indices[MASK_particle_is_in_state_0_AND_HAS_INDEED_neighbours_in_tactile_radius]=states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state0_that_is_in_tactile_contact_with_a_neighbour
    npwhere_array_with_states_dims_to_place_generated_states_for_0andT_NOT_particles_at_their_indices=np.zeros_like(new_states_0, dtype=float)
    npwhere_array_with_states_dims_to_place_generated_states_for_0andT_NOT_particles_at_their_indices[MASK_particle_is_in_state_0_AND_HAS_NOT_neighbours_in_tactile_radius]=states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state0_that_is_NOT_in_tactile_contact_with_a_neighbour    
    

    new_states_0=np.where(MASK_particle_is_in_state_0_AND_HAS_INDEED_neighbours_in_tactile_radius ,npwhere_array_with_states_dims_to_place_generated_states_for_0andT_particles_at_their_indices,new_states_0)
    new_states_0=np.where(MASK_particle_is_in_state_0_AND_HAS_NOT_neighbours_in_tactile_radius ,npwhere_array_with_states_dims_to_place_generated_states_for_0andT_NOT_particles_at_their_indices,new_states_0)

    ##Now update the particles in state 2
    new_states_2=np.copy(states)
    dist_Rrep=tree.sparse_distance_matrix(tree, max_distance=Rrep,output_type='coo_matrix')

    vx_expanded_over_cols_dist_Rrep=velocity_array_N2[dist_Rrep.col,0]
    vy_expanded_over_cols_dist_Rrep=velocity_array_N2[dist_Rrep.col,1]
    #print("data_vel_rep_x",data_vel_rep_x)
    #print("data_vel_rep_y",data_vel_rep_y)
    #print("dist_Rrep.col",dist_Rrep.col)
    #xnan=np.isnan(data_vel_rep_x)
    #ynan=np.isnan(data_vel_rep_y)
    #print("xnan",xnan)
    #print("ynan",ynan)


    MASK_get_rid_of_self_contributions_to_local_sums_for_Rrep = dist_Rrep.row != dist_Rrep.col
    vx_Rrep_NO_self_contributions_rows_are_full_local_velocity_sets_within_Rrep = sparse.coo_matrix((vx_expanded_over_cols_dist_Rrep,(dist_Rrep.row,dist_Rrep.col)), shape=dist_Rrep.get_shape())


    vy_Rrep_NO_self_contributions_rows_are_full_local_velocity_sets_within_Rrep = sparse.coo_matrix((vy_expanded_over_cols_dist_Rrep,(dist_Rrep.row,dist_Rrep.col)), shape=dist_Rrep.get_shape())


    POSx_expanded_over_cols_dist_Rrep=pos[dist_Rrep.col,0]
    POSy_expanded_over_cols_dist_Rrep=pos[dist_Rrep.col,1]   


    POSx_Rrep_NO_self_contributions_rows_are_full_local_position_sets_within_Rrep = sparse.coo_matrix((POSx_expanded_over_cols_dist_Rrep,(dist_Rrep.row,dist_Rrep.col)), shape=dist_Rrep.get_shape())    
    POSy_Rrep_NO_self_contributions_rows_are_full_local_position_sets_within_Rrep = sparse.coo_matrix((POSy_expanded_over_cols_dist_Rrep,(dist_Rrep.row,dist_Rrep.col)), shape=dist_Rrep.get_shape())
    TRANSPOSED_POSx_Rrep_NO_self_contributions_rows_are_full_local_position_sets_within_Rrep=POSx_Rrep_NO_self_contributions_rows_are_full_local_position_sets_within_Rrep.T
    TRANSPOSED_POSy_Rrep_NO_self_contributions_rows_are_full_local_position_sets_within_Rrep=POSy_Rrep_NO_self_contributions_rows_are_full_local_position_sets_within_Rrep.T

    LOCAL_SETS_MATRIX_OF_RELATIVE_X_POSITIONS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_Rrep_csr=POSx_Rrep_NO_self_contributions_rows_are_full_local_position_sets_within_Rrep.tocsr()-TRANSPOSED_POSx_Rrep_NO_self_contributions_rows_are_full_local_position_sets_within_Rrep.tocsr()
    LOCAL_SETS_MATRIX_OF_RELATIVE_Y_POSITIONS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_Rrep_csr=POSy_Rrep_NO_self_contributions_rows_are_full_local_position_sets_within_Rrep.tocsr()-TRANSPOSED_POSy_Rrep_NO_self_contributions_rows_are_full_local_position_sets_within_Rrep.tocsr()

    POSx_times_vx=LOCAL_SETS_MATRIX_OF_RELATIVE_X_POSITIONS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_Rrep_csr.multiply(vx_Rrep_NO_self_contributions_rows_are_full_local_velocity_sets_within_Rrep.tocsr())
    POSy_times_vy=LOCAL_SETS_MATRIX_OF_RELATIVE_Y_POSITIONS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_Rrep_csr.multiply(vy_Rrep_NO_self_contributions_rows_are_full_local_velocity_sets_within_Rrep.tocsr())
    
    LOCAL_SETS_MATRIX_OF_RELPOS_AND_PROPV_DOT_PRODUCTS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_Rrep=POSx_times_vx+POSy_times_vy

    #THIS STEP IS POSSIBLY_REDUNDANT_NOW_SINCE_SPARSE_MATRIX_RELATIVE_DISTANCE_CALC_WOULD_HAVE_TAKEN_CARE_OF_THAT
    MASK_elimination_of_self_distances_from_dist_Rrep_data=dist_Rrep.tocsr().data!=0    
    
    data_togoin_Vcostheta_Rrep_NO_self_contributions_rows_are_full_local_velocity_projection_sets_within_Rrep=LOCAL_SETS_MATRIX_OF_RELPOS_AND_PROPV_DOT_PRODUCTS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_Rrep.data/dist_Rrep.tocsr().data[MASK_elimination_of_self_distances_from_dist_Rrep_data]

    Vcostheta_Rrep_NO_self_contributions_rows_are_full_local_velocity_projection_sets_within_Rrep=sparse.coo_matrix((data_togoin_Vcostheta_Rrep_NO_self_contributions_rows_are_full_local_velocity_projection_sets_within_Rrep,(dist_Rrep.row[MASK_elimination_of_self_distances_from_dist_Rrep_data],dist_Rrep.col[MASK_elimination_of_self_distances_from_dist_Rrep_data])),shape=dist_Rrep.get_shape())

    local_sum_NO_self_contributions_velocity_pressure_on_each_particle_Rrep=np.squeeze(np.asarray(Vcostheta_Rrep_NO_self_contributions_rows_are_full_local_velocity_projection_sets_within_Rrep.tocsr().sum(axis=1)))


    ###Now compute the attractive pressure

    dist_Ratt=tree.sparse_distance_matrix(tree, max_distance=Ratt,output_type='coo_matrix')
    dist_in_att_zone=(dist_Ratt.tocsr()-dist_Rrep.tocsr()).tocoo()

    vx_expanded_over_cols_dist_in_att_zone=velocity_array_N2[dist_in_att_zone.col,0]
    vy_expanded_over_cols_dist_in_att_zone=velocity_array_N2[dist_in_att_zone.col,1]


    MASK_get_rid_of_self_contributions_to_local_sums_for_in_att_zone = dist_in_att_zone.row != dist_in_att_zone.col
    vx_in_att_zone_NO_self_contributions_rows_are_full_local_velocity_sets_within_in_att_zone = sparse.coo_matrix((vx_expanded_over_cols_dist_in_att_zone,(dist_in_att_zone.row,dist_in_att_zone.col)), shape=dist_in_att_zone.get_shape())


    vy_in_att_zone_NO_self_contributions_rows_are_full_local_velocity_sets_within_in_att_zone = sparse.coo_matrix((vy_expanded_over_cols_dist_in_att_zone,(dist_in_att_zone.row,dist_in_att_zone.col)), shape=dist_in_att_zone.get_shape())


    POSx_expanded_over_cols_dist_in_att_zone=pos[dist_in_att_zone.col,0]
    POSy_expanded_over_cols_dist_in_att_zone=pos[dist_in_att_zone.col,1]   


    POSx_in_att_zone_NO_self_contributions_rows_are_full_local_position_sets_within_in_att_zone = sparse.coo_matrix((POSx_expanded_over_cols_dist_in_att_zone,(dist_in_att_zone.row,dist_in_att_zone.col)), shape=dist_in_att_zone.get_shape())    
    POSy_in_att_zone_NO_self_contributions_rows_are_full_local_position_sets_within_in_att_zone = sparse.coo_matrix((POSy_expanded_over_cols_dist_in_att_zone,(dist_in_att_zone.row,dist_in_att_zone.col)), shape=dist_in_att_zone.get_shape())
    TRANSPOSED_POSx_in_att_zone_NO_self_contributions_rows_are_full_local_position_sets_within_in_att_zone=POSx_in_att_zone_NO_self_contributions_rows_are_full_local_position_sets_within_in_att_zone.T
    TRANSPOSED_POSy_in_att_zone_NO_self_contributions_rows_are_full_local_position_sets_within_in_att_zone=POSy_in_att_zone_NO_self_contributions_rows_are_full_local_position_sets_within_in_att_zone.T

    LOCAL_SETS_MATRIX_OF_RELATIVE_X_POSITIONS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_in_att_zone_csr=POSx_in_att_zone_NO_self_contributions_rows_are_full_local_position_sets_within_in_att_zone.tocsr()-TRANSPOSED_POSx_in_att_zone_NO_self_contributions_rows_are_full_local_position_sets_within_in_att_zone.tocsr()
    LOCAL_SETS_MATRIX_OF_RELATIVE_Y_POSITIONS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_in_att_zone_csr=POSy_in_att_zone_NO_self_contributions_rows_are_full_local_position_sets_within_in_att_zone.tocsr()-TRANSPOSED_POSy_in_att_zone_NO_self_contributions_rows_are_full_local_position_sets_within_in_att_zone.tocsr()

    POSx_times_vx=LOCAL_SETS_MATRIX_OF_RELATIVE_X_POSITIONS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_in_att_zone_csr.multiply(vx_in_att_zone_NO_self_contributions_rows_are_full_local_velocity_sets_within_in_att_zone.tocsr())
    POSy_times_vy=LOCAL_SETS_MATRIX_OF_RELATIVE_Y_POSITIONS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_in_att_zone_csr.multiply(vy_in_att_zone_NO_self_contributions_rows_are_full_local_velocity_sets_within_in_att_zone.tocsr())
    
    LOCAL_SETS_MATRIX_OF_RELPOS_AND_PROPV_DOT_PRODUCTS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_in_att_zone=POSx_times_vx+POSy_times_vy

    #THIS STEP IS POSSIBLY_REDUNDANT_NOW_SINCE_SPARSE_MATRIX_RELATIVE_DISTANCE_CALC_WOULD_HAVE_TAKEN_CARE_OF_THAT
    MASK_elimination_of_self_distances_from_dist_in_att_zone_data=dist_in_att_zone.tocsr().data!=0    
    
    data_togoin_Vcostheta_in_att_zone_NO_self_contributions_rows_are_full_local_velocity_projection_sets_within_in_att_zone=LOCAL_SETS_MATRIX_OF_RELPOS_AND_PROPV_DOT_PRODUCTS_OF_ALL_NEAREST_NEIGHBOURS_WITHIN_in_att_zone.data/dist_in_att_zone.tocsr().data[MASK_elimination_of_self_distances_from_dist_in_att_zone_data]

    Vcostheta_in_att_zone_NO_self_contributions_rows_are_full_local_velocity_projection_sets_within_in_att_zone=sparse.coo_matrix((data_togoin_Vcostheta_in_att_zone_NO_self_contributions_rows_are_full_local_velocity_projection_sets_within_in_att_zone,(dist_in_att_zone.row[MASK_elimination_of_self_distances_from_dist_in_att_zone_data],dist_in_att_zone.col[MASK_elimination_of_self_distances_from_dist_in_att_zone_data])),shape=dist_in_att_zone.get_shape())

    local_sum_NO_self_contributions_velocity_pressure_on_each_particle_in_att_zone=np.squeeze(np.asarray(Vcostheta_in_att_zone_NO_self_contributions_rows_are_full_local_velocity_projection_sets_within_in_att_zone.tocsr().sum(axis=1)))

    ###Now filter for where each particle has too many nearest neighbours

    MASK_nearest_neighbours_within_Ralign_ismorethan_Kalign=local_sum_NO_self_contributions_number_of_neighbours_within_Ralign>Kalign

    local_sum_NO_self_contributions_velocity_pressure_on_each_particle_in_att_zone=np.where(MASK_nearest_neighbours_within_Ralign_ismorethan_Kalign,0,local_sum_NO_self_contributions_velocity_pressure_on_each_particle_in_att_zone)

    #Ok this should save a ton of time actually, and will probably be better for hpc

    #Now store the updated states for the particles in state 2
    #We re use the tactile adjacency from before

    Visual_rep_adjacency=np.zeros(N)
    Visual_rep_adjacency=np.where(local_sum_NO_self_contributions_velocity_pressure_on_each_particle_Rrep<=-d1,1,0)
    Visual_att_adjacency=np.zeros(N)
    Visual_att_adjacency=np.where(local_sum_NO_self_contributions_velocity_pressure_on_each_particle_in_att_zone>=d3,1,0)

   #Now come the conditions

    #remove the stat definition because the redifinition of that variable makes each step inherently sequential

    MASK_particle_is_in_state_2_AND_HAS_INDEED_neighbours_in_tactile_radius=np.logical_and(new_states_2 == STATES[2],tactile_adjacency==1)
    MASK_particle_is_in_state_2_AND_HAS_NOT_neighbours_in_tactile_radius=np.logical_and(new_states_2 == STATES[2], tactile_adjacency==0)

    MASK_there_is_INDEED_enough_velocity_pressure_for_visually_stimulated_movement=np.logical_or(Visual_rep_adjacency==1,Visual_att_adjacency==1)
    MASK_there_is_NOT_enough_velocity_pressure_for_visually_stimulated_movement=np.logical_not(MASK_there_is_INDEED_enough_velocity_pressure_for_visually_stimulated_movement)

    MASK_there_can_be_only_spontaneous_movement=np.logical_and(MASK_particle_is_in_state_2_AND_HAS_NOT_neighbours_in_tactile_radius,MASK_there_is_NOT_enough_velocity_pressure_for_visually_stimulated_movement)
    MASK_there_can_also_be_tactile_stimulated_movement=np.logical_and(MASK_particle_is_in_state_2_AND_HAS_INDEED_neighbours_in_tactile_radius,MASK_there_is_NOT_enough_velocity_pressure_for_visually_stimulated_movement)
    MASK_there_can_also_be_visual_stimulated_movement=np.logical_and(MASK_particle_is_in_state_2_AND_HAS_NOT_neighbours_in_tactile_radius,MASK_there_is_INDEED_enough_velocity_pressure_for_visually_stimulated_movement)
    MASK_there_can_be_both_tactile_stimulated_movement_AND_visual_stimulated_movement=np.logical_and(MASK_particle_is_in_state_2_AND_HAS_INDEED_neighbours_in_tactile_radius,MASK_there_is_INDEED_enough_velocity_pressure_for_visually_stimulated_movement)




    states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state2_that_is_NEITHER_in_tactile_contact_with_a_neighbour_NOR_under_sufficient_pressure=np.random.choice(STATES,size= MASK_there_can_be_only_spontaneous_movement.sum(),p=s_S)
    npwhere_array_with_states_dims_to_place_generated_states_for_2andS_particles_at_their_indices=np.zeros_like(new_states_2, dtype=float)
    npwhere_array_with_states_dims_to_place_generated_states_for_2andS_particles_at_their_indices[ MASK_there_can_be_only_spontaneous_movement]=states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state2_that_is_NEITHER_in_tactile_contact_with_a_neighbour_NOR_under_sufficient_pressure

    
    states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state2_that_IS_INDEED_in_tactile_contact_BUT_NOT_under_sufficient_pressure=np.random.choice(STATES,size= MASK_there_can_also_be_tactile_stimulated_movement.sum(),p=s_T)
    npwhere_array_with_states_dims_to_place_generated_states_for_2andT_particles_at_their_indices=np.zeros_like(new_states_2, dtype=float)
    npwhere_array_with_states_dims_to_place_generated_states_for_2andT_particles_at_their_indices[ MASK_there_can_also_be_tactile_stimulated_movement]=states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state2_that_IS_INDEED_in_tactile_contact_BUT_NOT_under_sufficient_pressure

    states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state2_that_IS_NOT_in_tactile_contact_BUT_IS_INDEED_under_sufficient_pressure=np.random.choice(STATES,size= MASK_there_can_also_be_visual_stimulated_movement.sum(),p=s_V)
    npwhere_array_with_states_dims_to_place_generated_states_for_2andV_particles_at_their_indices=np.zeros_like(new_states_2, dtype=float)
    npwhere_array_with_states_dims_to_place_generated_states_for_2andV_particles_at_their_indices[MASK_there_can_also_be_visual_stimulated_movement]=states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state2_that_IS_NOT_in_tactile_contact_BUT_IS_INDEED_under_sufficient_pressure

    
    states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state2_that_is_BOTH_in_tactile_contact_with_a_neighbour_AND_under_sufficient_pressure=np.random.choice(STATES,size= MASK_there_can_be_both_tactile_stimulated_movement_AND_visual_stimulated_movement.sum(),p=s_TV)
    npwhere_array_with_states_dims_to_place_generated_states_for_2andTV_particles_at_their_indices=np.zeros_like(new_states_2, dtype=float)
    npwhere_array_with_states_dims_to_place_generated_states_for_2andTV_particles_at_their_indices[ MASK_there_can_be_both_tactile_stimulated_movement_AND_visual_stimulated_movement]=states_obtained_from_rolling_for_a_new_state_for_each_particle_in_state2_that_is_BOTH_in_tactile_contact_with_a_neighbour_AND_under_sufficient_pressure



    new_states_2=np.where(MASK_there_can_be_only_spontaneous_movement ,npwhere_array_with_states_dims_to_place_generated_states_for_2andS_particles_at_their_indices,new_states_2)
    new_states_2=np.where(MASK_there_can_also_be_tactile_stimulated_movement ,npwhere_array_with_states_dims_to_place_generated_states_for_2andT_particles_at_their_indices,new_states_2)
    new_states_2=np.where(MASK_there_can_also_be_visual_stimulated_movement,npwhere_array_with_states_dims_to_place_generated_states_for_2andV_particles_at_their_indices,new_states_2)
    new_states_2=np.where(MASK_there_can_be_both_tactile_stimulated_movement_AND_visual_stimulated_movement ,npwhere_array_with_states_dims_to_place_generated_states_for_2andTV_particles_at_their_indices,new_states_2)


    ###Now update the particles in state 1
    ###we will not recalculate the order parameter

    #NO_self_contributions_local_order_parameter_within_Ralign_normalised
    new_states_1=np.copy(states)

    local_dot_product_between_self_orientation_and_local_order_parameter_within_Ralign_normalised=cos*local_sum_NO_self_contributions_total_x_velocity_in_vicinity_of_each_particle_within_Ralign_including_only_moving+sin*    local_sum_NO_self_contributions_total_y_velocity_in_vicinity_of_each_particle_within_Ralign_including_only_moving
    MASK_self_is_pointed_SAME_WAY_as_local_average_direction=local_dot_product_between_self_orientation_and_local_order_parameter_within_Ralign_normalised>=0
    MASK_self_is_pointed_OPPOSITE_WAY_as_local_average_direction=local_dot_product_between_self_orientation_and_local_order_parameter_within_Ralign_normalised<0

    local_turn_probabilities_calculated_from_piecewise_ariel_function_for_all_paticles=np.zeros(N)
    local_turn_probabilities_calculated_from_piecewise_ariel_function_for_all_paticles[MASK_self_is_pointed_SAME_WAY_as_local_average_direction]=alpha+betaA*NO_self_contributions_local_order_parameter_within_Ralign_normalised[MASK_self_is_pointed_SAME_WAY_as_local_average_direction]
    local_turn_probabilities_calculated_from_piecewise_ariel_function_for_all_paticles[MASK_self_is_pointed_OPPOSITE_WAY_as_local_average_direction]=alpha+betaW*NO_self_contributions_local_order_parameter_within_Ralign_normalised[MASK_self_is_pointed_OPPOSITE_WAY_as_local_average_direction]
    

    MASK_particle_is_in_state_1=(new_states_1 == STATES[1])
    random_values_to_to_compare_with_custom_distribution_of_turn_probabilities = np.random.rand(N)
    MASK_particle_in_state_1_HAS_INDEED_TURNED = np.logical_and((MASK_particle_is_in_state_1) , (random_values_to_to_compare_with_custom_distribution_of_turn_probabilities <= local_turn_probabilities_calculated_from_piecewise_ariel_function_for_all_paticles))
    MASK_particle_in_state_1_HAS_NOT_TURNED=np.logical_and((MASK_particle_is_in_state_1) , (random_values_to_to_compare_with_custom_distribution_of_turn_probabilities > local_turn_probabilities_calculated_from_piecewise_ariel_function_for_all_paticles))
    new_states_1=np.where(MASK_particle_in_state_1_HAS_INDEED_TURNED,3,new_states_1)
    new_states_1=np.where(MASK_particle_in_state_1_HAS_NOT_TURNED,0,new_states_1)


    #update the particles in state 3 that have turned
    MASK_particle_is_in_state_3=new_states_3==STATES[3]
    new_states_3=np.where(MASK_particle_is_in_state_3,0,new_states_3)


    #Now combine all the updates together to produce the updated state vector


    NSC0=states==STATES[0]
    NSC2=states==STATES[2]
    NSC1=states==STATES[1]
    NSC3=states==STATES[3]

    new_states=np.zeros(N)


    new_states=np.where(NSC0,new_states_0,new_states)
    new_states=np.where(NSC2,new_states_2,new_states)
    new_states=np.where(NSC1,new_states_1,new_states)
    new_states=np.where(NSC3,new_states_3,new_states)

    states=new_states


    
    e=time.time()

    print(e-s)

    ##print(orient)

    return qv,


anim = FuncAnimation(fig,animate,np.arange(1, 200),interval=1, blit=True)
plt.show()











