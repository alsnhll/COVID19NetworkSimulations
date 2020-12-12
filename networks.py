""" Functions to create the various layers of the network """
import numpy as np2
import math
import itertools
import scipy.stats as ss

def create_fully_connected(dist_groups, indices):
    """ Divide the subset of the total population as given by the indices into fully connected groups 
    depending upon their distribution of sizes.
    @param dist_groups : Sizes of the groups in the population
    @type : list or 1D array
    @param indices : Indices of the subset of the population to be grouped together
    @type : list or 1D array
    @return : Adjacency matrix in sparse format [rows, cols, data]
    @type : list of lists
    """
    rows = []
    cols = []
    data = []
    current_indx = 0
    for size in dist_groups:
        group = indices[int(current_indx):int(current_indx+size)]
        current_indx += size
        comb = list(itertools.combinations(group,2))
        for i,j in comb:
            rows.extend([i,j])
            cols.extend([j,i])
            data.extend([1,1])
    return [rows, cols, data]
    
def create_external(pop,degree_dist):
    """ Create random external connections for the whole population given a degree distribution
    @param pop : Total population size
    @type : int
    @param degree_dist : Degree distribution for this layer
    @type : list or 1D array
    @return : Sparse adjacency matrix 
    @type : List of lists [rows, cols, data]
    """
    # Create external stubs that are randomly connected 
    rows = []
    cols = []
    data = []
    external_stubs = []
    
    for i in range(pop):
        stubs = degree_dist[i]
        external_stubs.extend([i for j in range(stubs)])
    
    # Fix the seed
    np2.random.seed(789)
    # Attach the random stubs        
    external_pairs = np2.random.choice(external_stubs, size = (int(len(external_stubs)/2),2), replace = False)
    for pairs in range(len(external_pairs)):
        i = external_pairs[pairs][0]
        j = external_pairs[pairs][1] 
        rows.extend([i,j])
        cols.extend([j,i])
        data.extend([1,1])

    return [rows, cols, data]
    
def create_external_corr(pop,pop_subset,degree_dist,n,r,indx_list,correlation_group):
    """ Create correlated external connections for either the whole population or a subset
    @param pop : Total population size
    @type : int
    @param pop_subset : Subset of the population involved in these external layers
    @type : int
    @param degree_dist : Degree distribution for this layer
    @type : list or 1D array
    @param n : Number of equal sized quantiles the correlated connections are divided into
    @type : int
    @param r : Amount of positive correlation between members of the same quantile
    @type : float
    @param indx_list : Array of indices of the individuals to be connected in the layer
    @type : list or 1D array
    @param correlation_group : Array of traits that are used to preferentially connect individuals
    @type : 1D array
    @return : Sparse adjacency matrix 
    @type : List of lists [rows, cols, data]
    """
    # Assign random and correlated stubs for each individual
    correlation = []
    np2.random.seed(789)
    for i in range(pop_subset):
        correlation.append(np2.random.binomial(1, r, size = degree_dist[i]))
    # Create external stubs that are randomly connected and the ones that are correlated for age groups
    rows = []
    cols = []
    data = []
    zero_stubs = []
    one_stubs = {}
    
    for i in range(pop_subset):
        ones = np2.count_nonzero(correlation[i])
        zeros = degree_dist[i] - ones
        zero_stubs.extend([indx_list[i] for j in range(zeros)])
        if ones != 0:
            one_stubs[(indx_list[i],ones)] = correlation_group[i]
      
    # Attach the random stubs        
    zero_pairs = np2.random.choice(zero_stubs, size = (int(len(zero_stubs)/2),2), replace = False)
    for pairs in range(len(zero_pairs)):
        i = zero_pairs[pairs][0]
        j = zero_pairs[pairs][1] 
        rows.extend([i,j])
        cols.extend([j,i])
        data.extend([1,1])

        
    if r > 0:
        # Order correlated stubs according to trait to be correlated
        ordered_ones = sorted(one_stubs, key=one_stubs.__getitem__)
        sorted_ones = []
        for pairs in range(len(ordered_ones)):
            index = ordered_ones[pairs][0]
            sorted_ones.extend([index for i in range(ordered_ones[pairs][1])])
    
        # Divide into n_school number of equal sized quantiles
        n_q = math.ceil(len(sorted_ones)/n)
        n_quantiles = [sorted_ones[i:i + n_q] for i in range(0, len(sorted_ones), n_q)]

        # Attach the correlated nodes
        for quantile in range(len(n_quantiles)):
            one_pairs = np2.random.choice(n_quantiles[quantile], size = (int(len(n_quantiles[quantile])/2),2), replace = False)
            for pairs in range(len(one_pairs)):
                i = one_pairs[pairs][0]
                j = one_pairs[pairs][1]
                rows.extend([i,j])
                cols.extend([j,i])
                data.extend([1,1])
    return [rows, cols, data]

def create_external_different_mixing(pop_subset, p_mixing, degree_dist):
    """ Create external connections where each person has an average probability of interaction with other people in the population depending upon their neighborhood/cluster.
    @param pop_subset: Contains the individual indices of people belonging to each neighborhood
    @type pop_subset: List of n lists where n is the number of neighborhoods
    @param p_mixing: Average probability of having external contacts in each neighborhood
    @type p_mixing: List of n lists where n is the number of neighborhoods
    @param degree_dist: Contains the number of external connections for each individual
    @type degree_dist: List of n lists where n is the number of neighborhoods
    @return: Sparse adjacency matrix
    @type: List of lists [rows, cols, data]
    """
  
    no_neigh = len(pop_subset)
    correlation = [[] for i in range(no_neigh)]
    rows = []
    cols = []
    data = []
  
    # For each neighborhood in cyclic order for eg. 1-1, 1-2, 1-3, 2-1, 2-2, 2-3, 3-1, 3-2, 3-3
    stub_types = [[[],[],[]] for i in range(no_neigh)]

    np2.random.seed(789)
    for neigh in range(no_neigh):
      for i in range(len(pop_subset[neigh])):
       # For eg: 0 - stays in the same neigh, 1 - goes to the next in cyclic order etc
        stubs = np2.random.choice([j for j in range(no_neigh)], p=p_mixing[neigh], size = int(degree_dist[neigh][i]))
        stub_indx, stub_counts = np2.unique(stubs, return_counts=True)
        for unique in range(len(stub_counts)):
          stub_types[neigh][stub_indx[unique]].extend([pop_subset[neigh][i] for j in range(stub_counts[unique])])

    # Within cluster connections
    for neigh in range(no_neigh):
      within = stub_types[neigh][neigh]
      within_pairs = np2.random.choice(within, size = (int(len(within)/2),2), replace = False)
      for pairs in range(len(within_pairs)):
        i = within_pairs[pairs][0]
        j = within_pairs[pairs][1] 
        rows.extend([i,j])
        cols.extend([j,i])
        data.extend([1,1])

    # Outside cluster connections 
    diff_cluster_connections = list(itertools.combinations([j for j in range(no_neigh)],2))
    for neigh1, neigh2 in diff_cluster_connections:
      stub1 = stub_types[neigh1][neigh2]
      stub2 = stub_types[neigh2][neigh1]
      stub1.extend(stub2)
      between_neigh = sorted(stub1)
      n_q = math.ceil(len(between_neigh)/2)
      n_quantiles = [between_neigh[i:i + n_q] for i in range(0, len(between_neigh), n_q)]

      for quantile in range(math.ceil(len(n_quantiles)/2)):
        pair_one = np2.random.choice(n_quantiles[quantile], size = len(n_quantiles[quantile]), replace = False)
        pair_two = np2.random.choice(n_quantiles[-1-quantile], size = len(n_quantiles[-1-quantile]), replace = False)
        if len(pair_one) > len(pair_two):
          no_pairs = len(pair_two)
        else:
          no_pairs = len(pair_one)
        for pairs in range(no_pairs):
          i = pair_one[pairs]
          j = pair_two[pairs]
          rows.extend([i,j])
          cols.extend([j,i])
          data.extend([1,1])

    return [rows,cols,data]
    
def create_friend_groups(para,age_grp_size,indices):
    """ Create age dependent distributions of sizes of friend groups and assign individuals to them
    @param para : List of parameters for the negative binomial distribution [n,p]
    @type : list
    @param age_grp_size : Number of individuals in an age group
    @type : int
    @param indices : Indices of the subset of the population to be grouped together
    @type : list or 1D array
    @return : Sparse adjacency matrix per age group
    @type : List of lists [rows, cols, data]
    
    """
    group_sizes = []
    pop_group = 0
    n = para[0]
    p = para[1]
    
    np2.random.seed(789)
    while pop_group <= age_grp_size:
        size = np2.random.negative_binomial(n,p, size=1)
        group_sizes.append(size)
        pop_group += size

    group_sizes[-1] -= pop_group-age_grp_size
    sparse_matrix = create_fully_connected(group_sizes,indices)
    return sparse_matrix
