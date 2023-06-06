# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:19:50 2021

@author: Zach


Class definition for a Firm-Worker Bipartite network (based on networkx graphs)

Note that currently this simply wraps a networkx Graph and provides some 
convenience methods for updating it in the context of the laborFlow package. 

In the future, this may just inherit the networkx object and update it locally
Or it may implement a different package entirely for large networks where 
networkx gets slow

"""

## Import Necessary Dependencies
from debugpy import connect
import igraph as ig
import numpy as np
import random as rd
from collections import Counter


class laborNetwork():
    """
    The laborNetwork class wraps a networkx Bipartite Graph and provides 
    methods for quickly creating/updating it in the context of the laborFlow 
    package
    
    Currently, the laborNetwork supports only a max of two Firm edges for each 
    Worker. This is based on the Guererro, Lopez, and Axtel paper "Labor FLow 
    Networks"
    """
    
    def __init__(self,
                 name  = None # Optionally, give the network a name (this is convenient for separate sectors)
                 ):
        """
        Class Constructor for a Labor Network
        
        
        Inputs:
        -------
            name: An optional name for the network (i.e. the sector)
        
        """
        
        if name != None:
            
            self.name = name
        
        else:

            self.name = None
            

        ## Placeholder for the bipartite graph
        self.graph = ig.Graph()
                
            
    
    def initializeRandomly(self,
                           labor, # Workers to be included in the network
                           firms  # Firms to be included in the network
                           # alpha = .95 # Initial proportion of workers with a current firm edge 
                           ):
        """
        Randomly Initialize a Network. Sets the previous firm-worker edge for 
        all workers. Note that the current firm-worker edge (not set here) may
        be "None"


        Parameters
        ----------
        labor : list of laborFlow.core.Labor.Worker objects
            Worker objects in a single sector of the Economy already 
            instantiated
        firms : list of strings
            Names/IDs of Firms in a single sector of the Economy already 
            to be included in the Network.

        Returns
        -------
        None. Graph is modified in place

        """
        
        """ ## Create an inital network
        self.graph.add_nodes_from(labor, bipartite = "Labor")
        self.graph.add_nodes_from(firms, bipartite = "Firm")
        
        
        ## Add previous/current edges to appropriate firms and set previous employer for respective worker
        old_edge = []
        new_edge = []
        for w in labor:
            
            ## If the Worker is currently employed, add that to the "new" edge list
            if w.firm != None:
                
                new_edge = new_edge + [[w.ID, w.firm]]
                
            
            ## Remove the current firm from the previous firm list
            tmp_firm_old = np.setdiff1d(firms, labor.firm)
            
            
            ## This should only ever be an issue in a 1 firm model...which would probably break everything anyway
            if len(tmp_firm_old) == 0:
                
                next
                
            # Get the previous firm
            prev_firm = str(np.random.choice(tmp_firm_old, 1)[0]) # Annoying, but a string is necessary here
            w.setPreviousFirm(prev_firm)                          # Set the chosen firm as the previous one for the worker 
                
            
            ## Create an inital bipartite network (with all Workers and all firms) and set the appropriate firm/worker edges
            old_edge = old_edge + [[w.ID, prev_firm]]
        
        
        ## Add the edges
        self.graph.add_edges_from(old_edge) # Previous employer
        self.graph.add_edges_from(new_edge) # Current employer """

        pass
    


    def addNodes(self, 
                 node_names, # List of the names of nodes to be added 
                 node_types  # List of the corresponding node type
                 ):
        """
        Add nodes to the graph. Note that because this is a bipartite graph, the type must be specified.
        Currently the only allowable specifications are "firm" or "worker"

        Parameters
        ----------
        node_names : List
            List of Worker (Labor Object) and Firm IDs to be added to the graph. Note that no 
            checks are made against the IDs aside from the ones implemented by python-igraph
        
        node_types : List 
            List consisting of "worker" and "firm" corresponding to the node_names

        Returns
        -------
        None. Object is modified in place
        """

        #### Input Checking ####

        ## Ensure that the two lists are the same length
        if len(node_names) != len(node_types):

            raise Exception("node_names must be the same length as node_types")


        ## Ensure that node types are either in "worker" or "firm"
        node_types = [ii.lower() for ii in node_types]
        type_bool = [1 if n == 'firm' else 0 for n in node_types if n in ['firm', 'worker']]
        if len(type_bool) != len(node_types):

            raise Exception("node_types must be either 'firm' or 'worker'")             



        #### Begin Function ####

        ## Add the new nodes, set their types
        self.graph.add_vertices(node_names, attributes={'type': type_bool, 'typeID': node_types})



    def removeNodes(self,
                    node_names # List of names that should be removed
                    ):
        """
        Remove nodes from the graph. 

        Parameters
        ----------
        node_names : List
            List of Worker (Labor Object) and Firm IDs to be added to the graph. Note that no 
            checks are made against the IDs aside from the ones implemented by python-igraph

        Returns
        -------
        None. Object is modified in place
        """

        #### Input Checking ####


        #### Begin Function ####
                
        self.graph.delete_vertices(node_names)
    

    
    def updateEdges(self,
                    rem_edges = None, # A list of tuples of edges to be added
                    add_edges = None  # A list of tuples of edges to be removed
                    ):
        """
        Update the network edges based on workers that have possibly changed jobs
        
        For speed, no additional error handling beyond the python-igraph functionality 
        is added here


        Parameters
        ----------
        rem_edges : list, tuples (None)
            A list of tuples representing current edges to be removed. These must already
            be present in the graph (if any are passed)
        
        add_edges : list, tuples (None)
            A list of tuples representing edges to be added


        Returns
        -------
        
        None. Graph is modified in place
        """
        
        
        ## If requested, remove edges
        if rem_edges != None:
            
            # eids = self.graph.get_eids(rem_edges, error = False)
            # eids = [eid for eid in eids if eid != -1]            # This could cause a performance issue if many eids are being removed...
            eids = self.graph.get_eids(rem_edges, error = False)
            for ii, eid in enumerate(eids):
                if eid == -1:
                    print(rem_edges[ii])
            self.graph.delete_edges(eids)
            
        
        ## If requested, add edges
        if add_edges != None:
            
            self.graph.add_edges(add_edges) # This should replace below since the inputs should be all updated to lists...
            # if len(add_edges) == 1:

                # ig.Graph.add_edges defaults to different behaviour if you try to pass a list than a list of tuples, 
                # which is what happens with the way I've strucuted the code here....
                # self.graph.add_edge(add_edges[0][0], add_edges[1])

            # else:

            #    self.graph.add_edges(add_edges)
               
       
        
       
    def getConnectedFirms(self,
                          firms,               # The firm name focus
                          firm_excl   = None,  # An optional list of firm_names to exclude from the returned connected firms
                          prob_subset = False, # A flag to subset the list of returned firms by probability
                          prob_val    = .8     # The percentage of firms that the list should be reduced to (i.e. 10 to 8)
                          ):
        """
        Project the Firm-Firm edges from the bipartite graph using Workers 
        as the connections for a given firm.
        
        Optionally, randomly subset the list of returned firms to some percentage 
        of the identified connected firms. This is applied as a weighted random 
        sampling, where the weights are the weights of the edges in the bipartite 
        projection 

        
        Parameters:
        ----------
        
        firms: list, str
            A list of the name(s) of the firms of focus. Firms connected to this firm will be 
            returned
        
        firm_excl: list, str
            An optional list of firm names that should be excluded from the output

        prob_subset : boolean
            A flag to subset the list of returned firms by random choice, weighted by the
            edge weights of the firm-firm projection

        prob_val : float
            The percentage of firms that the list should be reduced to (i.e. 10 to 8)
            Expressed as a decimal between 0 and 1
        
        Returns:
        -------
        
        list of linked firms
        """
        
        #### Input Checking ####



        #### Begin Function ####
            
        
        ## Project the bipartite graph and get the connected neighbors of that firm
        B = self.projectFirmGraph()
        all_connected_firms = []
        for firm in firms:
            vids = B.neighbors(firm)
            connected_firms = B.vs[vids]['name']
            
            ## Exclude any firms, if requested
            if firm_excl != None:

                connected_firms = [frm for frm in connected_firms if connected_firms not in firm_excl]


            ## If requested, subset the connected firms 
            if prob_subset:

                ## Get the edge weights for each connected firm
                wts = B.es.select(_source = firm, _target = connected_firms)['weight']
                

                ## get the ceiling of the number of items to choose from
                n = np.ceil(len(connected_firms)*prob_val)

                ## Subset the connected firms
                if n < len(connected_firms):

                    connected_firms = np.random.choice(connected_firms, size = n, replace = False, p = wts).tolist()           
            
            all_connected_firms.extend(connected_firms) # extend instead of append here because we want a flat list for np.unique() below


        ## Only keep the unique firms among the connections
        unique_connected_firms = np.unique(all_connected_firms).tolist()
        

        return unique_connected_firms
           
    
    
    def projectFirmGraph(self):
        """
        Return a bipartite projection of the firm-firm graph from the full 
        firm-worker graph.
        
        Useful for point-in-time network visualizations
        
        
        Parameters:
        ----------
        
        None.
        
        Returns:
        -------
        
        The igraph.Graph projection of the firm-firm network
        """
        
        
        ## Project the bipartite graph for some subset of firms
        B = self.graph.bipartite_projection(which = 1) # Note the 1 is returned because we set 'firm' as 1 when we add nodes...
        
        
        return B
    
    
    
    def visualizeFirmGraph(self):
        """
        placeholder for visualizing the firm-firm projection of the graph 
        
        """
        
        
        pass
       

    def saveGraphML(self,
                    save_name,         # Full path and name of the graph to be saved
                    save_full = False, # By default, save only the projected Firm-Firm graph. Set to True to save the whole graph
                    compresslevel = 1  # 1 is the least compressed and fastest. 9 is the most compressed and slowest
                    ):
        """
        Save either the full or the projected graph to graphML format

        Parameters
        ----------
        save_name : string
            The filepath and filename where the graph should be saved to

        save_full : boolean
            By default, the saved graph will be the projected Firm-Firm graph. Set this to 
            True to save the full graph


        Returns
        -------
        None        
        """


        if not save_full:

            g = self.projectFirmGraph()

        else:

            g = self.graph

        g.write_graphmlz(save_name, compresslevel = compresslevel)

    
    
    def recordStats(self,
                    firm_names # A list of firm names (or a subset) within the firm (typically by sector)
                    ):
        """
        Record the network statistics of the bipartite projection of the Firm-Firm
        network
        

        Returns
        -------
        stats: dict
            A dictionary of Node (firm)-specific statistics
            Can be merged with the output of Firms.recordStats() 
            
        stats_smry: dict
            A dictionary of Network Summary measures 
        """
                
        ## Project the Firm-Firm network
        B = self.projectFirmGraph()

        ## Note here that I may be paying a bit of a performance penalty by forcing the user to supply the firm_names explicitly
        ## That said, firms may come and go during a simulation, so unless the performance becomes incredibly burdensome, I think
        ## I'm willing to pay the price


        ## Get the node indices of interest, matching the order of the firm names
        tmp = B.vs['name']
        vids = [ii for n in firm_names for ii, v_n in enumerate(tmp) if v_n == n]        
        
        ## Calculate v-shapes
        ## This is part of the Clustering coefficient, but for some visualizations its nice to use it 
        degree = B.degree(vids)
        v_shapes = [(ii * (ii - 1 )) / 2 for ii in degree]
        
        all_triangles = B.cliques(min = 3, max = 3)                                           # Find all unique triangles (returns a list of tuples)
        # triangles = [Counter(vid for t in all_triangles)[vid] for vid in vids]              # This seems stupidly inefficient...and wrong
        triangles = [Counter([vid for t in all_triangles if vid in t])[vid] for vid in vids]  # Right (I think) but still not super efficient
        
        ## For whatever reason, eigenvector centrality and betweenness don't have the option to subset by vertice index. So we have to do it manually...
        eigs_all = B.eigenvector_centrality(directed = False)
        eig = [eigs_all[v] for v in vids]
        
        between_all = B.betweenness(directed = False) 
        between = [between_all[v] for v in vids]

        ## Clustering
        clust = [triangles[ii] / v_shapes[ii] if v_shapes[ii] > 0 else 0 for ii in range(len(v_shapes))]
        

        ## Get Firm/Node statistics
        stats = {'Firm Name': dict(zip(firm_names, firm_names)), 
                 'Sector': self.name,                                                 # In theory this is the sector. If this is standalone it might not be, but since it isn't really meant for that, keeping it like this
                 'Node Triangles': dict(zip(firm_names, triangles)),
                 'Node Degree': dict(zip(firm_names, degree)),
                 'Node V-Shapes': dict(zip(firm_names, v_shapes)),
                 'Clustering': dict(zip(firm_names, clust)),                           # Note this is equivalent to triangles / v-shapes
                 'PageRank': dict(zip(firm_names, B.pagerank(vids, directed = False))),
                 'Degree Centrality': dict(zip(firm_names, (np.array(degree) / (len(B.vs) - 1)).tolist())), 
                 'Eigenvector Centrality': dict(zip(firm_names, eig)),                 
                 'Betweenness Centrality': dict(zip(firm_names, between))                                     # can also included weights=[] argument...not sure if that is necessary at this point or not
                 }
        
        
        ## Calculate Global Clusering
        denom = sum(v_shapes)
        if denom > 0:
            gc = sum(triangles) / denom
        else:
            gc = 0
        
        ## Get Network summary statistics
        stats_smry = {'Total Nodes': B.vcount(),
                      'Total Edges': B.ecount(),
                      # 'Total Self-loops': nx.number_of_selfloops(B), # This should be zero...
                      'Density': B.density(),
                      'Global Clustering': gc,
                      'Sector': self.name
                      }
        
        
        return stats, stats_smry
    
    