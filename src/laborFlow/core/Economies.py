# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 08:01:08 2021

@author: Zach

Top-level functions for instantiating a world and simulating an Agent-Based 
Model implementation of the Moretti (2010) "Local Labor Markets" which is a 
general equillibrium approach to understanding the impacts of wage on labor 
flows with agent utility a function of wage, rent, local ammenity, and 
idiosynratic preferences. 

"""


## Import libraries here
# from laborFlows import Labor.Worker, Firms.Firm, Locations.Location # Import model base functionality
import random as rd # This is nice for some of the n choose k type uses, since similar numpy modules want arrays
import string as st

import itertools as itertools

import numpy as np
import pandas as pd

import sys
from math import log

import tqdm


## Import Classes defined in other modules within this package
from laborFlow.core.Locations import Location
from laborFlow.core.Firms import Firm
from laborFlow.core.Labor import Worker
from laborFlow.core.Networks import laborNetwork


## Import necessary statistics helper functions
from laborFlow.utils.statistics import statsToDataFrame


"""
Define the top level "Economy" class that instantiates a world and provides 
lists (of pointers) to Locations, Firms, and Workers
"""
class Economy():
    
    def __init__(self):
        """ 
        World class constructor method
        """
        
        self.Population = {} # Store the population of worker objects here
        self.Locations = {}  # Store the locations here
        self.Firms = {}      # Store the firm objects here

        """
        Note that here we should really have all of the connections/locations in a simple list (or dataframe)
        This will make our agents (firms and workers) a little bit more light-weight
        
        Hmmm.. maybe networkx is a good place to start...
        """
        
        self.Network_FirmWorker = {}  # A dictionary of laborFlows.Network objects by sector (bipartite graphs)
        
        
        self.sector_firm_hash = {}  # A quick look-up table to store Firms belonging to each sector in the model 
        self.sector_labor_hash = {} # A look-up dictionary of workers/labor per sector
        


    def addLocation(self, 
                    housing_elasticity,         # How elastic is the local housing supply? This value is greater than or equal to 0 (0 is perfectly elastic)
                    local_ammenities,           # A measure of local ammenities - for now this is static, and applies to all agents within the model (which may be ok - we capture idiosyncratic preferences already, which may be enough)
                    productivity_shifter,       # This is dictionary of productivity shifters for each industry sector in the model
                    location_ID           = [], # Optionally, provide a locality Name. If one is not provided, one will be created
                    gamma                 = 0   # Provide an optional agglomeration strength
                    ):
        """
        Add a location to the model with location-specific parameters
        
        Note that this simply describes a given city/rural area. Use this method to 
        iteratively add as many locals as requested for the given model. 
        
        Currently localities are static - once the parameters have been added, they aren't ammended
        This may be a useful feature to add in the future, particularly if localities get their 
        ammenities from some local tax base/policy choices...
        """
        
        ## If a location ID was not provided, create one
        if location_ID == []:
            
            ## Just randomly creates an 8-digit alpha-numeric string
            location_ID = ''.join((rd.choice(st.ascii_letters + st.digits) for ii in range(8)))
        
        
        ## Make sure the location ID is unique (to do. For now let's live dangerously)
        # if any([id_check for lc in ])
        
        self.Locations[location_ID] = Location(housing_elasticity, local_ammenities, productivity_shifter, location_ID, gamma = gamma)




    def addAllLocations(self):
        """
        Placeholder - a conveinence function for generating all locations
        """
        
        
        pass



    def addFirm(self, 
                name,                    # Provide a name for the firm
                sector,                  # Provide a name of the industry sector
                location,                # Pass a location name to assign the firm to initially
                productivity_shift = 0,  # This adds (or subtracts) from the city-specific productivity shifter 
                remote_pref        = 0   # Set some preference for workers working remotely (0 for none, 1 for max, or some value in between)
                ):
        """
        Add a firm individually 
        
        Appends to the Firms list of the Economy class
        
        
        Eventually add capital here...
        """
        
        ## Add the Firm object to the Firms list
        self.Firms[name] = Firm(sector = sector, 
                                name = name, 
                                location = location, 
                                productivity_shift = productivity_shift, 
                                remote_pref = remote_pref)
        
        
        ## Set a pointer in the appropriate location
        self.Locations[location].updateFirm(self.Firms[name], by = 'add')
        

        ## Update the firm lookup table for quick reference
        self.sector_firm_hash[sector].append(name)

        ## Add the node to the appropriate graph
        self.Network_FirmWorker[sector].addNodes(node_names = [name], node_types = ['firm'])



    def addAllFirms(self, 
                    n_firms,            # Total number of firms to be distributed
                    loc_pr,             # Provide probability for each location for distribution (must sum to 1) in dictionary form i.e. {'loc1': .2, 'loc2', .8}
                    sec_pr,             # Provide the proportion of firms that are in each sector in dictionary form(these will be randomly distributed among locations)
                    remote_prefs = None # Provide some random remote preferences by sector for each firm (0 is none, 1 is max). Firms will have individual prefereneces
                    ):
        """
        Convenience wrapper to create some number of firms and deal them to 
        locations (that have already been built)
        
        To Do:
            
            - Add a dictionary input and code to instantiate firms by sector 
              and location
            - Add the ability to pass in a firm-specific productivity 
              shifter...   
    
        Returns
        -------
        None.
        
        
        Eventually add capital here...    
        """
        
        #### Input Checking ####
        
        ## Ensure that Locations exist and that there are an equal number of locations as the size of the loc_pr
        if len(self.Locations) != len(loc_pr):
            
            raise Exception("Number of locations does not equal number of proportions passed in")
            
        elif sum(loc_pr.values()) != 1:
            
            raise Exception("Sum of location proportions must be equal to 1")
        
        
        ## Esnure that the sector inputs are suitable
        if sum(sec_pr.values()) != 1:
            
            raise Exception("Sum of sector proportions must be equal to 1")
        
        
        
        #### Begin Function ####
        
        ## Housekeeping - instantiate all appropriate sector hashes for labor and firms, as well as the sector firm network
        for sector in sec_pr.keys():
            
            self.sector_firm_hash[sector] = []
            self.sector_labor_hash[sector] = []
            self.Network_FirmWorker[sector] = laborNetwork(name = sector)
        
        
        ## Rounding is possibly an issue based on inputs. Set Update this and make sure location and sector add to the same value
        
        
        ## Multiply the proportions by the number of firms, with appropriate rounding. 
        loc_pr.update((key, ii * n_firms) for key, ii in loc_pr.items())
        sec_pr.update((key, ii * n_firms) for key, ii in sec_pr.items())
        
        
        ## Need these in the event that rounding goes a little off-kilter
        sec_keys = [key for key in sec_pr.keys()]
        
        
        ## Now create a list of the names (and flatten it, since the list comprehension creats a list of lists...)
        loc_pr = list(itertools.chain.from_iterable([[key] * int(ii) for key, ii in loc_pr.items()]))
        sec_pr = list(itertools.chain.from_iterable([[key] * int(ii) for key, ii in sec_pr.items()]))
        
        ## Sometimes the sec_pr will end up too short or too long by 1. Too long is fine, too short is a problem
        ## This will occurr if the int(ii) rounds to .5 exactly and the numbers happen to both round down 
        ## In this case, just randomly select another key to fill out the length (but only if it is exactly less than 1)
        if (n_firms - len(sec_pr)) == 1: 
            
            sec_pr.append(rd.choice(sec_keys))
        
        
        ## Finally, randomize the sector list so that they get put in locations randomly
        rd.shuffle(sec_pr)
        rd.shuffle(loc_pr)
        
        
        ## Now, go through and add the firms to the locations and remote working preferences of interest iteratively
        for ii in range(n_firms):
            
            ## Just randomly creates a 16-digit alpha-numeric string for the name
            name = ''.join((rd.choice(st.ascii_letters + st.digits) for jj in range(16)))
                        
            ## Randomly select the remote preference adding some gaussian noise
            if remote_prefs == None:
                
                ## If no preferences have been passed, set the remote preference to 0 (negative 1 to be sure we are outside the bounds of the uniform draw)
                r_pref = -1
                
            else:
                
                pref_base = remote_prefs[sec_pr[ii]]       # Get the base preference based on the sector
                r_pref = np.random.normal(pref_base, .075) # Add the Gaussian noise to the preference

            
            self.addFirm(name = name, sector = sec_pr[ii], location = loc_pr[ii], remote_pref = r_pref)
            
            


    def modifyFirm(self,
                   name,            # Name of Firm to be modified
                   
                   ## Pass in any value other than None to update the firms value for that property
                   sector             = None, # Should the sector be updated?
                   location           = None, # Should the location be updated?
                   labor_elasticity   = None, # Should the Labor Elasticity be updated?
                   productivity_shift = None, # Should the Productivity Shift be udpated?
                   remote_pref        = None  # Should the remote tolerance be updated?
                   ):
        """
        Wrapper to modify Firm parameters. Used to shock the economy after 
        reaching some steady state, i.e. provide the ability to change remote 
        tolerance individually
        
    
        Returns
        -------
        None.
    
        """
    
        ## Check to see if any inputs have been passed, if not, exit the function
        
        
        ## If some have, make sure the firm exists (and get its index) 
        
        
        ## Finally, update the appropriate parameters as requested
    
        pass
    
    
    
    def modifyFirms(self):
        """
        Wrapper for modifying multiple firms (perhaps all or by Sector or location)
        
        Returns
        -------
        None.
        
        """
        
        pass
    
    

    def addLabor(self,
                 n_workers,                # How many workers are in the world (this is not currently added to/subtracted from in the simulation),
                 location_prefs,           # Pass a dictionary with endpoints for preferences for each location, random preferefences for each location will be drawn from a uniform distribution for each worker (for now)
                 location_prop,            # Pass a dictionary with one entry per location, which sums to 1 (this is the inital distibution for the workers)
                 sector_prop,              # Pass a dictionary with one entry per sector, which sums to 1. This defines the relative sector sizes
                 remote_prefs      = None, # Pass a list [mean, sd].The population will have their location pref set by random draws from a guassian distribution, with min and max values of [0,1] If None are passed then all values will be set as 0
                 unemployment_rate = .05,  # Set an initial unemployment rate
                 firm_eta          = .8    # Exponential weighting for the distribution of firms within each sector
                 ):
        """
        Convienently populate the world with all workers 
        
        Note that this function must be run after firms have been created
        """
        
        #### Input Checking ####
        
        ## Ensure that Firms exist, otherwise raise an exception
        if len(self.Firms) == 0:
            
            raise Exception("Firms must be created before Labor can be added")
        
        
        #### Begin Function ####
        
        ## Create 3 arrays: location, sector, and employed/unemployed
        location_prop.update((key, ii * n_workers) for key, ii in location_prop.items())
        sector_prop.update((key, ii * n_workers) for key, ii in sector_prop.items())
        employed = np.random.choice([True, False], size = n_workers, p = [1 - unemployment_rate, unemployment_rate])
        
        ## Now create a list of the names (and flatten it, since the list comprehension creats a list of lists...)
        loc_pr = list(itertools.chain.from_iterable([[key] * int(ii) for key, ii in location_prop.items()]))
        sec_pr = list(itertools.chain.from_iterable([[key] * int(ii) for key, ii in sector_prop.items()]))
               
        
        ## Set the list of remote working preferences based on the input
        if remote_prefs == None:
            
            rem_pr = [0] * n_workers
            
        else:
            
            rem_pr = np.random.normal(remote_prefs[0], remote_prefs[1], n_workers) # Generate the distribution of preferences
            rem_pr[rem_pr < 0] = 0                                                 # Set a lower bound of 0
            rem_pr[rem_pr > 1] = 1                                                 # Set an upper bound of 1
        
        
        ## Randomly shuffle the location and sector orders
        rd.shuffle(loc_pr)
        rd.shuffle(sec_pr)
        
        
        ## An array of firms (which get created using an exponential distribution draw for each location/sector), assuming the worker is employed (unemployed is None)
        ## For each location/sector, generate some initial probability weights based on the number of firms
        frm_pr = {}
        for l_name in self.Locations.keys():
            
            frm_pr[l_name] = {}
            all_firms = self.Locations[l_name].Firms

            ## If there are any firms in the sector, in the location, then add workers to the firms 
            if len(all_firms) > 0:
                
                for s_name in sector_prop.keys():
                    
                    ## Count the number of firms that are in the sector at that location
                    f_avail = [all_firms[ii].name for ii in range(len(all_firms)) if all_firms[ii].sector == s_name]
                    
                    
                    ## If there are no firms in the sector in the location, set the worker to unemployed
                    if len(f_avail) == 0:
                
                        frm_pr[l_name]['Unemployed'] = None
                    
                    else :
                        
                        n_firms = len(f_avail)
                        
                        # If requested, draw from some pseudo exponential distribution
                        if firm_eta > 0:
                            
                            # n_firms = len(f_avail)
                            
                            # This is good in the future when firms have some way of firing underpreforming workers, however, right now
                            # it can be a bit harsh in long term outcomes. Instead, I'll try to be much more gradual about the probabilities
                            pr_firm = [(ii + 1)**(firm_eta) for ii in range(n_firms)]
                            tmp_tot = sum(pr_firm)
                            pr_firm = [ii / tmp_tot for ii in pr_firm]
                                                
                        
                            frm_pr[l_name][s_name] = np.random.choice(f_avail, size = n_workers, p = pr_firm)
                            
                            
                        ## Otherwise, just draw from a random normal
                        else :
                            
                            pr_firm = 1 / n_firms                        
                        
                            frm_pr[l_name][s_name] = np.random.choice(f_avail, size = n_workers)
                    
            else:
                
                frm_pr[l_name]['Unemployed'] = None
        
        
        
        # empty_firms = [self.Firms[ii].name for ii in range(len(self.Firms))] # Get a list of all firms - we will try to add a firm by sector/location, but over-ride the addition until all firms have been added at least once
        ## Now, create all of the workers
        for ii in range(n_workers):
            
            ## Create a dictionary of the location preferences, then pass the rest of the information by index
            ## This could probably be optimized...
            lp = {}
            for l_name in location_prefs.keys():
                
                lp[l_name] = rd.uniform(location_prefs[l_name][0], location_prefs[l_name][1]) # Selects a single floating point preference based on the endpoints and a random uniform distribution, for each location
            
            ## If the worker is employed, choose a firm in the location
            if employed[ii]:
                               
                # ## If there are firms that have not had a worker added yet in the sector/location, over ride above and add one to one of those firms
                # if len(empty_firms) > 0:
                    
                #     ## Get the firms in the sector that are also empty
                #     firm_sector = [self.Firms[jj].name for jj in range(len(self.Firms)) if self.Firms[jj].sector == sec_pr[ii] if self.Firms[jj].location == loc_pr[ii]]
                #     empty_sector = np.intersect1d(empty_firms, firm_sector)
                    
                    
                #     if len(empty_sector) > 0:
                        
                #         ## Randomply select a firm from the empty firms in the sector
                #         firm = np.random.choice(empty_sector, size = 1)[0]                          
                        
                #         ## Remove the firm from the empty firms list...
                #         empty_firms.remove(firm)
                    
                #     else:
                        
                #         ## Otherwise, just select the firm that was randomly assigned with the appropriate weight
                #         firm = frm_pr[loc_pr[ii]][sec_pr[ii]][ii]
                    
                    
                # else:
                
                #     ## Otherwise, just select the firm that was randomly assigned with the appropriate weight
                #     firm = frm_pr[loc_pr[ii]][sec_pr[ii]][ii]
                new_firm = frm_pr[loc_pr[ii]][sec_pr[ii]][ii]
                    
            else:
            
                new_firm = None
                
            ## Add the worker to the population
            self.addWorker(loc_pr[ii], lp, sector = sec_pr[ii], firm = new_firm, remote_pref = rem_pr[ii])
    
    
    
    def addWorker(self,            
                  location,              # The name of a location where this worker lives
                  location_pref,         # A dictionary of the idiosyncratic worker preferences for each location
                  sector         = None, # Which sector should the worker be in?
                  firm           = None, # If employed, pass the current firmID this worker is associated with, if unemployed, pass None
                  ID             = None, # Pass a worker ID. If None is passed an id will be assigned based on the list element length at the time of being appended
                  remote_pref    = 0     # Set some remote working preference (0 for none, 1 for max, or some value in between)
                  ):
        """
        Method for adding a worker to the model
        """
        
        ## Add the worker to the end of the array
        if ID == None:
            
            ID = 'w' + str(len(self.Population))
            # ID = str(len(self.Population))
            
            
        ## If no sector is passed (in the case of a single sector model) specify none as a string...
        if sector == None:
            
            sector = 'none'
            
       
        ## Add the worker object to the Population 
        self.Population[ID] = Worker(location, 
                                     location_pref, 
                                     sector = sector, 
                                     firm = firm, 
                                     ID = ID, 
                                     remote_pref = remote_pref)
        
        
        ## Add the worker name to the lookup table
        self.sector_labor_hash[sector] = ID
        
        
        ## Add the worker-firm edge to the sector-specific labor-firm network
        self.Network_FirmWorker[sector].addNodes(node_names = [ID], node_types = ['worker'])
        if firm != None:
            self.Network_FirmWorker[sector].updateEdges(add_edges = [[self.Population[ID].ID, self.Population[ID].firm]])
        
        
        ## Add a pointer to the Worker in the Location
        self.Locations[location].updateLabor(self.Population[ID], by = 'add')
        
        
        ## Add a pointer to the Worker at the firm
        if firm != None:
            
            self.Firms[firm].setWorker(self.Population[ID], by = 'add')


    
    
    def modifyWorker(self):
        """
        Modify a single worker        

        Returns
        -------
        None.

        """
        
        pass
    
    
        
    def modifyLabor(self):
        """
        Modify multiple workers, by sector/location etc

        Returns
        -------
        None.

        """
        
    
        pass

    
    
    
    def shockFirms(self):
        """
        Placeholder for allowing a firm shock to be passed
        """
        
        pass
    
    
    def shockLabor(self):
        """
        Placeholder for allowing a labor shock to be passed
        """
        
        
        pass
    
    
    def shockLocation(self):
        """
        Placeholder for allowing a location shock to be passed
        
        Returns
        -------
        None.
    
        """
        
        pass
        
    
    
    def simulate(self, 
                 steps = 50,          # Define some finite number of steps that the simulation should run for
                 alpha = .05,         # What percentage of the employed population should look for a new job each time step?
                 beta  = .02,         # What percetage of Firms should contemplate moving cities each time step?
                 start_firm = .05,    # The probability a worker who has been fired (or who's firm has moved) will start a new solo firm, assuming that a 'slot' exists
                 gamma = 0,           # Set this to any value other than zero to specify the strength of the agglomeration economy. If this is zero, no agglomeration is considered
                 prob_subset = False, # Optionally, limit the number of firms a worker may "apply" to probabilistically by edge weight
                 prob_val = .8,       # Works in conjuction with prob_subset. The percentage of firms that the list should be reduced to (i.e. 10 to 8)
                 is_shock = False,    # Set this value to True to skip time step zero. i.e. this is meant to be used when a shock occurs to restart the system
                 time_offset = 0      # Use this parameter to offset the Time Step column each iteration. Meant for shock scenarios
                 ):
        """
        Simulate the model for some number of time steps
    
    
        Returns
        -------
        stats_loc: pandas.DataFrame
            A pandas DataFrame of Location Statistics
            
        stats_firm: pandas.DataFrame
            A pandas DataFrame of Firm Statistics
            
        stats_labor: pandas.DataFrame
            A pandas DataFrame of Labor Statistics
            
        stats_network: pandas.DataFrame
            A pandas DataFrame of Network Statistics
    
        """
        
        ## Set the previous connections here (assuming this is the start of the model)
        if not is_shock:
            
            self.__setPreviousFirmWorkerConnections(alpha = .95)
        
        
        
        ## Loop simulating each step of the model by calling simulateOneStep
        ## Be sure to update/record the simulation step results here, aggregated
        ## May want to also write these results to a database...
        
        ## In order to add some variance to the model, create an array of the number of random workers/firms to move each time step
        w_sample = np.floor(np.random.normal(alpha, .01, steps) * len(self.Population)).astype(int)
        f_sample = np.floor(np.random.normal(beta, .01, steps) * len(self.Firms)).astype(int)
        
        
        ## Set a floor of 1 firm/worker to test the location against in the model
        f_sample[f_sample < 1] = 1
        w_sample[w_sample < 1] = 1
        
        
        ## When running from time = 0, this is run to initialize model parameters
        ## During a shock, this is skipped because those parameters already exist from the last run and this will double-count the last run
        if not is_shock:
            
            ## Calculate the initial wages offered by each firm and set them for the firm and each associated worker
            wages = self.__calcWages(gamma = gamma)
            rents = self.__calcRents()
            for f_name in wages.keys():
                
                for w in self.Firms[f_name].labor:
                    
                    ## Set the wage
                    w.setWage(max(0, wages[f_name]))
                    w.setRealWage(w.wage / rents[w.location])
                    
                    ## Set the utility as well (for nicer charts)
                    w.setUtility(w.calcUtility(wage = w.wage,
                                               rent = rents[w.location],
                                               A = self.Locations[w.location].Ammenities,
                                               pref = w.location_prefs[w.location]))
        
                
            ## Set initial utility values    
            self.__updateUtilities(wages, rents)
            
            
            ## Record the initial statistics
            stats_loc, stats_firm, stats_labor, stats_network = self.__recordStats(time_step = 0)
            
                
        for ii in tqdm.trange(steps):
        # for ii in range(steps): # For debugging with print statements
        
            ## Wake up labor and firms
            labor = rd.sample(list(self.Population.values()), w_sample[ii]) # Pointer to worker objects
            labor = labor.copy()                                            # Otherwise we are modifying lists in place which is apparently a big old problem in python
            f_names = rd.sample(list(self.Firms.keys()), f_sample[ii])      # Firm names 
            
            
            ## Should all unemployed workers also search for jobs??
            
            
            ## Perform a simulation step
            self.simulateOneStep(labor, f_names, start_firm = start_firm,
                                 prob_subset = prob_subset, prob_val = prob_val,
                                 gamma = gamma)
            
            
            ## Gather step statistics
            df_loc, df_firm, df_labor, df_network = self.__recordStats(time_step = (ii + 1 + time_offset))            
            
            if is_shock and (ii == 0):
                
                ## For the case of shock and initial step, simply set the statistics 
                stats_loc = df_loc
                stats_firm = df_firm
                stats_labor = df_labor
                stats_network = df_network
                
            else:
                
                ## Append the appropriate dictionaries
                stats_loc = pd.concat([stats_loc, df_loc], ignore_index = True)
                stats_firm = pd.concat([stats_firm, df_firm], ignore_index = True)
                stats_labor = pd.concat([stats_labor, df_labor], ignore_index = True)
                stats_network = pd.concat([stats_network, df_network], ignore_index = True)
    
        ## To Do - write the dictionaries to sql database tables. Maybe this should be its own method with the outputs
        
        
        ## Return the dictionaries
        return stats_loc, stats_firm, stats_labor, stats_network
        
        
        
    
    
    def simulateOneStep(self,
                        labor,               # Array of Worker objects that will be re-evaluating their choices
                        f_names,             # A list of firm names (strings)
                        start_firm = .05,    # The probability that each worker will attempt to start a new firm
                        prob_subset = False, # Optionally, limit the number of firms a worker may "apply" to probabilistically by edge weight
                        prob_val = .8,       # Works in conjuction with prob_subset. The percentage of firms that the list should be reduced to (i.e. 10 to 8)
                        gamma = 0            # Specify something other than 0 for agglommeration economies, otherwise the static productivity shifter is used
                        ):
        """
        Simulate one time-step in the model
        
        Parameters
        ----------
        labor: list, laborFlow.Worker
            A list of Worker objects of interest
        
        f_names: list, str
            A list of the names (strings) of firms of interest
        
        vote_rank: int, None
            Optionally, limit the number of firms a worker may "apply" to each 
            step by only keeping the top n connected firms by the voteRank 
            score.
            
        rand_conn: float, .1
            Works in conjunction with vote_rank. Add back random firms (by 
            percentage) from the initial connected firms list
        
        prob_subset : boolean
            A flag to subset the list of returned firms by random choice, weighted by the
            edge weights of the firm-firm projection

        prob_val : float
            The percentage of firms that the list should be reduced to (i.e. 10 to 8)
            Expressed as a decimal between 0 and 1
            
        gamma: float, 0
            Strength of agglommeration effects. If this value is 0, 
            agglomeration effects will be ignored, defaulting to the static
            location-specific productivity shifter
    
        Returns
        -------
        stats_loc: pandas.DataFrame
            A pandas DataFrame of Location Statistics
            
        stats_firm: pandas.DataFrame
            A pandas DataFrame of Firm Statistics
            
        stats_labor: pandas.DataFrame
            A pandas DataFrame of Labor Statistics
            
        stats_network: pandas.DataFrame
            A pandas DataFrame of Network Statistics
        """
        
        ## DataFrames were chosen for their smaller memory footprint:
        ## https://www.joeltok.com/blog/2021-6/memory-optimisation-python-dataframes-vs-json-like
        

        ## Calculate the current wages offered by each firm
        wages = self.__calcWages(gamma = gamma)
        
        
        ## Calculate the current rents offered in each location
        rents = self.__calcRents()
        
        
        ## Update labor supply (mean wage) on a per location, per sector basis
        ## I think this is actually demand...the calculation is correct but I'll need to change the names...
        wage_supply = self.__getLaborSupply()
        
        
        ## Get the firms that are hiring
        firms_hiring = self.__getHiringFirms(wage_supply, wages, gamma = gamma)
        
        
        ## Decide how many firms will move and update wages accordingly
        wages = self.__moveFirms(f_names, wages, rents, firms_hiring,
                                 prob_subset = prob_subset, prob_val = prob_val, gamma = gamma)
        
        
        ## Update labor supply (mean wage) on a per location, per sector basis
        wage_supply = self.__getLaborSupply()
        
        
        ## Get the firms that are hiring
        firms_hiring = self.__getHiringFirms(wage_supply, wages, gamma = gamma)
                
        
        ## Allow Workers to search for new firms/locations
        ## Note that this updates the appropriate firm-worker network implicitly
        self.__matchWorkers(labor, wages, rents, firms_hiring, alpha = start_firm,
                            prob_subset = prob_subset, prob_val = prob_val)
               
        
        ## Get all of the wages/utilities/ammenities/current location preferences for each worker
        self.__updateUtilities(wages, rents)
        
        
        ## And Finally, calcululate and store the firm components of production/profit/hiring quota
        _ = self.__getHiringFirms(wage_supply, wages, gamma = gamma, store_vals = True)
        
    
    
    def __moveFirms(self,
                    f_names,             # A list of firm names (strings)
                    wages,               # A list of wages offered by firm - in the case where a firm moves, this will need to be updated
                    rents,               # A list of rents by location. This is necessary to allow workers in firms that move to make an informed choice about whether they will stay with the firm
                    firms_hiring,        # A list of the firms that are currently hiring
                    alpha   = .05,       # The probability a worker will start a new firm
                    prob_subset = False, # Optionally, limit the number of firms a worker may "apply" to probabilistically by edge weight
                    prob_val = .8,       # Works in conjuction with prob_subset. The percentage of firms that the list should be reduced to (i.e. 10 to 8)
                    gamma = 0            # Specify something other than 0 for agglommeration economies, otherwise the static productivity shifter is used
                    ):
        """
        Have firms evaluate their production functions and decide to move 
        locations. If a firm moves, co-located workers move with the firm
        
        Note that any firms subject to a move will also naturally add all of 
        their workers to the list of workers to re-evaluate their current 
        situation 
        
        Parameters:
        ----------
        
        fnames: list, str
            A list of firm names that will evaluate their location options and
            potentially move
            
        wages: dict, float
            A dictionary where firm names are the keys and wages are float 
            values offered by each firm
            
        rents: dict, float
            A dictionary where location names are the keys and the cost of rent
            at each location are the values
            
        firms_hiring: list, str
            A list of the firms that are currently hiring
            
        alpha: float, .05
            The probability a worker will decide to start their own firm. This
            only happens if there is an empty firm in the sector (this can be 
            thought of as a proxy for market saturation)
        
        prob_subset : boolean
            A flag to subset the list of returned firms by random choice, weighted by the
            edge weights of the firm-firm projection

        prob_val : float
            The percentage of firms that the list should be reduced to (i.e. 10 to 8)
            Expressed as a decimal between 0 and 1
            
        gamma: float, 0
            Strength of agglommeration effects. If this value is 0, 
            agglomeration effects will be ignored, defaulting to the static
            location-specific productivity shifter
            
        
        Returns:
        -------
        
        wages: dict
            An updated wage dictionary, by firm, after the firms have moved and
            updated their production/wage
            
        """
        
        #### Input Checking ####
        
        ## If the length of id_Firm is 0, break the function and return an empty list
        if len(f_names) == 0:
            
            # return [], wages
            return wages
        
        
        #### Begin Function ####
        
        ## Get the mean wage for each location/sector and the agglommeration production shifter
        w_s = {}
        X_ls = {}
        for loc in self.Locations.keys():
            
            w_s[loc] = {}
            X_ls[loc] = {}
            for sector in self.sector_firm_hash.keys():
                
                ## Mean Wages
                ls_wages = [wages[f.name] for f in self.Locations[loc].Firms if f.sector == sector]
                if len(ls_wages) == 0:
                    w_s[loc][sector] = 0
                
                else:
                    w_s[loc][sector] = np.mean(ls_wages)

                
                
                ## Location and Sector-specific productivity shifter
                n_workers = sum([1 for w in self.Locations[loc].Workers if w.sector == sector])
                X_ls[loc][sector] = self.Locations[loc].calcProductivityShift(n_workers)
                        
                
        for f_name in f_names:
            firm_moved = False

            ## Get the number of workers
            n_workers = len(self.Firms[f_name].labor)
            
            
            ## If there are no workers, move on to the next firm
            if n_workers == 0:
                
                continue
                        
            
            ## Calculate the total production for each location (this would be a good place to parallelize with threads)
            ## This should be calcProfit() for each location (though right now this scales linearly with production, so not a problem)
            tot_profit = {}
            for l_name in self.Locations.keys():
                
                
                ## In the case gamma is equal to zero, simply use the static  productivity ammenity for the location
                if gamma == 0:
                    
                    X = self.Locations[l_name].Productivity[self.Firms[f_name].sector]
                                    
                else:
                    
                    X = X_ls[l_name][self.Firms[f_name].sector]
                
                
                ## Calculate the location-specific profit
                w_f = self.Firms[f_name].calcWage(X = X, N = n_workers) # Due to agglomeration effects, wage is location-specific
                tot_profit[l_name] = self.Firms[f_name].calcProfit(X = X, 
                                                                   N = n_workers, 
                                                                   w_s = w_s[l_name][self.Firms[f_name].sector], 
                                                                   w_f = w_f)
            
            ## Should the firm move?
            if len(np.unique(list(tot_profit.values()))) > 1:
                
                new_loc = max(tot_profit, key = tot_profit.get)                
            
            else:
                
                new_loc = self.Firms[f_name].location
            
            ## Store the old location
            old_loc = self.Firms[f_name].location
            
            
            ## If the new location is different than the old location, move the firm (and the associated workers)
            if new_loc != old_loc:
                
                firm_moved = True # If any firms have moved, this flag gets set to recalculate wages...
                
                self.Firms[f_name].setLocation(new_loc)                               # Move the firm                
                self.Locations[new_loc].updateFirm(self.Firms[f_name], by = 'add')    # Add the firm to the location pointer in the new location
                self.Locations[old_loc].updateFirm(self.Firms[f_name], by = 'remove') # Remove the pointer in the old location
                
                
                ## Should put this here eventually for increased speed, but it is easier to add to the end if any firms have moved...
                # wages[f_name] = self.Firms[idx].calcWage()
                flag = False
                ## Now, get all of the workers and move them, but also give them the opportunity to search for a job (i.e. decide if they'll go)
                for worker in self.Firms[f_name].labor:
                    
                    if not worker.is_remote:
                    
                        self.Population[worker.ID].setLocation(new_loc)            # Move the Worker
                        self.Locations[new_loc].updateLabor(worker, by = 'add')    # Update the location pointer  
                        self.Locations[old_loc].updateLabor(worker, by = 'remove') # Remove old location pointer 
                
                ## Allow the workers to search for a new job, but stay with the firm if that is their best option
                ## Note that this explicitly allows the workers to search in a "not fired" context
                # self.__matchWorkers(self.Firms[f_name].labor,
                firm_labor = self.Firms[f_name].labor.copy()
                self.__matchWorkers(firm_labor,
                                    wages, 
                                    rents, 
                                    firms_hiring,
                                    alpha = alpha, 
                                    firm_moved = True,
                                    prob_subset = prob_subset,
                                    prob_val = prob_val
                                    )
            
        ## Calculate the new wages offered by the firms if any have moved
        if firm_moved:
                
            wages = self.__calcWages(gamma = gamma)
            
        
        return wages
    
    
    
    def __fireWorkers(self):
        """
        Placeholder - this is implicit in the __matchWorkers() function. May or
        may not ever actually be implemented
        """
        
        pass



    def __calcWages(self, gamma = 0, update_workers = False):
        """
        Calculate the wages for all firms in the model based on the current 
        conditions and return a dictionary with the firm name and the wage
        """
                                
        
        ## If gamma is not zero, calculate the location productivity based on agglommeration
        if gamma != 0:
            
            X_ls = {}
            for loc in self.Locations.keys():
                
                X_ls[loc] = {}
                for sector in self.sector_firm_hash.keys():
                    
                    ## Location and Sector-specific productivity shifter
                    n_workers = sum([1 for w in self.Locations[loc].Workers if w.sector == sector])
                    X_ls[loc][sector] = self.Locations[loc].calcProductivityShift(n_workers)
        
        
        all_wages = {}
        for f_name in self.Firms.keys():
                       
                            
            ## Calculate the inputs (may be able to get this faster from the network...)
            n_workers = len(self.Firms[f_name].labor) # Number of workers
            
            ## If the firm has no workers, give it machine zero so that log doesn't break...
            if n_workers == 0:
                
                n_workers = sys.float_info.epsilon
            
                        
            ## If applicable, grab the location and sector specific productivity shifter
            if  gamma != 0:
                
                # need to get sector and pass it to locations
                X = X_ls[self.Firms[f_name].location][self.Firms[f_name].sector]
                
                
            ## Otherwise, simply grab the static location productivity
            else:
                
                X = self.Locations[self.Firms[f_name].location].Productivity[self.Firms[f_name].sector]
                        
                
            ## New - calculate the current wage
            all_wages[f_name] = self.Firms[f_name].calcWage(X = X,
                                                            N = n_workers)
            
            
            ## For all the workers in the firm, update their wage
            for w in self.Firms[f_name].labor:
                
                w.setWage(all_wages[f_name])
                w.setRealWage(all_wages[f_name] / self.Locations[self.Firms[f_name].location].Rent)
            
        return all_wages
    
    
    
    def __updateUtilities(self, wages, rents):
        """
        Update the wage, rent, ammenity, and current location preference values 
        for all workers in the model
        """
                    
            
        for w in self.Population.values():
            
            ## Get the wage (or set it as zero)
            if w.firm != None:
                
                wage = wages[w.firm]
                
            else:
                
                wage = 0
            
            
            ## Calculate the utility and update it
            w.calcUtility(wage, rents[w.location],
                          A  = self.Locations[w.location].Ammenities, pref = w.location_prefs[w.location], 
                          update_vals = True)
        
    
    
    
    
    def __calcRents(self):
        
        """
        Calculate the rent for every location based on the starting values 
        of the loop (internal method that should be called after __calcWages())
        """
        
        rents = {}
        for l_name in self.Locations.keys():
            
            ## Get the total number of workers in the city
            n_workers = len(self.Locations[l_name].Workers)                       
                
            
            ## Calculate the rent
            ## Note that z could possibly be updated here at some point - need to assign it in the Locations themselves
            rents[l_name] = self.Locations[l_name].calcHousingSupply(n_workers)
            
            
        return rents
    
    
    
    def __startNewFirm(self,
                       worker,      # Worker object from laborFlow package (pointer to element in Population)
                       firm_avail,  # A list of strings of available firms for the worker to "start"
                       wages,       # A dictionary of wages offered by firms within the worker's sector
                       rents        # A dictionary of rents by location (produced by __calcRents method)
                       ):
        """
        Move a worker to a firm with no workers to create a "new" firm
        
        i.e. this worker will try to establish a new firm in a market that is 
        not oversaturated
        
        """
        
        ## If there are, select one randomly and move the agent there
        firm_new = np.random.choice(firm_avail, size = 1)[0] # Returns the string instead of an array of one dimension
        firm_old = None
               
        
        ## Update the firm objects appropriately
        self.Firms[firm_new].setLocation(worker.location)             # Update the new firm location
        self.Firms[firm_new].setWorker(worker, by = 'add')            # Update the new firm with the worker
        
        if worker.firm != None:
            
            self.Firms[worker.firm].setWorker(worker, by = 'remove') # Remove the worker from the "old" firm
        
        
        ## Udate Worker with old/new firms
        if worker.firm != None:
            
            ## Only update the previous firm if they were currently employed when they started their new company
            firm_old = worker.firm_last
            worker.setPreviousFirm(worker.firm)
            
            
        worker.setFirm(firm_new)
        
        ## Ensure the firm is in the appropriate location 
        ## Note that this section needs to stay here, but the worker/firm updates can probably be 
        ## universally moved to the bottom of this function
        for loc in self.Locations.keys():
            
            ## If the firm is in the location of interest, remove the pointer from the list
            if self.Firms[firm_new] in self.Locations[loc].Firms:
                
                self.Locations[loc].updateFirm(self.Firms[firm_new], by = 'remove')
        
        
        ## Put the Firm pointer in the appropriate location
        # self.Locations[worker.location].Firms.updateFirm(self.Firms[firm_new], by = 'add')
        self.Locations[worker.location].updateFirm(self.Firms[firm_new], by = 'add')              
        
        ## Calculate and set the Utility for the worker
        utility = worker.calcUtility(wage = wages[firm_new],
                                     rent = rents[worker.location],
                                     A = self.Locations[worker.location].Ammenities,
                                     pref = worker.location_prefs[worker.location])
        worker.setUtility(utility)
        worker.setWage(wages[firm_new])
        worker.setRealWage(wages[firm_new] / rents[worker.location])
        
        
        ## This should not ever be remote...
        isRemote = worker.location != self.Firms[worker.firm].location
        worker.updateRemote(isRemote)    
    
        ## Return the add_edge and rem_edge values as tuples with the worker name
        add_edge = [worker.ID, firm_new]
        rem_edge = [worker.ID, firm_old]

        
        return add_edge, rem_edge
    
    
    
    def __matchWorker(self,
                      worker,              # Worker object from laborFlow package (pointer to element in Population)
                      wages,               # A dictionary of wages offered by firms within the worker's sector
                      rents,               # A dictionary of rents by location (produced by __calcRents method)
                      firms_hiring,        # A list of the firms currently hiring
                      alpha   = .05,       # The probability a worker will start a new firm
                      firm_moved = False,  # An option to allow the worker to search within their own firm if the firm has moved (that is the worker was not fired, they just explored their option when a firm moved)
                      prob_subset = False, # Optionally, limit the number of firms a worker may "apply" to probabilistically by edge weight
                      prob_val = .8        # Works in conjuction with prob_subset. The percentage of firms that the list should be reduced to (i.e. 10 to 8)
                      ):
        """
        Match a single worker to a firm (or to None, in which case they will 
        be unemployed) by maximizing their utility across firms within their 
        labor flow network
        
        For speed, this iteration of the code pre-calculates wages for all 
        firms and the offers a homogenous wage to each job-seeker on a 
        per-firm basis. This will need to be updated in the future if the 
        production function changes form to O-ring, or any other functional 
        that takes into account heterogenous skill/pay
        
        Parameters:
        ----------
        
        worker: laborFlow.Worker object 
            Worker (pointer to) in the Economy.Population to be matched
        
        wages: dict
            Dictionary of wages being offered by each Firm in the Economy 
        
        rents: dict
            Dictionary of current rents by Location in the Economy
            
        firms_hiring: list, str
            A list of the firms that are currently hiring
            
        alpha: float, .05
            The probability the worker will start a new firm
            
        firm_moved: bool, False
            Did the Firm move? If the firm moved, the worker gets a "free pass"
            to explore options and can go back to their own firm
        
        prob_subset : boolean, False
            A flag to subset the list of returned firms by random choice, weighted by the
            edge weights of the firm-firm projection

        prob_val : float, .8
            The percentage of firms that the list should be reduced to (i.e. 10 to 8)
            Expressed as a decimal between 0 and 1
        
        
        Returns:
        -------
            add_edge : list
                [worker.ID, new_firm]
                Edge list to be added to the appropriate network (preferably in bulk)
                Note that the new_firm may be None 

            rem_edge : list
                [worker.ID, rem_firm]
                Edge list to be removed from the appropriate network (preferably in bulk)
                Note that the rem_firm may be None 

            Additional relevant Economy, Firm, and Worker objects are modified in place
        """
        
        #### Argument Checking ####
        
        
        
        #### Begin Function ####
        
        ## Check to see if the worker will start a new company
        if np.random.uniform(0, 1) < alpha:
            
            
            ## To Do: Calculate Wage, Utility, and set isRemote as False
            ##        i.e. make sure all worker stats are updated
            
            
            ## Get any available 'firms' within the sector
            firm_avail = [fn for fn in self.sector_firm_hash[worker.sector] if len(self.Firms[fn].labor) < 1]
            
            ## Check to see if there any empty firms
            if len(firm_avail) > 0:
                                
                ## Move the worker to a "new" firm, if one is available
                add_edge, rem_edge = self.__startNewFirm(worker, firm_avail, wages, rents)                
                                
                ## Exit the function, returning the edges to be added/removed from the appropriate network
                return add_edge, rem_edge
        
        
        ## Apparently, numpy throws a fit matching with NoneType despite it working outside of this loop - so remove NoneType...
        worker_firms = [w for w in [worker.firm, worker.firm_last] if w is not None]
        
        ## If both the initial and last firm were None (will rarely happen at the beginning of the simulation), set conn_firms to []
        if len(worker_firms) == 0:
            
            conn_firms = []
            
        else:
        
            ## Get a list of connected firms in the worker's sector
            conn_firms = self.Network_FirmWorker[worker.sector].getConnectedFirms(worker_firms,                         # Only returns the edges to these two firms
                                                                                  self.sector_firm_hash[worker.sector], # The firms in the worker's sector
                                                                                  prob_subset = prob_subset,
                                                                                  prob_val = prob_val
                                                                                  )
            conn_firms = list(conn_firms)
        
        
            # Make sure old or current firms don't somehow end up in the list
            # conn_firms = list(np.setdiff1d(conn_firms, worker_firms))
            
                
        ## If there aren't any connected firms, need to randomly match firms
        ## Note that there may be an issue if there are no edges, at which point all edges would be returned. In that case, we could set the search the same way here
        if len(conn_firms) < 1:
            
            ## For now, set a limit on randomly matching with random firms at 25%
            n = int(np.ceil(len(self.sector_firm_hash[worker.sector]) * .25))
            conn_firms = list(np.random.choice(self.sector_firm_hash[worker.sector], size = n, replace = False)) # List matches the output of the getConnectedFirms() method above                     
                
                
        ## Except if the Firm moved..., add it to the list of firms to maximize utility across
        if firm_moved:
            
            conn_firms.append(worker.firm)
        
        
        ## Only search through firms that are actually hiring
        conn_firms = [c_firm for c_firm in conn_firms if c_firm in firms_hiring]
        
        
        ## Make sure that somehow a None value didn't sneak in
        ## Note that this will return [] if there are no connected firms in firms_hiring (from above) which will then skip the
        ## for new_firm in conn_firms: loop, only cycling through unemployment options
        conn_firms = list(filter(None, conn_firms))
        
        
        ## Calculate the Utility for each firm (and possible remote arrangement)
        utilities = [] # Store the utility
        locs = []      # Store the associated location
        fnames = []    # Store the associated firm
        WFH = []       # Is the worker in a remote situation?
        new_wage = []  # Store the new wage                
        for new_firm in conn_firms:
            
            ## Set the location of the firm for calculating utilities - this is done explicitly because we may want to add locations if a remote work arrangement is agreed upon
            lnames = [self.Firms[new_firm].location]
            isRemote = False                         # Default setting for remote work is False
            
            
            ## Will the firm and worker be in a remote arrangement?
            isRemote, partialRemote = self.__checkRemote(worker, self.Firms[new_firm], partial = False)
            
                           
            ## If both the firm and worker are ok with remote work, add every location in the model to the location
            if isRemote:
                    
                lnames = self.Locations.keys()
                
            
            ## Calculate the utilities for each firm and location of interest
            for loc in lnames:
                
                ## Get the parameters for calculating the worker's utility for each firm/location
                A = self.Locations[loc].Ammenities 
                pref = worker.location_prefs[loc]
                wage = wages[new_firm]    
                rent = rents[loc]
                
                
                ## Calculate the utility
                utility = worker.calcUtility(wage, rent, A, pref)
                
                ## The firm must offer a positive wage, otherwise it doesn't make sense to work
                ## If they don't, don't consider the firm
                if wage > 0:
                
                    ## Store the parameters
                    utilities.append(utility)
                    locs.append(loc)
                    fnames.append(new_firm)
                    WFH.append(isRemote)
                    new_wage.append(wage)
        
        
        ## Add utilities for no-work...
        for loc in self.Locations.keys():
            
            ## Get the parameters for calculating the worker's utility for each firm/location
            A = self.Locations[loc].Ammenities 
            pref = worker.location_prefs[loc]
            wage = 0
            rent = rents[loc]
            
            
            ## Calculate the utility
            utility = worker.calcUtility(wage, rent, A, pref)
        
            ## Store the parameters
            utilities.append(utility)
            locs.append(loc)
            fnames.append(None)
            WFH.append(False)
            new_wage.append(0)
            
        
        ## Get the index of the maximum utility and the associated values
        idx = np.random.choice(np.flatnonzero(utilities == np.max(utilities)))
        utility = utilities[idx]
        new_loc = locs[idx]
        new_firm = fnames[idx]
        isRemote = WFH[idx]
        new_wage = new_wage[idx]
        
                
        ## There are cases where the firm might not change, but the utility and remote status does
        ## In this case, we don't want to update the old firm, so to avoid that we'll explicitly get 
        ## the old firm here for setting it below
        if new_firm == worker.firm:
            
            old_firm = worker.firm_last # This will modify (or not in this case) the Labor object
            firm_rem = None             # This will be passed to the network to modify (or not in this case) the edges in bulk
            firm_add = None             
            changed_jobs = False
        
        else:
            
            old_firm = worker.firm
            if old_firm != None:
                firm_rem = worker.firm_last # This will be passed to the network to modify (or not in this case) the edges in bulk
            else:
                firm_rem = None
            firm_add = new_firm             # Same as above, but a new edge is added
            changed_jobs = True
            
            ## Only update the old/new firms if the new firm changed
            worker.setPreviousFirm(worker.firm)
            worker.setFirm(new_firm)
        
        
        ## Store the old location
        old_loc = worker.location
        
        ## Finally, update Worker and Firm, as well as appropriate pointers in Locations and the Economy        
        worker.setUtility(utility)
        worker.setWage(new_wage)
        worker.setRealWage(new_wage / rents[new_loc])
        worker.updateRemote(isRemote)
        
        worker.setLocation(new_loc)


        ## If the worker changed jobs, update the Firm lists (to the Worker pointers) appropriately
        if changed_jobs:
            
            ## Update the Firm objects appropriately, assuming there was a previous firm
            if (new_firm != None) and (new_firm != old_firm):
            
                self.Firms[new_firm].setWorker(worker, by = 'add')    # Update the new firm with the worker
                        
                
            ## A problem can be caused here if a worker never had a current firm, but was initialized with an old firm, then maximizes their utility with no firm again
            if old_firm != None:
                
                self.Firms[old_firm].setWorker(worker, by = 'remove') # Remove the worker from the old firm
            
        
        ## Update the Location objects appropriately, assuming the worker moved locations
        if new_loc != old_loc:
            
            self.Locations[new_loc].updateLabor(worker, by = 'add')
            self.Locations[old_loc].updateLabor(worker, by = 'remove')
            

        ## Return the add_edge and rem_edge values as tuples with the worker name
        add_edge = [worker.ID, firm_add]
        rem_edge = [worker.ID, firm_rem]


        return add_edge, rem_edge
        
        
        
    
    def __matchWorkers(self,
                       workers,            # List of Worker objects from laborFlow package (pointer to element in Population)
                       wages,              # A dictionary of wages offered by firms within the worker's sector
                       rents,              # A dictionary of rents by location (produced by __calcRents method)
                       firms_hiring,       # A list of the firms currently hiring
                       alpha   = .05,      # The probability a worker will start a new firm
                       firm_moved = False, # An option to allow the worker to search within their own firm if the firm has moved (that is the worker was not fired, they just explored their option when a firm moved)
                       prob_subset = False, # Optionally, limit the number of firms a worker may "apply" to probabilistically by edge weight
                       prob_val = .8        # Works in conjuction with prob_subset. The percentage of firms that the list should be reduced to (i.e. 10 to 8)
                       ):
        """
        Match an array of workers to firms (or to None, in which case they will 
        be unemployed) by maximizing their utility across firms within their 
        labor flow network
        
        For speed, this iteration of the code pre-calculates wages for all 
        firms and the offers a homogenous wage to each job-seeker on a 
        per-firm basis. This will need to be updated in the future if the 
        production function changes form to O-ring, or any other functional 
        that takes into account heterogenous skill/pay
        
        Parameters:
        ----------
        
        workers: list, laborFlow.Worker object 
            list of Workers (pointer to) in the Economy.Population to be matched
        
        wages: dict
            Dictionary of wages being offered by each Firm in the Economy 
        
        rents: dict
            Dictionary of current rents by Location in the Economy
            
        firms_hiring: list, str
            A list of the firms that are currently hiring
            
        alpha: float, .05
            The probability the worker will start a new firm
            
        firm_moved: bool, False
            Did the Firm move? If the firm moved, the worker gets a "free pass"
            to explore options and can go back to their own firm            
        
        prob_subset : boolean, False
            A flag to subset the list of returned firms by random choice, weighted by the
            edge weights of the firm-firm projection

        prob_val : float, .8
            The percentage of firms that the list should be reduced to (i.e. 10 to 8)
            Expressed as a decimal between 0 and 1
        
        
        Returns:
        -------
            None. 
            Relevant Economy, Firm, and Worker objects are modified in place
        """
        
        
        ## Note that for now I don't parallelize this, but this might be a useful place for speedups...
        edges_add = {k:None for k in self.Network_FirmWorker.keys()}
        edges_rem = {k:None for k in self.Network_FirmWorker.keys()}
        for worker in workers:
            
            if worker.firm is not None: 
                
                f_old = worker.firm
                
            else:
                
                f_old = None
            
            ## Match the worker with a new firm
            add_edge, rem_edge = self.__matchWorker(worker, wages, rents, firms_hiring, alpha = alpha, firm_moved = firm_moved,
                                                    prob_subset = prob_subset, prob_val = prob_val)
            
            ## If the new and old firm is not None, then add the edges to the output
            if add_edge[1] != None:

                # This could be avoided by setting k:[] instead of k:None above, but that would then require a few
                # changes in the Network.updateEdges() method. This should be straightforward and should be handled soon(ish)
                if edges_add[worker.sector] == None:

                    edges_add[worker.sector] = [add_edge]

                else:

                    edges_add[worker.sector].append(add_edge)

            ## If the new and old firm is not None, then add the edges to the output
            if rem_edge[1] != None:

                # This could be avoided by setting k:[] instead of k:None above, but that would then require a few
                # changes in the Network.updateEdges() method. This should be straightforward and should be handled soon(ish)
                if edges_rem[worker.sector] == None:

                    edges_rem[worker.sector] = [rem_edge]

                else:

                    edges_rem[worker.sector].append(rem_edge)
        
        
            ## If the worker changed firms, update that firm's quota (and remove from the hiring pool if the quota is zero)
            if worker.firm is not None:
                
                f_new = worker.firm
                
            else:
                
                f_new = None
               
                
            ## If a worker has started a new firm, the firm won't be in the hiring list. Skip the step below to avoid errors
            if f_new not in firms_hiring:
                
                continue
                
                
            if (f_old != f_new) and (f_new != None):
                
                ## Subtract 1 from the hiring quota
                self.Firms[worker.firm].updateHiringQuota(1)
                
                
                ## If that satisfies the quota, remove the firm from the hiring list
                if self.Firms[worker.firm].hiringQuota == 0:
                    
                    firms_hiring.remove(worker.firm)
            
            ## Note that if there are no firms_hiring left, this will be automatically handled by __matchWorker() 
            ## That is, an approporiate migration to a preferred location while unemployed will occurr
            
        
        ## Update the Networks if there are any edges to be added or removed
        for sector in edges_add.keys():

            # if edges_add[sector] != None:
            #     
            #     self.Network_FirmWorker[sector].updateEdges(rem_edges = edges_rem[sector], 
            #                                                 add_edges = edges_add[sector])
            self.Network_FirmWorker[sector].updateEdges(rem_edges = edges_rem[sector],
                                                        add_edges = edges_add[sector])
                
    
    
    
    def __checkRemote(self,
                      worker,         # pointer to Worker object in Population Field
                      firm,           # pointer to Firm object in Population Field
                      partial = False # (placeholder) option to set some partial remote working conditions
                      ):
        """
        Check whether the worker and firm will enter a remote working 
        arrangment 
        
        Note that a future feature of this function will allow for partial 
        remote work. This will be implemented by constraining the number of 
        locations for a worker to live to some "close" linked locations to the 
        firm. i.e. this will allow for further moves to "suburbs" but not 
        to other locations far away
        
        Parameters:
        ----------
        
            worker: laborFlow.Worker object (pointer)
                A (pointer to a) Worker in the Population field of the Economy
                
            firm: laborFlow.Firm object (pointer)
                A (pointer to a) Firm in the Firms field of the Economy
        
        
        Returns:
        -------
        
            isRemote: bool
                Indicates whether the firm-worker have entered a remote 
                arrangment
                
            isPartial: bool
                (placeholder) Indicates whether this should be a partial 
                remote working arrangment
        """    
        
        #### Input Checking ####
        
        
        
        #### Begin Function ####
        
        ## Evaluate whether the worker will have a remote preference for this iteration
        worker_pref = np.random.uniform(0, 1) < worker.remote_pref # Does the worker have a remote preference this iteration?
        firm_pref = np.random.uniform(0, 1) < firm.remote_pref     # Will the firm tolerate remote work this iteration?
        
        isRemote = worker_pref & firm_pref
        isPartial = False                  # Placeholder
        
        ## Placeholder - if a remote arrangement exists, check to see whether it will be partial or not
        if isRemote:
            
            isPartial = False
            if partial:
                
                ## Placeholder for checking for partial remote arrangements in future iterations of the code
                pass 
        
        
        return isRemote, isPartial
    
    
    
    def __getLaborSupply(self):
        """
        Get the mean wage per location, per sector in the model and return 
        the appropriate location/sector dictionary        

        Returns
        -------
        w_s: dict
            A dictionary of a dictionary of mean wages by sector and location
            
        
        Note - think about this some. If this is the consumption supply(?), 
               shouldn't this be the mean wage by location not sector and 
               location??
        """
        
        
        w_s = {}
        w_s_out = dict.fromkeys(self.Locations.keys())
        sectors = self.sector_labor_hash
        for l_name in self.Locations.keys():
            
            ## Instantiate the dicionary with sector names and no wages
            w_s[l_name] = {key: [0] for key in sectors}
            for w in self.Locations[l_name].Workers:
                
                ## Add the wage to the appropriate list
                w_s[l_name][w.sector].append(w.wage)
        
        
            ## Get the mean wage per sector
            w_s_out[l_name] = {key: np.mean(val) for key, val in w_s[l_name].items()}
            
            
        return w_s_out
    
    
    
    def __getHiringFirms(self, 
                         w_s,                # Supply of labor in the sector/location (returned from self.__getLaborSupply())
                         w_f,                # Curernt wage offered by each firm
                         gamma = 0,          # If gamma is a value other than zero, calculate agglommeration-based productivity
                         store_vals = False  # Optionally, store the values for the firm
                         ):
        """
        Get a list of the firms that have openings            

        Returns
        -------
        firms_hiring: list, str
            A list of the firm names (strings) that are hiring
        """
        
        ## If agglommeration impacts are required, calculate location/sector pairs once and re-use them in the loop below
        if gamma != 0:
            
            X = {}
            for loc in self.Locations.keys():
                
                X[loc] = {}
                for sector in self.sector_labor_hash.keys():
                    
                    n_workers = len(self.sector_labor_hash[sector])
                    X[loc][sector] = self.Locations[loc].calcProductivityShift(n_workers)
                    
        
        
        firms_hiring = []             # Instantiate the firm list
        for f in self.Firms.values(): # Cycle through all the firms in the model
        
        
            ## Only care about firms with employees...
            if len(f.labor) > 0:
        
                if gamma == 0:
                    
                    # calculate the optimal number of labor based on current conditions
                    N_hire = f.calcOptimumLabor(w_s[f.location][f.sector], 
                                                w_f[f.name],
                                                self.Locations[f.location].Productivity[f.sector]
                                                )
                    
                    ## Calculate the profit for statistics tracking purposes
                    prof = f.calcProfit(X = self.Locations[f.location].Productivity[f.sector],
                                        N = len(f.labor),
                                        w_s = w_s[f.location][f.sector],
                                        w_f = w_f[f.name],
                                        set_vals = store_vals
                                        )
                    
                    f.setProfit(prof)
                
                else:
                    
                    X_ls = X[f.location][f.sector]
                    
                    # calculate the optimal number of labor based on current conditions
                    N_hire = f.calcOptimumLabor(w_s[f.location][f.sector], 
                                                w_f[f.name],
                                                X_ls
                                                )
                    
                    ## Calculate the profit for statistics tracking purposes
                    prof = f.calcProfit(X = X_ls,
                                        N = len(f.labor),
                                        w_s = w_s[f.location][f.sector],
                                        w_f = w_f[f.name],
                                        set_vals = store_vals
                                        )
                    
                    f.setProfit(prof)
                    
                
                ## Update the hiring quota
                f.setHiringQuota(N_hire)
                
                
                ## If the firm is hiring, add it to the list to return
                if f.hiringQuota > 0:
                    
                    firms_hiring.append(f.name)
                    
                    
            else:
                
                if store_vals:
                    
                    if gamma == 0:
                    
                        f.setProfitComponentVals(0, X = self.Locations[f.location].Productivity[f.sector], 
                                                 w_s = w_s[f.location][f.sector], w_f = 0, k_s = 1)
                    
                    else:
                        
                        f.setProfitComponentVals(0, X = X[f.location][f.sector], 
                                                 w_s = w_s[f.location][f.sector], w_f = 0, k_s = 1)
                        
                        
                else:
                
                    f.setProfit(0)
                   
            
        
        
        return firms_hiring
    
    
    
    def __setPreviousFirmWorkerConnections(self,
                                           alpha = .95 # How many workers should have a previous connection?
                                           ):
        
        
        """
        Set the worker connections for the previous firm for all workers with 
        some probability, alpha, that the previous firm will be None (Unemployed)
                
        """
        
        for worker in self.Population.values():
            
            ## Was the worker previously employed?
            was_employed = np.random.uniform(0, 1) < alpha
            
            
            ## If so, randomly grab a firm from the worker's sector and append it to the previous list
            if was_employed:
                
                sector = worker.sector
                firm_match = True
                while firm_match:
                    
                    ## Be sure that the old firm isn't the same as the new one...
                    old_firm = np.random.choice(self.sector_firm_hash[sector], size = 1)[0]                     
                    firm_match = old_firm == worker.firm
                    
                    
                worker.setPreviousFirm(old_firm)
                
                
                ## Add the edge to the appropriate graph
                self.Network_FirmWorker[sector].updateEdges(add_edges = [[worker.ID, old_firm]])
                
                
            ## Otherwise, check for a None/None situation. It may rarely happen, but if so add the worker node to the graph 
            if (not was_employed) and (worker.firm == None):
                
                self.Network_FirmWorker[worker.sector].addNodes(node_names = [worker.ID], node_types = ['worker'])
                
        
        
    def __recordStats(self,
                      time_step # Pass in a time step
                      ):
        """
        Calculate and Record all of the statistics at the end of each 
        simulation step
        
        Parameters:
        ----------
        
        time_step: int
            Pass in the integer value of the time step. 
            
            
        Returns:
        -------
            
        stats_loc: pandas.DataFrame
            A pandas DataFrame of Location Statistics
            
        stats_firm: pandas.DataFrame
            A pandas DataFrame of Firm Statistics
            
        stats_labor: pandas.DataFrame
            A pandas DataFrame of Labor Statistics
            
        stats_network: pandas.DataFrame
            A pandas DataFrame of Network Statistics    
        
        """
        
        ## Get Labor Statistics
        labor_dict = {key: w.recordStats() for key, w in self.Population.items()}
        stats_labor = statsToDataFrame(labor_dict, time_step = time_step)
        
        
        ## Get Firm Statistics 
        firm_dict = {key: f.recordStats() for key, f in self.Firms.items()}
        stats_firm = statsToDataFrame(firm_dict, time_step = time_step)
        
        
        ## Get Both types of network Statistics
        net_dict = {}
        net_smry_dict = {}
        stats_net = {}
        for key, n in self.Network_FirmWorker.items():
            
            
            net_dict[key], net_smry_dict[key] = n.recordStats(self.sector_firm_hash[key])
        
            stats_net[key] = statsToDataFrame(net_dict[key], time_step, orient = 'columns')
        
        stats_network = statsToDataFrame(net_smry_dict, time_step)
        
        ## Unpack the firm network data frame and then merge with the firm statistics
        stats_net = pd.concat(stats_net.values(), ignore_index = True)
        stats_firm = stats_firm.merge(stats_net, on = ['Firm Name', 'Time Step', 'Sector'], copy = False) # Merge Node/Firm network statistics with Firm Statistics
        
        
        
        ## Get Location Statistics
        sectors = list(self.sector_firm_hash.keys())
        loc_dict = {key: l.recordStats(sectors, firm_stats = firm_dict, worker_stats = labor_dict) for key, l in self.Locations.items()}
        loc_dict = {key: statsToDataFrame(l.recordStats(sectors, firm_stats = firm_dict, worker_stats = labor_dict), time_step) for key, l in self.Locations.items()}
        stats_loc = pd.concat(loc_dict.values(), ignore_index = True)
        
        
        ## Return all of the appropriate data
        return stats_loc, stats_firm, stats_labor, stats_network
        