# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 23:25:06 2021

@author: Zach


Functions for Locations within the laborFlow package

"""


## Import necessary dependencies here
import numpy as np
import sys
eps = sys.float_info.epsilon

"""
Define a 'City' agent class

Note here that city can be urban or rural, this is simply a class that stores 
the appropriate attributes of the city and links to the Firms/Workers in it

This could even further be generalized to encompass a location within a location...
"""
class Location():
    
    def __init__(self,  
                 housing_elasticity,       # How elastic is the local housing supply? This value is greater than or equal to 0 (0 is perfectly elastic)
                 local_ammenities,         # A measure of local ammenities - for now this is static, and applies to all agents within the model (which may be ok - we capture idiosyncratic preferences already, which may be enough)
                 productivity_shifter,     # This is dictionary of productivity shifters for each industry sector in the model
                 location_ID,              # Set a unique character input for the location Name/ID (e.g., 'Urban' or 12345)
                 gamma                 = 0 # What is the strength of the agglomeration effect?
                 ):
        
        """
        Class Constructor for Location
        
        To Do: Add appropriate type checks against inputs....
        """
        
        ## General info
        self.Location_ID = location_ID # Set the location ID - note that there are no uniqueness checks within this class...
        # self.History = []              # Placeholder - will likely be a pandas dataframe in the future to store summary data of the firm on a per-time step basis

        ## City Specific information
        self.Productivity       = productivity_shifter # Productivity Shifter (this needs to be generalized to the firm, or made into some policy choice in the future when trying to attract a specific type of firm...)
        self.Housing_Elasticity = housing_elasticity   
        self.Ammenities         = local_ammenities
        
        
        ## Agglomeration strength
        self.agglomeration_strength = gamma


        ## Current Rent (Housing Supply)
        self.Rent = eps
        
        
        ## May get tracked here, may get tracked at world level (could cause uneccessary memory to be taken up)
        self.Firms = []                # Pointer to the firms in the city. This is a point-in-time list
        self.Workers = []              # Pointer to the workers in the city. This is a point-in-time list

        ## Set some additional Location-specific information (not currently utilized)
        # self.Total_Land = 10000000     # Set some cap on housing, optionally. This is expressed in terms of people (1 person = 1 land)  
    
    
    
    
    def calcHousingSupply(self,
                          N,      # Total Number of workers in the location, note for now I leave this as an explicit input, but may change this in the future depending on how I handle the networks and whether a city "knows" which firms and workers occupy t
                          z =   0 # Set some baseline to move the housing supply up or down 
                          ):
        """
        Calculate the housing supply according to eq (7):
            
            r = z + kN
        
        Returns
        -------
        r: the rent supply for the location
        """
        
        # Straightforward calculation according to eq (7) from Moretti
        # r = z + self.Housing_Elasticity * N
        r = z + N**(self.Housing_Elasticity)

        ## Store the rent. Note that if it is 0, this will cause downstream problems when calculating real wages
        ## Eventually this should be sorted out, but for now use this hack workaround just in case
        if r == 0:
            r_record = eps
        else:
            r_record = r

        self.Rent = r_record
        

        return r

    


    def calcHousingDemand(self):
        """
        Calculate the housing demand for the city according to eq (6):
            
            rb = (wb - wa) + ra + (Ab - Aa) - s * (Nb - Na) / N
            
            
        Note sure this is actually needed here because the housing demand is 
        just the number of people in a given location...
        
        Maybe something here is needed to provide some sort of "locality" adjustment...
        """
                
        pass
    
    
    
    def updateAttributes(self, **kwargs):
        """
        A method for updating the ammenities, housing elasticity, productivity multiplier        

        Returns
        -------
        None.

        """
        
        ## To Do: Perhaps add a check against the keyword args? And raise an exception if none exist...
        
        
        ## Update housing_elasticity if it was passed
        if 'housing_elasticity' in kwargs.keys():
            
            self.Housing_Elasticity = kwargs['housing_elasticity']
        
        
        ## Update Ammenities if it was passed
        if 'local_ammenities' in kwargs.keys():
            
            self.Ammenities = kwargs['local_ammenities']
            
            
        ## Update productivity multiplier if it was passed
        if 'productivity_shifter' in kwargs.keys():
            
            self.Productivity = kwargs['productivity_shifter']
            
            
        ## Update the agglommeration effect if it was passed
        if 'gamma' in kwargs.keys():
            
            self.agglomeration_strength = kwargs['gamma']

        
        ## Update the Rent if it was passed
        if 'Rent' in kwargs.keys():

            self.Rent = kwargs['Rent']
        
        
        ## Doesn't return anything...modifies object in place
        return None
        


    def calcProductivityShift(self,
                              N_s        # Number of workers in the location in the sector
                              # sector
                              ):
        """
        Calculate the productivity shift as a function of the number of workers
        in the sector in the location
        """
        
        ## This isn't super effecient. For very large models could be parallelized
        # N = sum([1 for w in self.Workers if w.sector == sector]) # Get total workers in the sector in the location
        X = N_s**self.agglomeration_strength                       # productivity shift 
        
        
        ## Set the value of the shift??
        
        
        return X



    def updateFirm(self,
                   Firm,       # Pointer to a Firm object (i.e. the object itself)
                   by =  'add' # 'add' or 'remove'
                   ):
        """
        Add or remove pointers to a Firm from the Firms list
        
        """
        
        if by == "add":
            
            # self.labor.append(worker_ID)
            self.Firms.append(Firm)
            
        elif by == "remove":
            
            # self.labor.remove(worker_ID)
            self.Firms.remove(Firm)
            
        else:
            
            raise Exception("Only 'add' or 'remove' are acceptable inputs to 'by'") 
        


    def updateLabor(self,
                    Worker,         # Pointer to a Worker object (i.e. the object itself)
                    by      = 'add' # 'add' or 'remove'
                    ):
        """
        Add or remove pointers to a Firm from the Firms list
        
        """
        
        if by == "add":
            
            # self.labor.append(worker_ID)
            self.Workers.append(Worker)
            
        elif by == "remove":
            
            # self.labor.remove(worker_ID)
            self.Workers.remove(Worker)
            
        else:
            
            raise Exception("Only 'add' or 'remove' are acceptable inputs to 'by'") 
            



    def recordStats(self,
                    sectors,     # Names of each sector in the model
                    firm_stats,  # A dictionary of statistics by Firm
                    worker_stats # A dictionary of statistics by worker 
                    ):
        """
        Calculate the current location statistics
        Parameters
        ----------
        sectors: list
            List of the names of each sector in the model.
            
        firm_stats: dictionary
            aggregated dictionary of statistics by Firm. 
            Output of Economy.getFirmStats() which aggregates Firms.Firm.recordStats()
            
        worker_stats: dictionary
            aggregated dictionary of statistics by Worker. 
            Output of Economy.getLaborStats() which aggregates Labor.Worker.recordStats()
            
            
        Returns
        -------
        
        Returns a nested dictionary by sector  
        """
        
        ## Determine total number of workers in the city (and set a workaround if 0)
        N = len(self.Workers)
        
           
        stats = {}        
        n_rent = np.log(N + eps)
        rent = self.calcHousingSupply(n_rent, z = 0)
        ## Calculate average wage/utility by Sector (include a "Total" as well) 
        ## This *might* be an ok place to parallelize once I move to c++
        for sector in sectors:
            
            stats[sector] = {}
            
            ## Append basic statistics
            
            stats[sector]['Name'] = self.Location_ID
            stats[sector]['rent'] = rent
            stats[sector]['Ammenities'] = self.Ammenities
            stats[sector]['Sector'] = sector
            
            
            ## Record Productivity Shift
            ## To Do: Update this to sector-specific when finally incorporating agglomeration impacts
            stats[sector]['Productivity Shifter'] = self.Productivity[sector] 
            
            
            ## Calculate Mean Firm Stats
            tmp_profit = []
            tmp_production = []
            tmp_capital = []
            tmp_labor = []
            tmp_remote_tol = []
            tmp_remote_w = []
            tmp_preferred_w = []
            tmp_prod_shift = []
            tmp_coloc = []
            tmp_firm_wage = []
            tmp_firm_realWage = []
            tmp_loc_wage = []
            tmp_loc_capital = []
            tmp_loc_prod_shift = []
            tmp_hiring_quota = []
            n = 0
            for fID in firm_stats.keys():
                
                if firm_stats[fID]['Sector'] == sector:
                    
                    if firm_stats[fID]['Location'] == self.Location_ID:
                                                
                        tmp_profit = tmp_profit + [firm_stats[fID]['Profit']]
                        tmp_production = tmp_production + [firm_stats[fID]['Production']]
                        tmp_capital = tmp_capital + [firm_stats[fID]['Capital']]
                        tmp_labor = tmp_labor + [firm_stats[fID]['Total Workers']]
                        tmp_remote_tol = tmp_remote_tol + [firm_stats[fID]['Remote Tolerance']]
                        tmp_remote_w = tmp_remote_w + [firm_stats[fID]['Remote Workers']]
                        tmp_preferred_w = tmp_preferred_w + [firm_stats[fID]['Preferred Location']]
                        tmp_prod_shift = tmp_prod_shift + [firm_stats[fID]['Productivity Shift']]
                        tmp_coloc = tmp_coloc + [firm_stats[fID]['Co-Located']]
                        tmp_firm_wage = tmp_firm_wage + [firm_stats[fID]['Wage']]
                        tmp_firm_realWage = tmp_firm_realWage + [firm_stats[fID]['Wage'] / rent]
                        tmp_loc_wage = tmp_loc_wage + [firm_stats[fID]['Location Mean Wage']]
                        tmp_loc_capital = tmp_loc_capital + [firm_stats[fID]['Location Mean Capital']]
                        tmp_loc_prod_shift = tmp_loc_prod_shift + [firm_stats[fID]['Location Productivity Shift']]
                        tmp_hiring_quota = tmp_hiring_quota + [firm_stats[fID]['Hiring Quota']]
                        
                        n += 1
                    
            if n > 0:
                
                
                stats[sector]['Firm Mean Profit'] = np.mean(tmp_profit)
                stats[sector]['Firm Mean Production'] = np.mean(tmp_production)
                stats[sector]['Firm Mean Capital'] = np.mean(tmp_capital)
                stats[sector]['Firm Mean Workers'] = np.mean(tmp_labor)
                stats[sector]['Firm Mean Remote Tolerance'] = np.mean(tmp_remote_tol) 
                stats[sector]['Firm Mean Remote Workers'] = np.mean(tmp_remote_w)
                stats[sector]['Firm Mean Worker Preferred Loc'] = np.mean(tmp_preferred_w)
                stats[sector]['Firm Mean Productivity Shift'] = np.mean(tmp_prod_shift) 
                stats[sector]['Firm Mean Co-Located'] = np.mean(tmp_coloc)
                stats[sector]['Total Firms'] = n
                stats[sector]['Firm Mean Wage'] = np.mean(tmp_firm_wage)
                stats[sector]['Firm Mean Real Wage'] = np.mean(tmp_firm_realWage)
                stats[sector]['Location Mean Wage'] = np.mean(tmp_loc_wage)
                stats[sector]['Location Mean Capital'] = np.mean(tmp_loc_capital)
                stats[sector]['Location Productivity Shift'] = np.mean(tmp_loc_prod_shift)
                stats[sector]['Firm Mean Hiring Quota unfulfilled'] = np.mean(tmp_hiring_quota)
                
                
            else:
                
                stats[sector]['Firm Mean Profit'] = 0
                stats[sector]['Firm Mean Production'] = 0
                stats[sector]['Firm Mean Capital'] = 0
                stats[sector]['Firm Mean Workers'] = 0
                stats[sector]['Firm Mean Remote Tolerance'] = 0
                stats[sector]['Firm Mean Remote Workers'] = 0
                stats[sector]['Firm Mean Worker Preferred Loc'] = 0
                stats[sector]['Firm Mean Productivity Shift'] = 0
                stats[sector]['Firm Mean Co-Located'] = 0
                stats[sector]['Total Firms'] = 0
                stats[sector]['Firm Mean Wage'] = 0
                stats[sector]['Firm Mean Real Wage'] = 0
                stats[sector]['Location Mean Wage'] = 0
                stats[sector]['Location Mean Capital'] = 0
                stats[sector]['Location Productivity Shift'] = 0
                stats[sector]['Firm Mean Hiring Quota unfulfilled'] = 0
            
            
            ## Calculate mean labor stats by sector
            ## Accumulate wage, utility, preferred location, and remote status as lists by sector
            tmp_wage = []
            tmp_realWage = []
            tmp_utility = []
            tmp_isRemote = []
            tmp_isFavorite = []
            tmp_isEmployed = []
            tmp_ammenities = []
            tmp_cur_loc_pref = []
            tmp_wage_emp = []
            tmp_realWage_emp = []
            tmp_utility_emp = []
            n = 0
            n_emp = 0
            for wID in worker_stats.keys():
                
                if worker_stats[wID]['sector'] == sector:
                    
                    if worker_stats[wID]['location'] == self.Location_ID:
                        
                        tmp_wage = tmp_wage + [worker_stats[wID]['wage']]
                        tmp_realWage = tmp_realWage + [worker_stats[wID]['realWage']]
                        tmp_utility = tmp_utility + [worker_stats[wID]['utility']]
                        tmp_isRemote = tmp_isRemote + [worker_stats[wID]['isRemote']]
                        tmp_isFavorite = tmp_isFavorite + [worker_stats[wID]['isFavorite']]
                        tmp_isEmployed = tmp_isEmployed + [worker_stats[wID]['isEmployed']]
                        tmp_ammenities = tmp_ammenities + [worker_stats[wID]['Ammenity']]
                        tmp_cur_loc_pref = tmp_cur_loc_pref + [worker_stats[wID]['Current Location Preference']]

                        ## Track the wage of only the employed
                        if worker_stats[wID]['isEmployed']:

                            tmp_wage_emp = tmp_wage_emp + [worker_stats[wID]['wage']]
                            tmp_realWage_emp = tmp_realWage_emp + [worker_stats[wID]['realWage']]
                            tmp_utility_emp = tmp_utility_emp + [worker_stats[wID]['utility']]

                            n_emp += 1


                        n += 1 
                    
            ## Need to add number of unemployed...
            if n > 0:
                
                stats[sector]['mean wage'] = np.mean(tmp_wage)
                stats[sector]['mean real wage'] = np.mean(tmp_realWage)
                stats[sector]['mean utility'] = np.mean(tmp_utility)

                ## If there was at least 1 employed worker, calculate the nominal/real wage and utility
                if n_emp > 0:

                    stats[sector]['employed mean wage'] = np.mean(tmp_wage_emp)
                    stats[sector]['employed mean real wage'] = np.mean(tmp_realWage_emp)
                    stats[sector]['employed mean utility'] = np.mean(tmp_utility_emp)

                else:

                    ## Otherwise, set the wages to 0 so we don't break anything...
                    stats[sector]['employed mean wage'] = 0
                    stats[sector]['employed mean real wage'] = 0
                    stats[sector]['employed mean utility'] = 0

                
                stats[sector]['N workers'] = n
                stats[sector]['N Employed'] = sum(tmp_isEmployed) 
                
                ## Total Number of workers in preferred location and remote
                stats[sector]['remote workers'] = sum(tmp_isRemote)
                stats[sector]['preferred location'] = sum(tmp_isFavorite)
                
                ## Record mean worker preferences
                stats[sector]['Ammenities'] = np.mean(tmp_ammenities)
                stats[sector]['Current Location Preference'] = np.mean(tmp_cur_loc_pref)
                
            else:
                
                stats[sector]['mean wage'] = 0
                stats[sector]['mean real wage'] = 0
                stats[sector]['mean utility'] = 0

                stats[sector]['employed mean wage'] = 0
                stats[sector]['employed mean real wage'] = 0
                stats[sector]['employed mean utility'] = 0
                
                stats[sector]['N workers'] = 0
                stats[sector]['N Employed']  = 0
                
                ## Total Number of workers in preferred location and remote
                stats[sector]['remote workers'] = 0
                stats[sector]['preferred location'] = 0
                
                ## Record mean worker preferences
                stats[sector]['Ammenities'] = 0
                stats[sector]['Current Location Preference'] = 0 
            
        
        ## The steps below are probably better suited for a step above
        ## Convert these to a pandas Data Frame
        
        
        ## Aggregate Mean and Total Stats for the location and append as a row
        
        
        # ## Finally, calculate mean wage/utility and Productivity/Profit
        # if N > 0:
            
        #     ## Labor Stats
        #     stats['mean wage Total'] = np.mean(worker_stats['wage'])
        #     stats['mean utility Total'] = np.mean(worker_stats['utility'])
        #     stats['remote workers Total'] = sum(worker_stats['isRemote'])
        #     stats['preferred location Total'] = sum(worker_stats['isFavorite'])
        #     stats['Total Employed'] = sum(worker_stats['isEmployed'])
            
        # else:
            
        #     ## Labor Stats
        #     stats['mean wage Total'] = 0
        #     stats['mean utility Total'] = 0
        #     stats['remote workers Total'] = 0
        #     stats['preferred location Total'] = 0
        #     stats['Total Employed'] = 0          


        return stats       
    
    