# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 23:25:06 2021

@author: Zach


Functions for Firms within the laborFlow package

"""


## Import Necessary libraries
import numpy as np
from math import log, exp
import sys
eps = sys.float_info.epsilon



"""
Define a 'Firm' agent class (placeholder)

To Do - Add a pointer array to the Firm workers (which will be modified globally)
Maybe. Need to think about this a bit

"""
class Firm():
    
    def __init__(self, 
                 location_pref      = [],  # A dictionary of the idiosyncratic firm preferences for each location
                 sector             = [],  # Provide an optional industry sector 
                 name               = [],  # Provide an optional Firm Name
                 location           = [],  # Optionally, provide a name of the location where the city resides                 
                 labor_elasticity   = .75, # Labor elasticity in the Cobb-Douglass production function
                 productivity_shift = 0,   # This will be addititive (or subtractative(?)) from the locality specific productivity shifter
                 remote_pref        = 0    # How tolerant of remote working is the firm? Value between 0 and 1 with 1 being fully remote, 0 being no remote work. Both worker and firm must randomly select remote work to enter into such a relationship
                 ):
        
        """
        Firm Class constructor method
        
        
        Note that for now Capital Input in the Cobb-Douglas production function is ignored. This is consistent with 
        Moretti for a single industry type, but could possibly provide additional color if more than a single 
        industry type is provided. Either way, that is a detail for another follow-up model
        """
        
        self.name = name         # Give the Firm a name
        
        self.sector   = sector   # Placeholder for a potential industry sector
        self.location = location # Where is the firm located? (i.e. which city)        
        self.capital  = 1        # What is the capital input (perhaps it won't be considered?) - since the log of 1 is 0...


        ## This variable provides a baseline to keep firms attached to a certain location.
        ## Without it, if there is an imbalance in city-specific productivity multipliers, then eventually all of the firms
        ## in the sector will move to the more productive city
        ## This feature could be removed if there were costs to move, tax implications, etc
        self.location_prefs = location_pref # A dictionary of firm preferences for each location


        self.productivity_shift = productivity_shift
        self.labor_elasticity   = labor_elasticity
        
        self.remote_pref = remote_pref # Global value of how favorable firm is to remote working

        self.labor = []    # Which/how many workers are associated with the firm (this )
        
        
        self.production = 0 # What is the firm's production?
        self.profit = 0     # What is the firm's profit?
        
        self.price = 1      # Unit price of good. For now set this to 1
        
        self.hiringQuota = 0 # How many workers need to be hired to optimize labor at the current wage supply      
        
        ## Store values that go into the Production/Profit calculations
        self.location_productivity_shift = 0
        self.w_s = 0                         # Average wage in the city the firm is in
        self.wage = 0                        # Wage offered by the firm
        self.k_s = 0                         # 


    def calcProduction(self, 
                       X,    # Productivity Ammenity
                       N     # Number of workers
                       ):
        """
        Calculate the production output of the firm. The marginal product of
        labor:
            
            Y = X_f * X * aN^(a - 1) * K^b
                
            
        Note that both here and for wage, under agglommeration economies the 
        Number of workers in the firm and the number of workers in the 
        sector and location are different. I don't believe they should be 
        combined when taking the derivative with respect to N. That is, the 
        value of X can be determined by the location itself
        """
        
        ## Ensure inputs have a floor of 1 
        X_f = max(1, self.productivity_shift) # Firm's productivity shifter
        K = max(1, self.capital)              # Firm's capital
        X = max(1, X)                         # Location Productivity shifter
        
        h = self.labor_elasticity
        
        pr = X_f * X * N**(h) * K**(1-h)
                
        # self.production = log(pr) # Store the production value for quick access
        
        
        return pr
    
    
    

    def calcProfit(self, 
                   X,                # Productivity Ammenity
                   N,                # Total Number of workers
                   w_s,              # average wage in the city in the sector
                   w_f,              # Current wage offered by the firm
                   k_s = 1,          # Cost of capital, assumed to be 1 for the time being while I ignore it...
                   set_vals = False  # Should the appropriate values be updated?
                   ):
        """
        Calculate the profit the firm will take:
            
            pi = p*X*K^(b)*N^a - w_s*N - k_s*K
            
        Note that both here and for wage, under agglommeration economies the 
        Number of workers in the firm and the number of workers in the 
        sector and location are different. I don't believe they should be 
        combined when taking the derivative with respect to N. That is, the 
        value of X can be determined by the location itself
        """
        
        ## Get values and set appropriate wage floors
        price = max(1, self.price)
        X = max(1, X)
        K = max(1, self.capital)
        X_f = max(1, self.productivity_shift)
        
        w_s = max(0, max(w_s, w_f)) # Hmmm...think about this a little more...
        # w_s = max(0, w_f) # The actual profit is actually based on the wage paid by the individual firm... 
        
        h = self.labor_elasticity
        
        
        ## The question is, has wage already been scaled by the number of workers? 
        ## I think the issue is that each firm needs technology/productivity shifter that ultimately helps scale all firm sizes.
        ## This should also be tuned to get an appropriate equilibrium unemployment level
        profit = price * X * X_f * K**(1-h) * N**(h) - w_s * N - k_s * K # Pretty sure this is correct. It gives a max. The alternative below, does not
        # profit = price * X * X_f * K**(1-h) * N**(h) - w_s - k_s       # Probably wrong
        
        
        ## Store the current profit and also return it
        # self.profit = profit
        
        ## If requested, simply set the appropriate values
        if set_vals:
            
            
            self.profit = profit
            self.production = self.calcProduction(X, N)
            
            self.location_productivity_shift = X
            self.w_s = w_s                         # Average wage in the city the firm is in
            self.wage = w_f                        # Wage offered by the firm
            self.k_s = k_s                         # 
        
        
        ## Return the profit
        return profit




    def calcWage(self, 
                 X,    # Productivity Ammenity
                 N     # Total Number of workers
                 # gamma = 0 # Agglommeration input - move this to Locations where the agglomeration value can be calculated based on total workers in the sector in the city
                 ):
        
        """
        The wage is the wage offered for all employees (and will therefore be continually adjusted...)
        
        Labor Demand (i.e. wage) is defined by equation (5) in moretti
        
        Future: Update the wages in some idiosyncratic fashion as described below
        
        Calculate the wage that will be offered to new employees
        
        Note here the wage stagnates for a worker if they choose not to explore other options. Raises should 
        be considered at some point...
        
        It may be useful to use o-ring extension of Cobb-Dougloass here
        
        Note that both here and for wage, under agglommeration economies the 
        Number of workers in the firm and the number of workers in the 
        sector and location are different. I don't believe they should be 
        combined when taking the derivative with respect to N. That is, the 
        value of X can be determined by the location itself
        """
        
        ## If N is zero, the wage is zero...
        if N == 0:
            
            return 0
        
        
        ## Ensure inputs have a floor of 1 
        X_f = max(1, self.productivity_shift) # Firm's productivity shifter
        K = max(1, self.capital)              # Firm's capital
        X = max(1, X)                         # Location Productivity shifter
        
        h = self.labor_elasticity
        
        # Calculate labors marginal contribution to production
        w = X_f * X * h * N**(h-1) * K**(1-h)
        w = w * N
        
        ## If I want to scale back, uncomment this.
        ## Note that the units shouldn't really matter here so long as I'm consistent....
        w = log(max(w, 1))
        
    
        return w
    
    
    
    def calcOptimumLabor(self,
                         w_s,      # Mean wage in the city
                         w_f,      # Current wage offered by the firm
                         X         # Productivity Ammenity
                         ):
        """
        Calculate the optimum number of workers the firm should have based on 
        the firm's production and the mean local wage for the sector
        
        Here, the optimum number of workers is solved by taking the partial 
        derivative with respect to labor (N) of the profit function, pi:
            
            pi = p*X*K^(b)*N^a - w_s*N - k_s*K
            
        Giving:
            
            N = exp((log(w_s) - log(a * p * X) - (b)log(K)) / (a - 1))
            
        Where X is fixed. Note that when X is an agglomeration function of the
        density of local workers, this will possibly need to be updated
        
        Note that currently the term w_s will actually be the maximum of the 
        mean wage in the city in the sector and the wage currently offered by 
        the firm, since firms pay wage in marginal product of labor without 
        trying to minimize labor costs. This may be a useful area to explore 
        in the future...
        
        w_s: mean wage of workers in the city and sector
        a: labor elasticity, here = h
        b: capital elasticity, here = 1 - h
        p: price of good
        X: productivity shifter (Firm plus Local)
        K: capital
        
        
        Return:
        ------
        
        N: Optimal number of workers for the firm to maximize profit
        
        Note that both here and for wage, under agglommeration economies the 
        Number of workers in the firm and the number of workers in the 
        sector and location are different. I don't believe they should be 
        combined when taking the derivative with respect to N. That is, the 
        value of X can be determined by the location itself
        """
        
        ## Get firm values
        ## Get values and set appropriate wage floors
        price = max(1, self.price)
        X = max(1, X)
        K = max(1, self.capital)
        X_f = max(1, self.productivity_shift)
        
        w_s = max(eps, max(w_s, w_f)) # Hmmm...think about this a little more...
        # w_s = max(eps, w_s)
        
        h = self.labor_elasticity
        
        
        ## Calculate the optimal number of workers
        N = (log(w_s) - log(h * price * X * X_f) - (1 - h) * log(K)) / (h - 1) # This is shifted to log space for ease of calculations
        N = exp(N)                                                             # Move back to the units we're working in everywhere else
        
        N = np.ceil(N)                                                         # Whole number of workers...
        
        
        return N
    
    
    
    def setHiringQuota(self, 
                       N     # Optimal number of workers based on current conditions to maximize profit - from calcOptimumLabor()
                       ):
        """
        Set the hiring quota for the period 
        
        N: optimal number of workers calculate from calcOptimumLabor() method
        
        
        sets the hiring quota
        
        Note that workers currently aren't fired if doing so optimizes profits
        Possibly update this in the future
        """
        
        
        ## The maximum number of workers to hire has a floor of zero
        self.hiringQuota = max(0, N - len(self.labor))
    
    
    
    def updateHiringQuota(self, N):
        """
        setter method for updating the hiring quota 

        If hiring a worker, this is +1. If firing a worker, this is -1        
        """
    
        self.hiringQuota = self.hiringQuota - N
        
    
    
    def setLocation(self, location):
        """
        Location setter method - this gets used when a firm chooses to change locations
        """
        
        self.location = location
        
        
        
    def setWorker(self, 
                  # worker_ID,
                  worker,            # pointer to worker object
                  by         = "add" # "add" (hire) or "remove" (fired, quits, retires)
                  ):
        """
        Add or subtract a worker from the firm
        """
        
        if by == "add":
            
            # self.labor.append(worker_ID)
            self.labor.append(worker)
            
        elif by == "remove":
            
            # self.labor.remove(worker_ID)
            self.labor.remove(worker)
            
        else:
            
            raise Exception("Only 'add' or 'remove' are acceptable inputs to 'by'") 
            
            
    def setCapital(self,
                   K):
        """
        Setter method to update the capital (K) input that goes into the firm's
        production function
        """
        
        self.capital = K
            
            
            
    def updateRemotePref(self, remote_pref):
        """
        Setter method for updating the remote preference
        """
        
        self.remote_pref = remote_pref
        
        
    def setProfit(self, profit):
        """
        Setter method for updating profit (since it may be called repeatedly without moving...)
        """
        
        self.profit = profit


    def setProfitComponentVals(self, profit, X, w_s, w_f, k_s):
        """
        Set the values for the components of profit
        """
        
        self.profit = profit
        self.production = self.calcProduction(X, len(self.labor))
        
        self.location_productivity_shift = X
        self.w_s = w_s                         # Average wage in the city the firm is in
        self.wage = w_f                        # Wage offered by the firm
        self.k_s = k_s                         # 
        
        
    def recordStats(self):
        """
        Record point-in-time firm statistics 
        """
        
        ## Get the number of workers
        N = len(self.labor)
        
        
        ## Record some basic statistics
        stats = {'Total Workers': N,
                 'Location': self.location,
                 'Sector': self.sector,
                 'Firm Name': self.name,
                 'Remote Tolerance': self.remote_pref,
                 'Productivity Shift': self.productivity_shift,
                 'Labor Elasticity': self.labor_elasticity,
                 'Profit': self.profit,
                 'Production': self.production,
                 'Capital': self.capital,
                 'Hiring Quota': self.hiringQuota,
                 'Wage': self.wage,
                 'Location Mean Wage': self.w_s,
                 'Location Mean Capital': self.k_s,
                 'Location Productivity Shift': self.location_productivity_shift 
                 }
        
        
        ## Calculate additional statistics, if the firm has workers
        remote_workers = 0
        preferred_location = 0
        co_located = 0
        if N > 0:
            
            for ii in range(N):
            
                remote_workers += self.labor[ii].is_remote  
                preferred_location += self.labor[ii].inPreferredLocation
                co_located += (self.location == self.labor[ii].location)
                
            
        stats['Remote Workers'] = remote_workers
        stats['Preferred Location'] = preferred_location
        stats['Co-Located'] = co_located
        
        
        return stats
        