# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 23:25:06 2021

@author: Zach


Functions for Labor/Workers/Agents within the laborFlow package

"""

from math import log, exp
import sys
eps = sys.float_info.epsilon

"""
Define a 'Worker' agent class
"""
class Worker():
    
    def __init__(self,            
                 location,              # The name of a location where this worker lives
                 location_pref,         # A dictionary of the idiosyncratic worker preferences for each location
                 sector         = None, # What sector is the worker in?
                 firm           = None, # If employed, pass the current firmID this worker is associated with, if unemployed, pass None
                 remote_pref    = 0,    # Placeholder to set some preference for remote work (between 0 and 1)
                 ID             = None  # Provide an optional worker ID (necessary for creating the networks)
                 ):
        """
        Worker Class Constructor method
        """
        
        self.wage = 0            # Store the workers wage (possibly wage history in the future)
        self.realWage = 0        # Nominal wage / rent
        self.location = location # Store the location that the worker is in (possibly history as well in the future)

        self.rent = 0            # This ultimately may be a location-specific thing for the initial model, but perhaps we could update this in the future
        
        self.utility = 0         # Store the workers utility
        
        self.location_prefs = location_pref # A dictionary of location preferences
        
        ## Get (and set) the preferred location.
        ## Note that currently this does not change in the model
        self.preferredLocation = max(location_pref, key = location_pref.get)
        self.inPreferredLocation = self.preferredLocation == location
        
        self.firm = firm      # Who does the worker work for (if empty, the worker is unemployed)
        self.firm_last = None # Who was the last firm the worker was associated with? 
        
        self.sector = sector  # What sector is the worker in? For simplicity, a worker will not change sectors after initially set
        
        self.remote_pref = remote_pref
        
        self.ID = ID # Set the worker ID
        
        self.is_remote = False # Is the worker remote? (This identifier is necessary because remote work does not necessarily imply different location than firm)
        
        
        self.curr_loc_pref = 0 # What is the preference for the current location the worker is in
        self.Ammenity = 0      # What is the ammenity of the current location where the worker resides?
        
        
        ## Future attributes (placeholders for a more complicated model)
        ## O-ring model is probably a good place and works within a Cobb-Douglass framework
        ## Perhaps an explanation here is that "productivity" is some combination of skill/effort (and perhaps that value could change...)
        # self.skillset     = [] # Give the worker some skill
        self.ability      = [] # Give the worker some ability to match each skill
        self.productivity = [] # How valuabable is the employee in terms of total output?

       
    
    
    def calcUtility(self, wage, rent, A, pref, update_vals = False):
        """
        Straightforward calculation of utility: 
        
            u_i = w_{if} - r_l + A_l + e_{il}
            
            for worker i, firm f, and location l
        
        Returns
        -------
        utility - a floating point representation of the Worker's utility
        
        """
        
        ## Scale rent (or comment out to unscale...)
        rent = log(rent + eps)
        
        # utility = log(wage + eps) - log(rent + eps) + A + pref 
        utility = wage - rent + A + pref 
        # utility = wage - rent + exp(A) + exp(pref) 
        
        
        ## If requested, update the worker preferences that go into the utility calculation
        if update_vals:
            
            self.wage = wage
            self.rent = rent
            self.Ammenity = A
            self.curr_loc_pref = pref
            self.utility

            if wage == 0:

                self.realWage = 0

            else:

                self.realWage = wage / rent
        
        else:
        
            return utility
    
    
    
    def setLocation(self, location):
        """
        Set (or update) the location where the worker currently resides, as well as the preferred Location identifier
        
        Returns
        -------
        None
        
        """
        
        self.location = location
        self.inPreferredLocation = location == self.preferredLocation
    
    
    
    def setLocationPrefs(self, loc_prefs):
        """
        Pass in a dictionary of location names and preferences - this will set the 
        location_pref dictionary
        
        Represents idiosyncratic preferences for each location in the model:
            
            e_{il}
            
            for worker i, location l
        
        Returns
        -------
        None
        
        """
        
        self.location_prefs = loc_prefs
    
    
    
    def setFirm(self, firm):
        """
        Who does the worker work for? If unemployed, None
        
        Pass the firm ID where the worker is currently employed 
        
        
        Returns
        -------
        None
        
        """        
        
        self.firm = firm
        
        
    def setPreviousFirm(self, firm):
        """
        Set the previous firm
        
        Pass the firm ID where the worker was previously employed (only a single previous instance is saved)
        
        Returns
        -------
        None
        
        """
        
        if firm == None:
            
            return None
        
        else:
            
            self.firm_last = firm
    
    
    def updateRemote(self, is_remote):
        """
        Update the is_remote attribute indicating whether the firm/worker are engaged in a remote working relationship
        
        Returns
        -------
        None
        
        """
        
        self.is_remote = is_remote
        
        
        
    def setWage(self, wage):
        """
        Setter method for wage
        """
        
        self.wage = wage
        

    def setRealWage(self, realWage):
        """
        Setter method for wage
        """
        
        self.wage = realWage
        
        
    def setRent(self, rent):
        """
        Setter method for rent

        Parameters
        ----------
        rent : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.rent = rent
        
        
    def setUtility(self, utility):
        """
        

        Parameters
        ----------
        utility : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.utility = utility
               
        
        
    def recordStats(self):
        """
        Record and return some basic statistics for the worker
        
        Call this method from either the Locations.Location class or from the 
        Economy module and aggregate appropriately
        
        Returns
        -------
        stats: a dictionary of point-in-time statistics for each worker
        """
        
        ## Store Statistics
        stats = {'location': self.location,
                 'rent': self.rent,
                 'utility': self.utility,
                 'isFavorite': self.inPreferredLocation,
                 'firm': self.firm,
                 'wage': self.wage,
                 'realWage': self.realWage,
                 'isRemote': self.is_remote,
                 'sector': self.sector,
                 'ID': self.ID,
                 'isEmployed': self.firm != None,
                 # Need to determine and store if the worker is co-located even if remote
                 
                 'Ammenity': self.Ammenity,
                 'Current Location Preference': self.curr_loc_pref
                 }
        
        
        return stats