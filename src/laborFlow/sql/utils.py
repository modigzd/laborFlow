# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:37:20 2021

@author: Zach


Utilities for connecting to and writing to a local database. 

Psycopg2:
    https://towardsdatascience.com/python-and-postgresql-how-to-access-a-postgresql-database-like-a-data-scientist-b5a9c5a0ea43
    
"""


## Note that this should probably become a class...
## Though maybe not. Maybe only the top-level writeStatsTables and 
## writeSimulationParameters tables get called



def connectDB():
    """
    Placeholder - connect to a database by creating a cursor object
    """
    
    pass



def writeStatTables(close_conn = True):
    """
    Placeholder - write the simulation results from a run to the appropriate 
    tables in the database
    """
    pass



def writeLaborStats(close_conn = True):
    """
    Placeholder - write the labor results from a simulation run to the labor 
    tables (raw and mean)
    """
    
    pass


def writeFirmStats(close_conn = True):
    """
    Placeholder - write the firm results from a simulation run to the firm 
    tables (raw and mean)
    """
    
    pass


def writeLocationStats(close_conn = True):
    """
    Placeholder - write the location results from a simulation run to the 
    location tables (raw and mean)
    """
    
    pass


def writeNetworkStats(close_conn = True):
    """
    Placeholder - write the network results from a simulation run to the 
    network tables (raw and mean)
    """
    
    pass



def writeSimulationParameters(close_conn = True):
    """
    Placeholder - write the simulation parameters (locations, labor, firms) to
    the appropriate simulation parameters tables
    """
    
    pass



def addSimulationID(close_conn = True,
                    over_write = False
                    ):
    """
    Placeholder - this writes to a top level table that creates a simulation ID
    after checking to ensure that the seed and parameters do not already exist
    in the database
    
    If the parameters match, this will give the user the option to 
    delete/overwrite previous results
    """
    pass



def addShockParameters(close_conn = True,
                    over_write = False
                    ):
    """
    Placeholder - write any shock parameters to the separate shock tables
    (Labor, Firm, and Location shocks for now)
    """
    
    
    pass



def deleteResults(simulation_id, 
                  close_conn = True
                  ):
    """
    Search for all tables by simulation_id (global primary key) and delete the
    associated results
    """
    
    pass

