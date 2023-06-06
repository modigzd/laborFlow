# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:34:10 2021

@author: Zach

Utilities for aggregating/calculating statistics for Firms, Labor, Locations,
and Economies
"""


import pandas as pd



def statsToDataFrame(stats_dict,      # A Dictionary of statistics, to be converted to a data.frame
                     time_step,       # The time step that should be appended once converted
                     orient = 'index' # By default orient by index. For Graph measues, use columns
                     ):
    """
    Basic convienence function for casting a dictionary of individual stats 
    (Firm, Worker, Location, Network, etc) to a pandas DataFrame and appending
    the appropriate time-step to the output

    Parameters
    ----------
    stats_dict : dictionary of dictionaries
        A Dictionary of statistics created from a calcStats()/returnStats() 
        method, typically by Firm/Worker/Location key
    
    time_step : int
        Pass a time step to be appended to a column of the DataFrame
        
    orient : string
        Defaults to 'index' for most applications within this package. Set to 
        'columns' for network measures which are formulated with a different 
        underlying dictionary structure

    Returns
    -------
    df : pandas DataFrame
        A pandas DataFrame of individual statistics
    """

    ## Cast the data frame
    df = pd.DataFrame().from_dict(stats_dict, orient = orient)
    
    
    ## Append the Time-Step column
    df['Time Step'] = time_step
    
    
    ## Release the Index if 'index' was passed
    df.reset_index(drop = True, inplace = True)
        
    
    
    return df
    
    

def calcMeanStats(df # A pandas data frame, typically from returnStats() or calcStats() methods from laborFlow class objects
                  ):
    """    
    Calculate the mean statistics of various pandas data frame outputs from 
    laborFlows.Economy methods
    
    Parameters
    ----------
    df : pandas DataFrame
       Typically the output of returnStats() or calcStats() methods from 
       laborFlow class objects 
    
    Returns
    -------
    df : pandas DataFrame
        Mean of the objects passed in

    """
    
    ## Get the signature
    
    ## Calculate the mean on the appropriate numeric columns and set the 
    ## non-numeric columns as appropriate
    
    ## May also want to grab standard deviation and mode...
    
    pass




def getStatsSignature(df
                      ):
    """
    Pass a stats DataFrame and determine whether it is a Network, Labor, Firm,
    Economy, or Location output.
    
    Parameters
    ----------
    
    df: pandas DataFrame
        This is the output of a stats agglomeration method from 
        laborFlows.Economy objects
    
    Returns
    -------
    
    signature : string
        A string signifying which stat object the data frame belowngs to
        
    numeric_cols : list of strings
        Which columns are numeric?
        
    replace_col : list of strings
        Which columns can or should be replaced (i.e. worker IDs to a 
        placeholder)
    """
    
    ## Define columns of each signature type
    cols = {'Labor': ['location', 'rent', 'utility', 'isFavorite', 'firm', 'wage', 
                      'isRemote', 'sector', 'ID', 'isEmployed', 'Time Step'],
            
            ## Firm Statistics
            'Firms': ['Total Workers', 'Location', 'Sector', 'Firm Name', 'Remote Tolerance', 
                      'Productivity Shift', 'Labor Elasticity', 'Profit', 'Production',
                      'Capital', 'Remote Workers', 'Preferred Location', 'Co-Located', 'Time Step'],
            
            ## Network-specific Firm Statistics
            'Network': [],
            
            ## Summary Measures for a network
            'Network Summary': [],
            
            ## Location Statistics
            ## Note that currently networks are not summerized per location, think about adding this
            'Location': ['Name', 'rent', 'Ammenities', 'Sector', 'Productivity Shifter',
                         'Firm Mean Profit', 'Firm Mean Production', 'Firm Mean Capital',
                         'Firm Mean Workers', 'Firm Mean Remote Tolerance',
                         'Firm Mean Remote Workers', 'Firm Mean Worker Preferred Loc',
                         'Firm Mean Productivity Shift', 'Firm Mean Co-Located', 'Total Firms',
                         'mean wage', 'mean utility', 'N workers', 'N Employed', 'remote workers',
                         'preferred location', 'Time Step']
            }
        
    
    ## Check to see if all of the columns are in the dataframe
    df_type = {}
    for c in cols.keys():
        
        df_type[c] = set(df.columns).issubset(cols[c])
    
    ## If 0 or more than 1 match, throw an error 
    ## (more than 1 is a debugging tool and should probably be moved to tests eventually)
    
    
    
    ## Define the appropriate numeric and replace columns, then return everything
    
    pass






