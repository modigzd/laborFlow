# -*- coding: utf-8 -*-
"""
debug the laborFlow package

@author: zachm
"""



import sys
import random
import numpy as np
import time
# import pandas as pd

## Point to the location where the Game and Agent class definitions are
sys.path.insert(0, r'C:\Users\zachm\Documents\George Mason\Projects\Labor Flow Python Package\src\working_debug_results')


import laborFlow as lf
# from laborFlow.utils.statistics import statsToDataFrame


# Set a seed for reproducible results
## Note that I'm using both random and numpy throughout the code, so I need to seed both here...
## Should pass this as an argument to simulate...
random.seed(202012)
np.random.seed(202012)


## Create an economy
test = lf.Economy()


## Add Two locations
# test.addLocation(housing_elasticity = .5, local_ammenities = np.log(6), productivity_shifter = 8, location_ID = "Location 1")
# test.addLocation(housing_elasticity = .5, local_ammenities = np.log(3), productivity_shifter = 4, location_ID = "Location 2")
test.addLocation(housing_elasticity = .5, local_ammenities = np.log(6), productivity_shifter = 1, location_ID = "Location 1")
test.addLocation(housing_elasticity = .5, local_ammenities = np.log(3), productivity_shifter = 2, location_ID = "Location 2")


### Add 50 firms in two sectors
test.addAllFirms(n_firms = 150, 
                 loc_pr = {'Location 1': .7, 'Location 2': .3},
                 sec_pr = {'Sector 1': .6, 'Sector 2': .4},
                 # remote_prefs = {'Sector 1': .2, 'Sector 2': .5}
                 remote_prefs = None
                 )


## Add 2000 workers
test.addLabor(n_workers = 5000,
              location_prefs = {'Location 1': [-2, 2], 'Location 2': [-1, 3]}, 
              location_prop = {'Location 1': .8, 'Location 2': .2},
              sector_prop = {'Sector 1': .6, 'Sector 2': .4},
              remote_prefs = [.8, .8],
              # remote_prefs = None,
              unemployment_rate = .05,
              firm_eta = .8)


## Simulate the first 50 steps
start = time.time()
stats_loc, stats_firm, stats_labor, stats_network = test.simulate(steps = 100) 


## Update remote tolerances for workers
for f in test.Firms.values():
    
    ## Get the tolerance from a gaussian ditribution based on sector
    if f.sector == 'Sector 1':
        
        new_tol = np.random.normal(.1, .075)
    
    elif f.sector == 'Sector 2':
        
        new_tol = np.random.normal(.9, .075)
    
    
    ## Bound the values between 0 and 1 and udpate the firm
    new_tol = min(max(0, new_tol), 1)    
    f.updateRemotePref(new_tol)
        


## Simulate the last 50 steps
s_stats_loc, s_stats_firm, s_stats_labor, s_stats_network = test.simulate(steps = 100)

end = time.time()
print(end - start)

## Update the "steps" and bind the shocked data frames to the initial data frames
s_stats_loc['Time Step'] = s_stats_loc['Time Step'] + 75
s_stats_firm['Time Step'] = s_stats_firm['Time Step'] + 75
s_stats_labor['Time Step'] = s_stats_labor['Time Step'] + 75
s_stats_network['Time Step'] = s_stats_network['Time Step'] + 75


## Need to actually drop the last step from the original DF because the 1st step of the shocked df will have all of the same values except 
## Updated remote tolerances

stats_loc = stats_loc.append(s_stats_loc, ignore_index = True)
stats_firm = stats_firm.append(s_stats_firm, ignore_index = True)
stats_labor = stats_labor.append(s_stats_labor, ignore_index = True)
stats_network = stats_network.append(s_stats_network, ignore_index = True)


## Write these files to csvs in the results folder for looking at in R
fpath = r'C:\Users\zachm\Documents\George Mason\Projects\Labor Flow Python Package\bin\results'
stats_loc.to_csv(fpath + r'\location_results.csv')
stats_firm.to_csv(fpath + r'\firm_results.csv')
# stats_labor.to_csv(fpath + r'\labor_results.csv') # These are pretty big, let's avoid writing them...
stats_network.to_csv(fpath + r'\network_results.csv')


## One thing that isn't done is the explicit creation/save of an edge list to redraw the firm-firm projection at a later time


## Get the size of the objects on disk:
# from pympler import asizeof 
# asizeof.asizeof(test)
# asizeof.asizeof(test.Locations['Location 1'])
# asizeof.asizeof(test.Population[0])
# asizeof.asizeof(test.Firms[0])



