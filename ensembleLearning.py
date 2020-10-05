# %% - imports
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import xgboost


# own modules
from dataLoader import getInvComData, getYahFinData
from dataPrep import shiftColumns, addSmaEma, addSinglePeriodFinFeat, \
    addStochFast, addStochSlow, addUltOsc


# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"