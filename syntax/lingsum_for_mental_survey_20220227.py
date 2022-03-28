#import runpy
#runpy.run_path(path_name='C:/Users/Kasia/Documents/GitHub/LS-XAI/lingsum.py')

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:55:13 2022

@author: Kasia
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:15:29 2019

@author: k
"""

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import seaborn as sns
import os
from copy import deepcopy
import warnings
#from openpyxl.workbook import Workbook

#function definition
def membership_function(data, var_name, value, central=0, spread=0.1, plot=False, na_omit=False, 
           expert = False,use_central_and_spread=False):
    d = deepcopy(data)
    
    if na_omit:
        d = d.loc[~d[var_name].isna()]
    else:
        d = d.fillna(0)
        
    d = d[var_name]
    
    max_for_universe = np.max(d)  
    min_for_universe = np.min(d)
    
    universe = np.arange(min_for_universe, max_for_universe, 0.001)
    
    reg_name = var_name 
    
    reg = ctrl.Consequent(universe, reg_name)

    if use_central_and_spread:
        first_quartile = np.max([central-(spread),min_for_universe])
        median_quartile = central
        third_quartile = np.min([central+(spread),max_for_universe])
    else:        
        first_quartile = np.percentile(d, 25)
        median_quartile = np.percentile(d, 50)
        third_quartile = np.percentile(d, 75)
        
   #quartiles based fuzzification
    low = fuzz.trapmf(reg.universe, [min_for_universe, min_for_universe, first_quartile, median_quartile])
    medium = fuzz.trimf(reg.universe, [first_quartile, median_quartile, third_quartile])
    high = fuzz.trapmf(reg.universe, [median_quartile, third_quartile, max_for_universe, max_for_universe])
     
    if plot:     
        fig, (ax0) = plt.subplots(nrows=1, figsize=(5, 3))
        ax0.plot(universe, low, 'b', linewidth=2, label='low')
        ax0.plot(universe, medium, 'r', linewidth=2, label='medium')
        ax0.plot(universe, high, 'g', linewidth=2, label='high')
        ax0.set_title(str(var_name))
        ax0.legend()
        plt.tight_layout()
        plt.close()
        fig.savefig("LinguisticVariable_"+str(var_name)+"_spread_"+str(spread)+".png")
        #quit()

    return (fuzz.interp_membership(universe, low, value),
            fuzz.interp_membership(universe, medium, value),
            fuzz.interp_membership(universe, high, value)
            )

#Test stopnie    
def calculate_membership(data, var_name, plot=False, na_omit=True, expert=False, printout=False):
    column = data[var_name]
    result = pd.DataFrame(np.zeros(len(column)*3).reshape(-1,3))
    result.columns = [var_name + "_low", var_name + "_medium", var_name + "_high"]
    
    #for i in range(1):
    for i in range(len(column)):
        result.loc[i,] = membership_function(data, var_name, column[i], 0, 0, plot, na_omit, expert)
        if printout==True:
            print(str(result.loc[i,]))
            print(str(column[i]))

            
    return result

def calculate_membership_fixed(data, var_name, plot=False, na_omit=True, expert=False, printout=False,
                              use_central_and_spread=True, central=0, spread=0.1):
    column = data[var_name]
    result = pd.DataFrame(np.zeros(len(column)*3).reshape(-1,3))
    result.columns = [var_name + "_low", var_name + "_medium", var_name + "_high"]
    
    #for i in range(1):
    for i in range(len(column)):
        result.loc[i,] = membership_function(data, var_name, column[i], central, spread, 
                  plot, na_omit, expert, use_central_and_spread=True) 
        if printout==True:
            print(str(result.loc[i,]))
            print(str(column[i]))

            
    return result

def evolving_linguistic_terms(data, var_name, suffix,central_name, spread_name, plot=False, na_omit=True, printout=False):
    column = data[var_name]
    column_central = data[central_name]
    column_spread = data[spread_name]
    
    result = pd.DataFrame(np.zeros(len(column)*3).reshape(-1,3))
    result.columns = [var_name + suffix+"_low", var_name + suffix+"_medium", var_name +suffix+ "_high"]
    
    for i in range(len(column)):
    #for i in range(100):
        result.loc[i,] = membership_function(data, var_name, column[i], column_central[i], column_spread[i], 
                  plot, na_omit, expert,mina=np.min(data[var_name + suffix]),
                  maxa=np.max(data[var_name + suffix]), use_central_and_spread=True)
        if printout==True:
            print(str(result.loc[i,]))
            print(str(column[i]))
    return result

def kwantyfikator(x):
    czesc = np.arange(0, 1.01, 0.001)
    wiekszosc = fuzz.trapmf(czesc, [0.5, 0.7, 1, 1])
    mniejszosc = fuzz.trapmf(czesc, [0, 0, 0.3, 0.50])
    prawie_wszystkie = fuzz.trapmf(czesc, [0.8, 0.9, 1, 1])
    czesc_wiekszosc = fuzz.interp_membership(czesc, wiekszosc, x)
    czesc_mniejszosc = fuzz.interp_membership(czesc, mniejszosc, x)
    czesc_prawie_wszystkie =  fuzz.interp_membership(czesc, prawie_wszystkie, x)
    return dict(wiekszosc = czesc_wiekszosc, 
                mniejszosc = czesc_mniejszosc, 
                prawie_wszystkie = czesc_prawie_wszystkie)

def Degree_of_truth(d, Q = "wiekszosc", P = "", P2 = ""):
    """
    Stopień prawdy dla prostych podsumowan lingwistycznych
    """    
    if P2 == "":    
        p = np.mean(d[P])
    else:
        p = np.mean(np.fmin(d[P], d[P2]))
    return kwantyfikator(p)[Q]
    
def Degree_of_truth_ext(d, Q = "wiekszosc", P = "", R = "", P2 = ""):    
    """
    Stopień prawdy dla zlozonych podsumowan lingwistycznych
    """   
    #d=data3
    #P="infexp_low"
    #R="trans_low"
    #P2="ABG_low"
    if P2 == "":
        p = np.fmin(d[P], d[R])
        ###########tutaj zmieniamy t-norme!!!!#######
        #p = np.fmax(0,(d[P]+d[R]-1))
        r = d[R]
        t = np.sum(p)/np.sum(r)
        return kwantyfikator(t)[Q]
    else:
        p1 = np.fmin(d[P2], d[R])
        p = np.fmin(p1, d[P])
        r = d[R]
        ###########tutaj zmieniamy t-norme!!!!#######
        #p = np.fmax(0,(d[P]+d[R]-1))
        t = np.sum(p)/np.sum(r)
        return kwantyfikator(t)[Q]
            

def t_norm(a, b, ntype):
    """
    calculates t-norm for param a and b
    :param ntype:
        1 - minimum
        2 - product
        3 - Lukasiewicz t-norm
    """
    if ntype == 1:
        return np.minimum(a, b)
    elif ntype == 2:
        return a * b
    elif ntype == 3:
        return np.maximum(0, a + b - 1)

def Degree_of_support(d, Q = "wiekszosc", P = "", P2 = ""):
    #DoS = = np.mean(d[P][d[P] > 0])
    DoS = sum(d[P]>0)/ len(d)
    return DoS

def Degree_of_support_ext(d, Q = "wiekszosc", P = "", R = "", P2=""): 
    p = np.fmin(d[P], d[R])
    ###########tutaj zmieniamy t-norme!!!!#######
    #p = np.fmax(0,(d[P]+d[R]-1))
    DoS = sum(p>0)/ len(d)
    return DoS


def all_protoform(d, var_names, Q = "wiekszosc", desc = 'most', classtoprint="class"):
    """
    Funkcja wyznaczajoca stopnie prawdy dla wszystkich 
    podumowań lingwistycznych (prostych i zlozonych)    
    """
    
    pp = [var_names[0] + "_low", var_names[0] + "_medium", var_names[0] + "_high"]
    qq = [var_names[1] + "_low", var_names[1] + "_medium", var_names[1] + "_high"]
    #zz = [var_names[2] + "_low", var_names[2] + "_medium", var_names[2] + "_high"]
    qq_shap_print = ["against predicting "+classtoprint+" class", "around zero to predicting "+classtoprint +" class", "positively to predicting "+classtoprint+" class"]
    pp_print = [var_names[0], var_names[0],var_names[0]]
    pp_print1 = ["low", "medium","high"]
    
    protoform = np.empty(9, dtype = "object")
    Id = np.zeros(9)
    DoT = np.zeros(9)
    DoS = np.zeros(9)
    k = 0
    #for i in range(len(pp)):
        #print(i)
        #DoT[k] = Degree_of_truth(d = d, Q = Q, P = qq[i])
        #DoS[k] = Degree_of_support(d = d, Q = Q, P = qq[i])
        #protoform[k] = desc + " records are " + qq_shap_print[i]
        #k += 1
        #DoT[k] = Degree_of_truth(d = d, Q = Q, P = pp[i])
        #DoS[k] = Degree_of_support(d = d, Q = Q, P = pp[i])
        #protoform[k] = desc + " records are " + pp[i]
        #k += 1
        #DoT[k] = Degree_of_truth(d = d, Q = Q, P = pp[i], P2 = qq[i])
        #DoS[k] = Degree_of_support(d = d, Q = Q, P = pp[i], P2 = qq[i])
        #protoform[k] =  desc + " records are " + pp[i] + " and have " + qq_shap_print[i]
        #k += 1

    for i in range(len(pp)):
        for j in range(len(qq)):
            #DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = qq[j], R = pp[i])
            #DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = qq[j], R = pp[i])
            #protoform[k] = desc +" " + pp_print[i] + " records " + " have " + qq_shap_print[j] + "."
            #Id[k] = k
            #k += 1       
            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = pp[j], R = qq[i])
            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = pp[j], R = qq[i])
            protoform[k] = "Among records that contribute "+ qq_shap_print[i] + ", "+ desc + " of them have " + pp_print[j] + " feature at "+pp_print1[j]+" level."
            Id[k] = k
            k += 1


    #for i in range(len(pp)):
    #    for j in range(3):
    #        for l in range(3):
    #            DoT[k] = Degree_of_truth_ext(d = d, Q = Q, P = pp[j], R = qq[i], P2 = zz[l])
    #            DoS[k] = Degree_of_support_ext(d = d, Q = Q, P = pp[j], R = qq[i], P2 = zz[l])
    #            protoform[k] = "Among all "+ pp[j] + " records, " + desc + " are " + qq[i] + " and " + zz[l]
    #            if pp[j]=='trans_high':
    #                print(protoform[k])
    #                print("DoT "+ str(DoT[k]) + " DoS " + str(DoS[k]))
    #                print(" ")
    #            k += 1    
        #p q z
        #trans infexp abg
        #Among all trans_low records, most are infexp_low AND ABG_low"
            
    dd = {'Id': Id,
          'protoform': protoform,
            'DoT': DoT,
            'DoS': DoS}
    dd = pd.DataFrame(dd)   
    return dd[['Id', 'protoform', "DoT"]]


####################################################################################################################################################
# Import data
####################################################################################################################################################

#ShapFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/mental_surveys/shap_values_survey_class_1_onehead_treatments.csv'
#ShapFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/mental_surveys/shap_values_survey_class_0_onehead_treatments.csv'

#type_of_eval="baseline" #onehead
type_of_eval="onehead" #onehead
classcode=1

ShapFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/mental_surveys/shap_values_survey_class_'+str(classcode)+'_'+type_of_eval+'.csv'

DataFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/mental_surveys/data_shap_survey_baseline.csv'
ResultsDir = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/'

#import data
data = pd.read_csv(DataFile, sep=',')
shapdata = pd.read_csv(ShapFile, sep=',')
data=data.reset_index()
shapdata=shapdata.reset_index()
data.columns

####################################################################################################################################################
# Parameters
####################################################################################################################################################

plot=False
printout=False

if classcode==0: classtoprint='control'
if classcode==1: classtoprint='treatment'

expert = False #dictionary with expert opinion about
relative_LS = False #if relative LS is True, patient_no must be provided
spread=0.02

####################################################################################################################################################
# Basic stats
####################################################################################################################################################

if "var_names" in locals():
    del var_names
#variables to be summarized
var_names=['Age',	
'Gender',	
'self_employed',	
'family_history',	
'work_interfere',	
'no_employees',	
'remote_work',	
'tech_company',	
'benefits',	
'care_options',	
'wellness_program',	
'seek_help',	
'anonymity',
'leave','mental_health_consequence',	
'phys_health_consequence',	
'coworkers',
'supervisor',
'mental_health_interview',	
'phys_health_interview',	
'mental_vs_physical',	
'obs_consequence']

predicted_var=['class']

if plot==True:
    for zmienna in var_names:
            fig=plt.figure(figsize=(15,8))
            sns.boxplot(x="class", y=zmienna, data = data.loc[:,["class",zmienna]])
            fig.savefig("Stats_"+str(zmienna)+".png")
            fig=plt.figure(figsize=(15,8))
            sns.boxplot(shapdata.loc[:,[zmienna]])
            fig.savefig("Stats_shap_"+str(zmienna)+"_classtoprint_"+classtoprint+".png")

plot=False
      
#NA percentage
data2=data[var_names]
data2.columns = var_names
data2.agg(lambda x: np.mean(x.isna())).reset_index().rename(columns={'index': 'column', 0: 'NA_percentage'})
        
######################################################################################################################
# Linguistic summaries for individual parameters
######################################################################################################################

data3 = data2.copy()
data4 = shapdata.copy()
data3_full = data3.copy()

if "df_protoform_all" in locals():
    del df_protoform_all
    
#for name in var_names[0:1]:
for name in var_names:
        print(name)
        temp = calculate_membership(data3, name, plot,expert=expert, printout=printout)
        temp2 = calculate_membership_fixed(data4, name, plot,expert=expert, 
                                          printout=printout, use_central_and_spread=True, central=0, spread=spread)
        temp2.columns=["shap_"+name+"_low","shap_"+name+"_medium","shap_"+name+"_high",]
        data_for_lingsum = pd.concat([temp,temp2], axis=1)
        temp_var_names=[name, "shap_"+name]
        df_protoform = all_protoform(data_for_lingsum, temp_var_names, Q = 'wiekszosc', desc = 'most', classtoprint=classtoprint)
        if "df_protoform_all" in locals():
            df_protoform_all = df_protoform_all.append(df_protoform)
        else:
            df_protoform_all = df_protoform.copy()
        data3_full = pd.concat([data3_full, data_for_lingsum], axis=1)
        data3_full.to_csv("data_with_memberships_MentalSurveys_20220227_spread_"+str(spread)+".csv")
        df_protoform_all.to_csv("Protoforms_MentalSurveys33_20220227_spread_"+str(spread)+".csv")
        
# 50 most true linguistic summaries 
#df_protoform_all.sort('DoT', ascending = False).head(n = 100)
    
