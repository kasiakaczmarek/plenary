
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
    wiekszosc = fuzz.trapmf(czesc, [0.45, 0.6, 1, 1])
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
            protoform[k] = "Among records that contribute "+ qq_shap_print[i] + ", "+ desc + " of them have " + pp_print[j] + "-related features at "+pp_print1[j]+" level."
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

# set paths
TempDataDir = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/preprocessed_data/data_with_fcsts.csv'

classcode=1

type_of_model="symptom"
#type_of_model="class"
#type_of_eval="baseline" #onehead
type_of_eval="onehead" #onehead

if type_of_model=="symptom":
    DataFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/one_head_1/data_shap_onehead_symptom'+str(classcode)+'.csv'
    if classcode==0: classtoprint='anxiety'
    if classcode==1: classtoprint='decreased_activity'
    if classcode==4: classtoprint='elevated_activity'

if type_of_model=="class":
    if type_of_eval=="onehead":
        DataFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/one_head_1/data_shap_onehead_class'+str(classcode)+'.csv'
    else:
        DataFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/baseline_final/data_shap_'+str(type_of_eval)+'_class'+str(classcode)+'.csv'
    if classcode==0: classtoprint='euthymia'
    if classcode==1: classtoprint='depression'
    if classcode==2: classtoprint='mania'
    if classcode==3: classtoprint='mixed'
    
#DataFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/one_head_1/data_shap_onehead_class'+str(classcode)+'.csv'
#DataFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/one_head_1/data_shap_onehead_class1.csv'
#DataFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/one_head_1/data_shap_onehead_class2.csv'
#DataFile = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/one_head_1/data_shap_onehead_class3.csv'

GroupingsData = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/data_in/BD_grouping_acoustic_features - acoustic_features.csv'
ResultsDir = r'C:/Users/Kasia/Documents/GitHub/LS-XAI/'

#import data
data = pd.read_csv(DataFile, sep=';')
data=data.reset_index()
data.columns

####################################################################################################################################################
# Parameters
####################################################################################################################################################

plot=False#True
calculate_individual_LS=False
printout=False
expert = False #dictionary with expert opinion about
relative_LS = False #if relative LS is True, patient_no must be provided
spread=0.1
date_id="20220227"
####################################################################################################################################################
# Basic stats
####################################################################################################################################################
#stats about acoustic data
acoustic_var_names=['pcm_LOGenergy_sma', 'pcm_zcr_sma', 
                    'voiceprob_sma',
       'f0sma', 'f0env_sma', 'pcm_fftMag_fband0-250sma',
       'pcm_fftMag_fband0-650sma', 'pcm_fftMag_spectralRollOff250sma',
       'pcm_fftMag_spectralRollOff500sma',
       'pcm_fftMag_spectralRollOff750sma',
       'pcm_fftMag_spectralRollOff900sma', 'pcm_fftmag_spectralflux_sma',
       'pcm_fftmag_spectralcentroid_sma', 'pcm_fftmag_spectralmaxpos_sma',
       'pcm_fftmag_spectralminpos_sma', 'f0final_sma',
       'voicingfinalunclipped_sma', 'jitterlocal_sma', 'jitterddp_sma',
       'shimmerlocal_sma', 'loghnr_sma', 'audspec_lengthl1norm_sma',
       'audspecrasta_lengthl1norm_sma', 'pcm_rmsenergy_sma',
       'audSpec_Rfilt_sma_compare_0', 'audSpec_Rfilt_sma_compare_1',
       'audSpec_Rfilt_sma_compare_2', 'audSpec_Rfilt_sma_compare_3',
       'audSpec_Rfilt_sma_compare_4', 'audSpec_Rfilt_sma_compare_5',
       'audSpec_Rfilt_sma_compare_6', 'audSpec_Rfilt_sma_compare_7',
       'audSpec_Rfilt_sma_compare_8', 'audSpec_Rfilt_sma_compare_9',
       'audSpec_Rfilt_sma_compare_10', 'audSpec_Rfilt_sma_compare_11',
       'audSpec_Rfilt_sma_compare_12', 'audSpec_Rfilt_sma_compare_13',
       'audSpec_Rfilt_sma_compare_14', 'audSpec_Rfilt_sma_compare_15',
       'audSpec_Rfilt_sma_compare_16', 'audSpec_Rfilt_sma_compare_17',
       'audSpec_Rfilt_sma_compare_18', 'audSpec_Rfilt_sma_compare_19',
       'audSpec_Rfilt_sma_compare_20', 'audSpec_Rfilt_sma_compare_21',
       'audSpec_Rfilt_sma_compare_22', 'audSpec_Rfilt_sma_compare_23',
       'audSpec_Rfilt_sma_compare_24', 'audSpec_Rfilt_sma_compare_25',
       'pcm_fftMag_fband250-650sma_compare',
       'pcm_fftMag_fband1000-4000sma_compare',
       'pcm_fftmag_spectralentropy_sma_compare',
       'pcm_fftmag_spectralvariance_sma_compare',
       'pcm_fftmag_spectralskewness_sma_compare',
       'pcm_fftmag_spectralkurtosis_sma_compare',
       'pcm_fftmag_psysharpness_sma_compare',
       'pcm_fftmag_spectralharmonicity_sma_compare', 'loudness_sma3',
       'alpharatio_sma3', 'hammarbergindex_sma3', 'slope0-500sma3',
       'slope500-1500sma3', 'F0semitoneFrom275Hz_sma3nz',
       'logRelF0-H1-H2sma3nz', 'logRelF0-H1-A3sma3nz', 'f1frequency_sma3nz',
       'f1bandwidth_sma3nz', 'f1amplitudelogrelf0sma3nz',
       'f2frequency_sma3nz', 'f2amplitudelogrelf0sma3nz',
       'f3frequency_sma3nz', 'f3amplitudelogrelf0sma3nz',
       'pcm_fftMag_mfcc_0', 'pcm_fftMag_mfcc_1', 'pcm_fftMag_mfcc_2',
       'pcm_fftMag_mfcc_3', 'pcm_fftMag_mfcc_4', 'pcm_fftMag_mfcc_5',
       'pcm_fftMag_mfcc_6', 'pcm_fftMag_mfcc_7', 'pcm_fftMag_mfcc_8',
       'pcm_fftMag_mfcc_9', 'pcm_fftMag_mfcc_10', 'pcm_fftMag_mfcc_11',
       'pcm_fftMag_mfcc_12']

vars_y=['hamd_ymrs','patient_id','hamd_suma','yms_suma']

#data=data[[acoustic_var_names]].dropna().reset_index()

if plot==True:
    for zmienna in acoustic_var_names:
            fig=plt.figure(figsize=(15,8))
            sns.boxplot(x="patient_id", y=zmienna, data = data.loc[:,["patient_id",zmienna]])
            fig.savefig("Stats_"+str(zmienna)+".png")
            fig=plt.figure(figsize=(15,8))
            sns.boxplot(x="patient_id", y='shap_'+zmienna, data = data.loc[:,["patient_id",'shap_'+zmienna]])
            fig.savefig("Stats_"+str('shap_'+zmienna)+".png")


#NA percentage
data2=data[acoustic_var_names]
data2.columns = acoustic_var_names
data2.agg(lambda x: np.mean(x.isna())).reset_index().rename(columns={'index': 'column', 0: 'NA_percentage'})
    

data3_full = data2.copy()
data4 = data.copy()

######################################################################################################################
# Linguistic summaries for individual parameters
######################################################################################################################

if "df_protoform_all" in locals():
    del df_protoform_all
        
if calculate_individual_LS==True:
    #for name in acoustic_var_names:
    for name in acoustic_var_names[10:20]:
            temp = calculate_membership(data4, name, plot,expert=expert, printout=printout)
            temp2 = calculate_membership_fixed(data4, 'shap_'+name, plot,expert=expert, 
                                              printout=printout, use_central_and_spread=True, central=0, spread=spread)
            data_for_lingsum = pd.concat([temp,temp2], axis=1)
            var_names=[name, "shap_"+name]
            df_protoform = all_protoform(data_for_lingsum, var_names, Q = 'wiekszosc', desc = 'most', classtoprint=classtoprint)
            if "df_protoform_all" in locals():
                df_protoform_all = df_protoform_all.append(df_protoform)
            else:
                df_protoform_all = df_protoform.copy()
            data3_full = pd.concat([data3_full, data_for_lingsum], axis=1)
            data3_full.to_csv("data_with_memberships_"+str(date_id)+str(type_of_eval)+str(spread)+str(classtoprint)+"individual.csv")
            df_protoform_all.to_csv("Protoforms_"+str(date_id)+str(type_of_eval)+str(spread)+str(classtoprint)+"individual.csv")

#name=acoustic_var_names[12]
#df=data[name]        
#fig=df.hist(grid=False, bins=10)
#plt.xlabel('shap_pcm_fftmag_spectralcentroid_sma')
#plt.ylabel('# of acoustic frames')
#plt.title('SHAP value (classification of depression)') 
#fig.savefig("Hist_"+str('shap_'+name)+".png")
####################################################################################################################################################
# Grouping to high level features
####################################################################################################################################################
    
#groupings       
#data_groupings = pd.read_csv(GroupingsData, sep=',')		
#1. Laudness/energy-related features		
#2. Pitch-related features		
#3. Spectral-related features		
#4. Voice quality-related features		
#5. Lenght of speech		

#acoustic_types=['loudness_energy','pitch','spectral','voice_quality','speech_length']	

######################################################################################################################
# Linguistic summaries for grouped acoustic vars
######################################################################################################################

 
# 40 najbardzien prawdziwych podsumowan lingwistycznych 
#df_protoform.sort('DoT', ascending = False).head(n = 40)
    
acoustic_group_label='energy'
acoustic_group_energy=['pcm_LOGenergy_sma',
'pcm_fftMag_fband0-250sma',
'pcm_fftMag_fband0-650sma',
'audspec_lengthl1norm_sma',
'audspecrasta_lengthl1norm_sma',
'pcm_rmsenergy_sma',
'audSpec_Rfilt_sma_compare_0',
'audSpec_Rfilt_sma_compare_1',
'audSpec_Rfilt_sma_compare_2',
'audSpec_Rfilt_sma_compare_3',
'audSpec_Rfilt_sma_compare_4',
'audSpec_Rfilt_sma_compare_5',
'audSpec_Rfilt_sma_compare_6',
'audSpec_Rfilt_sma_compare_7',
'audSpec_Rfilt_sma_compare_8',
'audSpec_Rfilt_sma_compare_9',
'audSpec_Rfilt_sma_compare_10',
'audSpec_Rfilt_sma_compare_11',
'audSpec_Rfilt_sma_compare_12',
'audSpec_Rfilt_sma_compare_13',
'audSpec_Rfilt_sma_compare_14',
'audSpec_Rfilt_sma_compare_15',
'audSpec_Rfilt_sma_compare_16',
'audSpec_Rfilt_sma_compare_17',
'audSpec_Rfilt_sma_compare_18',
'audSpec_Rfilt_sma_compare_19',
'audSpec_Rfilt_sma_compare_20',
'audSpec_Rfilt_sma_compare_21',
'audSpec_Rfilt_sma_compare_22',
'audSpec_Rfilt_sma_compare_23',
'audSpec_Rfilt_sma_compare_24',
'audSpec_Rfilt_sma_compare_25',
'pcm_fftMag_fband250-650sma_compare',
'pcm_fftMag_fband1000-4000sma_compare',
'loudness_sma3']

if "data_for_lingsum_all" in locals():
    del data_for_lingsum_all

#for name in acoustic_group_energy[34:35]:
for name in acoustic_group_energy:
        temp = calculate_membership(data4, name, plot,expert=expert, printout=printout)
        temp.columns=[acoustic_group_label+'_low',acoustic_group_label+'_medium',acoustic_group_label+'_high']
        temp2 = calculate_membership_fixed(data4, 'shap_'+name, plot,expert=expert, 
                                          printout=printout, use_central_and_spread=True, central=0, spread=spread)
        temp2.columns=['shap_'+acoustic_group_label+'_low','shap_'+acoustic_group_label+'_medium','shap_'+acoustic_group_label+'_high']
        data_for_lingsum = pd.concat([temp,temp2], axis=1)
        var_names=[acoustic_group_label, 'shap_'+acoustic_group_label]
        if "data_for_lingsum_all" in locals():
            data_for_lingsum_all = data_for_lingsum_all.append(data_for_lingsum)
        else:
            data_for_lingsum_all = data_for_lingsum.copy()
        data_for_lingsum_all.to_csv("data_with_memberships_"+str(date_id)+str(type_of_eval)+str(spread)+"_"+acoustic_group_label+str(classtoprint)+".csv")
        
df_protoform = all_protoform(data_for_lingsum_all, var_names, Q = 'wiekszosc', desc = 'most', classtoprint=classtoprint)
df_protoform.to_csv("Protoforms_LS_"+str(date_id)+str(type_of_eval)+str(spread)+"_"+acoustic_group_label+str(classtoprint)+".csv")

if "df_protoform_all" in locals():
    df_protoform_all = df_protoform_all.append(df_protoform)
else:
    df_protoform_all = df_protoform.copy()
                
data_for_lingsum_all.head
 
######################################################################################################################
# Linguistic summaries for grouped acoustic vars: PITCH RELATED
######################################################################################################################
 
acoustic_group_label='pitch'
acoustic_group_pitch=['voiceprob_sma',
       'f0sma', 'f0env_sma',
      'f0final_sma', 'F0semitoneFrom275Hz_sma3nz',
      'f1frequency_sma3nz',
       'f1bandwidth_sma3nz', 'f1amplitudelogrelf0sma3nz',
       'f2frequency_sma3nz', 'f2amplitudelogrelf0sma3nz',
       'f3frequency_sma3nz', 'f3amplitudelogrelf0sma3nz']

if "data_for_lingsum_all" in locals():
    del data_for_lingsum_all

for name in acoustic_group_pitch:
        temp = calculate_membership(data4, name, plot,expert=expert, printout=printout)
        temp.columns=[acoustic_group_label+'_low',acoustic_group_label+'_medium',acoustic_group_label+'_high']
        temp2 = calculate_membership_fixed(data4, 'shap_'+name, plot,expert=expert, 
                                          printout=printout, use_central_and_spread=True, central=0, spread=spread)
        temp2.columns=['shap_'+acoustic_group_label+'_low','shap_'+acoustic_group_label+'_medium','shap_'+acoustic_group_label+'_high']
        data_for_lingsum = pd.concat([temp,temp2], axis=1)
        var_names=[acoustic_group_label, 'shap_'+acoustic_group_label]
        if "data_for_lingsum_all" in locals():
            data_for_lingsum_all = data_for_lingsum_all.append(data_for_lingsum)
        else:
            data_for_lingsum_all = data_for_lingsum.copy()
        data_for_lingsum_all.to_csv("data_with_memberships_"+str(date_id)+str(type_of_eval)+str(spread)+"_"+acoustic_group_label+str(classtoprint)+".csv")
        
df_protoform = all_protoform(data_for_lingsum_all, var_names, Q = 'wiekszosc', desc = 'most', classtoprint=classtoprint)
df_protoform.to_csv("Protoforms_LS_"+str(date_id)+str(type_of_eval)+str(spread)+"_"+acoustic_group_label+str(classtoprint)+".csv")

if "df_protoform_all" in locals():
    df_protoform_all = df_protoform_all.append(df_protoform)
else:
    df_protoform_all = df_protoform.copy()
                
data_for_lingsum_all.head


######################################################################################################################
# Linguistic summaries for grouped acoustic vars: Sprectral 
######################################################################################################################
 
acoustic_group_label='spectral'
acoustic_group_spectral=['pcm_fftMag_mfcc_0', 'pcm_fftMag_mfcc_1', 'pcm_fftMag_mfcc_2',
       'pcm_fftMag_mfcc_3', 'pcm_fftMag_mfcc_4', 'pcm_fftMag_mfcc_5',
       'pcm_fftMag_mfcc_6', 'pcm_fftMag_mfcc_7', 'pcm_fftMag_mfcc_8',
       'pcm_fftMag_mfcc_9', 'pcm_fftMag_mfcc_10', 'pcm_fftMag_mfcc_11',
       'pcm_fftMag_mfcc_12','pcm_fftMag_spectralRollOff250sma',
       'pcm_fftMag_spectralRollOff500sma',
       'pcm_fftMag_spectralRollOff750sma',
       'pcm_fftMag_spectralRollOff900sma', 'pcm_fftmag_spectralflux_sma',
       'pcm_fftmag_spectralcentroid_sma', 'pcm_fftmag_spectralmaxpos_sma',
       'pcm_fftmag_spectralminpos_sma','pcm_fftmag_spectralentropy_sma_compare',
       'pcm_fftmag_spectralvariance_sma_compare',
       'pcm_fftmag_spectralskewness_sma_compare',
       'pcm_fftmag_spectralkurtosis_sma_compare',
       'pcm_fftmag_psysharpness_sma_compare',
       'pcm_fftmag_spectralharmonicity_sma_compare',       
       'alpharatio_sma3', 'hammarbergindex_sma3', 'slope0-500sma3',
       'slope500-1500sma3', 
       'logRelF0-H1-H2sma3nz', 'logRelF0-H1-A3sma3nz']

if "data_for_lingsum_all" in locals():
    del data_for_lingsum_all
for name in acoustic_group_spectral:
        temp = calculate_membership(data4, name, plot,expert=expert, printout=printout)
        temp.columns=[acoustic_group_label+'_low',acoustic_group_label+'_medium',acoustic_group_label+'_high']
        temp2 = calculate_membership_fixed(data4, 'shap_'+name, plot,expert=expert, 
                                          printout=printout, use_central_and_spread=True, central=0, spread=spread)
        temp2.columns=['shap_'+acoustic_group_label+'_low','shap_'+acoustic_group_label+'_medium','shap_'+acoustic_group_label+'_high']
        data_for_lingsum = pd.concat([temp,temp2], axis=1)
        var_names=[acoustic_group_label, 'shap_'+acoustic_group_label]
        if "data_for_lingsum_all" in locals():
            data_for_lingsum_all = data_for_lingsum_all.append(data_for_lingsum)
        else:
            data_for_lingsum_all = data_for_lingsum.copy()
        data_for_lingsum_all.to_csv("data_with_memberships_"+str(date_id)+str(type_of_eval)+str(spread)+"_"+acoustic_group_label+str(classtoprint)+".csv")
        
df_protoform = all_protoform(data_for_lingsum_all, var_names, Q = 'wiekszosc', desc = 'most', classtoprint=classtoprint)
df_protoform.to_csv("Protoforms_LS_"+str(date_id)+str(type_of_eval)+str(spread)+"_"+acoustic_group_label+str(classtoprint)+".csv")

if "df_protoform_all" in locals():
    df_protoform_all = df_protoform_all.append(df_protoform)
else:
    df_protoform_all = df_protoform.copy()
                
data_for_lingsum_all.head

######################################################################################################################
# Linguistic summaries for grouped acoustic vars: quality 
######################################################################################################################
 
acoustic_group_label='quality'
acoustic_group_quality=['voicingfinalunclipped_sma', 'jitterlocal_sma', 'jitterddp_sma',
       'shimmerlocal_sma', 'loghnr_sma']

if "data_for_lingsum_all" in locals():
    del data_for_lingsum_all
    
for name in acoustic_group_quality:
        temp = calculate_membership(data4, name, plot,expert=expert, printout=printout)
        temp.columns=[acoustic_group_label+'_low',acoustic_group_label+'_medium',acoustic_group_label+'_high']
        temp2 = calculate_membership_fixed(data4, 'shap_'+name, plot,expert=expert, 
                                          printout=printout, use_central_and_spread=True, central=0, spread=spread)
        temp2.columns=['shap_'+acoustic_group_label+'_low','shap_'+acoustic_group_label+'_medium','shap_'+acoustic_group_label+'_high']
        data_for_lingsum = pd.concat([temp,temp2], axis=1)
        var_names=[acoustic_group_label, 'shap_'+acoustic_group_label]
        if "data_for_lingsum_all" in locals():
            data_for_lingsum_all = data_for_lingsum_all.append(data_for_lingsum)
        else:
            data_for_lingsum_all = data_for_lingsum.copy()
        data_for_lingsum_all.to_csv("data_with_memberships_"+str(date_id)+str(type_of_eval)+str(spread)+"_"+acoustic_group_label+str(classtoprint)+".csv")
        
df_protoform = all_protoform(data_for_lingsum_all, var_names, Q = 'wiekszosc', desc = 'most', classtoprint=classtoprint)
df_protoform.to_csv("Protoforms_LS_"+str(date_id)+str(type_of_eval)+str(spread)+"_"+acoustic_group_label+str(classtoprint)+".csv")

if "df_protoform_all" in locals():
    df_protoform_all = df_protoform_all.append(df_protoform)
else:
    df_protoform_all = df_protoform.copy()
                
data_for_lingsum_all.head

######################################################################################################################
# export outputs
######################################################################################################################
df_protoform_all.to_csv("Protoforms_LS_"+str(date_id)+str(type_of_eval)+str(spread)+str(classtoprint)+".csv")


