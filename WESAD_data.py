"""
Author: Michael Scheid 
Date: 1/25/2020

Source code to load, compute features and analyze data from WESAD
dataset:

https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29#

"""

import pickle
import numpy as np
import os
import datetime
import tensorflow as tf
from pathlib import Path

import sklearn
from  sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.models import load_model

class WESAD_data:
    # Path to the WESAD dataset
    ROOT_PATH = r'C:\Users\micha\Desktop\WESAD Project\WESAD'
    
    # pickle file extension for importing
    FILE_EXT = '.pkl'

    # IDs of the subjects
    SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    
    # Label values defined in the WESAD readme
    BASELINE = 1
    STRESS = 2
    AMUSE = 3
    
    FEATURE_KEYS =     ['max',  'min', 'mean', 'range', 'std']
    FEATURE_ACC_KEYS = ['maxx', 'maxy', 'maxz', 'mean', 'std']

    # Keys for measurements collected by the RespiBAN on the chest
    # minus the ones we don't want
    # RAW_SENSOR_VALUES = ['ACC','ECG','EDA','EMG','Resp','Temp']
    RAW_SENSOR_VALUES = ['ACC', 'EDA','Temp']
    
    FEATURES = {'a_mean': [], 'a_std': [], 'a_maxx': [], 'a_maxy': [], 'a_maxz': [],\
                'e_max': [],  'e_min': [], 'e_mean': [], 'e_range': [], 'e_std': [], \
                't_max': [],  't_min': [], 't_mean': [], 't_range': [], 't_std': [] }
    STRESS_FEATURES = {'a_mean': [], 'a_std': [], 'a_maxx': [], 'a_maxy': [], 'a_maxz': [],\
                'e_max': [],  'e_min': [], 'e_mean': [], 'e_range': [], 'e_std': [], \
                't_max': [],  't_min': [], 't_mean': [], 't_range': [], 't_std': [] }
    AMUSE_FEATURES = {'a_mean': [], 'a_std': [], 'a_maxx': [], 'a_maxy': [], 'a_maxz': [],\
                'e_max': [],  'e_min': [], 'e_mean': [], 'e_range': [], 'e_std': [], \
                't_max': [],  't_min': [], 't_mean': [], 't_range': [], 't_std': [] }
    
    # Dictionaries to store the two sets of data
    BASELINE_DATA = []
    STRESS_DATA = []
    AMUSE_DATA = []
    
    # the file name for the last created model
    last_saved=''
    
    def __init__(self, ignore_empatica=False, ignore_additional_signals=False):
        # denotes that we will be excluding the empatica data 
        # after loading those measurements
        self.ignore_empatica = ignore_empatica

    def get_subject_path(self, subject):
        """ 
        Parameters:
        subject (int): id of the subject
        
        Returns:
        str: path to the pickle file for the given subject number
             iff the path exists 
        """
        
        # subjects path looks like data_set + '<subject>/<subject>.pkl'
        path = os.path.join(WESAD_data.ROOT_PATH, 'S'+ str(subject), 'S' + str(subject) + WESAD_data.FILE_EXT)
        print('Loading data for S'+ str(subject))
        #print('Path=' + path)
        if os.path.isfile(path):
            return path
        else:
            print(path)
            raise Exception('Invalid subject: ' + str(subject))

    def load(self, subject):
        """ 
        Loads and saves the data from the pkl file for the provided subject
        
        Parameters:
        subject (int): id of the subject
        
        Returns: Baseline and stress data
        dict: {{'EDA': [###, ..], ..}, 
               {'EDA': [###, ..], ..} }
        """
       
        # change the encoding because the data appears to have been
        # pickled with py2 and we are in py3
        with open(self.get_subject_path(subject), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
            return self.extract_and_reform(data, subject)
    
    def load_all(self, subjects=SUBJECTS):
        """ 
        Parameters:
        subjects (list): subject ids to be loaded
        
        Returns:
        """
        for subject in subjects:
            self.load(subject)
                
    
    def extract_and_reform(self, data, subject):
        """ 
        Extracts and shapes the data from the pkl file
        for the provided subject.
        
        Parameters:
        data (dict): as loaded from the pickle file
        
        Returns: Baseline and stress data
        dict: {{'EDA': [###, ..], ..}, 
               {'EDA': [###, ..], ..} }
        """
                
        if self.ignore_empatica:
            del data['signal']['wrist']
        
        baseline_indices = np.nonzero(data['label']==WESAD_data.BASELINE)[0]   
        stress_indices = np.nonzero(data['label']==WESAD_data.STRESS)[0]
        amuse_indices = np.nonzero(data['label']==WESAD_data.AMUSE)[0]
        base = dict()
        stress = dict()
        amuse = dict()
        
        for value in WESAD_data.RAW_SENSOR_VALUES: 
            base[value] = data['signal']['chest'][value][baseline_indices]
            #base['subject'] = [subject for subject in len(baseline_indices)]
            
            stress[value] = data['signal']['chest'][value][stress_indices]
            #stress['subject'] = [subject for subject in len(stress_indices)]
            
            amuse[value] = data['signal']['chest'][value][amuse_indices]
            #amuse['subject'] = [subject for subject in len(amuse_indices)]
            
        WESAD_data.BASELINE_DATA.append(base)
        WESAD_data.STRESS_DATA.append(stress)
        WESAD_data.AMUSE_DATA.append(amuse)
        
        return base, stress, amuse
    
    def get_stats(self, values, window_size=42000, window_shift=175):
        """ 
        Calculates basic statistics including max, min, mean, and std
        for the given data
        
        Parameters:
        values (numpy.ndarray): list of numeric sensor values
        
        Returns: 
        dict: 
        """

        num_features = values.size - window_size
      
        max_tmp = []
        min_tmp = []
        mean_tmp = []
        dynamic_range_tmp = []
        std_tmp = []
        for i in range(0, num_features, window_shift):
            window = values[i:window_size + i]
            max_tmp.append(np.amax(window))
            min_tmp.append(np.amin(window))
            mean_tmp.append(np.mean(window))
            dynamic_range_tmp.append(max_tmp[-1] - min_tmp[-1])
            std_tmp.append(np.std(window))

        features = {}
        features['max'] = max_tmp
        features['min'] = min_tmp
        features['mean'] = mean_tmp
        features['range'] = dynamic_range_tmp
        features['std'] = std_tmp
        return features

    def get_features_for_acc(self, values, window_size=42000, window_shift=175):
        """ 
        Calculates statistics including mean and std
        for the given data and the peak frequency per axis and the
        body acceleration component (RMS)
        
        Parameters:
        values (numpy.ndarray): list of numeric sensor values [x, y, z]
        window_size (int): specifies size of the sliding window
        window_shift (int): Specifies the sliding window shift
        
        Returns: 
        dict: features for mean, max, std for given data
        """
        #print("There are ", len(values[:,1]), " samples being considered.")
        num_features = len(values[:,1]) - window_size
        #print("Computing ", num_features , " feature values with window size" \
        #              "of ", str(window_size) + "." )
        maxx_tmp = []
        maxy_tmp = []
        maxz_tmp = []
        mean_tmp = []
        std_tmp = []        
        for i in range(0, num_features, window_shift):
            windowx = values[i:window_size + i, 0]
            windowy = values[i:window_size + i, 1]
            windowz = values[i:window_size + i, 2]
                        
            meanx = np.mean(windowx)
            meany = np.mean(windowy)
            meanz = np.mean(windowz)
            mean_tmp.append( (meanx + meany + meanz) )

            stdx = np.std(windowx)
            stdy = np.std(windowy)
            stdz = np.std(windowz)
            std_tmp.append( (stdx + stdy + stdz) )
            
            maxx_tmp.append(np.amax(windowx))
            maxy_tmp.append(np.amax(windowy))
            maxz_tmp.append(np.amax(windowz))

        features = {}
        features['mean'] = mean_tmp
        features['std'] =  std_tmp
        features['maxx'] = maxx_tmp
        features['maxy'] = maxy_tmp
        features['maxz'] = maxz_tmp
        
        return features
    
    def compute_features(self, subjects=SUBJECTS, data=BASELINE_DATA, window_size=42000, window_shift=175):
        """ 
        Calculates features for the provided subjects given the data and
        window size/shift of interest. 
        
        Parameters:
        subjects (list): subject ids to be loaded
        data (list): List of dictionaries containing subjects as indices
                     and dictionaries with features for each 'ACC', 'EDA', 'Temp'
        window_size (int): specifies size of the sliding window
        window_shift (int): Specifies the sliding window shift
        Returns:
            
        """
        index = 0
        keys = list(WESAD_data.FEATURES.keys())
        print('Computing features..')
        for subject in subjects:
            print("\tsubject:", subject)
            
            key_index = 0
            acc = self.get_features_for_acc(data[index]['ACC'], window_size, window_shift)
            
            for feature in WESAD_data.FEATURE_ACC_KEYS:
                #print('computed ', len(acc[feature]), 'windows for acc ', feature)
                WESAD_data.FEATURES[keys[key_index]].extend(acc[feature])
                key_index = key_index + 1
            
            eda = self.get_stats(data[index]['EDA'], window_size, window_shift)
            for feature in WESAD_data.FEATURE_KEYS:
                #print('computed ', len(eda[feature]), 'windows for eda ', feature)
                WESAD_data.FEATURES[keys[key_index]].extend(eda[feature])
                key_index = key_index + 1

            temp = self.get_stats(data[index]['Temp'], window_size, window_shift)
            for feature in WESAD_data.FEATURE_KEYS:
                #print('computed ', len(temp[feature]), 'windows for temp ', feature)
                WESAD_data.FEATURES[keys[key_index]].extend(temp[feature])
                key_index = key_index + 1
            
            index  += 1
            
        return WESAD_data.FEATURES

    def compute_features_stress(self, subjects=SUBJECTS, data=STRESS_DATA, window_size=42000, window_shift=175):
        
        index = 0 
        keys = list(WESAD_data.STRESS_FEATURES.keys())
        print('conputing features..')    
        for subject in subjects:
            print("\tsubject:", subject)
            key_index = 0
            
            acc = self.get_features_for_acc(data[index]['ACC'], window_size, window_shift)
            for feature in WESAD_data.FEATURE_ACC_KEYS:
                #print('computed ', len(acc[feature]), 'windows for acc ', feature)
                WESAD_data.STRESS_FEATURES[keys[key_index]].extend(acc[feature])
                key_index = key_index + 1
            
            eda = self.get_stats(data[index]['EDA'], window_size, window_shift)
            for feature in WESAD_data.FEATURE_KEYS:
                #print('computed ', len(eda[feature]), 'windows for eda ', feature)
                WESAD_data.STRESS_FEATURES[keys[key_index]].extend(eda[feature])
                key_index = key_index + 1

            temp = self.get_stats(data[index]['Temp'], window_size, window_shift)
            for feature in WESAD_data.FEATURE_KEYS:
                #print('computed ', len(temp[feature]), 'windows for temp ', feature)
                WESAD_data.STRESS_FEATURES[keys[key_index]].extend(temp[feature])
                key_index = key_index + 1
                
            index += 1
        return WESAD_data.STRESS_FEATURES
        
    def compute_features_amuse(self, subjects=SUBJECTS, data=AMUSE_DATA, window_size=42000, window_shift=175):
        index = 0 
        keys = list(WESAD_data.AMUSE_FEATURES.keys())
        print('conputing features..')   
        
        for subject in subjects:
            print("\tsubject:", subject)
            key_index = 0
            
            acc = self.get_features_for_acc(data[index]['ACC'], window_size, window_shift)
            for feature in WESAD_data.FEATURE_ACC_KEYS:
                #print('computed ', len(acc[feature]), 'windows for acc ', feature)
                WESAD_data.AMUSE_FEATURES[keys[key_index]].extend(acc[feature])
                key_index = key_index + 1
            
            eda = self.get_stats(data[index]['EDA'], window_size, window_shift)
            for feature in WESAD_data.FEATURE_KEYS:
                #print('computed ', len(eda[feature]), 'windows for eda ', feature)
                WESAD_data.AMUSE_FEATURES[keys[key_index]].extend(eda[feature])
                key_index = key_index + 1

            temp = self.get_stats(data[index]['Temp'], window_size, window_shift)
            for feature in WESAD_data.FEATURE_KEYS:
                #print('computed ', len(temp[feature]), 'windows for temp ', feature)
                WESAD_data.AMUSE_FEATURES[keys[key_index]].extend(temp[feature])
                key_index = key_index + 1
                
            index += 1
            
        return WESAD_data.AMUSE_FEATURES

  