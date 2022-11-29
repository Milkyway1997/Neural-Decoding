#------------------------------------------Personal------------------------------------------------
import WU_MEG_DP
import Post_Processing
#------------------------------------------Sklearn------------------------------------------------
from sklearn.svm import SVC
from sklearn_rvm import EMRVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
#------------------------------------------Others------------------------------------------------
import pywt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from PIL import Image, ImageFilter
#------------------------------------------TensorFlow------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout

class ND:
    def __init__(self, window_size=25, file_info={'fileName': [''], 'fileDir': ''}, seed=25):
        self.message = "Script by Li Liu"
        self.channel = 64
        self.window_size = window_size
        self.seed = seed
        self.fileName = file_info['fileName']
        self.fileDir = file_info['fileDir']
        self.bad_channels = [18, 21]

    # ---------------------------------------------------------------------------------------------- Pre-processing --------------------------------------------------------------------------------------------------------------------------------------------
    # Remove mean from each channel, and datasets must be an 3D array.
    def mean_removal(self, tr_datasets, te_datasets):
        tr_data_array = tr_datasets.copy()
        te_data_array = te_datasets.copy()
        channels_mean = np.mean(tr_data_array, axis=(0, 2))
        channels_mean = list(channels_mean)

        for c in range(len(channels_mean)):
            tr_data_array[:, c, :] = tr_data_array[:, c, :] - channels_mean[c]
            te_data_array[:, c, :] = te_data_array[:, c, :] - channels_mean[c]
        return tr_data_array, te_data_array

    # Remove bad and useless epoch from raw data. And input_data must be an 3D array, and cannot be a list.
    def remove_bad_channels(self, input_data, bad_channels):
        train_data_removed = input_data.copy()
        for b in reversed(bad_channels):
            train_data_removed = np.concatenate((train_data_removed[:, :b, :], train_data_removed[:, b + 1:, :]), axis=1)
        return train_data_removed

    # Standardization method and datasets must be a list that contains 6 of 3D array(data of different images).
    def data_normalization_by_std(self, tr_datasets, te_datasets):
        tr_data_array = np.concatenate([d for d in tr_datasets], axis=0)
        te_data_array = np.concatenate([d for d in te_datasets], axis=0)
        tr_data_array_reshaped = tr_data_array.reshape((tr_data_array.shape[0] * tr_data_array.shape[1]), tr_data_array.shape[2])
        te_data_array_reshaped = te_data_array.reshape((te_data_array.shape[0] * te_data_array.shape[1]), te_data_array.shape[2])

        trData_std = np.std(tr_data_array_reshaped)
        trData_mean = np.mean(tr_data_array_reshaped)
        tr_normalized_data = (tr_data_array - trData_mean) / trData_std
        te_normalized_data = (te_data_array - trData_mean) / trData_std

        return tr_normalized_data, te_normalized_data

    # Standardization by SkLearn, and datasets must be a list that contains 6 of 3D array(data of different images).
    def sklearn_normalization(self, tr_datasets, te_datasets):
        tr_data_array = np.concatenate([d for d in tr_datasets], axis=0)
        te_data_array = np.concatenate([d for d in te_datasets], axis=0)
        tr_data_array_reshaped = tr_data_array.reshape((tr_data_array.shape[0] * tr_data_array.shape[1]), tr_data_array.shape[2])
        te_data_array_reshaped = te_data_array.reshape((te_data_array.shape[0] * te_data_array.shape[1]), te_data_array.shape[2])
        tr_original_shape = tr_data_array.shape
        te_original_shape = te_data_array.shape

        sc = StandardScaler()
        sc.fit(tr_data_array_reshaped)
        tr_normalized_data = sc.transform(tr_data_array_reshaped)
        te_normalized_data = sc.transform(te_data_array_reshaped)

        tr_original_shape_data = tr_normalized_data.reshape(tr_original_shape)
        te_original_shape_data = te_normalized_data.reshape(te_original_shape)
        return tr_original_shape_data, te_original_shape_data

    # Average trials that are come from same exemplar to reduce noise. Input data and output data are both 3D array.
    # The number of trials will be averaged depends on avg_num.
    def averging_trials(self, input_data, avg_num=4):
        inputData = input_data.copy()
        np.random.seed(self.seed)
        np.random.shuffle(inputData)

        num_pseudo_trials = int(inputData.shape[0] / avg_num)
        pseudo_trials = np.empty((num_pseudo_trials, inputData.shape[1], inputData.shape[2]))
        index = 0
        for i in range(num_pseudo_trials):
            temp = 0
            for j in range(avg_num):
                if index >= inputData.shape[0]:
                    break
                temp = temp + inputData[index, :, :]
                index += 1
            temp = temp / avg_num
            pseudo_trials[i, :, :] = temp
        return pseudo_trials

    # Function that used by slice_window, and used numpy's polyfit function.
    # User can change the degree of the fitting polynomial from Menu.
    def curve_fitting(self, in_x, in_y, cf_degree):
        z = np.polyfit(in_x, in_y, cf_degree)
        return z

    # Curve fitting is very core and significant step in my entire pre-processing pipeline.
    # It's used to reduce feature dimension and get most informational sample data. The purposes are similar to PCA but achieved from a different approach.
    # Menu['sliceWindowType'] = 'cf' or 'mean'
    # Menu['cfDegree'] = '1', '2', '3', '4'
    def slice_window(self, Menu, input_data, window_size):
        new_list_of_data = []
        sliceWindowType = Menu['sliceWindowType']
        cf_degree = int(Menu['cfDegree'])
        num_of_epochs = input_data.shape[0]
        channel = input_data.shape[1]
        all_time_points_in_data = input_data.shape[2]
        window_num = int(np.floor(all_time_points_in_data / window_size))

        for ep in range(num_of_epochs):
            temp_list = []
            for ch in range(channel):
                for wn in range(window_num):
                    y = input_data[ep][ch][(window_size*wn):window_size+(window_size*wn)]
                    x = np.array(range(window_size))
                    if sliceWindowType == 'cf':
                        cf = self.curve_fitting(in_x=x, in_y=y, cf_degree=cf_degree)
                        temp_list.extend(cf)
                    elif sliceWindowType == 'mean':
                        temp_list.append(np.mean(y))
            new_list_of_data.append(temp_list)

        return new_list_of_data

    # input_data must be a 2D array(n_Samples x n_features).
    def pca(self, train_data, test_data, components=0.95):

        pca = PCA(n_components=components, svd_solver='full')
        transformed_tr_data = pca.fit_transform(train_data)
        transformed_te_data = pca.transform(test_data)

        return transformed_tr_data, transformed_te_data

    # Dimensional reduction procedure named Singular Value Decomposition (SVD)
    def apply_svd(self, train_data, test_data, n_components=256):
        if len(train_data.shape) > 2:
            reshaped_train = train_data.reshape([train_data.shape[0], train_data.shape[1] * train_data.shape[2]])
            reshaped_test = test_data.reshape([test_data.shape[0], test_data.shape[1] * test_data.shape[2]])
        else:
            reshaped_train = train_data.copy()
            reshaped_test = test_data.copy()

        svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
        svd.fit(reshaped_train)

        reduced_train = svd.transform(reshaped_train)
        reduced_test = svd.transform(reshaped_test)

        recovered_train = svd.inverse_transform(reduced_train)
        recovered_test = svd.inverse_transform(reduced_test)

        return reduced_train, reduced_test, recovered_train, recovered_test

    # This method is used to generate time range list and baseline list for multivariate pattern analysis.
    # And this was called by the method below.
    def create_range_list(self, rangeIndex, start_time=0, range_length=100, increase_Length=50):

        rangeIndex = int(rangeIndex)

        range_a = start_time + rangeIndex*increase_Length
        range_b = range_a + range_length
        timeRangeList = [(range_a, range_b)]

        base = range_a
        baselineList = [(base, base)]

        return timeRangeList, baselineList

    # Function use for testing different stages(There are 4 stages in the experiment) and different time range.
    def different_range_options(self, Menu):
        stage = Menu['stage']
        rangeIndex = Menu['rangeIndex']
        # -1 -> Regular range from 0ms to 600ms
        # -2 -> Pre stimuli from -200ms to 0ms
        if stage == '1':
            if rangeIndex == '-1':
                timeRangeList = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600)]
                baselineList = [(0, 0), (100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
            elif rangeIndex == '-2':
                timeRangeList = [(-200, -100), (-100, 0)]
                baselineList = [(-200, -200), (-100, -100)]
            elif rangeIndex == '-3':
                timeRangeList = [(-200, -100)]
                baselineList = [(-200, -200)]
            else:
                timeRangeList, baselineList = self.create_range_list(rangeIndex=rangeIndex, start_time=0, range_length=20, increase_Length=10)
        elif stage == '2':
            if rangeIndex == '-1':
                timeRangeList = [(2000, 2100), (2100, 2200), (2200, 2300), (2300, 2400), (2400, 2500), (2500, 2600)]
                baselineList = [(2000, 2000), (2100, 2100), (2200, 2200), (2300, 2300), (2400, 2400), (2500, 2500)]
            elif rangeIndex == '-2':
                timeRangeList = [(1800, 1900), (1900, 2000)]
                baselineList = [(1800, 1800), (1900, 1900)]
            elif rangeIndex == '-3':
                timeRangeList = [(1800, 1900)]
                baselineList = [(1800, 1800)]
            else:
                timeRangeList, baselineList = self.create_range_list(rangeIndex=rangeIndex, start_time=2000, range_length=20, increase_Length=10)
        elif stage == '3':
            if rangeIndex == '-1':
                timeRangeList = [(4000, 4100), (4100, 4200), (4200, 4300), (4300, 4400), (4400, 4500), (4500, 4600)]
                baselineList = [(4000, 4000), (4100, 4100), (4200, 4200), (4300, 4300), (4400, 4400), (4500, 4500)]
            elif rangeIndex == '-2':
                timeRangeList = [(3800, 3900), (3900, 4000)]
                baselineList = [(3800, 3800), (3900, 3900)]
            elif rangeIndex == '-3':
                timeRangeList = [(3800, 3900)]
                baselineList = [(3800, 3800)]
            else:
                timeRangeList, baselineList = self.create_range_list(rangeIndex=rangeIndex, start_time=4000, range_length=20, increase_Length=10)
        elif stage == '4':
            if rangeIndex == '-1':
                timeRangeList = [(6000, 6100), (6100, 6200), (6200, 6300), (6300, 6400), (6400, 6500), (6500, 6600)]
                baselineList = [(6000, 6000), (6100, 6100), (6200, 6200), (6300, 6300), (6400, 6400), (6500, 6500)]
            elif rangeIndex == '-2':
                timeRangeList = [(5800, 5900), (5900, 6000)]
                baselineList = [(5800, 5800), (5900, 5900)]
            elif rangeIndex == '-3':
                timeRangeList = [(5800, 5900)]
                baselineList = [(5800, 5800)]
            else:
                timeRangeList, baselineList = self.create_range_list(rangeIndex=rangeIndex, start_time=6000, range_length=20, increase_Length=10)

        return timeRangeList, baselineList

    # Function that read data from pre-save data.
    def get_all_data_from_dataSgement(self, dataSegment, sourceType):
        epochTypes = ["010", "011", "100", "101", "110", "111"]
        datasets = []
        for i in epochTypes:
            new_data_from_seg = self.remove_bad_channels(input_data=dataSegment[i][sourceType][:, :64, :], bad_channels=self.bad_channels)
            datasets.append(new_data_from_seg)
        return datasets

    # Function that read data from pre-save data and used pair with get_all_data_from_dataSgement().
    # datasets must be a list that contains 6 of 3D array(data of different images)
    def concatenate_new_data_from_dataSgement_to_existed_dataSet(self, datasets, dataSegment, sourceType):
        epochTypes = ["010", "011", "100", "101", "110", "111"]
        new_output_datasets = []
        for i in range(len(epochTypes)):
            new_data_from_seg = self.remove_bad_channels(input_data=dataSegment[epochTypes[i]][sourceType][:, :64, :], bad_channels=self.bad_channels)
            new_output_datasets.append(np.concatenate([datasets[i], new_data_from_seg], axis=2))
        return new_output_datasets

    # Reshape list that contains raw data into 2D array for ML models.
    # datasets must be a list that contains 6 of 3D array(data of different images).
    def prepare_raw_datasets(self, datasets):
        raw_datasets = np.concatenate([d for d in datasets], axis=0)
        raw_datasets = raw_datasets.reshape(raw_datasets.shape[0], (raw_datasets.shape[1] * raw_datasets.shape[2]))
        return raw_datasets

    # Method used to separate raw dataset(single image) into train and test dataset.
    # img_data must be a 3D array, and 0.8 is the default of train_percentage.
    def separate_single_imageData(self, img_data, train_percentage=0.8):
        shuffle_array = list(range(img_data.shape[0]))
        np.random.seed(self.seed)
        np.random.shuffle(shuffle_array)

        separation_line = int(np.floor(train_percentage * img_data.shape[0]))
        train = img_data[shuffle_array[:separation_line], :, :]
        test = img_data[shuffle_array[separation_line:], :, :]
        return train, test

    # Perform data separation on entire data(all images).
    # train_datasets_lst must be a list that contains 6 of 3D array(data of different images).
    def separate_train_and_test(self, train_datasets_lst):
        test_datasets = []
        train_datasets = []

        for img_data in train_datasets_lst:
            train_data, test_data = self.separate_single_imageData(img_data=img_data)
            train_datasets.append(train_data)
            test_datasets.append(test_data)

        return train_datasets, test_datasets

    # Read pre-save data and receive raw train and test dataset(both array and list). How pre-save data get processed will depend on whether 'PGA' function is turned on.
    # Menu must be given as input, and there will be outputs.
    # raw_train_data_array, raw_test_data_array, new_train_datasets_lst, new_test_datasets_lst = self.Step1_dataPreparation(Menu=Menu)
    def Step1_dataPreparation(self, Menu, data_split=True):
        turnOnPga = Menu['turnOnPga']
        newTDT = Menu['newTDT']
        baseOrFlip = Menu['baseOrFlip']

        fileDir = self.fileDir
        fileName = self.fileName

        Settings = {}
        Settings["inDataDir"] = fileDir
        Settings["DataFileNames"] = fileName

        sl = WU_MEG_DP.Sumitomo_Decoding_Long(Settings, loadSettings='yes')
        segment_manager = sl.LoadSegmentData(Settings["DataFileNames"][0], Settings["inDataDir"])

        if segment_manager.GetTDT() == 'All(PGA)':
            segment_manager.ResetTDT(newTDT)

        timeRangeList, baselineList = self.different_range_options(Menu=Menu)
        segment_manager.AddSegments(timeRangeList, baselineList)  # Adds requested segments

        if turnOnPga == 'yes':
            for i in range(segment_manager.numberOfSegments):
                dataSegment_temp = segment_manager.dataSegmentList[i]
                if i == 0:
                    train_datasets_lst = self.get_all_data_from_dataSgement(dataSegment=dataSegment_temp, sourceType='train')
                    test_datasets_lst = self.get_all_data_from_dataSgement(dataSegment=dataSegment_temp, sourceType='test')
                else:
                    train_datasets_lst = self.concatenate_new_data_from_dataSgement_to_existed_dataSet(datasets=train_datasets_lst, dataSegment=dataSegment_temp, sourceType='train')
                    test_datasets_lst = self.concatenate_new_data_from_dataSgement_to_existed_dataSet(datasets=test_datasets_lst, dataSegment=dataSegment_temp, sourceType='test')
            # Raw datasets that after performed PGA
            raw_train_data_array = self.prepare_raw_datasets(datasets=train_datasets_lst)
            raw_test_data_array = self.prepare_raw_datasets(datasets=test_datasets_lst)
            return raw_train_data_array, raw_test_data_array, train_datasets_lst, test_datasets_lst
        else:
            for i in range(segment_manager.numberOfSegments):
                dataSegment_temp = segment_manager.dataSegmentList[i]
                if i == 0:
                    train_datasets_lst = self.get_all_data_from_dataSgement(dataSegment=dataSegment_temp, sourceType=baseOrFlip)
                else:
                    train_datasets_lst = self.concatenate_new_data_from_dataSgement_to_existed_dataSet(datasets=train_datasets_lst, dataSegment=dataSegment_temp, sourceType=baseOrFlip)
            if data_split == False:
                return train_datasets_lst
            # Raw datasets that without perform PGA
            new_train_datasets_lst, new_test_datasets_lst = self.separate_train_and_test(train_datasets_lst=train_datasets_lst)

            raw_train_data_array = self.prepare_raw_datasets(datasets=new_train_datasets_lst)
            raw_test_data_array = self.prepare_raw_datasets(datasets=new_test_datasets_lst)
            return raw_train_data_array, raw_test_data_array, new_train_datasets_lst, new_test_datasets_lst
        return 0

    # Method that allows the users to turn on and off Trials Averaging.
    # datasets_lst must be a list that contains 6 of 3D array(data of different images).
    def perform_avg_trials(self, Menu, datasets_lst):
        avg_trials = Menu['avgTrials']
        avg_num = int(Menu['avgNum'])
        if avg_trials == 'yes':
            output_list = []
            for dataset in datasets_lst:
                avg_dataset = self.averging_trials(input_data=dataset, avg_num=avg_num)
                output_list.append(avg_dataset)
        else:
            return datasets_lst
        return output_list

    # Method that allows the users to turn on and off Standardization.
    # datasets_lst must be a list that contains 6 of 3D array(data of different images).
    def Step2_perform_normalization_on_all_images(self, Menu, tr_datasets_lst, te_datasets_lst):
        normType = Menu['normType']
        if normType == 'my':
            tr_normalized_data, te_normalized_data = self.data_normalization_by_std(tr_datasets=tr_datasets_lst, te_datasets=te_datasets_lst)
        elif normType == 'sk':
            tr_normalized_data, te_normalized_data = self.sklearn_normalization(tr_datasets=tr_datasets_lst, te_datasets=te_datasets_lst)
        else:
            tr_datasets_lst_copy = tr_datasets_lst.copy()
            te_datasets_lst_copy = te_datasets_lst.copy()
            tr_normalized_data = np.concatenate([d for d in tr_datasets_lst_copy], axis=0)
            te_normalized_data = np.concatenate([d for d in te_datasets_lst_copy], axis=0)
        return tr_normalized_data, te_normalized_data

    # Method that allows the users to turn on and off Mean Removal.
    # datasets_array must be a 3D array.
    def Step3_perform_mean_removal_on_all_images(self, Menu, tr_datasets_array, te_datasets_array):
        turnOnMeanRemoval = Menu['turnOnMeanRemoval']
        if turnOnMeanRemoval == 'yes':
            tr_data_array, te_data_array = self.mean_removal(tr_datasets=tr_datasets_array, te_datasets=te_datasets_array)
        else:
            tr_data_array = tr_datasets_array.copy()
            te_data_array = te_datasets_array.copy()
        return tr_data_array, te_data_array

    # Method that allows the users to turn on and off Slice Window (Curve Fitting).
    # data must be a 3D array.
    def Step4_perform_slicing_window(self, Menu, data):
        turnOnSliceWindow = Menu['turnOnSliceWindow']
        if turnOnSliceWindow == 'yes':
            sliced_data = self.slice_window(Menu=Menu, input_data=data, window_size=self.window_size)
            sliced_data = np.array(sliced_data)
        else:
            sliced_data = data.copy()
            sliced_data = sliced_data.reshape(sliced_data.shape[0], (sliced_data.shape[1] * sliced_data.shape[2]))
        return sliced_data

    # Method that allows the users to turn on and off PCA.
    # input_dict must be a dictionary that contains 6 objects, which are train_data, test_data, tr_label_name, tr_label_category, te_label_name, te_label_category
    # output_dict will be the output for the function, and the new dictionary will also contain 6 objects in same order as above.
    def perform_pca_after_slicing_window(self, Menu, input_dict):
        pcaOn = Menu['pcaOn']
        if pcaOn == 'after':
            pcaComponents = float(Menu['pcaComponents'])
            train_data = input_dict['train_data'].copy()
            test_data = input_dict['test_data'].copy()

            tr_label_name = input_dict['tr_label_name'].copy()
            tr_label_category = input_dict['tr_label_category'].copy()

            te_label_name = input_dict['te_label_name'].copy()
            te_label_category = input_dict['te_label_category'].copy()


            transformed_tr_data, transformed_te_data = self.pca(train_data=train_data, test_data=test_data, components=pcaComponents)
            output_dict = {'train_data': transformed_tr_data, 'test_data': transformed_te_data,
                           'tr_label_name': tr_label_name, 'tr_label_category': tr_label_category,
                           'te_label_name': te_label_name, 'te_label_category': te_label_category}
        else:
            return input_dict
        return output_dict

    # Input train_dataset and test_dataset must be 3D array. And output transformed_tr_data and transformed_te_data will be two 2D array after performed PCA on original data.
    def perform_pca_before_slicing_window(self, train_dataset, test_dataset):
        train_data = train_dataset.reshape(train_dataset.shape[0], (train_dataset.shape[1] * train_dataset.shape[2]))
        test_data = test_dataset.reshape(test_dataset.shape[0], (test_dataset.shape[1] * test_dataset.shape[2]))

        transformed_tr_data, transformed_te_data = self.pca(train_data=train_data, test_data=test_data, components=0.999)
        return transformed_tr_data, transformed_te_data

    # Same concept with slice window method above but the new one is use for PCA before slicing window only.
    # Input dataset must be a 2D array because after perform PCA, the shape of the data would be 2D not 3D anymore.
    def slice_window_for_PCA(self, Menu, dataset, window_size):
        new_list_of_data = []
        sliceWindowType = Menu['sliceWindowType']
        cf_degree = int(Menu['cfDegree'])
        num_of_epochs = dataset.shape[0]
        all_time_points_in_data = dataset.shape[1]
        window_num = int(np.floor(all_time_points_in_data / window_size))

        for ep in range(num_of_epochs):
            temp_list = []
            for wn in range(window_num):
                y = dataset[ep][(window_size * wn):window_size + (window_size * wn)]
                x = np.array(range(window_size))
                if sliceWindowType == 'cf':
                    cf = self.curve_fitting(in_x=x, in_y=y, cf_degree=cf_degree)
                    temp_list.extend(cf)
                elif sliceWindowType == 'mean':
                    temp_list.extend(np.mean(y))
            new_list_of_data.append(temp_list)
        new_data = np.array(new_list_of_data)
        return new_data

    # Create object classification and category classification labels for ML.
    # datasets_lst must be a list that contains 6 of 3D array(data of different images).
    def data_labelling(self, datasets_lst):
        datasets = datasets_lst.copy()
        img_tuple = [np.ones(datasets[k].shape[0])*k for k in range(len(datasets))]
        label_name = np.concatenate(img_tuple, axis=0)

        food_category = np.zeros(datasets[0].shape[0] + datasets[1].shape[0] + datasets[2].shape[0], dtype='int')
        object_category = np.ones(datasets[3].shape[0] + datasets[4].shape[0] + datasets[5].shape[0], dtype='int')
        label_category = np.concatenate((food_category, object_category), axis=0)
        return label_name, label_category

    # Shuffle train/test data and its labels in same order, so that the data and label pairs would not be messed up.
    # input_dict must be a dictionary that contains 6 objects, which are train_data, test_data, tr_label_name, tr_label_category, te_label_name, te_label_category
    # output_dict will be the output for the function, and the new dictionary will also contain 6 objects in same order as above.
    def data_shuffling(self, input_dict):
        train_data = input_dict['train_data'].copy()
        test_data = input_dict['test_data'].copy()

        tr_label_name = input_dict['tr_label_name'].copy()
        tr_label_category = input_dict['tr_label_category'].copy()

        te_label_name = input_dict['te_label_name'].copy()
        te_label_category = input_dict['te_label_category'].copy()

        shuffle_array1 = list(range(train_data.shape[0]))
        shuffle_array2 = list(range(test_data.shape[0]))

        np.random.shuffle(shuffle_array1)
        np.random.shuffle(shuffle_array2)

        shuffled_train_data = train_data[shuffle_array1, :].copy()
        shuffled_tr_label_name = tr_label_name[shuffle_array1].copy()
        shuffled_tr_label_category = tr_label_category[shuffle_array1].copy()

        shuffled_test_data = test_data[shuffle_array2, :].copy()
        shuffled_te_label_name = te_label_name[shuffle_array2].copy()
        shuffled_te_label_category = te_label_category[shuffle_array2].copy()

        output_dict = {'train_data': shuffled_train_data, 'test_data': shuffled_test_data,
                       'tr_label_name': shuffled_tr_label_name, 'tr_label_category': shuffled_tr_label_category,
                       'te_label_name': shuffled_te_label_name, 'te_label_category': shuffled_te_label_category}
        return output_dict

    # -----------------------------------------------------------------------------------------Main-----------------------------------------------------------------------------------------------------------------------
    # Run the entire pre-processing pipeline here.
    # From Read pre-save data -> Doubling data -> Trials Averaging ->
    # Data Labelling -> Standardization -> Slice Window (Curve Fitting) -> Shuffling data
    def work_flow(self, Menu):
        getRawData = Menu['getRawData']
        doublingData = Menu['doublingData']
        pcaOn = Menu['pcaOn']
        raw_train_data_array, raw_test_data_array, train_datasets_lst, test_datasets_lst = self.Step1_dataPreparation(Menu=Menu)

        if doublingData == 'yes':
            doubled_train_datasets = self.doubling_train_data(train_datasets_lst=train_datasets_lst)
            old_label_name, old_label_category = self.data_labelling(datasets_lst=train_datasets_lst)
            tr_label_name, tr_label_category = self.doubling_train_label(tr_label_name=old_label_name, tr_label_category=old_label_category)
            new_train_datasets = doubled_train_datasets
        else:
            averaged_train_datasets = self.perform_avg_trials(Menu=Menu, datasets_lst=train_datasets_lst)
            tr_label_name, tr_label_category = self.data_labelling(datasets_lst=averaged_train_datasets)
            new_train_datasets = averaged_train_datasets

        averaged_test_datasets = self.perform_avg_trials(Menu=Menu, datasets_lst=test_datasets_lst)
        te_label_name, te_label_category = self.data_labelling(datasets_lst=averaged_test_datasets)
        new_test_datasets = averaged_test_datasets

        if getRawData == 'yes':
            input_dict = {'train_data': raw_train_data_array, 'test_data': raw_test_data_array,
                           'tr_label_name': tr_label_name, 'tr_label_category': tr_label_category,
                           'te_label_name': te_label_name, 'te_label_category': te_label_category}
            shuffled_dict = self.data_shuffling(input_dict=input_dict)
            return shuffled_dict
        else:
            normalized_train_datasets, normalized_test_datasets = self.Step2_perform_normalization_on_all_images(Menu=Menu, tr_datasets_lst=new_train_datasets, te_datasets_lst=new_test_datasets)
            mean_removed_train_datasets, mean_removed_test_datasets = self.Step3_perform_mean_removal_on_all_images(Menu=Menu, tr_datasets_array=normalized_train_datasets, te_datasets_array=normalized_test_datasets)

            if pcaOn == 'before':
                transformed_tr_data, transformed_te_data = self.perform_pca_before_slicing_window(train_dataset=mean_removed_train_datasets, test_dataset=mean_removed_test_datasets)
                train_data = self.slice_window_for_PCA(Menu=Menu, dataset=transformed_tr_data, window_size=self.window_size)
                test_data = self.slice_window_for_PCA(Menu=Menu, dataset=transformed_te_data, window_size=self.window_size)
            else:
                train_data = self.Step4_perform_slicing_window(Menu=Menu, data=mean_removed_train_datasets)
                test_data = self.Step4_perform_slicing_window(Menu=Menu, data=mean_removed_test_datasets)

            input_dict = {'train_data': train_data, 'test_data': test_data,
                          'tr_label_name': tr_label_name, 'tr_label_category': tr_label_category,
                          'te_label_name': te_label_name, 'te_label_category': te_label_category}

            input_dict_ = self.perform_pca_after_slicing_window(Menu=Menu, input_dict=input_dict)
            shuffled_dict = self.data_shuffling(input_dict=input_dict_)
            return shuffled_dict
        return 0

    # Menu will contain 17 objects as below.
    # Menu = {}
    # Menu['windowSize'] = self.window_size
    # Menu['getRawData'] = ['yes', 'no']
    # Menu['doublingData'] = ['yes', 'no']
    # Menu['stage'] = ['1', '2', '3', '4']
    # Menu['rangeIndex'] = ['-1'] or inx through a loop
    #
    # Menu['avgTrials'] = ['yes', 'no']
    # Menu['avgNum'] = ['1', '2', '3', '4', '5']
    #
    # Menu['normType'] = ['my', 'sk', 'no']
    # Menu['turnOnMeanRemoval'] = ['yes', 'no']
    #
    # Menu['turnOnSliceWindow'] = ['yes', 'no']
    # Menu['sliceWindowType'] = ['cf', 'mean']
    # Menu['cfDegree'] = ['1', '2', '3', '4']
    #
    # # Only clustered1 and half_pga1 can use flip
    # Menu['baseOrFlip'] = ['base', 'flip']
    # Menu['turnOnPga'] = ['yes', 'no']
    # Menu['newTDT'] = ['half_pga', 'unknown', 'pga_gauss', 'clustered1', ...]
    #
    # Menu['pcaOn'] = ['before', 'after', 'no']
    # Menu['pcaComponents'] = [0.8, 0.85, '0.9', '0.95', '0.99']

    def main(self, Menu):
        np.random.seed(self.seed)
        Menu_copy = Menu.copy()
        normType = Menu_copy['normType']
        turnOnPga = Menu_copy['turnOnPga']
        doublingData = Menu_copy['doublingData']
        getRawData = Menu_copy['getRawData']

        if getRawData == 'yes':
            Menu_copy['avgTrials'] = 'no'
            Menu_copy['doublingData'] = 'no'
        if normType == 'sk':
            Menu_copy['turnOnMeanRemoval'] = 'no'
        if turnOnPga == 'yes':
            Menu_copy['avgTrials'] = 'no'
            Menu_copy['doublingData'] = 'no'
        if doublingData == 'yes':
            Menu_copy['avgTrials'] = 'no'

        output_dict = self.work_flow(Menu=Menu_copy)
        return output_dict

    # -----------------------------------------------------------------------------------------Doubling Data-----------------------------------------------------------------------------------------------------------------------
    # Doubling data means combine original train data and train data that have been reversed (0.1 -> -0.1).
    # The reason for doubling and reversing data is that to remove anti-correlation between different exemplars/epochs when performing trials averaging.
    # Thus, afterward, train data size would be doubled compare to the original train data. This process doesn't need to apply to test data.
    # input and output are both list, but input has a length of 6 list and output has a length of 12 list.
    def doubling_train_data(self, train_datasets_lst):
        reversed_list = []
        for datasets in train_datasets_lst:
            reversed_train_data = -datasets
            reversed_list.append(reversed_train_data)
        output_list = train_datasets_lst + reversed_list
        return output_list

    # Double labels as well for those doubled train data that get from method above.
    # inputs are original labels, and outputs are doubled train labels.
    def doubling_train_label(self, tr_label_name, tr_label_category):

        reversed_tr_label_name = tr_label_name + 6
        reversed_tr_label_category = tr_label_category + 2

        new_tr_label_name = np.concatenate((tr_label_name, reversed_tr_label_name), axis=0)
        new_tr_label_category = np.concatenate((tr_label_category, reversed_tr_label_category), axis=0)

        return new_tr_label_name, new_tr_label_category

        # ML model(Linear SVM) use for doubled train data only. Cannot be used for regular train data.
        # When it's computing final accuracy, predictions are reversed back to original label range(0-11 -> 0-5).
        # classification_type can be either 'name' or 'category'
        # model_type can be either 'rvm', 'svm-linear', and 'svm-rbf'

    def model_for_doubled_data(self, input_dict, classification_type):

        tr_data = input_dict['train_data'].copy()
        te_data = input_dict['test_data'].copy()

        tr_label_name = input_dict['tr_label_name'].copy()
        tr_label_category = input_dict['tr_label_category'].copy()

        te_label_name = input_dict['te_label_name'].copy()
        te_label_category = input_dict['te_label_category'].copy()

        if classification_type == 'name':
            train_data = tr_data
            test_data = te_data
            tr_label = tr_label_name
            te_label = te_label_name
        elif classification_type == 'category':
            train_data = tr_data
            test_data = te_data
            tr_label = tr_label_category
            te_label = te_label_category

        correct_prediction = 0
        total_number_of_test_data = test_data.shape[0]

        model = SVC(kernel='linear', C=0.01, gamma=0.001)
        model.fit(train_data, tr_label)
        score = model.score(test_data, te_label)
        prediction = model.predict(test_data)
        reversed_prediction = prediction.copy()

        if classification_type == 'name':
            reversed_prediction[reversed_prediction > 5] = reversed_prediction[reversed_prediction > 5] - 6
        elif classification_type == 'category':
            reversed_prediction[reversed_prediction > 1] = reversed_prediction[reversed_prediction > 1] - 2

        for i in range(total_number_of_test_data):
            if reversed_prediction[i] == te_label[i]:
                correct_prediction += 1
        accuracy = correct_prediction / total_number_of_test_data

        return accuracy, score, prediction

    # A special examination for half PGA procedure alone. This method average 5 trials in test dataset but do nothing with train dataset.
    # This test that whether PGA procedure help increase the test accuracy and whether it's doing any weired things.
    def halfPga_averging_test_prep(self, Menu):
        raw_train_data_array, raw_test_data_array, train_datasets_lst, test_datasets_lst = self.Step1_dataPreparation(Menu=Menu)

        averaged_train_datasets = train_datasets_lst
        tr_label_name, tr_label_category = self.data_labelling(datasets_lst=averaged_train_datasets)

        averaged_test_datasets = self.perform_avg_trials(Menu=Menu, datasets_lst=test_datasets_lst)
        te_label_name, te_label_category = self.data_labelling(datasets_lst=averaged_test_datasets)

        normalized_train_datasets, normalized_test_datasets = self.Step2_perform_normalization_on_all_images(Menu=Menu, tr_datasets_lst=averaged_train_datasets, te_datasets_lst=averaged_test_datasets)

        train_data = self.Step4_perform_slicing_window(Menu=Menu, data=normalized_train_datasets)
        test_data = self.Step4_perform_slicing_window(Menu=Menu, data=normalized_test_datasets)

        input_dict = {'train_data': train_data, 'test_data': test_data,
                      'tr_label_name': tr_label_name, 'tr_label_category': tr_label_category,
                      'te_label_name': te_label_name, 'te_label_category': te_label_category}

        shuffled_dict = self.data_shuffling(input_dict=input_dict)

        return shuffled_dict

    # -----------------------------------------------------------------------------------------Cross-Subjects-----------------------------------------------------------------------------------------------------------------------
    # To get data from individual subjects
    def cross_subjects_prep_data_inner(self, Menu, fileDir, fileName, data_split=True):
        newTDT = Menu['newTDT']
        baseOrFlip = Menu['baseOrFlip']
        turnOnPga = Menu['turnOnPga']

        Settings = {}
        Settings["inDataDir"] = fileDir
        Settings["DataFileNames"] = fileName

        sl = WU_MEG_DP.Sumitomo_Decoding_Long(Settings, loadSettings='yes')
        segment_manager = sl.LoadSegmentData(Settings["DataFileNames"][0], Settings["inDataDir"])

        if segment_manager.GetTDT() == 'All(PGA)':
            segment_manager.ResetTDT(newTDT)

        timeRangeList, baselineList = self.different_range_options(Menu=Menu)
        segment_manager.AddSegments(timeRangeList, baselineList)  # Adds requested segments

        # for i in range(segment_manager.numberOfSegments):
        #     dataSegment_temp = segment_manager.dataSegmentList[i]
        #     if i == 0:
        #         train_datasets_lst = self.get_all_data_from_dataSgement(dataSegment=dataSegment_temp, sourceType=baseOrFlip)
        #     else:
        #         train_datasets_lst = self.concatenate_new_data_from_dataSgement_to_existed_dataSet(datasets=train_datasets_lst, dataSegment=dataSegment_temp, sourceType=baseOrFlip)
        #     # Raw datasets that without perform PGA
        #     new_train_datasets_lst, new_test_datasets_lst = self.separate_train_and_test(train_datasets_lst=train_datasets_lst)

        if turnOnPga == 'yes':
            for i in range(segment_manager.numberOfSegments):
                dataSegment_temp = segment_manager.dataSegmentList[i]
                if i == 0:
                    train_datasets_lst = self.get_all_data_from_dataSgement(dataSegment=dataSegment_temp, sourceType='train')
                    test_datasets_lst = self.get_all_data_from_dataSgement(dataSegment=dataSegment_temp, sourceType='test')
                else:
                    train_datasets_lst = self.concatenate_new_data_from_dataSgement_to_existed_dataSet(datasets=train_datasets_lst, dataSegment=dataSegment_temp, sourceType='train')
                    test_datasets_lst = self.concatenate_new_data_from_dataSgement_to_existed_dataSet(datasets=test_datasets_lst, dataSegment=dataSegment_temp, sourceType='test')
            if data_split:
                return train_datasets_lst, test_datasets_lst
            else:
                dataset = []
                for inx, _ in enumerate(train_datasets_lst):
                    dataset.append(np.concatenate((train_datasets_lst[inx], test_datasets_lst[inx]), axis=0))
                return dataset
        else:
            for i in range(segment_manager.numberOfSegments):
                dataSegment_temp = segment_manager.dataSegmentList[i]
                if i == 0:
                    train_datasets_lst = self.get_all_data_from_dataSgement(dataSegment=dataSegment_temp, sourceType=baseOrFlip)
                else:
                    train_datasets_lst = self.concatenate_new_data_from_dataSgement_to_existed_dataSet(datasets=train_datasets_lst, dataSegment=dataSegment_temp, sourceType=baseOrFlip)
            if data_split:
                # Raw datasets that without perform PGA
                new_train_datasets_lst, new_test_datasets_lst = self.separate_train_and_test(train_datasets_lst=train_datasets_lst)
            else:
                return train_datasets_lst

        return new_train_datasets_lst, new_test_datasets_lst

    # Combine all seven subjects data together to exam cross-subjects.
    # Train and Test datasets are 80 and 20 percent split from each participant, so the final datasets contained data from all seven subjects.
    def cross_subjects_prep_data_ver1(self, Menu, fileDir):

        fileNames = ['sub1002_PGA.pkl', 'sub1003_PGA.pkl', 'sub1004_PGA.pkl', 'sub1005_PGA.pkl', 'sub1006_PGA.pkl', 'sub1007_PGA.pkl', 'sub1008_PGA.pkl']
        for i, file in enumerate(fileNames):
            if i == 0:
                new_train_datasets_lst, new_test_datasets_lst = self.cross_subjects_prep_data_inner(Menu=Menu, fileDir=fileDir, fileName=[file])
                train_dataset = new_train_datasets_lst
                test_dataset = new_test_datasets_lst
            else:
                new_train_datasets_lst, new_test_datasets_lst = self.cross_subjects_prep_data_inner(Menu=Menu, fileDir=fileDir, fileName=[file])
                for inx, _ in enumerate(train_dataset):
                    train_dataset[inx] = np.concatenate((train_dataset[inx], new_train_datasets_lst[inx]), axis=0)
                    test_dataset[inx] = np.concatenate((test_dataset[inx], new_test_datasets_lst[inx]), axis=0)
        return train_dataset, test_dataset

    # Leave one subject out cross-validation
    # loso_inx stands index of leave one subject out, since there are 7 subjects, there will be 7 probabilities for cross-validation test.
    # loso_inx could be an int number from 0-6.
    def cross_subjects_prep_data_ver2(self, Menu, fileDir, loso_inx):

        fileNames = ['sub1002_PGA.pkl', 'sub1003_PGA.pkl', 'sub1004_PGA.pkl', 'sub1005_PGA.pkl', 'sub1006_PGA.pkl', 'sub1007_PGA.pkl', 'sub1008_PGA.pkl']

        testData_fileName = fileNames[loso_inx]
        trainData_fileName = fileNames.copy()
        trainData_fileName.remove(testData_fileName)

        for inx, file in enumerate(trainData_fileName):
            if inx == 0:
                train_dataset_temp = self.cross_subjects_prep_data_inner(Menu=Menu, fileDir=fileDir, fileName=[file], data_split=False)
                train_dataset = train_dataset_temp
            else:
                train_dataset_temp = self.cross_subjects_prep_data_inner(Menu=Menu, fileDir=fileDir, fileName=[file], data_split=False)
                for i, _ in enumerate(train_dataset):
                    train_dataset[i] = np.concatenate((train_dataset[i], train_dataset_temp[i]), axis=0)

        test_dataset = self.cross_subjects_prep_data_inner(Menu=Menu, fileDir=fileDir, fileName=[testData_fileName], data_split=False)

        return train_dataset, test_dataset

    # Entire workflow for cross-subject examination, including data preparation, data labeling, normalization, Slice Window, and etc.
    def cross_subjects_workflow(self, Menu, fileDir, ver=2, loso_inx=0):

        if ver == 1:
            train_dataset, test_dataset = self.cross_subjects_prep_data_ver1(Menu=Menu, fileDir=fileDir)
        elif ver == 2:
            train_dataset, test_dataset = self.cross_subjects_prep_data_ver2(Menu=Menu, fileDir=fileDir, loso_inx=loso_inx)

        tr_label_name, tr_label_category = self.data_labelling(datasets_lst=train_dataset)
        te_label_name, te_label_category = self.data_labelling(datasets_lst=test_dataset)

        normalized_train_datasets, normalized_test_datasets = self.Step2_perform_normalization_on_all_images(Menu=Menu, tr_datasets_lst=train_dataset, te_datasets_lst=test_dataset)

        train_data = self.Step4_perform_slicing_window(Menu=Menu, data=normalized_train_datasets)
        test_data = self.Step4_perform_slicing_window(Menu=Menu, data=normalized_test_datasets)

        input_dict = {'train_data': train_data, 'test_data': test_data,
                      'tr_label_name': tr_label_name, 'tr_label_category': tr_label_category,
                      'te_label_name': te_label_name, 'te_label_category': te_label_category}

        shuffled_dict = self.data_shuffling(input_dict=input_dict)

        return shuffled_dict

        # Get ready for training ML
        # Steps: Get test and train data -> Get test and train label (both name and cate) -> shuffle the data
    def separate_train_test_cross_subjects(self, data_list, label_list, loso_inx=0):

        te_data = data_list[loso_inx]
        tr_data = data_list.copy()
        tr_data.remove(te_data)

        te_label = label_list[loso_inx]
        tr_label = label_list.copy()
        tr_label.remove(te_label)

        train_data = np.concatenate([d for d in tr_data], axis=0)
        test_data = te_data

        tr_label_name = np.concatenate([d['label_name'] for d in tr_label], axis=0)
        tr_label_category = np.concatenate([d['label_category'] for d in tr_label], axis=0)
        te_label_name = te_label['label_name']
        te_label_category = te_label['label_category']

        print('Ready to shuffle')
        input_dict = {'train_data': train_data, 'test_data': test_data,
                      'tr_label_name': tr_label_name, 'tr_label_category': tr_label_category,
                      'te_label_name': te_label_name, 'te_label_category': te_label_category}
        shuffled_dict = self.data_shuffling(input_dict=input_dict)
        return shuffled_dict

    # -----------------------------------------------------------------------------------------CWT-----------------------------------------------------------------------------------------------------------------------
    # Continuous Wavelet Transform (CWT)
    # Wavelet 'morl' is the most suitable one for MEG data said by a previous study.
    # Input data should be a 3D array (samples * ch * data points)
    # output coef is 4D array (samples, 62, 128, 128). Output freqs is 3D array (samples, 62, 128)
    def cwt(self, data, wavelet='morl', scale=64):
        # scales = np.arange(1, data.shape[2] + 1)
        scales = np.arange(1, scale + 1)
        coef_output = np.zeros(shape=(data.shape[0], data.shape[1], scale, data.shape[2]))
        freqs_output = np.zeros(shape=(data.shape[0], data.shape[1], scale))

        for inx_ep, epoch in enumerate(data):
            for inx_ch, ch in enumerate(epoch):
                coef, freqs = pywt.cwt(data=ch, scales=scales, wavelet=wavelet)
                coef_output[inx_ep, inx_ch, :, :] = coef
                freqs_output[inx_ep, inx_ch, :] = freqs

        # Use skimage to resize the coef matrix
        shape = (coef_output.shape[0], coef_output.shape[1], scale, scale)
        rescaled_coef = np.zeros(shape=shape)
        for i in range(coef_output.shape[0]):
            for j in range(coef_output.shape[1]):
                rescaled_coef[i, j, :, :] = resize(coef_output[i][j], (scale, scale), mode='constant')
        return rescaled_coef, freqs_output

    # Combine 64 images (64 channels) into one image
    # data should be a single epoch, and size of input data should be 3D array (channels, scale, scale)
    def image_combination_inner(self, data, saving_address='C:\\Users\Sakai Lab\Desktop', fileName='image', saveImage=True):
        horizontal_holder = []
        for i in range(1, 9):
            imgs_comb = np.hstack([i for i in data[8 * (i - 1):8 * i]])
            horizontal_holder.append(imgs_comb)
        img_final = np.vstack([d for d in horizontal_holder])

        if saveImage:
            fig = plt.figure(frameon=False)
            fig.set_size_inches(15, 15)

            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            ax.imshow(img_final, cmap='coolwarm', aspect='auto')
            fig.savefig(saving_address + '\\' + fileName + '.png', format="png")
            plt.clf()
            plt.close()
        return img_final

    # compute all epochs by calling image_combination for a participant
    # data should contain all epochs and size of input data should be 4D array (samples, channels, scale, scale)
    def image_combination(self, data, saving_address='C:\\Users\Sakai Lab\Desktop', subject_number='sub1002', saveImage=True):

        scale = data[subject_number].shape[2]
        image_width = scale * 8
        image_height = image_width

        subject = np.zeros(shape=(data[subject_number].shape[0], 64, data[subject_number].shape[2], data[subject_number].shape[3]))
        for inx in range(data[subject_number].shape[1]):
            subject[:, inx, :, :] = data[subject_number][:, inx, :, :]

        output = np.zeros(shape=(data[subject_number].shape[0], image_width, image_height))
        for inx, epoch in enumerate(subject):
            name = 'epoch' + str(inx+1)
            final_coef = self.image_combination_inner(data=epoch, saving_address=saving_address, fileName=name, saveImage=saveImage)
            output[inx, :, :] = final_coef

        return output

    # compute all participants
    # two output lists would contain 7 array, and the dimension of each array would be 4D (output from method above).
    def image_combination_for_all_subjects(self, saving_address='C:\\Users\Sakai Lab\Desktop', readFileName='transformed_data_cwt_stage1.pkl', saveImage=True, image_combine=True):
        pp = Post_Processing.PP()
        original_results = pp.read_file(fileName=readFileName)
        stage = original_results[0]

        dataset_dict = stage['dataset']
        label_dict = stage['label']

        fileNames = ['sub1002', 'sub1003', 'sub1004', 'sub1005', 'sub1006', 'sub1007', 'sub1008']
        data_list = []
        label_list = []
        for d in fileNames:
            if not os.path.exists(saving_address + '\\' + d):
                os.makedirs(saving_address + '\\' + d)
            add = saving_address + '\\' + d
            if image_combine:
                output = self.image_combination(data=dataset_dict, saving_address=add, subject_number=d, saveImage=saveImage)
                data_list.append(output)
            else:
                data_list.append(dataset_dict[d])
            label_list.append(label_dict[d])

        return data_list, label_list

    # Read/load pre-saved CWT images for all seven subjects, and return the data dictionary.
    def read_image_to_array(self, address='C:\\Users\Sakai Lab\Desktop\\Nov\stage1\\', scale=256):
        folderNames = ['sub1002', 'sub1003', 'sub1004', 'sub1005', 'sub1006', 'sub1007', 'sub1008']
        save_list = []
        for folder in folderNames:
            dir = address + folder + '\*.png'
            filelist = glob.glob(dir)
            subject = np.array([np.array(Image.open(fname).convert('RGB')) for fname in filelist])
            resized_subject = np.zeros(shape=(subject.shape[0], scale, scale, subject.shape[3]))
            for i in range(subject.shape[0]):
                resized_subject[i] = resize(subject[i], (scale, scale), mode='constant')
            save_list.append(resized_subject)

        dataset_dict = {'sub1002': save_list[0], 'sub1003': save_list[1], 'sub1004': save_list[2],
                        'sub1005': save_list[3], 'sub1006': save_list[4], 'sub1007': save_list[5],
                        'sub1008': save_list[6]}
        return dataset_dict

    # -----------------------------------------------------------------------------------------TensorFlow-----------------------------------------------------------------------------------------------------------------------
    # Build my CNN model using tensorFlow, but it's not perform well compare to other CNN models, such as AlexNet and ResNet.
    # Input_shape must be a tuple that contained 3 numbers corresponds to (image_width, image_height, and channels).
    def build_cnn_model(self, input_shape, activation='relu', ObjOrCate='obj'):
        model = Sequential()
        # 3 Convolution layer with Max polling
        model.add(Conv2D(64, (5, 5), activation=activation, padding='same', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (5, 5), activation=activation, padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (5, 5), activation=activation, padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())

        # 3 Full connected layer
        model.add(Dense(1024, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation=activation))
        model.add(Dropout(0.5))
        if ObjOrCate == 'obj':
            model.add(Dense(6, activation='softmax'))  # 6 classes
        elif ObjOrCate == 'cate':
            model.add(Dense(1, activation='sigmoid'))  # 2 classes

        # summarize the model
        print(model.summary())
        return model

    # Build simple version of AlexNet with my own full connected layers.
    def build_AlexNnet(self, input_shape, activation='relu', ObjOrCate='obj'):
        model = Sequential()
        # AlexNet
        model.add(Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation=activation, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation=activation, padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation=activation, padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation=activation, padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation=activation, padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

        # 3 Full connected layer
        model.add(Dense(512, activation=activation))
        model.add(Dropout(0.5))
        # model.add(Dense(1024, activation=activation))
        # model.add(Dropout(0.5))
        if ObjOrCate == 'obj':
            model.add(Dense(6, activation='softmax')) # 6 classes
        elif ObjOrCate == 'cate':
            model.add(Dense(1, activation='sigmoid')) # 2 classes

        # summarize the model
        print(model.summary())
        return model

    # Build vgg16 with my own full connected layers.
    def build_vgg16(self, input_shape, activation='relu', ObjOrCate='obj'):
        model = Sequential()
        # Layer 1: Convolutional
        model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 2: Convolutional
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 3: MaxPooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 4: Convolutional
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 5: Convolutional
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 6: MaxPooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 7: Convolutional
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 8: Convolutional
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 9: Convolutional
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 10: MaxPooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 11: Convolutional
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 12: Convolutional
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 13: Convolutional
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 14: MaxPooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 15: Convolutional
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 16: Convolutional
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 17: Convolutional
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=activation))
        # Layer 18: MaxPooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 19: Flatten
        model.add(Flatten())
        # # Layer 20: Fully Connected Layer
        # model.add(Dense(units=4096, activation=activation))
        # Layer 21: Fully Connected Layer
        model.add(Dense(units=512, activation=activation))
        # Layer 22: Softmax Layer
        if ObjOrCate == 'obj':
            model.add(Dense(6, activation='softmax'))  # 6 classes
        elif ObjOrCate == 'cate':
            model.add(Dense(1, activation='sigmoid'))  # 2 classes

        # summarize the model
        print(model.summary())
        return model

    # Compile and fit data to the CNN model.
    # compile_type need to pair with its model to have a good accuracy, such as alex.
    def compile_and_fit_model(self, model, X_train, y_train, X_vali, y_vali, batch_size, n_epochs, LR=0.01, compile_type='alex'):
        # compile the model
        if compile_type == 'my':
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])
        elif compile_type == 'alex':
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=tf.optimizers.SGD(learning_rate=LR),
                metrics=['accuracy'])

        # fit the model
        history = model.fit(x=X_train,
                            y=y_train,
                            batch_size=batch_size,
                            epochs=n_epochs,
                            verbose=1,
                            validation_data=(X_vali, y_vali))
        return model, history

    # Separate training data and label into train and validation dataset to fit the CNN model.
    # Input data can be either 3D or 4D array.
    def separate_train_and_validation_dataset(self, data, label, train_percentage=0.8):
        shuffle_array = list(range(data.shape[0]))
        continue_run = True
        seed = self.seed
        while continue_run:
            np.random.seed(seed)
            # shuffle_array = list(range(data.shape[0]))
            np.random.shuffle(shuffle_array)

            separation_line = int(np.floor(train_percentage * data.shape[0]))
            X_train = data[shuffle_array[:separation_line], :, :]
            X_validation = data[shuffle_array[separation_line:], :, :]

            Y_train = label[shuffle_array[:separation_line]]
            Y_validation = label[shuffle_array[separation_line:]]

            te_unique = np.unique(Y_validation, return_counts=True)
            te_class_inx, te_class_count = te_unique
            min_count, max_count = np.min(te_class_count), np.max(te_class_count)
            if (max_count - min_count) < 10:
                continue_run = False
            seed += 1
        return X_train, X_validation, Y_train, Y_validation

    # The method used to resize the all images within train/test dataset.
    # For example, resize an array with  (1500, 1024, 1024, 3) to (1500, 256, 256, 3)
    # For type 0, input data should be 3D array (samples, image_width, image_height)
    # For type 1, input data should be 4D array (samples, meg_channels, image_width, image_height)
    # For type 2, input data should be 4D array (samples, image_width, image_height, rgb)
    def data_resize(self, data, new_size, type='0'):
        sample_size = data.shape[0]
        if type == '0':
            shape = (sample_size, new_size, new_size)
            resized_data = np.zeros(shape=shape)
            for inx, d in enumerate(data):
                resized_data[inx] = resize(d, (new_size, new_size), mode='constant')

        elif type == '1':
            channels = data.shape[1]
            shape = (sample_size, channels, new_size, new_size)
            resized_data = np.zeros(shape=shape)
            for inx, d in enumerate(data):
                resized_data[inx] = resize(d[:], (new_size, new_size), mode='constant')

        elif type == '2':
            rgb = data.shape[3]
            shape = (sample_size, new_size, new_size, rgb)
            resized_data = np.zeros(shape=shape)
            for inx, d in enumerate(data):
                resized_data[inx] = resize(d, (new_size, new_size), mode='constant')

        return resized_data

    # The method used to perform data scaling for train and test data.
    # For type 0, input data should be 3D array (samples, image_width, image_height)
    # For type 1, input data should be 4D array (samples, meg_channels, image_width, image_height)
    def data_normalization_by_std_for_cnn(self, train_data_4d, test_data_4d, type='0'):
        if type == '0':
            tr_data_array_reshaped = train_data_4d.reshape(train_data_4d.shape[0],
                                                           (train_data_4d.shape[1] * train_data_4d.shape[2]))
            te_data_array_reshaped = test_data_4d.reshape(test_data_4d.shape[0],
                                                          (test_data_4d.shape[1] * test_data_4d.shape[2]))
        elif type == '1':
            tr_data_array_reshaped = train_data_4d.reshape((train_data_4d.shape[0] * train_data_4d.shape[1]),
                                                           (train_data_4d.shape[2] * train_data_4d.shape[3]))
            te_data_array_reshaped = test_data_4d.reshape((test_data_4d.shape[0] * test_data_4d.shape[1]),
                                                          (test_data_4d.shape[2] * test_data_4d.shape[3]))

        trData_std = np.std(tr_data_array_reshaped)
        trData_mean = np.mean(tr_data_array_reshaped)
        tr_normalized_data = (train_data_4d - trData_mean) / trData_std
        te_normalized_data = (test_data_4d - trData_mean) / trData_std

        return tr_normalized_data, te_normalized_data

    # ---------------------------------------------------------------------------------------------- Machine Learning Models --------------------------------------------------------------------------------------------------------------------------------------------
    # Method 1 used for searching best kernel and hyper-parameters for ML model(SVM).
    # Grid search using validation dataset
    def model_search_svm_by_ValidationSet(self, train_data, test_data, tr_label, te_label):
        # datasets separation
        x_train, x_val, y_train, y_val = train_test_split(train_data, tr_label, train_size=0.8, random_state=self.seed)
        best_score = 0
        best_parameters = {}

        param_list = [0.001, 0.01, 0.1, 1, 10]
        # kernel_list = ['linear', 'rbf']
        kernel_list = ['linear']
        for kernel in kernel_list:
            for gamma in param_list:
                for C in param_list:
                    model = SVC(kernel=kernel, C=C, gamma=gamma)
                    model.fit(x_train, y_train)
                    score = model.score(x_val, y_val)
                    if score > best_score:
                        best_score = score
                        best_parameters = {'kernel': kernel, 'gamma': gamma, 'C': C}

        svm = SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'])
        svm.fit(train_data, tr_label)
        final_score = svm.score(test_data, te_label)

        print('Best Validation Score: {}'.format(best_score))
        print('Final Score: {}'.format(final_score))
        print('Best Parameters: {}'.format(best_parameters))
        return final_score

    # Method 2 used for searching best kernel and hyper-parameters for ML model(SVM).
    # Grid search using cross validation
    def model_search_svm_by_CrossValidation(self, train_data, test_data, tr_label, te_label):
        best_score = 0
        best_parameters = {}

        param_list = [0.001, 0.01, 0.1, 1, 10]
        # kernel_list = ['linear', 'rbf']
        kernel_list = ['linear']
        for kernel in kernel_list:
            for gamma in param_list:
                for C in param_list:
                    model = SVC(kernel=kernel, C=C, gamma=gamma)
                    scores = cross_val_score(model, train_data, tr_label, cv=5)
                    score = np.mean(scores)
                    if score > best_score:
                        best_score = score
                        best_parameters = {'kernel': kernel, 'gamma': gamma, 'C': C}

        svm = SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'])
        svm.fit(train_data, tr_label)
        final_score = svm.score(test_data, te_label)

        print('Best Validation Score: {}'.format(best_score))
        print('Final Score: {}'.format(final_score))
        print('Best Parameters: {}'.format(best_parameters))
        return final_score

    # sklearn Support Vector Machine with rbf kernel
    def rbf_svm(self, train_data, test_data, tr_label, te_label, random_label=False, confusion_matrix_ON=False):
        train_label = tr_label.copy()
        test_label = te_label.copy()
        if random_label == True:
            np.random.seed(self.seed)
            np.random.shuffle(train_label)
            np.random.shuffle(test_label)

        model = SVC(kernel='rbf', C=10.0, gamma=0.001)
        model.fit(train_data, train_label)

        final_score = model.score(test_data, test_label)
        prediction = model.predict(test_data)

        if confusion_matrix_ON:
            conf_mat = confusion_matrix(y_true=test_label, y_pred=prediction)
            return final_score, conf_mat

        # print('Final Score: {}'.format(final_score))
        return final_score

    # sklearn Support Vector Machine with linear kernel
    def linear_svm(self, train_data, test_data, tr_label, te_label, random_label=False, confusion_matrix_ON=False):
        train_label = tr_label.copy()
        test_label = te_label.copy()
        if random_label == True:
            np.random.seed(self.seed)
            np.random.shuffle(train_label)
            np.random.shuffle(test_label)

        model = SVC(kernel='linear', C=0.01, gamma=0.001)
        model.fit(train_data, train_label)

        final_score = model.score(test_data, test_label)
        prediction = model.predict(test_data)

        if confusion_matrix_ON:
            conf_mat = confusion_matrix(y_true=test_label, y_pred=prediction)
            return final_score, conf_mat

        # print('Final Score: {}'.format(final_score))
        return final_score

    # sklearn Relevance Vector Machine with rbf kernel
    def rvm(self, train_data, test_data, tr_label, te_label):

        model = EMRVR(kernel='rbf', gamma='auto')
        model.fit(train_data, tr_label)

        score = model.score(test_data, te_label)
        prediction = model.predict(test_data)
        prediction = np.rint(prediction)

        correct_prediction = 0
        total_number_of_test_data = test_data.shape[0]

        for i in range(total_number_of_test_data):
            if prediction[i] == te_label[i]:
                correct_prediction += 1
        accuracy = correct_prediction / total_number_of_test_data

        # return score, accuracy, prediction
        return accuracy

    # sklearn Linear Discriminant Analysis.
    def lda(self, train_data, test_data, tr_label, te_label):

        lda = LinearDiscriminantAnalysis()
        lda.fit(train_data, tr_label)

        score = lda.score(test_data, te_label)
        prediction = lda.predict(test_data)

        correct_prediction = 0
        total_number_of_test_data = test_data.shape[0]

        for i in range(total_number_of_test_data):
            if prediction[i] == te_label[i]:
                correct_prediction += 1
        accuracy = correct_prediction / total_number_of_test_data

        return accuracy

    # sklearn Gaussian Naive Bayes.
    def gnb(self, train_data, test_data, tr_label, te_label):

        gnb = GaussianNB()
        gnb.fit(train_data, tr_label)

        score = gnb.score(test_data, te_label)
        prediction = gnb.predict(test_data)

        correct_prediction = 0
        total_number_of_test_data = test_data.shape[0]

        for i in range(total_number_of_test_data):
            if prediction[i] == te_label[i]:
                correct_prediction += 1
        accuracy = correct_prediction / total_number_of_test_data
        return accuracy

    # Five-fold cross-validation for linear SVM
    def corss_validation_svm(self, train_data, test_data, tr_label, te_label):

        X = np.concatenate((train_data, test_data), axis=0)
        y = np.concatenate((tr_label, te_label), axis=0)
        model = SVC(kernel='linear', C=0.01, gamma=0.001)
        scores = cross_val_score(model, X, y, cv=5)
        # return np.mean(scores), np.std(scores)
        return np.mean(scores)

    # Output accuracies for each object, such as accuracy of Sushi or Gyoza.
    def accuracy_per_object(self, train_data, test_data, tr_label, te_label):
        true_label = list(te_label.copy())
        obj_inx = np.unique(true_label)
        count_each_label = [true_label.count(i) for i in obj_inx]
        correct_prediction = [0] * len(obj_inx)

        model = SVC(kernel='linear', C=0.01, gamma=0.001)
        model.fit(train_data, tr_label)
        prediction = model.predict(test_data)
        final_score = model.score(test_data, te_label)

        for true, pred in zip(te_label, prediction):
            if true == pred:
                correct_prediction[int(true)] += 1

        # print(correct_prediction)
        accuracy_per_object = [correct_prediction[int(i)] / count_each_label[int(i)] for i in obj_inx]
        return accuracy_per_object, final_score

    # sklearn Neural Network
    # Neural Network requires lots of data samples, so this is only for cross-subjects.
    # Since the PGA have lots of more data samples than base data, PGA will have two hidden layers and base will have only one.
    # Input Neurons would be 1860, and Output Neurons would be 6, so hidden neurons should be between 6 and 1860.
    # Fist, test out 2/3 of input neurons for hidden neurons, which is 1240.
    def neural_network(self, train_data, test_data, tr_label, te_label, baseOrPga='base', random_label=False, confusion_matrix_ON=False):
        if baseOrPga == 'base':
            hidden_layer_sizes = (1240, )
        else:
            hidden_layer_sizes = (1240, 1240)

        train_label = tr_label.copy()
        test_label = te_label.copy()
        if random_label == True:
            np.random.seed(self.seed)
            np.random.shuffle(train_label)
            np.random.shuffle(test_label)

        nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=self.seed)
        nn.fit(train_data, train_label)

        score = nn.score(test_data, test_label)
        prediction = nn.predict(test_data)
        if confusion_matrix_ON:
            conf_mat = confusion_matrix(y_true=test_label, y_pred=prediction)
            return score, conf_mat
        return score

    def Search_best_parameters_for_NN(self, train_data, test_data, tr_label, te_label, baseOrPga='base'):
        if baseOrPga == 'base':
            hidden_layer_sizes = [(1240, ), (930, ), (620, ), (1240, 930), (930, 620), (620, 620)]
        else:
            hidden_layer_sizes = [(1240, 1240), (930, 930), (620, 620), (1240, 930, 620), (930, 930, 620), (620, 620, 620)]
        parameters = {'solver': ('lbfgs', 'sgd', 'adam'), 'hidden_layer_sizes': hidden_layer_sizes}
        mlp = MLPClassifier()
        nn = GridSearchCV(estimator=mlp, param_grid=parameters, cv=StratifiedKFold(n_splits=2), n_jobs=2)
        nn.fit(train_data, tr_label)
        score = nn.score(test_data, te_label)
        prediction = nn.predict(test_data)
        details = nn.cv_results_

        return score, details