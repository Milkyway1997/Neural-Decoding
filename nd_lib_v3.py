import WU_MEG_DP
import numpy as np
from sklearn.svm import SVC
from sklearn_rvm import EMRVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

class ND:
    def __init__(self, window_size=25, file_info={'fileName': [''], 'fileDir': ''}):
        self.message = "Script by Li Liu"
        self.channel = 64
        self.window_size = window_size
        self.seed = 25
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

    # Function use for testing different stages(There are 4 stages in the experiment) and different time range.
    def different_range_options(self, Menu):
        rangeOption = Menu['rangeOption']

        if rangeOption == '0':
            timeRangeList = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600)]
            baselineList = [(0, 0), (100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
        elif rangeOption == '1':
            timeRangeList = [(2000, 2100), (2100, 2200), (2200, 2300), (2300, 2400), (2400, 2500), (2500, 2600)]
            baselineList = [(2000, 2000), (2100, 2100), (2200, 2200), (2300, 2300), (2400, 2400), (2500, 2500)]
        elif rangeOption == '2':
            timeRangeList = [(4000, 4100), (4100, 4200), (4200, 4300), (4300, 4400), (4400, 4500), (4500, 4600)]
            baselineList = [(4000, 4000), (4100, 4100), (4200, 4200), (4300, 4300), (4400, 4400), (4500, 4500)]
        elif rangeOption == '3':
            timeRangeList = [(6000, 6100), (6100, 6200), (6200, 6300), (6300, 6400), (6400, 6500), (6500, 6600)]
            baselineList = [(6000, 6000), (6100, 6100), (6200, 6200), (6300, 6300), (6400, 6400), (6500, 6500)]

        # if rangeOption == '0':
        #     timeRangeList = [(0, 200), (200, 400), (400, 600)]
        #     baselineList = [(0, 0), (200, 200), (400, 400)]

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
    def Step1_dataPreparation(self, Menu):
        turnOnPga = Menu['turnOnPga']
        newTDT = Menu['newTDT']

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
                    train_datasets_lst = self.get_all_data_from_dataSgement(dataSegment=dataSegment_temp, sourceType='base')
                else:
                    train_datasets_lst = self.concatenate_new_data_from_dataSgement_to_existed_dataSet(datasets=train_datasets_lst, dataSegment=dataSegment_temp, sourceType='base')
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

    # Menu will contain 12 objects as below.
    # Menu['getRawData'] = ['yes', 'no']
    # Menu['turnOnPga'] = ['yes', 'no']
    # Menu['normType'] = ['my', 'sk', 'no']
    # Menu['turnOnSliceWindow'] = ['yes', 'no']
    # Menu['turnOnMeanRemoval'] = ['yes', 'no']
    # Menu['rangeOption'] = ['1', '2', '3']
    # Menu['sliceWindowType'] = ['cf', 'mean']
    # Menu['cfDegree '] = '1', '2', '3', '4'
    # Menu['avgTrials'] = 'yes', 'no
    # Menu['avgNum'] = '2', '3', '4'
    # Menu['pcaOn'] = 'before', 'after', 'no'
    # Menu['doublingData'] = 'yes', 'no'
    # Menu['newTDT'] = 'unknown', 'pga_gauss'
    def main(self, Menu):
        np.random.seed(self.seed)
        Menu_copy = Menu.copy()
        normType = Menu_copy['normType']
        turnOnPga = Menu_copy['turnOnPga']
        doublingData = Menu['doublingData']

        if normType == 'sk':
            Menu_copy['turnOnMeanRemoval'] = 'no'
        if turnOnPga == 'yes':
            Menu_copy['avgTrials'] = 'no'
            Menu['doublingData'] = 'no'
        if doublingData == 'yes':
            Menu_copy['avgTrials'] = 'no'

        output_dict = self.work_flow(Menu=Menu_copy)
        return output_dict

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

    # ---------------------------------------------------------------------------------------------- Machine Learning Models --------------------------------------------------------------------------------------------------------------------------------------------
    # Method 1 used for searching best kernel and hyper-parameters for ML model(SVM).
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

    def rbf_svm(self, train_data, test_data, tr_label, te_label):

        model = SVC(kernel='rbf', C=10.0, gamma=0.001)
        model.fit(train_data, tr_label)

        final_score = model.score(test_data, te_label)

        # print('Final Score: {}'.format(final_score))
        return final_score

    def linear_svm(self, train_data, test_data, tr_label, te_label):

        model = SVC(kernel='linear', C=0.01, gamma=0.001)
        model.fit(train_data, tr_label)

        final_score = model.score(test_data, te_label)

        # print('Final Score: {}'.format(final_score))
        return final_score

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

    def corss_validation(self, train_data, test_data, tr_label, te_label):

        X = np.concatenate((train_data, test_data), axis=0)
        y = np.concatenate((tr_label, te_label), axis=0)
        model = SVC(kernel='linear', C=0.01, gamma=0.001)
        scores = cross_val_score(model, X, y, cv=5)
        # return np.mean(scores), np.std(scores)
        return np.mean(scores)
