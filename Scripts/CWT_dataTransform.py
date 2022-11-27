import nd_lib_v3
import numpy as np

class CWT:
    def __init__(self, window_size=20, seed=25):
        self.window_size = window_size
        self.message = "Script by Li Liu"
        self.channel = 64
        self.seed = seed

    def data_transform(self, dir, stage='0'):

        files = ['sub1002_PGA.pkl', 'sub1003_PGA.pkl', 'sub1004_PGA.pkl', 'sub1005_PGA.pkl', 'sub1006_PGA.pkl',
                 'sub1007_PGA.pkl', 'sub1008_PGA.pkl']

        transformed_data = []
        labels = []
        for f in files:
            file_info = {'fileName': [f], 'fileDir': dir}
            nd = nd_lib_v3.ND(file_info=file_info, seed=self.seed)

            Menu = {}
            Menu['stage'] = stage
            Menu['rangeIndex'] = '-1'

            Menu['baseOrFlip'] = 'base'
            Menu['turnOnPga'] = 'no'
            Menu['newTDT'] = 'clustered1'

            train_datasets_lst = nd.Step1_dataPreparation(Menu=Menu, data_split=False)

            label_name, label_category = nd.data_labelling(datasets_lst=train_datasets_lst)
            temp_label_dict = {'label_name': label_name, 'label_category': label_category}
            labels.append(temp_label_dict)

            train_datasets_arr = np.concatenate([d for d in train_datasets_lst], axis=0)
            # rescaled_coef is a 4D array.
            rescaled_coef, _ = nd.cwt(data=train_datasets_arr, scale=64)
            transformed_data.append(rescaled_coef)

        dataset_dict = {'sub1002': transformed_data[0], 'sub1003': transformed_data[1], 'sub1004': transformed_data[2],
                       'sub1005': transformed_data[3], 'sub1006': transformed_data[4], 'sub1007': transformed_data[5],
                       'sub1008': transformed_data[6]}

        label_dict = {'sub1002': labels[0], 'sub1003': labels[1], 'sub1004': labels[2],
                       'sub1005': labels[3], 'sub1006': labels[4], 'sub1007': labels[5],
                       'sub1008': labels[6]}
        return dataset_dict, label_dict

    def main(self):
        output_lst = []
        # directions = ['D:\\Data\OCT_TDT\half_pga1\Stage1\data_PGA', 'D:\\Data\OCT_TDT\half_pga1\Stage2\data_PGA',
        #               'D:\\Data\OCT_TDT\half_pga1\Stage3\data_PGA', 'D:\\Data\OCT_TDT\half_pga1\Stage4\data_PGA']
        # directions = ['D:\\Data\OCT_TDT\clustered1\Stage1\data_PGA', 'D:\\Data\OCT_TDT\clustered1\Stage2\data_PGA',
        #               'D:\\Data\OCT_TDT\clustered1\Stage3\data_PGA', 'D:\\Data\OCT_TDT\clustered1\Stage4\data_PGA']
        directions = ['D:\\Data\OCT_TDT\clustered1\Stage1\data_PGA']

        for inx, dir in enumerate(directions):
            print("--------------------------------------------------------------")
            print("This is Stage " + str(inx + 1))
            print("--------------------------------------------------------------")
            dataset_dict, label_dict = self.data_transform(dir=dir, stage=str(inx+1))
            dict = {'dataset': dataset_dict, 'label': label_dict}
            output_lst.append(dict)

        return output_lst