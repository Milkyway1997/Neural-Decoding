import nd_lib_v3
import numpy as np
from numba import jit, cuda


class Run:
    def __init__(self, window_size=25, seed=25):
        self.window_size = window_size
        self.message = "Script by Li Liu"
        self.channel = 64
        self.seed = seed

    def run_conditions(self, file_info, stage, classifier, pgaOn):
        nd = nd_lib_v3.ND(window_size=self.window_size, file_info=file_info, seed=self.seed)
        results = []
        avgNum = ['1', '2', '3', '4', '5']
        cfDegree = ['0', '1', '2', '3', '4']
        doublingData = ['yes', 'no']
        pca = ['before', 'after', 'no']
        pcaComponents = ['0.8', '0.85', '0.9', '0.95', '0.99']
        turnOnSliceWindow = ['yes', 'no']
        EmptyLayer = ['Empty']
        # Not Getting Raw data for Range Option 1
        for layer in doublingData:
            Menu = {}
            Menu['getRawData'] = 'no'
            Menu['doublingData'] = layer
            Menu['stage'] = stage
            Menu['rangeIndex'] = '-1'

            Menu['avgTrials'] = 'no'
            Menu['avgNum'] = '3'

            Menu['normType'] = 'sk'
            Menu['turnOnMeanRemoval'] = 'no'

            Menu['turnOnSliceWindow'] = 'yes'
            Menu['sliceWindowType'] = 'cf'
            Menu['cfDegree'] = '1'

            # Only clustered1 and half_pga1 can use flip
            Menu['baseOrFlip'] = 'flip'
            Menu['turnOnPga'] = pgaOn
            Menu['newTDT'] = 'clustered1'

            Menu['pcaOn'] = 'no'
            Menu['pcaComponents'] = '0.9'

            output_dict = nd.main(Menu=Menu)

            doublingData = Menu['doublingData']
            if doublingData == 'yes':
                name, score, prediction = nd.model_for_doubled_data(input_dict=output_dict, classification_type='name')
                cate, score_, prediction_ = nd.model_for_doubled_data(input_dict=output_dict, classification_type='category')
            else:
                train_data = output_dict['train_data']
                test_data = output_dict['test_data']

                tr_label_name = output_dict['tr_label_name']
                tr_label_category = output_dict['tr_label_category']

                te_label_name = output_dict['te_label_name']
                te_label_category = output_dict['te_label_category']

                if classifier == 'LDA':
                    name = nd.lda(train_data, test_data, tr_label_name, te_label_name)
                    cate = nd.lda(train_data, test_data, tr_label_category, te_label_category)

                elif classifier == 'rbf-SVM':
                    name = nd.rbf_svm(train_data, test_data, tr_label_name, te_label_name)
                    cate = nd.rbf_svm(train_data, test_data, tr_label_category, te_label_category)

                elif classifier == 'linear-SVM':
                    name = nd.linear_svm(train_data, test_data, tr_label_name, te_label_name, random_test_label=False)
                    cate = nd.linear_svm(train_data, test_data, tr_label_category, te_label_category, random_test_label=False)

                elif classifier == 'RVM':
                    name = nd.rvm(train_data, test_data, tr_label_name, te_label_name)
                    cate = nd.rvm(train_data, test_data, tr_label_category, te_label_category)

                elif classifier == 'GNB':
                    name = nd.gnb(train_data, test_data, tr_label_name, te_label_name)
                    cate = nd.gnb(train_data, test_data, tr_label_category, te_label_category)

            temp_dict = {'Menu': Menu, 'Accuracy_Name': name, 'Accuracy_Category': cate}
            results.append(temp_dict)
        return results

    # bop stands for base or pga, input of bop should be either 'pga' or 'base'
    # results should contain all subjects' data not just one person
    def get_std_and_mean(self, results, bop):
        results_name = []
        results_cate = []
        if bop == 'base':
            for result in results:
                if result['Menu']['turnOnPga'] == 'no':
                    results_name.append(result['Accuracy_Name'])
                    results_cate.append(result['Accuracy_Category'])
            output_dict = {'mean_name': np.mean(results_name), 'std_name': np.std(results_name),
                           'mean_cate': np.mean(results_cate), 'std_cate': np.std(results_cate),
                           'original_results': results}
            return output_dict
        elif bop == 'pga':
            for result in results:
                if result['Menu']['turnOnPga'] == 'yes':
                    results_name.append(result['Accuracy_Name'])
                    results_cate.append(result['Accuracy_Category'])
            output_dict = {'mean_name': np.mean(results_name), 'std_name': np.std(results_name),
                           'mean_cate': np.mean(results_cate), 'std_cate': np.std(results_cate),
                           'original_results': results}
            return output_dict
        return 0

    # results_pga, results_base each contains 4 dictionary that contains 4 objects, which are mean_name, std_name, mean_cate, and std_cate.
    def main(self, classifier='rbf-SVM', pgaOn='no'):
        files = ['sub1002_PGA.pkl', 'sub1003_PGA.pkl', 'sub1004_PGA.pkl', 'sub1005_PGA.pkl', 'sub1006_PGA.pkl', 'sub1007_PGA.pkl', 'sub1008_PGA.pkl']
        # files = ['sub1002_Stage2_EMD.pkl', 'sub1003_Stage2_EMD.pkl', 'sub1004_Stage2_EMD.pkl', 'sub1005_Stage2_EMD.pkl', 'sub1006_Stage2_EMD.pkl', 'sub1007_Stage2_EMD.pkl', 'sub1002_Stage2_EMD.pkl']

        results_pga = []
        results_base = []

        directions = ['D:\\Data\OCT_TDT\clustered1\Stage1\data_PGA', 'D:\\Data\OCT_TDT\clustered1\Stage2\data_PGA',
                        'D:\\Data\OCT_TDT\clustered1\Stage3\data_PGA', 'D:\\Data\OCT_TDT\clustered1\Stage4\data_PGA']
        # directions = ['D:\\Data\OCT_TDT\half_pga1\Stage1\data_PGA']
        for d in range(len(directions)):
            # temp_holder contains all results from stage1, then stage2, and so
            print("--------------------------------------------------------------")
            print("This is Stage " + str(d+1))
            print("--------------------------------------------------------------")
            temp_holder = []
            for f in files:
                file_info = {'fileName': [f], 'fileDir': directions[d]}
                results = self.run_conditions(file_info=file_info, stage=str(d+1), classifier=classifier, pgaOn=pgaOn)
                # results = self.run_conditions(file_info=file_info)
                temp_holder.extend(results)
            results_pga.append(self.get_std_and_mean(results=temp_holder, bop='pga'))
            results_base.append(self.get_std_and_mean(results=temp_holder, bop='base'))
        return results_pga, results_base

