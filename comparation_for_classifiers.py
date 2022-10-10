import nd_lib_v3
import numpy as np


class Run:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.message = "Script by Li Liu"
        self.channel = 64
        self.seed = 25

    def run_all_classifiers(self, file_info, pgaOn='no', rangeOp='0'):
        nd = nd_lib_v3.ND(window_size=self.window_size, file_info=file_info)
        results = []

        for layer1 in rangeOp:
            Menu = {}
            Menu['getRawData'] = 'no'
            Menu['doublingData'] = 'no'
            Menu['rangeOption'] = layer1

            Menu['avgTrials'] = 'no'
            Menu['avgNum'] = '4'

            Menu['normType'] = 'sk'
            Menu['turnOnMeanRemoval'] = 'no'

            Menu['turnOnSliceWindow'] = 'yes'
            Menu['sliceWindowType'] = 'cf'
            Menu['cfDegree'] = '1'

            Menu['turnOnPga'] = pgaOn
            Menu['newTDT'] = 'clustered1'

            Menu['pcaOn'] = 'no'
            Menu['pcaComponents'] = '0.9'

            output_dict = nd.main(Menu=Menu)

            train_data = output_dict['train_data']
            test_data = output_dict['test_data']

            tr_label_name = output_dict['tr_label_name']
            tr_label_category = output_dict['tr_label_category']

            te_label_name = output_dict['te_label_name']
            te_label_category = output_dict['te_label_category']

            # print('----------------------------------------------linear----------------------------------------------------')
            name_linear = nd.linear_svm(train_data, test_data, tr_label_name, te_label_name)
            cate_linear = nd.linear_svm(train_data, test_data, tr_label_category, te_label_category)
            # print('----------------------------------------------rbf----------------------------------------------------')
            name_rbf = nd.rbf_svm(train_data, test_data, tr_label_name, te_label_name)
            cate_rbf = nd.rbf_svm(train_data, test_data, tr_label_category, te_label_category)
            # print('----------------------------------------------lda----------------------------------------------------')
            name_lda = nd.lda(train_data, test_data, tr_label_name, te_label_name)
            cate_lda = nd.lda(train_data, test_data, tr_label_category, te_label_category)
            # print('----------------------------------------------rvm----------------------------------------------------')
            name_rvm = nd.rvm(train_data, test_data, tr_label_name, te_label_name)
            cate_rvm = nd.rvm(train_data, test_data, tr_label_category, te_label_category)
            # print('----------------------------------------------gnb----------------------------------------------------')
            name_gnb = nd.gnb(train_data, test_data, tr_label_name, te_label_name)
            cate_gnb = nd.gnb(train_data, test_data, tr_label_category, te_label_category)


            temp_dict = {'Menu': Menu, 'Accuracy_Name_linear': name_linear, 'Accuracy_Category_linear': cate_linear,
                         'Accuracy_Name_rbf': name_rbf, 'Accuracy_Category_rbf': cate_rbf, 'Accuracy_Name_lda': name_lda, 'Accuracy_Category_lda': cate_lda,
                         'Accuracy_Name_rvm': name_rvm, 'Accuracy_Category_rvm': cate_rvm, 'Accuracy_Name_gnb': name_gnb, 'Accuracy_Category_gnb': cate_gnb}
            results.append(temp_dict)

        return results


    def main(self, pgaOn='no'):
        files = ['sub1002_PGA.pkl', 'sub1003_PGA.pkl', 'sub1004_PGA.pkl', 'sub1005_PGA.pkl', 'sub1006_PGA.pkl', 'sub1007_PGA.pkl', 'sub1008_PGA.pkl']
        output_lst = []

        directions = ['D:\\Data\OCT_TDT\clustered1\Stage1\data_PGA', 'D:\\Data\OCT_TDT\clustered1\Stage2\data_PGA',
                      'D:\\Data\OCT_TDT\clustered1\Stage3\data_PGA', 'D:\\Data\OCT_TDT\clustered1\Stage4\data_PGA']
        # directions = ['D:\\Data\OCT_TDT\clustered1\data_PGA']
        for d in range(len(directions)):
            # temp_holder contains all results from stage1, then stage2, and so on so forth
            temp_holder = []
            for f in files:
                file_info = {'fileName': [f], 'fileDir': directions[d]}
                results = self.run_all_classifiers(file_info=file_info, pgaOn=pgaOn, rangeOp=str(d))
                temp_holder.extend(results)
            output_lst.append(temp_holder)

        return output_lst