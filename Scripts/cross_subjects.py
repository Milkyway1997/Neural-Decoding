import nd_lib_v3
import numpy as np

class Cross_Subjects:
    def __init__(self, window_size=20, seed=25):
        self.window_size = window_size
        self.message = "Script by Li Liu"
        self.channel = 64
        self.seed = seed

    def cross_subjects_examination(self, fileDir, stage, pgaOn='no', baseOrFlip='base', ver=2):
        nd = nd_lib_v3.ND(window_size=self.window_size, seed=self.seed)
        results = []
        if ver == 1:
            loop = 1
        elif ver == 2:
            # loop = 7
            loop = 1

        for inx in range(loop):
            print('Index Of: {}'.format(inx))
            Menu = {}
            Menu['windowSize'] = self.window_size
            Menu['getRawData'] = 'no'
            Menu['doublingData'] = 'no'
            Menu['stage'] = stage
            Menu['rangeIndex'] = '-1'

            Menu['avgTrials'] = 'no'
            Menu['avgNum'] = '4'

            Menu['normType'] = 'sk'
            Menu['turnOnMeanRemoval'] = 'no'

            Menu['turnOnSliceWindow'] = 'yes'
            Menu['sliceWindowType'] = 'cf'
            Menu['cfDegree'] = '1'

            # Only clustered1 and half_pga1 can use flip
            Menu['baseOrFlip'] = baseOrFlip
            Menu['turnOnPga'] = pgaOn
            Menu['newTDT'] = 'clustered1'

            Menu['pcaOn'] = 'no'
            Menu['pcaComponents'] = '0.9'

            if ver == 1:
                output_dict = nd.cross_subjects_workflow(Menu=Menu, fileDir=fileDir, ver=ver)
            elif ver == 2:
                output_dict = nd.cross_subjects_workflow(Menu=Menu, fileDir=fileDir, ver=ver, loso_inx=inx)

            train_data = output_dict['train_data']
            test_data = output_dict['test_data']

            tr_label_name = output_dict['tr_label_name']
            tr_label_category = output_dict['tr_label_category']

            te_label_name = output_dict['te_label_name']
            te_label_category = output_dict['te_label_category']

            # print('----------------------------------------------linear----------------------------------------------------')
            name_linear = nd.linear_svm(train_data, test_data, tr_label_name, te_label_name)
            cate_linear = nd.linear_svm(train_data, test_data, tr_label_category, te_label_category)

            # print('----------------------------------------------Neural Network----------------------------------------------------')
            if pgaOn == 'yes':
                # name_nn = nd.neural_network(train_data, test_data, tr_label_name, te_label_name, baseOrPga='pga')
                # cate_nn = nd.neural_network(train_data, test_data, tr_label_category, te_label_category, baseOrPga='pga')
                name_nn, details_name = nd.Search_best_parameters_for_NN(train_data, test_data, tr_label_name, te_label_name, baseOrPga='pga')
                cate_nn, details_cate = nd.Search_best_parameters_for_NN(train_data, test_data, tr_label_category, te_label_category, baseOrPga='pga')
                nn_name = [name_nn, details_name]
                nn_cate = [cate_nn, details_cate]
            else:
                # name_nn = nd.neural_network(train_data, test_data, tr_label_name, te_label_name, baseOrPga='base')
                # cate_nn = nd.neural_network(train_data, test_data, tr_label_category, te_label_category, baseOrPga='base')
                name_nn, details_name = nd.Search_best_parameters_for_NN(train_data, test_data, tr_label_name, te_label_name, baseOrPga='base')
                cate_nn, details_cate = nd.Search_best_parameters_for_NN(train_data, test_data, tr_label_category, te_label_category, baseOrPga='base')
                nn_name = [name_nn, details_name]
                nn_cate = [cate_nn, details_cate]

            temp_dict = {'Menu': Menu, 'Accuracy_Name_linear': name_linear, 'Accuracy_Category_linear': cate_linear,
                         'Accuracy_Name_NN': nn_name, 'Accuracy_Category_NN': nn_cate}
            results.append(temp_dict)
        return results

    def main(self, pgaOn='no', baseOrFlip='base', ver=2):
        output_lst = []
        # directions = ['D:\\Data\OCT_TDT\clustered1\Stage1\data_PGA', 'D:\\Data\OCT_TDT\clustered1\Stage2\data_PGA',
        #               'D:\\Data\OCT_TDT\clustered1\Stage3\data_PGA', 'D:\\Data\OCT_TDT\clustered1\Stage4\data_PGA']
        directions = ['D:\\Data\OCT_TDT\clustered1\Stage1\data_PGA', 'D:\\Data\OCT_TDT\clustered1\Stage2\data_PGA']

        for inx, dir in enumerate(directions):
            # temp_holder contains all results from stage1, then stage2, and so on so forth
            print("--------------------------------------------------------------")
            print("This is Stage " + str(inx + 1))
            print("--------------------------------------------------------------")
            results = self.cross_subjects_examination(fileDir=dir, stage=str(inx+1), pgaOn=pgaOn, baseOrFlip=baseOrFlip, ver=ver)
            output_lst.append(results)

        return output_lst