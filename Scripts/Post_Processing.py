import numpy as np
import pickle
import pandas as pd
from tabulate import tabulate
import Window_Size
from scipy import stats
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


class PP:
    def __init__(self):
        self.message = "Script by Li Liu"
        self.channel = 64
        self.seed = 25
        self.number_of_time_range = 15

    def read_file(self, fileName):
        with open(fileName, 'rb') as f:
            original_results = pickle.load(f)
        return original_results

    def inner(self, results):
        result = results.copy()
        holder_name = []
        holder_cate = []
        for r in result:
            temp_holder_name = []
            temp_holder_cate = []
            for ori in r:
                temp_holder_name.append(ori['Accuracy_Name'])
                temp_holder_cate.append(ori['Accuracy_Category'])
            holder_name.append(temp_holder_name)
            holder_cate.append(temp_holder_cate)
        return holder_name, holder_cate

    def clean_original_data(self, original_results):
        new_results = []
        results = original_results.copy()
        for r in results:
            temp = []
            name, cate = self.inner(r)
            temp.append(name)
            temp.append(cate)
            new_results.append(temp)
        new_results = np.array(new_results)
        return new_results

    def multiples(self, value, length):
        return [value * i for i in range(1, length + 1)]

    def multi_Str(self, value, length):
        return [str(value * i) for i in range(1, length + 1)]

    # -----------------------------------------------------------------------------------------Graph Per Window Size-----------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------Single Layer-----------------------------------------------------------------------------------------------------------------------
    # Subject1002-Subject 1008 -> 0-6
    def create_graph(self, original_results, x_str={'value': 10, 'length': 20}, stage=0, subject=0,
                     saving_address='C:\\Users\Sakai Lab\Desktop'):
        value = x_str['value']
        length = x_str['length']

        results = self.clean_original_data(original_results=original_results)
        classification_object = results[:, 0, stage, subject]
        classification_category = results[:, 1, stage, subject]

        x = self.multi_Str(value=value, length=length)
        plt.errorbar(x, classification_object, fmt='-^', label='Object')
        plt.errorbar(x, classification_category, fmt='-^', label='Category')
        plt.grid('on')

        plt.axhline(y=0.1667, color='#1f77b4', linestyle='--', label='Obj. Chance')
        plt.axhline(y=0.5, color='#ff7f0e', linestyle='--', label='Cat. Chance')
        plt.xlabel("Window Sizes(ms)")
        plt.ylabel("Accuracy Of Subject 100" + str(subject + 2))
        plt.title("Stage " + str(stage + 1) + " With Different Window Sizes")
        plt.ylim(ymax=1, ymin=0)
        # plt.legend(loc='best')
        plt.legend(ncol=2)

        plt.savefig(saving_address + '\sub100' + str(subject + 2) + '.png')
        plt.clf()
        return 0

    def mean_and_SD(self, original_results, stage=0):
        results = []
        cleaned_data = self.clean_original_data(original_results=original_results)
        for data in cleaned_data:
            classification_object = data[0, stage, :]
            classification_category = data[1, stage, :]

            temp = {'mean_object': np.mean(classification_object), 'std_object': np.std(classification_object),
                    'mean_cate': np.mean(classification_category), 'std_cate': np.std(classification_category)}
            results.append(temp)
        return results

    def create_mean_std_graph(self, original_results, stage=0, saving_address='C:\\Users\Sakai Lab\Desktop',
                              x_str={'value': 10, 'length': 20}):

        results = self.mean_and_SD(original_results=original_results, stage=stage)
        value = x_str['value']
        length = x_str['length']

        name_mean_lst_ws = []
        name_std_lst_ws = []
        cate_mean_lst_ws = []
        cate_std_lst_ws = []
        for r in results:
            name_mean_lst_ws.append(r['mean_object'])
            name_std_lst_ws.append(r['std_object'])
            cate_mean_lst_ws.append(r['mean_cate'])
            cate_std_lst_ws.append(r['std_cate'])

        x = self.multi_Str(value=value, length=length)

        plt.errorbar(x, name_mean_lst_ws, yerr=name_std_lst_ws, fmt='-^', label='Object')
        plt.errorbar(x, cate_mean_lst_ws, yerr=cate_std_lst_ws, fmt='-^', label='Category')
        plt.grid('on')

        plt.axhline(y=0.1667, color='#1f77b4', linestyle='--', label='Obj. Chance')
        plt.axhline(y=0.5, color='#ff7f0e', linestyle='--', label='Cat. Chance')

        plt.legend(ncol=2)
        plt.ylim(ymax=1, ymin=0)

        plt.xlabel("Window Sizes(ms)")
        plt.ylabel("Mean Accuracy Of All Subjects")
        plt.title("Stage " + str(stage + 1) + " With Different Window Sizes")

        plt.savefig(saving_address + '\Mean_Accuracy.png')
        plt.clf()
        return 0

    def auto_graph_saving(self, original_results, stage=0, total_subject_number=7,
                          saving_address='C:\\Users\Sakai Lab\Desktop', x_str={'value': 10, 'length': 20}):

        for i in range(total_subject_number):
            self.create_graph(original_results=original_results, stage=stage, subject=i, saving_address=saving_address,
                              x_str=x_str)

        self.create_mean_std_graph(original_results=original_results, stage=stage, saving_address=saving_address,
                                   x_str=x_str)
        return 0

    # -----------------------------------------------------------------------------------------Graph Per Window Size-----------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------Multiple Layer-----------------------------------------------------------------------------------------------------------------------

    def compare_graph(self, multi_results, sec_layer_length, x_str={'value': 10, 'length': 20}, stage=0, subject=0,
                      saving_address='C:\\Users\Sakai Lab\Desktop', label='Case'):
        value = x_str['value']
        length = x_str['length']
        x = self.multi_Str(value=value, length=length)

        for i in range(sec_layer_length):
            classification_object = multi_results[:, 0, stage, i]
            classification_category = multi_results[:, 1, stage, i]
            plt.errorbar(x, classification_object, fmt='-^', label='Object ' + label + str(i + 1))
            plt.errorbar(x, classification_category, fmt='-^', label='Category ' + label + str(i + 1))
        plt.grid('on')

        plt.axhline(y=0.1667, color='#1f77b4', linestyle='--')
        plt.axhline(y=0.5, color='#ff7f0e', linestyle='--')
        plt.xlabel("Window Sizes(ms)")
        plt.ylabel("Accuracy Of Subject 100" + str(subject + 2))
        plt.title("Stage " + str(stage + 1) + " With Different Window Sizes")
        plt.ylim(ymax=1, ymin=0)
        # plt.legend(loc='best')
        # ax = plt.subplot(111)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend below current axis
        plt.legend(loc='upper left', bbox_to_anchor=(0.9, 1.1), ncol=1, fancybox=True, shadow=True, fontsize=5)

        plt.savefig(saving_address + '\sub100' + str(subject + 2) + '.png')
        plt.clf()

        return 0

    def create_graph_for_multi_layer(self, original_results, sec_layer_length=5, stage=0, total_subject_number=7,
                                     address='C:\\Users\Sakai Lab\Desktop', label='Case',
                                     x_str={'value': 10, 'length': 20}):

        clean_data = self.clean_original_data(original_results=original_results)

        for i in range(total_subject_number):
            multi_results = clean_data[:, :, :, (i * sec_layer_length):(i * sec_layer_length) + sec_layer_length]
            self.compare_graph(multi_results=multi_results, sec_layer_length=sec_layer_length,
                               x_str=x_str, stage=stage, subject=i, saving_address=address, label=label)

        self.create_graph_for_multi_layer_mean(original_results=original_results, sec_layer_length=sec_layer_length,
                                               x_str=x_str, stage=stage, saving_address=address, label=label)
        return 0

    def reshape_data_for_multi_layer_mean(self, original_results, sec_layer_length=5, stage=0):
        clean_data = self.clean_original_data(original_results=original_results)
        number_of_subjects = int(len(original_results[0][0]) / sec_layer_length)
        multi_layer_clean_data = clean_data[:, :, stage, :]
        multi_layer_clean_data_list = []

        for sll in range(sec_layer_length):
            multi_layer_temp = []
            for num in range(number_of_subjects):
                temp = clean_data[:, :, stage, sll + sec_layer_length * num:sll + sec_layer_length * num + 1]
                multi_layer_clean_data = np.concatenate((multi_layer_clean_data, temp), axis=2)
                multi_layer_temp.append(temp)
            multi_layer_clean_data_list.append(multi_layer_temp)

        half = int(len(multi_layer_clean_data[0][0]) / 2)
        multi_layer_clean_data = multi_layer_clean_data[:, :, half:]

        multi_layer_clean_data_list = np.array(multi_layer_clean_data_list)
        output_list = []
        for data in multi_layer_clean_data_list:
            multi_layer_temp = []
            for i in range(data.shape[1]):
                classification_object = data[:, i, 0]
                classification_category = data[:, i, 1]
                temp = {'mean_object': np.mean(classification_object), 'std_object': np.std(classification_object),
                        'mean_cate': np.mean(classification_category), 'std_cate': np.std(classification_category)}
                multi_layer_temp.append(temp)
            output_list.append(multi_layer_temp)

        return output_list, multi_layer_clean_data_list, multi_layer_clean_data

    def create_graph_for_multi_layer_mean(self, original_results, sec_layer_length=5, x_str={'value': 10, 'length': 20},
                                          stage=0, saving_address='C:\\Users\Sakai Lab\Desktop', label='Case'):

        value = x_str['value']
        length = x_str['length']
        x = self.multi_Str(value=value, length=length)

        output_list, multi_layer_clean_data_list, multi_layer_clean_data = self.reshape_data_for_multi_layer_mean(
            original_results=original_results, sec_layer_length=sec_layer_length, stage=stage)

        count = 1
        for output in output_list:
            name_mean_lst_ws = []
            name_std_lst_ws = []
            cate_mean_lst_ws = []
            cate_std_lst_ws = []
            for r in output:
                name_mean_lst_ws.append(r['mean_object'])
                name_std_lst_ws.append(r['std_object'])
                cate_mean_lst_ws.append(r['mean_cate'])
                cate_std_lst_ws.append(r['std_cate'])
            plt.errorbar(x, name_mean_lst_ws, yerr=name_std_lst_ws, fmt='-^', label='Object ' + label + str(count))
            plt.errorbar(x, cate_mean_lst_ws, yerr=cate_std_lst_ws, fmt='-^', label='Category ' + label + str(count))
            count += 1

        plt.grid('on')
        plt.ylim(ymax=1, ymin=0)
        plt.axhline(y=0.1667, color='#1f77b4', linestyle='--')
        plt.axhline(y=0.5, color='#ff7f0e', linestyle='--')
        plt.legend(loc='upper left', bbox_to_anchor=(0.9, 1.1), ncol=1, fancybox=True, shadow=True, fontsize=5)

        plt.xlabel("Window Sizes(ms)")
        plt.ylabel("Mean Accuracy Of All Subjects")
        plt.title("Stage " + str(stage + 1) + " With Different Window Sizes")
        plt.savefig(saving_address + '\Mean_Accuracy.png')
        plt.clf()
        return 0

    # -----------------------------------------------------------------------------------------Bar Chart-----------------------------------------------------------------------------------------------------------------------

    def create_bar_chart(self, results, classifier='linear', saving_address='C:\\Users\Sakai Lab\Desktop',
                         dataType2='1'):
        dir = ['base_groupBy_stages', 'base_groupBy_subjects', 'pga_groupBy_stages', 'pga_groupBy_subjects', 'flip_groupBy_stages', 'flip_groupBy_subjects']
        dir2 = ['groupBy_stages', 'groupBy_subjects']
        if dataType2 == '1':
            for d in dir:
                if not os.path.exists(saving_address + '\\' + d):
                    os.makedirs(saving_address + '\\' + d)
            index = 0
            self.create_bar_chart_for_all_stages(results=results, BaseOrPGA='base', groupBy='stages',
                                                 classifier=classifier,
                                                 saving_address=saving_address + '\\' + dir[index])
            index += 1
            self.create_bar_chart_for_all_stages(results=results, BaseOrPGA='base', groupBy='subjects',
                                                 classifier=classifier,
                                                 saving_address=saving_address + '\\' + dir[index])
            index += 1
            self.create_bar_chart_for_all_stages(results=results, BaseOrPGA='pga', groupBy='stages',
                                                 classifier=classifier,
                                                 saving_address=saving_address + '\\' + dir[index])
            index += 1
            self.create_bar_chart_for_all_stages(results=results, BaseOrPGA='pga', groupBy='subjects',
                                                 classifier=classifier,
                                                 saving_address=saving_address + '\\' + dir[index])
            index += 1
            self.create_bar_chart_for_all_stages(results=results, BaseOrPGA='flip', groupBy='stages',
                                                 classifier=classifier,
                                                 saving_address=saving_address + '\\' + dir[index])
            index += 1
            self.create_bar_chart_for_all_stages(results=results, BaseOrPGA='flip', groupBy='subjects',
                                                 classifier=classifier,
                                                 saving_address=saving_address + '\\' + dir[index])

        #Type 2 contained not only mean but also standard deviation
        elif dataType2 == '2':
            for d in dir2:
                if not os.path.exists(saving_address + '\\' + d):
                    os.makedirs(saving_address + '\\' + d)
            results_name = results[0]
            results_cate = results[1]
            index = 0
            self.create_bar_chart_for_all_stages_dataType2(results_name=results_name, results_cate=results_cate,
                                                           groupBy='stages',
                                                           saving_address=saving_address + '\\' + dir2[index])
            index += 1
            self.create_bar_chart_for_all_stages_dataType2(results_name=results_name, results_cate=results_cate,
                                                           groupBy='subjects',
                                                           saving_address=saving_address + '\\' + dir2[index])
        return 0

    # BaseOrPGA could be 'pga' or 'base'. groupBy could be 'subjects' or 'stages'.
    def create_bar_chart_for_all_stages(self, results, BaseOrPGA='base', groupBy='subjects', classifier='linear',
                                        saving_address='C:\\Users\Sakai Lab\Desktop'):
        if BaseOrPGA == 'pga':
            data = results['PGA']
        elif BaseOrPGA == 'base':
            data = results['BASE']
        elif BaseOrPGA == 'flip':
            data = results['FLIP']

        results_name_ = []
        results_cate_ = []
        if groupBy == 'subjects':
            for i in range(len(data)):
                temp_holder_name = []
                temp_holder_cate = []
                for j in range(len(data[0])):
                    temp_holder_name.append(data[i][j]['Accuracy_Name_' + classifier])
                    temp_holder_cate.append(data[i][j]['Accuracy_Category_' + classifier])
                results_name_.append(temp_holder_name)
                results_cate_.append(temp_holder_cate)
            results_name_ = np.array(results_name_)
            results_cate_ = np.array(results_cate_)

            self.create_bar_chart_groupBy_subjects(data=results_name_, NameOrCategory='name',
                                                   saving_address=saving_address)
            self.create_bar_chart_groupBy_subjects(data=results_cate_, NameOrCategory='category',
                                                   saving_address=saving_address)
        else:
            for i in range(len(data[0])):
                temp_holder_name = []
                temp_holder_cate = []
                for j in range(len(data)):
                    temp_holder_name.append(data[j][i]['Accuracy_Name_' + classifier])
                    temp_holder_cate.append(data[j][i]['Accuracy_Category_' + classifier])
                results_name_.append(temp_holder_name)
                results_cate_.append(temp_holder_cate)
            results_name_ = np.array(results_name_)
            results_cate_ = np.array(results_cate_)

            self.create_bar_chart_groupBy_stages(data=results_name_, NameOrCategory='name',
                                                 saving_address=saving_address)
            self.create_bar_chart_groupBy_stages(data=results_cate_, NameOrCategory='category',
                                                 saving_address=saving_address)

        return 0

    # NameOrCategory could be 'name' or 'category'.
    def create_bar_chart_groupBy_subjects(self, data, NameOrCategory='name',
                                          saving_address='C:\\Users\Sakai Lab\Desktop'):

        colors = ['#1f77b4', '#7f7f7f', '#2ca02c', '#ff7f0e']
        width = 0.2
        numberOfStages = 4
        numberOfSubjects = 7
        x = np.array(self.multiples(1, numberOfSubjects))
        y = data
        for i in range(numberOfStages):
            plt.bar(x + width * i, y[i], width, color=colors[i])

        plt.xticks(x + width * 1.5, ['Sub1002', 'Sub1003', 'Sub1004', 'Sub1005', 'Sub1006', 'Sub1007', 'Sub1008'])
        plt.xlabel("Subjects")
        if NameOrCategory == 'name':
            plt.ylabel("Accuracy For Object Classification")
            plt.axhline(y=0.1667, color='#1f77b4', linestyle='--')
            plt.legend(['Obj. Chance', "Stage 1", "Stage 2", "Stage 3", "Stage 4"], loc='upper left',
                       bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=4)
            plt.ylim(ymax=1, ymin=0)
            plt.savefig(saving_address + '\Object_Classification.png')
        else:
            plt.ylabel("Accuracy For Category Classification")
            plt.axhline(y=0.5, color='#ff7f0e', linestyle='--')
            plt.legend(['Cat. Chance', "Stage 1", "Stage 2", "Stage 3", "Stage 4"], loc='upper left',
                       bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=4)
            plt.ylim(ymax=1, ymin=0)
            plt.savefig(saving_address + '\Category_Classification.png')
        plt.clf()
        return 0

    # NameOrCategory could be 'name' or 'category'.
    def create_bar_chart_groupBy_stages(self, data, NameOrCategory='name',
                                        saving_address='C:\\Users\Sakai Lab\Desktop'):

        colors = ['#1f77b4', '#7f7f7f', '#2ca02c', '#ff7f0e', '#17becf', '#e377c2', '#bcbd22']
        width = 0.1
        numberOfStages = 4
        numberOfSubjects = 7
        x = np.array(self.multiples(1, numberOfStages))
        y = data
        for i in range(numberOfSubjects):
            plt.bar(x + width * i, y[i], width, color=colors[i])

        plt.xticks(x + width * 3, ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
        plt.xlabel("Subjects")
        if NameOrCategory == 'name':
            plt.ylabel("Accuracy For Object Classification")
            plt.axhline(y=0.1667, color='#1f77b4', linestyle='--')
            plt.legend(['Obj. Chance', 'Sub1002', 'Sub1003', 'Sub1004', 'Sub1005', 'Sub1006', 'Sub1007', 'Sub1008'],
                       loc='upper left', bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=4)
            plt.ylim(ymax=1, ymin=0)
            plt.savefig(saving_address + '\Object_Classification.png')
        else:
            plt.ylabel("Accuracy For Category Classification")
            plt.axhline(y=0.5, color='#ff7f0e', linestyle='--')
            plt.legend(['Cat. Chance', 'Sub1002', 'Sub1003', 'Sub1004', 'Sub1005', 'Sub1006', 'Sub1007', 'Sub1008'],
                       loc='upper left', bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=4)
            plt.ylim(ymax=1, ymin=0)
            plt.savefig(saving_address + '\Category_Classification.png')
        plt.clf()
        return 0

    def bar_chart_for_cross_subjects_v1(self, data, classifier='linear', saving_address='C:\\Users\Sakai Lab\Desktop'):
        data_copy = data.copy()
        dir = ['BASE', 'FLIP', 'PGA']
        for d in dir:
            if not os.path.exists(saving_address + '\\' + d):
                os.makedirs(saving_address + '\\' + d)

        for d_inx, d in enumerate(dir):
            # -----------------------------------------Base-----------------------------------------
            if d_inx == 0:
                address = saving_address + '\\' + d
                self.bar_chart_for_cross_subjects_v1_inner(data=data_copy, classifier=classifier, dataType='base', saving_address=address)

            # -----------------------------------------Flip-----------------------------------------
            elif d_inx == 1:
                address = saving_address + '\\' + d
                self.bar_chart_for_cross_subjects_v1_inner(data=data_copy, classifier=classifier, dataType='flip', saving_address=address)

            # -----------------------------------------Pga-----------------------------------------
            elif d_inx == 2:
                address = saving_address + '\\' + d
                self.bar_chart_for_cross_subjects_v1_inner(data=data_copy, classifier=classifier, dataType='pga', saving_address=address)

        return 0

    def bar_chart_for_cross_subjects_v1_inner(self, data, classifier='linear', dataType='base', saving_address='C:\\Users\Sakai Lab\Desktop'):
        object_result, category_result = self.clean_data_step1(results=data, dataType=dataType, classifier=classifier)
        object_result = np.squeeze(np.array(object_result))
        category_result = np.squeeze(np.array(category_result))
        colors = ['#1f77b4', '#ff7f0e']
        width = 0.3
        numberOfStages = 4
        numberOfCategory = 2

        x = np.array(self.multiples(1, numberOfStages))
        y = np.zeros(shape=(object_result.shape[0], numberOfCategory))
        y[:, 0] = object_result
        y[:, 1] = category_result

        plt.axhline(y=0.1667, color='#1f77b4', linestyle='--')
        plt.axhline(y=0.5, color='#ff7f0e', linestyle='--')

        for i in range(numberOfCategory):
            plt.bar(x + width * i, y[:, i], width, color=colors[i])

        plt.xticks(x + width * 0.5, ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
        plt.xlabel("Stages")
        plt.ylabel("Accuracy For Cross-Subjects V1")
        plt.legend(['Obj. Chance', 'Cat. Chance', 'Object', 'Category'],
                   loc='upper left', bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=6)
        plt.ylim(ymax=1, ymin=0)
        plt.savefig(saving_address + '\Cross-Subjects_V1.png')
        plt.clf()

        return 0

    # -----------------------------------------------------------------------------------------Random Seeds-----------------------------------------------------------------------------------------------------------------------

    def random_seed_clean_data(self, original_results):
        clean_data = self.clean_original_data(original_results=original_results)
        regular_pool_name = []
        regular_pool_cate = []
        stages_name = []
        stages_cate = []
        #ttest of individual subject
        ind_sub_name = []
        ind_sub_cate = []
        for i in range(clean_data.shape[2]):
            temp_holder_name = []
            temp_holder_cate = []
            temp_pool_name = []
            temp_pool_cate = []
            temp_ind_sub_name = []
            temp_ind_sub_cate = []
            for j in range(clean_data.shape[3]):
                temp_pool_name.extend(clean_data[:, 0, i, j])
                temp_pool_cate.extend(clean_data[:, 1, i, j])

                temp_ind_sub_name.append(clean_data[:, 0, i, j])
                temp_ind_sub_cate.append(clean_data[:, 1, i, j])

                name_classification = {'mean': np.mean(clean_data[:, 0, i, j]), 'std': np.std(clean_data[:, 0, i, j])}
                cate_classification = {'mean': np.mean(clean_data[:, 1, i, j]), 'std': np.std(clean_data[:, 1, i, j])}
                temp_holder_name.append(name_classification)
                temp_holder_cate.append(cate_classification)

            stages_name.append(temp_holder_name)
            stages_cate.append(temp_holder_cate)
            regular_pool_name.append(temp_pool_name)
            regular_pool_cate.append(temp_pool_cate)
            ind_sub_name.append(temp_ind_sub_name)
            ind_sub_cate.append(temp_ind_sub_cate)

        regular_pool_name = np.array(regular_pool_name)
        regular_pool_cate = np.array(regular_pool_cate)
        stages_name = np.array(stages_name)
        stages_cate = np.array(stages_cate)
        ind_sub_name = np.array(ind_sub_name)
        ind_sub_cate = np.array(ind_sub_cate)

        return regular_pool_name, regular_pool_cate, stages_name, stages_cate, ind_sub_name, ind_sub_cate

    # Regular & Random pool contains a 2d array (4 stages * samples)
    def students_ttest(self, regular_pool_name, random_pool_name, regular_pool_cate, random_pool_cate, p_value=0.05):
        results = []
        for i in range(4):
            statistic_name, pvalue_name = stats.ttest_rel(a=regular_pool_name[i], b=random_pool_name[i],
                                                          alternative='greater')
            statistic_cate, pvalue_cate = stats.ttest_rel(a=regular_pool_cate[i], b=random_pool_cate[i],
                                                          alternative='greater')
            temp_dict = {'pvalue_object': pvalue_name, 'pvalue_category': pvalue_cate,
                         'significant_object': pvalue_name < p_value, 'significant_category': pvalue_cate < p_value}
            results.append(temp_dict)

        results = np.array(results)

        return results

    def get_fixation_cross(self, ind_sub_name, ind_sub_cate, ind_sub_name_random, ind_sub_cate_random):
        ind_sub_results = []
        for i in range(ind_sub_name.shape[1]):
            ind_result = self.students_ttest(regular_pool_name=ind_sub_name[:, i, :],
                                           random_pool_name=ind_sub_name_random[:, i, :],
                                           regular_pool_cate=ind_sub_cate[:, i, :],
                                           random_pool_cate=ind_sub_cate_random[:, i, :], p_value=0.05)
            ind_sub_results.append(ind_result)

        cross = []
        for stage in range(4):
            stages_name_ = []
            stages_cate_ = []
            for sub in range(7):
                pvalue_name = ind_sub_results[sub][stage]['pvalue_object']
                pvalue_cate = ind_sub_results[sub][stage]['pvalue_category']

                if pvalue_name < 0.01:
                    stages_name_.append('**')
                elif pvalue_name < 0.05:
                    stages_name_.append('*')
                else:
                    stages_name_.append('')

                if pvalue_cate < 0.01:
                    stages_cate_.append('**')
                elif pvalue_cate < 0.05:
                    stages_cate_.append('*')
                else:
                    stages_cate_.append('')
            comb = [stages_name_, stages_cate_]
            cross.append(comb)
        cross = np.array(cross)
        return cross

    # Type2 is using for Random Seed testing that it will contain STD for each bar.
    # BaseOrPGA could be 'pga' or 'base'. groupBy could be 'subjects' or 'stages'.
    def create_bar_chart_for_all_stages_dataType2(self, results_name, results_cate, groupBy='subjects',
                                                  saving_address='C:\\Users\Sakai Lab\Desktop', cross=None):

        mean_name = []
        std_name = []
        mean_cate = []
        std_cate = []

        if groupBy == 'subjects':
            for i in range(results_name.shape[0]):
                temp_mean_name = []
                temp_std_name = []
                temp_mean_cate = []
                temp_std_cate = []
                for j in range(results_name.shape[1]):
                    mean_name_ = results_name[i][j]['mean']
                    std_name_ = results_name[i][j]['std']
                    mean_cate_ = results_cate[i][j]['mean']
                    std_cate_ = results_cate[i][j]['std']

                    temp_mean_name.append(mean_name_)
                    temp_std_name.append(std_name_)
                    temp_mean_cate.append(mean_cate_)
                    temp_std_cate.append(std_cate_)
                mean_name.append(temp_mean_name)
                std_name.append(temp_std_name)
                mean_cate.append(temp_mean_cate)
                std_cate.append(temp_std_cate)

            mean_name = np.array(mean_name)
            std_name = np.array(std_name)
            mean_cate = np.array(mean_cate)
            std_cate = np.array(std_cate)

            self.create_bar_chart_groupBy_subjects_dataType2(mean=mean_name, std=std_name, NameOrCategory='name',
                                                             saving_address=saving_address, fixation_cross=cross)
            self.create_bar_chart_groupBy_subjects_dataType2(mean=mean_cate, std=std_cate, NameOrCategory='category',
                                                             saving_address=saving_address, fixation_cross=cross)
        else:
            for i in range(results_name.shape[1]):
                temp_mean_name = []
                temp_std_name = []
                temp_mean_cate = []
                temp_std_cate = []
                for j in range(results_name.shape[0]):
                    mean_name_ = results_name[j][i]['mean']
                    std_name_ = results_name[j][i]['std']
                    mean_cate_ = results_cate[j][i]['mean']
                    std_cate_ = results_cate[j][i]['std']

                    temp_mean_name.append(mean_name_)
                    temp_std_name.append(std_name_)
                    temp_mean_cate.append(mean_cate_)
                    temp_std_cate.append(std_cate_)
                mean_name.append(temp_mean_name)
                std_name.append(temp_std_name)
                mean_cate.append(temp_mean_cate)
                std_cate.append(temp_std_cate)

            mean_name = np.array(mean_name)
            std_name = np.array(std_name)
            mean_cate = np.array(mean_cate)
            std_cate = np.array(std_cate)

            self.create_bar_chart_groupBy_stages_dataType2(mean=mean_name, std=std_name, NameOrCategory='name',
                                                           saving_address=saving_address, fixation_cross=cross)
            self.create_bar_chart_groupBy_stages_dataType2(mean=mean_cate, std=std_cate, NameOrCategory='category',
                                                           saving_address=saving_address, fixation_cross=cross)

        return 0

    # NameOrCategory could be 'name' or 'category'.
    def create_bar_chart_groupBy_subjects_dataType2(self, mean, std, NameOrCategory='name',
                                                    saving_address='C:\\Users\Sakai Lab\Desktop', fixation_cross=None):
        if NameOrCategory == 'name':
            type = 0
        else:
            type = 1
        colors = ['#1f77b4', '#7f7f7f', '#2ca02c', '#ff7f0e']
        width = 0.2
        numberOfStages = 4
        numberOfSubjects = 7
        x = np.array(self.multiples(1, numberOfSubjects))
        y = mean
        fig, ax = plt.subplots()
        for i in range(numberOfStages):
            pps = ax.bar(x=x + width * i, height=y[i], width=width, color=colors[i])
            if fixation_cross is not None:
                text_sub = fixation_cross[i, type, :]
                for inx, p in enumerate(pps):
                    text_stage = text_sub[inx]
                    height = p.get_height()
                    ax.text(x=p.get_x() + p.get_width() / 2, y=height + .06, s="{}".format(text_stage), ha='center')

        ax.set_xticks(x + width * 1.5, ['Sub1002', 'Sub1003', 'Sub1004', 'Sub1005', 'Sub1006', 'Sub1007', 'Sub1008'])
        ax.set_xlabel("Subjects")
        if NameOrCategory == 'name':
            ax.axhline(y=0.1667, color='#1f77b4', linestyle='--')
            ax.set_ylabel("Accuracy For Object Classification")
            ax.legend(["Obj. Chance", "Stage 1", "Stage 2", "Stage 3", "Stage 4"], loc='upper left',
                       bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=4)
            for i in range(numberOfStages):
                ax.errorbar(x=x + width * i, y=y[i], yerr=std[i], capsize=2, capthick=1, color='dimgrey', fmt=' ')
            ax.set_ylim(ymax=1, ymin=0)
            fig.savefig(saving_address + '\Object_Classification.png')
        else:
            ax.set_ylabel("Accuracy For Category Classification")
            ax.axhline(y=0.5, color='#ff7f0e', linestyle='--')
            ax.legend(["Cat. Chance", "Stage 1", "Stage 2", "Stage 3", "Stage 4"], loc='upper left',
                       bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=4)
            for i in range(numberOfStages):
                ax.errorbar(x=x + width * i, y=y[i], yerr=std[i], capsize=2, capthick=1, color='dimgrey', fmt=' ')
            ax.set_ylim(ymax=1, ymin=0)
            fig.savefig(saving_address + '\Category_Classification.png')
        plt.clf()
        return 0

    # NameOrCategory could be 'name' or 'category'.
    def create_bar_chart_groupBy_stages_dataType2(self, mean, std, NameOrCategory='name',
                                                  saving_address='C:\\Users\Sakai Lab\Desktop', fixation_cross=None):
        if NameOrCategory == 'name':
            type = 0
        else:
            type = 1
        colors = ['#1f77b4', '#7f7f7f', '#2ca02c', '#ff7f0e', '#17becf', '#e377c2', '#bcbd22']
        width = 0.1
        numberOfStages = 4
        numberOfSubjects = 7
        x = np.array(self.multiples(1, numberOfStages))
        y = mean
        fig, ax = plt.subplots()
        for i in range(numberOfSubjects):
            pps = ax.bar(x + width * i, y[i], width, color=colors[i])
            if fixation_cross is not None:
                text_sub = fixation_cross[:, type, i]
                for inx, p in enumerate(pps):
                    text_stage = text_sub[inx]
                    height = p.get_height()
                    ax.text(x=p.get_x() + p.get_width() / 2, y=height + .06, s="{}".format(text_stage), ha='center')

        ax.set_xticks(x + width * 3, ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
        ax.set_xlabel("Subjects")
        if NameOrCategory == 'name':
            ax.axhline(y=0.1667, color='#1f77b4', linestyle='--')
            ax.set_ylabel("Accuracy For Object Classification")
            ax.legend(["Obj. Chance", 'Sub1002', 'Sub1003', 'Sub1004', 'Sub1005', 'Sub1006', 'Sub1007', 'Sub1008'],
                      loc='upper left', bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=4)
            for i in range(numberOfSubjects):
                ax.errorbar(x=x + width * i, y=y[i], yerr=std[i], capsize=2, capthick=1, color='dimgrey', fmt=' ')

            ax.set_ylim(ymax=1, ymin=0)
            fig.savefig(saving_address + '\Object_Classification.png')
        else:
            ax.set_ylabel("Accuracy For Category Classification")
            ax.axhline(y=0.5, color='#ff7f0e', linestyle='--')
            ax.legend(["Cat. Chance", 'Sub1002', 'Sub1003', 'Sub1004', 'Sub1005', 'Sub1006', 'Sub1007', 'Sub1008'],
                      loc='upper left', bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=4)
            for i in range(numberOfSubjects):
                ax.errorbar(x=x + width * i, y=y[i], yerr=std[i], capsize=2, capthick=1, color='dimgrey', fmt=' ')
            ax.set_ylim(ymax=1, ymin=0)
            fig.savefig(saving_address + '\Category_Classification.png')
        plt.clf()
        return 0

    # -----------------------------------------------------------------------------------------Temporal examination-----------------------------------------------------------------------------------------------------------------------
    def clean_data_step1(self, results, dataType='base', classifier='linear'):
        if dataType == 'pga':
            data = results['PGA']
        elif dataType == 'base':
            data = results['BASE']
        elif dataType == 'flip':
            data = results['FLIP']

        results_name = []
        results_cate = []
        for i in range(len(data)):
            temp_holder_name = []
            temp_holder_cate = []
            for j in range(len(data[0])):
                temp_holder_name.append(data[i][j]['Accuracy_Name_' + classifier])
                temp_holder_cate.append(data[i][j]['Accuracy_Category_' + classifier])
            results_name.append(temp_holder_name)
            results_cate.append(temp_holder_cate)
        return results_name, results_cate

    def clean_data_step2(self, results, number_of_subjects=7, number_of_time_range=15, dataType='base',
                         classifier='linear'):
        results_name, results_category = self.clean_data_step1(results=results, dataType=dataType,
                                                               classifier=classifier)
        ranges = number_of_time_range

        new_results_name = []
        new_results_category = []
        for stage_name, stage_cate in zip(results_name, results_category):
            temp_holder_stages_name = []
            temp_holder_stages_cate = []
            for subject in range(number_of_subjects):
                temp_holder_stages_name.append(stage_name[subject * ranges:subject * ranges + ranges])
                temp_holder_stages_cate.append(stage_cate[subject * ranges:subject * ranges + ranges])
            new_results_name.append(temp_holder_stages_name)
            new_results_category.append(temp_holder_stages_cate)

        new_results_name = np.array(new_results_name)
        new_results_category = np.array(new_results_category)

        return new_results_name, new_results_category

    def temporal_exam_graph(self, results, stage=0, subject=0, saving_address='C:\\Users\Sakai Lab\Desktop',
                            dataType='base', classifier='linear', range_length=100, increase_Length=50, number_of_time_range=15):
        new_results_object, new_results_category = self.clean_data_step2(results=results, dataType=dataType,
                                                                         classifier=classifier,
                                                                         number_of_time_range=number_of_time_range)

        classification_object = new_results_object[stage, subject, :]
        classification_category = new_results_category[stage, subject, :]

        number_of_x_label = new_results_object.shape[2]
        x = self.create_x_label(num_of_ranges=number_of_x_label, range_length=range_length, increase_Length=increase_Length)

        # x = ['0-100', '50-150', '100-200', '150-250', '200-300', '250-350', '300-400', '350-450', '400-500', '450-550',
        #      '500-600', '550-650', '600-700', '650-750', '700-800']

        # plt.figure(figsize=(20,5))
        plt.errorbar(x, classification_object, fmt='-', label='Object')
        plt.errorbar(x, classification_category, fmt='-', label='Category')
        plt.grid('on')

        plt.axhline(y=0.1667, color='#1f77b4', linestyle='--', label='Obj. Chance')
        plt.axhline(y=0.5, color='#ff7f0e', linestyle='--', label='Cat. Chance')

        # plt.xlabel("Time Ranges")
        plt.ylabel("Accuracy Of Subject 100" + str(subject + 2))
        plt.title("Stage " + str(stage + 1) + " With Different Time Ranges")
        plt.ylim(ymax=1, ymin=0)
        if number_of_x_label > 25:
            plt.xticks(x[::5], rotation=90, fontsize=6)
        else:
            plt.xticks(rotation=90, fontsize=6)
        plt.legend(ncol=2)

        plt.savefig(saving_address + '\sub100' + str(subject + 2) + '.png')
        plt.clf()
        return 0

    def temporal_exam_mean_graph(self, results, stage=0, saving_address='C:\\Users\Sakai Lab\Desktop', dataType='base',
                                 classifier='linear', range_length=100, increase_Length=50, number_of_time_range=15):
        new_results_object, new_results_category = self.clean_data_step2(results=results, dataType=dataType,
                                                                         classifier=classifier,
                                                                         number_of_time_range=number_of_time_range)

        classification_object_mean = []
        classification_object_std = []
        classification_category_mean = []
        classification_category_std = []
        for i in range(number_of_time_range):
            classification_object_mean.append(np.mean(new_results_object[stage, :, i], axis=0))
            classification_object_std.append(np.std(new_results_object[stage, :, i], axis=0)*0)
            classification_category_mean.append(np.mean(new_results_category[stage, :, i], axis=0))
            classification_category_std.append(np.std(new_results_category[stage, :, i], axis=0)*0)

        number_of_x_label = new_results_object.shape[2]
        x = self.create_x_label(num_of_ranges=number_of_x_label, range_length=range_length, increase_Length=increase_Length)

        # x = ['0-100', '50-150', '100-200', '150-250', '200-300', '250-350', '300-400', '350-450', '400-500', '450-550',
        #      '500-600', '550-650', '600-700', '650-750', '700-800']

        plt.errorbar(x, classification_object_mean, yerr=classification_object_std, fmt='-', label='Object')
        plt.errorbar(x, classification_category_mean, yerr=classification_category_std, fmt='-', label='Category')
        plt.grid('on')

        plt.axhline(y=0.1667, color='#1f77b4', linestyle='--', label='Obj. Chance')
        plt.axhline(y=0.5, color='#ff7f0e', linestyle='--', label='Cat. Chance')

        # plt.xlabel("Time Ranges")
        plt.ylabel("Mean Accuracy Of All Subjects")
        plt.title("Stage " + str(stage + 1) + " With Different Time Ranges")
        plt.ylim(ymax=1, ymin=0)
        if number_of_x_label > 25:
            plt.xticks(x[::5], rotation=90, fontsize=6)
        else:
            plt.xticks(rotation=90, fontsize=6)
        plt.legend(ncol=2)

        plt.savefig(saving_address + '\Mean_Accuracy.png')
        plt.clf()
        return 0

    def temporal_exam_auto_graph_stages(self, results, stage=0, total_subject_number=7,
                                        saving_address='C:\\Users\Sakai Lab\Desktop', classifier='linear',
                                        dataType='base', range_length=100, increase_Length=50, number_of_time_range=15):

        for i in range(total_subject_number):
            self.temporal_exam_graph(results=results, stage=stage, subject=i, saving_address=saving_address,
                                     dataType=dataType, classifier=classifier, range_length=range_length, increase_Length=increase_Length, number_of_time_range=number_of_time_range)

        self.temporal_exam_mean_graph(results=results, stage=stage, saving_address=saving_address, dataType=dataType,
                                      classifier=classifier, range_length=range_length, increase_Length=increase_Length, number_of_time_range=number_of_time_range)
        return 0

    def temporal_exam_auto_graph(self, results, saving_address, classifier='linear', range_length=100, increase_Length=50, number_of_time_range=15):
        dir = ['BASE', 'FLIP', 'PGA']
        inner_dir = ['Stage1', 'Stage2', 'Stage3', 'Stage4']
        for d in dir:
            if not os.path.exists(saving_address + '\\' + d):
                os.makedirs(saving_address + '\\' + d)
                for id in inner_dir:
                    if not os.path.exists(saving_address + '\\' + d + '\\' + id):
                        os.makedirs(saving_address + '\\' + d + '\\' + id)

        for d_inx, d in enumerate(dir):
            # -----------------------------------------Base-----------------------------------------
            if d_inx == 0:
                for stage_inx, id in enumerate(inner_dir):
                    address = saving_address + '\\' + d + '\\' + id
                    self.temporal_exam_auto_graph_stages(results=results, stage=stage_inx, total_subject_number=7,
                                                         saving_address=address, classifier=classifier, dataType='base',
                                                         range_length=range_length, increase_Length=increase_Length, number_of_time_range=number_of_time_range)

            # -----------------------------------------Flip-----------------------------------------
            elif d_inx == 1:
                for stage_inx, id in enumerate(inner_dir):
                    address = saving_address + '\\' + d + '\\' + id
                    self.temporal_exam_auto_graph_stages(results=results, stage=stage_inx, total_subject_number=7,
                                                         saving_address=address, classifier=classifier, dataType='flip',
                                                         range_length=range_length, increase_Length=increase_Length, number_of_time_range=number_of_time_range)

            # -----------------------------------------Pga-----------------------------------------
            elif d_inx == 2:
                for stage_inx, id in enumerate(inner_dir):
                    address = saving_address + '\\' + d + '\\' + id
                    self.temporal_exam_auto_graph_stages(results=results, stage=stage_inx, total_subject_number=7,
                                                         saving_address=address, classifier=classifier, dataType='pga',
                                                         range_length=range_length, increase_Length=increase_Length, number_of_time_range=number_of_time_range)

        return 0

    def create_range_list(self, rangeIndex, start_time=0, range_length=100, increase_Length=50):

        range_a = start_time + rangeIndex * increase_Length
        range_b = range_a + range_length
        timeRangeList = [(range_a, range_b)]

        base = range_a
        baselineList = [(base, base)]

        return timeRangeList, baselineList

    def create_x_label(self, num_of_ranges, range_length=100, increase_Length=50):
        output = []

        for inx in range(num_of_ranges):
            timeRangeList, baselineList = self.create_range_list(rangeIndex=inx, start_time=0, range_length=range_length,
                                                            increase_Length=increase_Length)
            a, b = timeRangeList[0]
            label = '' + str(a) + '-' + str(b)
            output.append(label)

        return output

    # -----------------------------------------------------------------------------------------ML Classifiers Table-----------------------------------------------------------------------------------------------------------------------
    def getMean(self, cleaned_data_object, cleaned_data_category, number_of_time_range, stage):
        classification_object_mean = []
        classification_object_std = []
        classification_category_mean = []
        classification_category_std = []
        for i in range(number_of_time_range):
            classification_object_mean.append(np.mean(cleaned_data_object[stage, :, i], axis=0))
            classification_object_std.append(np.std(cleaned_data_object[stage, :, i], axis=0))
            classification_category_mean.append(np.mean(cleaned_data_category[stage, :, i], axis=0))
            classification_category_std.append(np.std(cleaned_data_category[stage, :, i], axis=0))

        return classification_object_mean, classification_object_std, classification_category_mean, classification_category_std

    def prep_create_table(self, results, dataType='base', number_of_time_range=1, stage=0):
        output_object_mean = []
        output_category_mean = []
        classifiers = ['linear', 'rbf', 'rvm', 'lda', 'gnb']
        for classifier in classifiers:
            new_results_object, new_results_category = self.clean_data_step2(results=results, dataType=dataType, classifier=classifier, number_of_time_range=number_of_time_range)
            object_mean, _, category_mean, _ = self.getMean(cleaned_data_object=new_results_object, cleaned_data_category=new_results_category, number_of_time_range=number_of_time_range, stage=stage)
            output_object_mean.extend([round(item, 3) for item in object_mean])
            output_category_mean.extend([round(item, 3) for item in category_mean])

        return output_object_mean, output_category_mean

    def create_table(self, results, dataType='base'):
        columns = ['Linear SVM', 'RBF SVM', 'RVM', 'LDA', 'GNB']
        rows = ['Stage1-Object', 'Stage1-Category', 'Stage2-Object', 'Stage2-Category', 'Stage3-Object',
                'Stage3-Category', 'Stage4-Object', 'Stage4-Category']

        results_list = []
        for i in range(4):
            output_object_mean, output_category_mean = self.prep_create_table(results=results, number_of_time_range=1, stage=i,
                                                                    dataType=dataType)
            results_list.append(output_object_mean)
            results_list.append(output_category_mean)

        df = pd.DataFrame(results_list)
        df.columns = columns
        df.index = rows
        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
        return df

    # -----------------------------------------------------------------------------------------Accuracy Per Individual Objects-----------------------------------------------------------------------------------------------------------------------
    def AccPerObj_Mean_of_subjects(self, reshaped_data, NameOrCategory):
        if NameOrCategory == 'name':
            numOfObj = 6
        elif NameOrCategory == 'cate':
            numOfObj = 2

        stages = []
        for stage in range(4):
            objects = []
            for object in range(numOfObj):
                objects.append(np.mean(reshaped_data[stage, :, object]))
            stages.append(objects)
        stages = np.array(stages)

        return stages

    def AccPerObj_reshape_data(self, data_object, data_category):
        reshaped_object = data_object.reshape((4, 7, 6))
        new_object_data = self.AccPerObj_Mean_of_subjects(reshaped_data=reshaped_object, NameOrCategory='name')
        reshaped_object = np.transpose(new_object_data)

        reshaped_category = data_category.reshape((4, 7, 2))
        new_category_data = self.AccPerObj_Mean_of_subjects(reshaped_data=reshaped_category, NameOrCategory='cate')
        reshaped_category = np.transpose(new_category_data)

        return reshaped_object, reshaped_category

    def AccPerObj(self, original_results, saving_address, dataType='base'):
        object_result, category_result = self.clean_data_step2(results=original_results, dataType=dataType, classifier='AccuracyPerObject', number_of_time_range=1)
        reshaped_object_result, reshaped_category_result = self.AccPerObj_reshape_data(data_object=object_result, data_category=category_result)

        self.AccPerObj_BarChart(data=reshaped_object_result, NameOrCategory='name', saving_address=saving_address)
        self.AccPerObj_BarChart(data=reshaped_category_result, NameOrCategory='cate', saving_address=saving_address)

        return 0

    def AccPerObj_auto_plot(self, original_results, saving_address):
        dir = ['BASE', 'FLIP', 'PGA']
        for d in dir:
            if not os.path.exists(saving_address + '\\' + d):
                os.makedirs(saving_address + '\\' + d)

        for d_inx, d in enumerate(dir):
            # -----------------------------------------Base-----------------------------------------
            if d_inx == 0:
                address = saving_address + '\\' + d
                self.AccPerObj(original_results=original_results, saving_address=address, dataType='base')

            # -----------------------------------------Flip-----------------------------------------
            elif d_inx == 1:
                address = saving_address + '\\' + d
                self.AccPerObj(original_results=original_results, saving_address=address, dataType='flip')

            # -----------------------------------------Pga-----------------------------------------
            elif d_inx == 2:
                address = saving_address + '\\' + d
                self.AccPerObj(original_results=original_results, saving_address=address, dataType='pga')

        return 0

    def AccPerObj_BarChart(self, data, NameOrCategory, saving_address):
        if NameOrCategory == 'name':
            colors = ['#1f77b4', '#7f7f7f', '#2ca02c', '#ff7f0e', '#17becf', '#bcbd22']
            width = 0.1
            numberOfStages = 4
            numberOfObject = 6
            x = np.array(self.multiples(1, numberOfStages))
            y = data
            for i in range(numberOfObject):
                plt.bar(x + width * i, y[i], width, color=colors[i])

            plt.xticks(x + width * 2.5, ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
            plt.xlabel("Individual Objects")
            plt.ylabel("Accuracy For Object Classification")
            plt.axhline(y=0.1667, color='#1f77b4', linestyle='--')
            plt.legend(['Obj. Chance', 'Sushi', 'Gyoza', 'Cookie', 'Knife', 'Kushi', 'Pencil'],
                       loc='upper left', bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=6)
            plt.ylim(ymax=1, ymin=0)
            plt.savefig(saving_address + '\Object_Classification.png')
        else:
            colors = ['#17becf', '#bcbd22']
            width = 0.3
            numberOfStages = 4
            numberOfCategory = 2
            x = np.array(self.multiples(1, numberOfStages))
            y = data
            for i in range(numberOfCategory):
                plt.bar(x + width * i, y[i], width, color=colors[i])

            plt.xticks(x + width * 0.5, ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
            plt.xlabel("Food vs Tool")
            plt.ylabel("Accuracy For Category Classification")
            plt.axhline(y=0.5, color='#ff7f0e', linestyle='--')
            plt.legend(['Cat. Chance', 'Food', 'Tool'],
                       loc='upper left', bbox_to_anchor=(0.95, 1.15), ncol=1, fancybox=True, shadow=True, fontsize=6)
            plt.ylim(ymax=1, ymin=0)
            plt.savefig(saving_address + '\Category_Classification.png')
        plt.clf()
        return 0