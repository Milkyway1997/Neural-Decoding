import numpy as np
import pickle
import pandas as pd
import Window_Size
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

class PP:
    def __init__(self):
        self.message = "Script by Li Liu"
        self.channel = 64
        self.seed = 25

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
    # Subject1002-Subject 1008 -> 0-6
    def create_graph(self, original_results, x_str={'value': 10, 'length': 20}, stage=0, subject=0, saving_address='C:\\Users\Sakai Lab\Desktop'):
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
        plt.ylabel("Accuracy Of Subject 100"+str(subject+2))
        plt.title("Stage " + str(stage+1) +" With Different Window Sizes")
        plt.ylim(ymax=1, ymin=0)
        # plt.legend(loc='best')
        plt.legend(ncol=2)

        plt.savefig(saving_address + '\sub100' + str(subject+2) + '.png')
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

    def create_mean_std_graph(self, original_results, stage=0, saving_address='C:\\Users\Sakai Lab\Desktop', x_str={'value': 10, 'length': 20}):

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
        plt.title("Stage " + str(stage+1) +" With Different Window Sizes")

        plt.savefig(saving_address + '\Mean_Accuracy.png')
        plt.clf()
        return 0

    def auto_graph_saving(self, original_results, stage=0, total_subject_number=7, saving_address='C:\\Users\Sakai Lab\Desktop', x_str={'value': 10, 'length': 20}):

        for i in range(total_subject_number):
            self.create_graph(original_results=original_results, stage=stage, subject=i, saving_address=saving_address, x_str=x_str)

        self.create_mean_std_graph(original_results=original_results, stage=stage, saving_address=saving_address, x_str=x_str)

        return 0

    def compare_graph(self, multi_results, sec_layer_length, x_str={'value': 10, 'length': 20}, stage=0, subject=0,
                      saving_address='C:\\Users\Sakai Lab\Desktop', label='Case'):
        value = x_str['value']
        length = x_str['length']
        x = self.multi_Str(value=value, length=length)

        for i in range(sec_layer_length):
            classification_object = multi_results[:, 0, stage, i]
            classification_category = multi_results[:, 1, stage, i]
            plt.errorbar(x, classification_object, fmt='-^', label='Object ' + label + str(i+1))
            plt.errorbar(x, classification_category, fmt='-^', label='Category ' + label + str(i+1))
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

    def create_graph_for_multi_layer(self, original_results, sec_layer_length=5, stage=0, total_subject_number=7, address='C:\\Users\Sakai Lab\Desktop', label='Case', x_str={'value': 10, 'length': 20}):

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

    def create_graph_for_multi_layer_mean(self, original_results, sec_layer_length=5, x_str={'value': 10, 'length': 20}, stage=0, saving_address='C:\\Users\Sakai Lab\Desktop', label='Case'):

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

    def create_bar_chart(self, results, classifier='linear', saving_address='C:\\Users\Sakai Lab\Desktop'):
        dir = ['base_groupBy_stages', 'base_groupBy_subjects', 'pga_groupBy_stages', 'pga_groupBy_subjects']
        for d in dir:
            if not os.path.exists(saving_address + '\\' + d):
                os.makedirs(saving_address + '\\' + d)
        index = 0
        self.create_bar_chart_for_all_stages(results=results, BaseOrPGA='base', groupBy='stages', classifier=classifier,
                                           saving_address=saving_address+'\\'+dir[index])
        index += 1
        self.create_bar_chart_for_all_stages(results=results, BaseOrPGA='base', groupBy='subjects', classifier=classifier,
                                             saving_address=saving_address + '\\' + dir[index])
        index += 1
        self.create_bar_chart_for_all_stages(results=results, BaseOrPGA='pga', groupBy='stages', classifier=classifier,
                                             saving_address=saving_address + '\\' + dir[index])
        index += 1
        self.create_bar_chart_for_all_stages(results=results, BaseOrPGA='pga', groupBy='subjects', classifier=classifier,
                                             saving_address=saving_address + '\\' + dir[index])
        return 0
    # BaseOrPGA could be 'pga' or 'base'. groupBy could be 'subjects' or 'stages'.
    def create_bar_chart_for_all_stages(self, results, BaseOrPGA='base', groupBy='subjects', classifier='linear', saving_address='C:\\Users\Sakai Lab\Desktop'):
        if BaseOrPGA == 'pga':
            data = results['PGA']
        else:
            data = results['BASE']

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

            self.create_bar_chart_groupBy_subjects(data=results_name_, NameOrCategory='name', saving_address=saving_address)
            self.create_bar_chart_groupBy_subjects(data=results_cate_, NameOrCategory='category', saving_address=saving_address)
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

            self.create_bar_chart_groupBy_stages(data=results_name_, NameOrCategory='name', saving_address=saving_address)
            self.create_bar_chart_groupBy_stages(data=results_cate_, NameOrCategory='category', saving_address=saving_address)

        return 0

    # NameOrCategory could be 'name' or 'category'.
    def create_bar_chart_groupBy_subjects(self, data, NameOrCategory='name', saving_address='C:\\Users\Sakai Lab\Desktop'):

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
            plt.legend(["Stage 1", "Stage 2", "Stage 3", "Stage 4"], loc='upper left', bbox_to_anchor=(0.9, 1.1), ncol=1, fancybox=True, shadow=True, fontsize=5)
            plt.axhline(y=0.1667, color='#1f77b4', linestyle='--', label='Obj. Chance')
            plt.ylim(ymax=1, ymin=0)
            plt.savefig(saving_address + '\Object_Classification.png')
        else:
            plt.ylabel("Accuracy For Category Classification")
            plt.legend(["Stage 1", "Stage 2", "Stage 3", "Stage 4"], loc='upper left', bbox_to_anchor=(0.9, 1.1), ncol=1, fancybox=True, shadow=True, fontsize=5)
            plt.axhline(y=0.5, color='#ff7f0e', linestyle='--', label='Cat. Chance')
            plt.ylim(ymax=1, ymin=0)
            plt.savefig(saving_address + '\Category_Classification.png')
        plt.clf()
        return 0

    # NameOrCategory could be 'name' or 'category'.
    def create_bar_chart_groupBy_stages(self, data, NameOrCategory='name', saving_address='C:\\Users\Sakai Lab\Desktop'):

        colors = ['#1f77b4', '#7f7f7f', '#2ca02c', '#ff7f0e', '#17becf', '#7f7f7f', '#bcbd22']
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
            plt.legend(['Sub1002', 'Sub1003', 'Sub1004', 'Sub1005', 'Sub1006', 'Sub1007', 'Sub1008'], loc='upper left', bbox_to_anchor=(0.9, 1.1), ncol=1, fancybox=True, shadow=True, fontsize=5)
            plt.axhline(y=0.1667, color='#1f77b4', linestyle='--', label='Obj. Chance')
            plt.ylim(ymax=1, ymin=0)
            plt.savefig(saving_address + '\Object_Classification.png')
        else:
            plt.ylabel("Accuracy For Category Classification")
            plt.legend(['Sub1002', 'Sub1003', 'Sub1004', 'Sub1005', 'Sub1006', 'Sub1007', 'Sub1008'], loc='upper left', bbox_to_anchor=(0.9, 1.1), ncol=1, fancybox=True, shadow=True, fontsize=5)
            plt.axhline(y=0.5, color='#ff7f0e', linestyle='--', label='Cat. Chance')
            plt.ylim(ymax=1, ymin=0)
            plt.savefig(saving_address + '\Category_Classification.png')
        plt.clf()
        return 0


