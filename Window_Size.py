import comparation_for_pga_all_sub

class window_size:
    def __init__(self):
        self.message = "Script by Li Liu"
        self.channel = 64
        self.seed = 25

    def multiples(self, value, length):
        return [value * i for i in range(1, length + 1)]

    def check_valid(self, lst, number):
        out_lst = []
        for i in lst:
            if (number % i) == 0:
                out_lst.append(i)
        return out_lst

    def accuracy_by_window_size(self, classifier='rbf-SVM', pgaOn='no'):

        window_lst = self.multiples(5, 20)
        # window_lst = self.check_valid(window_lst, 10)

        # window_lst = [20]
        name_mean_lst_ws = []
        name_std_lst_ws = []
        cate_mean_lst_ws = []
        cate_std_lst_ws = []
        original_results_ws = []

        for w in window_lst:
            run = comparation_for_pga_all_sub.Run(window_size=w)
            results_pga, results_base = run.main(classifier=classifier, pgaOn=pgaOn)

            name_mean_lst_stages = []
            name_std_lst_stages = []
            cate_mean_lst_stages = []
            cate_std_lst_stages = []
            original_results = []

            if pgaOn == 'yes':
                results = results_pga
            else:
                results = results_base

            for r in results:
                name_mean_lst_stages.append(r['mean_name'])
                name_std_lst_stages.append(r['std_name'])
                cate_mean_lst_stages.append(r['mean_cate'])
                cate_std_lst_stages.append(r['std_cate'])
                original_results.append(r['original_results'])

            name_mean_lst_ws.append(name_mean_lst_stages)
            name_std_lst_ws.append(name_std_lst_stages)
            cate_mean_lst_ws.append(cate_mean_lst_stages)
            cate_std_lst_ws.append(cate_std_lst_stages)
            original_results_ws.append(original_results)

            print("--------------------------------------------------------------")
            print("This is windows size " + str(w*2))
            print("--------------------------------------------------------------")

        return name_mean_lst_ws, name_std_lst_ws, cate_mean_lst_ws, cate_std_lst_ws, original_results_ws