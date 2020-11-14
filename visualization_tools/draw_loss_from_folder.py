import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
def draw_loss_folder(file_folder_for_read, file_folder_to_save, algorithm_type, Title=None, issave=False, linestyle=None, metrics_need_to_plot=None, firstNum=100,
                     xmin=None, xmax=None, ymin=None, ymax=None):
    file_list = []
    for file_name in os.listdir(file_folder_for_read):
        if file_name[-4:] == ".txt":
            file_total_path = file_folder_for_read + "\\" + file_name
            file_list.append(file_total_path)
    print(file_list)

    result_list_dir = {}

    # 初始化
    with open(file_list[0], 'rb') as f:
        test_history = pickle.load(f)
        for metrics in test_history:
            # if metrics[-3:] == 'acc' or metrics[-4:] == 'loss':
            # if metrics == 'loss':
            result_list_dir[metrics] = []
        time_stamp = np.arange(1, len(test_history['loss'])+1)[:firstNum]


    # print(len(result_list_dir))
    for file_total_path in file_list:
        # 对于该算法所有的实验数据
        with open(file_total_path, 'rb') as f:
            history = pickle.load(f)
            for metrics in history:
                # if metrics == 'loss':
                # print(metrics, history[metrics])
                result_list_dir[metrics].append(history[metrics][:firstNum])


    for metrics in result_list_dir:
        print(metrics, "in plot")

        if metrics_need_to_plot is not None:
            if metrics not in metrics_need_to_plot:
                continue

        print("result_list_dir", result_list_dir[metrics])
        result_all = np.vstack(result_list_dir[metrics])
        # print(result_all)
        result_average = np.mean(result_all, axis=0)

        if linestyle != None:
            plt.plot(time_stamp, result_average, label=metrics, linestyle=linestyle)
        else:
            plt.plot(time_stamp, result_average, label=metrics)
    if Title is not None:
        plt.title(Title)
    print(file_folder_to_save + "\\" + algorithm_type + "_average_with_errorbar.png")
    # plt.xlim(xmax=100)

    # if xmax is not None:
    plt.xlim(xmax=xmax, xmin=xmin)
    plt.ylim(ymax=ymax, ymin=ymin)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel("metrics")
    if issave:
        if metrics_need_to_plot != None:
            print(metrics_need_to_plot)
            plt.savefig(file_folder_to_save + "\\" + algorithm_type + "_average_with_errorbar_" + metrics_need_to_plot[0] + ".png")
        else:
            plt.savefig(file_folder_to_save + "\\" + algorithm_type + "_average_with_errorbar.png")
