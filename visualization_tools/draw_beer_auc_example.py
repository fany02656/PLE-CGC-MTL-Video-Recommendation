from visualization_tools.draw_auc_from_folder import draw_auc_Folder

# 构建文件路径
weight_of_loss = [1, 1]
tasks_name = ['income', 'marital']
metrics_need_to_plot = []

save_relative_address = ""
for index, task_name in enumerate(tasks_name):
    each_weight = weight_of_loss[index]
    save_relative_address += (task_name + str(each_weight))  # 相同方法构建保存路径
    if index != len(tasks_name)-1:
        save_relative_address += "_"

    if each_weight == 1:
        metrics_need_to_plot.append("train_" + task_name)  # 仅绘制受训练的标签
        metrics_need_to_plot.append("val_" + task_name)
        # metrics_need_to_plot.append("test_" + task_name)
print("metrics_need_to_plot", metrics_need_to_plot)
relative_path = save_relative_address + "/auc/"

algorithm_type = 'CGC'  # algorithm='MMoE' # 'OMoE' # 'shared_bottom'
file_folder_for_read = '../train_history/census/' + algorithm_type + "/" + relative_path
file_folder_to_save = '../train_history/census/' + algorithm_type + "/" + relative_path

draw_auc_Folder(file_folder_for_read, file_folder_to_save, algorithm_type, issave=True,  isplot=True,
                   firstNum=200, DrawList=metrics_need_to_plot, ymin=0, ymax=1)

