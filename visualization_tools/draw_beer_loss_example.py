from visualization_tools.draw_loss_from_folder import draw_loss_folder
import matplotlib.pyplot as plt

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
        metrics_need_to_plot.append(task_name + "_loss")  # 仅绘制受训练的标签
        metrics_need_to_plot.append("val_" + task_name + "_loss")
print("metrics_need_to_plot", metrics_need_to_plot)
relative_path = save_relative_address + "/loss/"


algorithm_type = 'CGC'  # algorithm='MMoE' # 'OMoE' # 'shared_bottom'
file_folder_for_read = '../train_history/census/' + algorithm_type + "/" + relative_path
file_folder_to_save = '../train_history/census/' + algorithm_type + "/" + relative_path
print("file_folder_for_read", file_folder_for_read)
draw_loss_folder(file_folder_for_read, file_folder_to_save, algorithm_type, issave=True, metrics_need_to_plot=metrics_need_to_plot,
                 firstNum=200)  #, ymin=0, ymax=2)
plt.show()
