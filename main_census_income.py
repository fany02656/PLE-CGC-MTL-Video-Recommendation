import argparse
import time
import pickle
import os
import random
import datetime
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.optimizers import Adam, SGD
import keras

from keras.initializers import VarianceScaling
from keras.layers import Dense, Dropout, Input, Embedding,Layer, Flatten
from keras.models import Model
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from bottom_model.mmoe import MMoE
from bottom_model.cgc import CGC
from bottom_model.omoe import OMoE
from data_preprocessing.census_data_preprocessing import data_preparation as data_preprocessing




filename_record_auc = ""  # os.path.join(save_address,'auc_'+file_sig+time_run)

# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)
        f = open(filename_record_auc, 'a')
        num_of_tasks = len(self.model.output_names)
        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            print("output_name", output_name)
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            f.write(str(train_roc_auc) + ",")
            f.write(str(validation_roc_auc) + ",")
            if index == num_of_tasks - 1:
                f.write(str(test_roc_auc))
            else:
                f.write(str(test_roc_auc) + ",")
        f.write("\n")
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return



def main(epochs=100, algorithm_type='CGC', SEED=2, dataset="census", save_relative_address="default", weight_of_loss_Dir=None,
         batch_size=128, tower_unit=8, shared=124, num_shared_experts=4, num_task_experts_list=None, expert_unit=16, tasks_num=5, embed=5, learning_rate=0.1, l2=1e-3, decay_step=1, decay_rate=0.6):
    if num_task_experts_list is None:
        num_task_experts_list = [2, 2]
    if weight_of_loss_Dir is None:
        weight_of_loss_Dir = {'income': 1, 'marital': 1}
    # 对字典进行排序，保证与训练集各个标签的先后顺序一致
    key_sorted = sorted(weight_of_loss_Dir)
    New_weight_of_loss_Dir = {}
    for key in key_sorted:
        New_weight_of_loss_Dir[key] = weight_of_loss_Dir[key]
    weight_of_loss_Dir = New_weight_of_loss_Dir
    print("weight_of_loss_Dir")
    print(weight_of_loss_Dir)

    # 构建存储文件路径
    timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S')
    fileSig = "_" + algorithm_type + '_epochs_' + str(epochs) + "_SEED_" + str(SEED) + "_" + timeRun

    metrics_to_train = "auc"
    auc_save_address = "train_history/" + dataset + "/" + algorithm_type + "/" + save_relative_address + "/" + metrics_to_train + "/"
    isExist = os.path.exists(auc_save_address)
    if not isExist:
        os.makedirs(auc_save_address)
    global filename_record_auc
    filename_record_auc = os.path.join(auc_save_address, metrics_to_train + fileSig + '.txt')
    # 构建列的名称
    with open(filename_record_auc, 'a') as f:
        for index, output_name in enumerate(weight_of_loss_Dir):
            # print(output_name)
            train_col_name = "train_" + output_name
            validation_col_name = "val_" + output_name
            test_col_name = "test_" + output_name

            f.write(train_col_name + ",")
            f.write(validation_col_name + ",")
            if index == len(weight_of_loss_Dir) - 1:
                f.write(test_col_name)
            else:
                f.write(test_col_name + ",")
        f.write("\n")
    # loss
    metrics_to_train = "loss"
    loss_save_address = "train_history/" + dataset + "/" + algorithm_type + "/" + save_relative_address + "/" + metrics_to_train + "/"
    isExist = os.path.exists(loss_save_address)
    if not isExist:
        os.makedirs(loss_save_address)
    filename_record_loss = os.path.join(loss_save_address, metrics_to_train + fileSig + '.txt')
    print("loss_save_address", loss_save_address)

    np.random.seed(SEED)
    random.seed(SEED)
    tf.set_random_seed(SEED)
    tf_session = tf.Session(graph=tf.get_default_graph())
    K.set_session(tf_session)

    # 读入数据
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preprocessing()
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    ############################ build model ##############################
    # tower_unit = 8, shared = 124, num_shared_experts = 8, num_task_experts_list, expert_unit = 16, tasks_num = 5, embed = 5, learning_rate = 0.1, l2 = 1e-3, decay_step = 1, decay_rate = 0.6
    # 构建网络结构 如果这里使用 函数传参构建模型的话，需要传入7个参数，考虑一下tf.Flags等等
    input_layer = Input(shape=(num_features,))
    output_layers = []

    assert algorithm_type in ["CGC", "MMoE", "OMoE", "SBDNN"]
    if algorithm_type == "CGC":
        cgc_layers = CGC(
            units=expert_unit,
            num_shared_experts=num_shared_experts,
            num_tasks=tasks_num,
            num_task_experts_list=num_task_experts_list,
            # expert_bias_regularizer=regularizers.l2(l2)
        )(input_layer)

        # Build tower layer from MMoE layer
        for index, task_layer in enumerate(cgc_layers):
            tower_layer = Dense(
                units=tower_unit,
                activation='relu',
                kernel_initializer=VarianceScaling())(task_layer)
            output_layer = Dense(
                units=output_info[index][0],
                name=output_info[index][1],
                activation='softmax',
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)
    elif algorithm_type == 'MMoE':
        print("in MMoE")
        # Set up MMoE layer
        mmoe_layers = MMoE(
            units=expert_unit,
            num_experts=num_shared_experts,
            num_tasks=tasks_num
        )(input_layer)
        # Build tower layer from MMoE layer
        for index, task_layer in enumerate(mmoe_layers):
            tower_layer = Dense(
                units=tower_unit,
                activation='relu',
                kernel_initializer=VarianceScaling())(task_layer)
            output_layer = Dense(
                units=output_info[index][0],
                name=output_info[index][1],
                activation='softmax',
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)
    elif algorithm_type == 'OMoE':
        print("in OMoE")
        # Set up OMoE layer
        omoe_layers = OMoE(
            units=expert_unit,  # 20200504
            num_experts=num_shared_experts,
            num_tasks=tasks_num
        )(input_layer)
        # Build tower layer from OMoE layer
        for index in range(len(output_info)):
            tower_layer = Dense(
                units=tower_unit,
                activation='relu',
                kernel_initializer=VarianceScaling())(omoe_layers)
            output_layer = Dense(
                units=output_info[index][0],
                name=output_info[index][1],
                activation='softmax',
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)
    elif algorithm_type == "SBDNN":
        print("in shared_bottom")
        shared_bottom_layer = Dense(
            units=113,
            activation='relu',  # mmoe中expert activation func 是 relu
            kernel_initializer=VarianceScaling())(input_layer)
        for index in range(len(output_info)):
            tower_layer = Dense(
                units=tower_unit,
                activation='relu',
                kernel_initializer=VarianceScaling())(shared_bottom_layer)
            output_layer = Dense(
                units=output_info[index][0],
                name=output_info[index][1],
                activation='softmax',
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)
    else:
        return

    model = Model(inputs=[input_layer], outputs=output_layers)

    ############################ build model end ##########################
    # optimizer = SGD(lr=learning_rate, decay=decay_rate)
    optimizer = Adam(lr=learning_rate)
    print("weight_of_loss in main", weight_of_loss_Dir)
    model.compile(
        loss={'income': 'binary_crossentropy', 'marital': 'binary_crossentropy'},
        loss_weights=weight_of_loss_Dir,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    # Print out model architecture summary
    model.summary()


    # 训练
    # Train the model
    history = model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label)
                # test_data=(validation_data, validation_label)  # 显示 test_data没有对齐
            )
        ],
        epochs=epochs,
        batch_size=batch_size
    )


    with open(filename_record_loss, 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--SEED", type=int, default=int(time.time()*100)%399, help="")
    parser.add_argument("--epochs", type=int, default=50, help="")
    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--weight_of_loss", nargs='*', default=['1', '1'])

    parser.add_argument("--model", type=str, default="CGC")
    parser.add_argument("--tower_unit", type=int, default=8, help="")
    parser.add_argument("--shared", type=int, default=124, help="")
    # parser.add_argument("--expert_num", type=int, default=8, help="")  # MMoE 使用 num_shared_experts 传入expert参数
    parser.add_argument("--expert_unit", type=int, default=4, help="")
    parser.add_argument("--tasks_num", type=int, default=2, help="")
    parser.add_argument("--embed", type=int, default=5, help="")
    parser.add_argument("--num_shared_experts", type=int, default=4, help="")
    parser.add_argument("--num_task_experts_list", nargs='*', default=None)

    parser.add_argument("--learning_rate", type=float, default=0.1, help="")
    parser.add_argument("--l2", type=float, default=0.001, help="")
    parser.add_argument("--decay_step", type=int, default=1, help="")
    parser.add_argument("--decay_rate", type=float, default=0.6, help="")
    parser.add_argument("--relative_path", type=str, default="", help="")
    parser.add_argument("--dataset", type=str, default="census", help="")

    args = parser.parse_args()

    print(args.epochs)
    print(args.SEED)
    tasks_name = ['income', 'marital']
    # 需要训练的任务
    weight_of_loss_Dir = {}
    for index, task_name in enumerate(tasks_name):
        weight_of_loss_Dir[task_name] = int(args.weight_of_loss[index])
    # print(weight_of_loss_Dir)
    # 生成保存路径字符串
    save_relative_address = ""
    for index, task_name in enumerate(tasks_name):
        each_weight = args.weight_of_loss[index]
        save_relative_address += (task_name + each_weight)
        if index != len(tasks_name)-1:
            save_relative_address += "_"
    save_relative_address = args.relative_path + "/" + save_relative_address
    # print(save_relative_address)

    num_task_experts_list = []
    if args.num_task_experts_list == None:
        num_task_experts_list = [0] * args.tasks_num
    else:
        for index, num_task_experts_each in enumerate(args.num_task_experts_list):
            num_task_experts_list.append(int(num_task_experts_each))
    print(num_task_experts_list)

    main(epochs=args.epochs, algorithm_type=args.model, SEED=args.SEED, weight_of_loss_Dir=weight_of_loss_Dir, save_relative_address=save_relative_address,
         batch_size=args.batch_size, tower_unit=args.tower_unit, shared=args.shared, num_shared_experts=args.num_shared_experts, num_task_experts_list=num_task_experts_list, expert_unit=args.expert_unit, tasks_num=args.tasks_num, embed=args.embed, learning_rate=args.learning_rate, l2=args.l2, decay_step=args.decay_step, decay_rate=args.decay_rate)  # algorithm='MMoE' # 'OMoE' # 'shared_bottom'
