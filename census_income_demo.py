"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""
import argparse

from matplotlib import pyplot
import pickle
import os
import random
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

from bottom_model.mmoe import MMoE
from bottom_model.omoe import OMoE
from bottom_model.cgc import CGC



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

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            print(
                'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                    output_name, round(train_roc_auc, 4),
                    output_name, round(validation_roc_auc, 4),
                    output_name, round(test_roc_auc, 4)
                )
            )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def data_preparation(SEED=2):
    # The column names are from
    # https://www2.1010data.com/documentationcenter/prod/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # Load the dataset in Pandas
    train_df = pd.read_csv(
        'data/census-income.data.gz',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    # TODO 需要将测试集进行拆分 拆分成 1:1 (拆分完之后怎么用) 原本代码已完成拆分
    other_df = pd.read_csv(
        'data/census-income.test.gz',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    """
    # print(other_df['age'])
    # print("other_df", other_df)
    # print("other_of[income_50k]", other_df['income_50k'].dtype)
    print("other_of[age]", other_df['age'].dtype)
    print("other_of[wage_per_hour]", other_df['wage_per_hour'].dtype)
    print("other_of[capital_gains]", other_df['capital_gains'].dtype)
    print("other_of[capital_losses]", other_df['capital_losses'].dtype)
    print("other_of[stock_dividends]", other_df['stock_dividends'].dtype)
    print("other_of[num_emp]", other_df['num_emp'].dtype)
    print("other_of[weeks_worked]", other_df['weeks_worked'].dtype)
    """
    numOfLower0 = 0
    numOfEqual0 = 0
    train_df.drop(columns=['instance_weight'])
    other_df.drop(columns=['instance_weight'])
    def continue2disperse(ori_feat_val):
        if ori_feat_val < 0:
            # print("出现小于0的异常值")
            # input()

            return 0
        if ori_feat_val == 0:
            # print("出现等于0的异常值")

            # input()
            return 0
        new_feat_val = np.floor( (np.log(ori_feat_val))**2 ).astype(int)
        return new_feat_val

    for algs in other_df:
        # print(algs)
        # input()
        if other_df[algs].dtype == "int64":
            # print("in int64")
            # print("other_df[algs]", other_df[algs])
            other_df[algs] = other_df[algs].map(continue2disperse)
            # print("train_df[algs]", train_df[algs])
            train_df[algs] = train_df[algs].map(continue2disperse)
            # input()


    # First group of tasks according to the paper
    label_columns = ['income_50k', 'marital_stat']

    # One-hot encoding categorical columns
    categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']
    # print("len(categorical_columns)", len(categorical_columns))
    train_raw_labels = train_df[label_columns]
    # print("train_raw_labels", train_raw_labels)  # 将这两列提取出来
    other_raw_labels = other_df[label_columns]

    # 仅对 包含在column中数据进行操作
    transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    print("transformed_train", [column for column in transformed_train])  # 将所有标志转换为onehot 编码
    transformed_other = pd.get_dummies(other_df.drop(label_columns, axis=1), columns=categorical_columns)
    # input()
    # Filling the missing column in the other set
    transformed_other['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    # One-hot encoding categorical labels
    train_income = to_categorical((train_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    # print("train_income", train_income)
    # input()
    train_marital = to_categorical((train_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)
    """
    print("train_marital", train_marital)
    # input()
    """
    other_income = to_categorical((other_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    other_marital = to_categorical((other_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)

    dict_outputs = {
        'income': train_income.shape[1],
        'marital': train_marital.shape[1]
    }
    """
    print("dict_outputs", dict_outputs)
    print("train_income.shape", train_income.shape)
    input()
    """
    dict_train_labels = {
        'income': train_income,
        'marital': train_marital
    }
    dict_other_labels = {
        'income': other_income,
        'marital': other_marital
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # Split the other dataset into 1:1 validation to test according to the paper
    validation_indices = transformed_other.sample(frac=0.5, replace=False, random_state=SEED).index
    test_indices = list(set(transformed_other.index) - set(validation_indices))
    validation_data = transformed_other.iloc[validation_indices]  # 这一行
    validation_label = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
    test_data = transformed_other.iloc[test_indices]
    test_label = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
    train_data = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info



def main(epochs=100, algorithm_type='MMoE', SEED=2):
    timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S')

    # SEED = 2
    # Fix numpy seed for reproducibility
    np.random.seed(SEED)
    # Fix random seed for reproducibility
    random.seed(SEED)
    # Fix TensorFlow graph-level seed for reproducibility
    tf.set_random_seed(SEED)
    tf_session = tf.Session(graph=tf.get_default_graph())
    K.set_session(tf_session)


    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation(SEED=SEED)
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))
    """
    # Set up the input layer
    input_layer = Input(shape=(num_features,))
    output_layers = []
    # Set up MMoE layer
    mmoe_layers = MMoE(
        units=4,
        num_experts=8,
        num_tasks=2
    )(input_layer)
    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=output_info[index][0],  # TODO
            name=output_info[index][1],
            activation='softmax',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    """

    input_layer = Input(shape=(num_features,))
    output_layers = []
    if algorithm_type == 'CGC':
        print("in CGC")
        # Set up MMoE layer
        cgc_layers = CGC(
            units=4,
            num_shared_experts=4,
            num_tasks=2,
            num_task_experts_list=[2, 2]
        )(input_layer)

        # Build tower layer from MMoE layer
        for index, task_layer in enumerate(cgc_layers):
            tower_layer = Dense(
                units=8,
                activation='relu',
                kernel_initializer=VarianceScaling())(task_layer)
            output_layer = Dense(
                units=output_info[index][0],  # 20200504
                name=output_info[index][1],
                activation='softmax',  # 20200504  # 这里 census 用的都是 softmax
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)
    elif algorithm_type == 'MMoE':
        print("in MMoE")
        # Set up MMoE layer
        mmoe_layers = MMoE(
            units=4,  # 20200504
            num_experts=8,
            num_tasks=2
        )(input_layer)

        # Build tower layer from MMoE layer
        for index, task_layer in enumerate(mmoe_layers):
            tower_layer = Dense(
                units=8,
                activation='relu',
                kernel_initializer=VarianceScaling())(task_layer)
            output_layer = Dense(
                units=output_info[index][0],  # 20200504
                name=output_info[index][1],
                activation='softmax',  # 20200504  # 这里 census 用的都是 softmax
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)
    elif algorithm_type == 'OMoE':
        print("in OMoE")
        # Set up OMoE layer
        omoe_layers = OMoE(
            units=4,  # 20200504
            num_experts=8,
            num_tasks=2
        )(input_layer)
        # Build tower layer from OMoE layer
        for index in range(len(output_info)):
            # print("index", index)
            tower_layer = Dense(
                units=8,
                activation='relu',
                kernel_initializer=VarianceScaling())(omoe_layers)
            output_layer = Dense(
                units=output_info[index][0],  # 20200504
                name=output_info[index][1],
                activation='softmax',  # 20200504
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)

    else:  # shared_bottom
        print("in shared_bottom")
        shared_bottom_layer = Dense(
            units=113,
            activation='relu',  # mmoe中expert activation func 是 relu
            kernel_initializer=VarianceScaling())(input_layer)
        for index in range(len(output_info)):
            tower_layer = Dense(
                units=8,
                activation='relu',
                kernel_initializer=VarianceScaling())(shared_bottom_layer)
            output_layer = Dense(
                units=output_info[index][0],  # 20200504
                name=output_info[index][1],
                activation='softmax',  # 20200504
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)
    ### 20200505 end

    # Compile model
    model = Model(inputs=[input_layer], outputs=output_layers)
    adam_optimizer = Adam()
    model.compile(
        loss={'income': 'binary_crossentropy', 'marital': 'binary_crossentropy'},
        optimizer=adam_optimizer,
        metrics=['accuracy']
    )

    # Print out model architecture summary
    model.summary()
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
            )
        ],
        epochs=epochs
    )



    save_address = "train_history/census/" + algorithm_type
    if not os.path.exists(save_address):
        os.makedirs(save_address)

    fileSig = algorithm_type + '_epochs_' + str(epochs) + '_' + timeRun

    # 记录Reward
    filenameWriteReward = os.path.join(save_address, fileSig + '.txt')
    with open(filenameWriteReward, 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == '__main__':
    # python census_income_demo.py --epochs 200 --SEED 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="")
    parser.add_argument("--SEED", type=int, default=0, help="")
    parser.add_argument("--model", type=str, default="CGC", help="")
    args = parser.parse_args()

    print(args.epochs)
    print(args.SEED)

    main(epochs=args.epochs, algorithm_type=args.model, SEED=args.SEED)  # algorithm='MMoE' # 'OMoE' # 'shared_bottom'
    # main(epochs=args.epochs, algorithm_type='MMoE', SEED=args.SEED)  # algorithm='MMoE' # 'OMoE' # 'shared_bottom'
    # main(epochs=args.epochs, algorithm_type='OMoE', SEED=args.SEED)  # algorithm='MMoE' # 'OMoE' # 'shared_bottom'
    # main(epochs=args.epochs, algorithm_type='shared_bottom', SEED=args.SEED)  # algorithm='MMoE' # 'OMoE' # 'shared_bottom'
