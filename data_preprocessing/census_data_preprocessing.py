
import pandas as pd
import numpy as np
from keras.utils import to_categorical
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

        if ori_feat_val <= 0:
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
    """
    print("train_df", train_df)
    print("train_df", train_df.columns)
    input()
    train_df_after_drop = train_df.drop(label_columns, axis=1)
    print("train_df after drop", train_df_after_drop)
    print("train_df after drop", train_df_after_drop.columns)
    input()
    transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    print("transformed_train", transformed_train)
    print("transformed_train", transformed_train.columns)
    input()
    """
    # 仅对 包含在column中数据进行操作
    transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    # print("transformed_train", [column for column in transformed_train])  # 将所有标志转换为onehot 编码
    transformed_other = pd.get_dummies(other_df.drop(label_columns, axis=1), columns=categorical_columns)
    # input()
    # Filling the missing column in the other set
    transformed_other['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    # One-hot encoding categorical labels
    train_income = to_categorical((train_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    # print("train_income", train_income)
    # input()
    train_marital = to_categorical((train_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)


    other_income = to_categorical((other_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    other_marital = to_categorical((other_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)
    # 维度为2的输出，label 为 01 或 10

    dict_outputs = {
        'income': train_income.shape[1],
        'marital': train_marital.shape[1]
    }
    # print("dict_outputs", dict_outputs)
    # input()
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
