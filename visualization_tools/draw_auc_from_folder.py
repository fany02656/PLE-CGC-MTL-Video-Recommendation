import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import os
def draw_auc_Folder(fileFolderForRead, fileFolderToSave, drawType="Default", Title=None, issave=False, isplot=False,
                       firstNum=10000, xmin=None, xmax=None, ymin=None, ymax=None, DrawList=None,
                       xlabel=None, ylabel=None, AlgName_2_NewNameColorLinestyle=None):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # 构造路径
    fileList = []
    for filename in os.listdir(fileFolderForRead):
        if os.path.isfile(fileFolderForRead + "/" + filename):
            # if (filename[:4] == "loss"):
            fileList.append(fileFolderForRead + "/" + filename)

    dataNpArray_list_dir = {}

    # TODO
    # 仅计算轮数足够的文件
    newfileList = []
    for fileTotalPath in fileList:
        print("fileTotalPath", fileTotalPath)
        dataInFrame = pd.read_csv(fileTotalPath, index_col=None)
        print("column", dataInFrame.columns)

        if dataInFrame.values.shape[0] >= firstNum:
            newfileList.append(fileTotalPath)
    if len(newfileList) > 0:
        fileList = newfileList
        print("轮数足够的文件数为", len(newfileList))
    else:
        print("没有轮数足够的文件")
        return

    # 初始化
    firstFilePath = fileList[0]
    dataInFrame = pd.read_csv(firstFilePath, index_col=None)
    for algsName in dataInFrame:
        # if algsName == 0:
        #     continue
        print("algsName", algsName)
        dataNpArray_list_dir[algsName] = []
    timeStamp = np.arange(1, dataInFrame.values.shape[0] + 1)[:firstNum]  # TODO 0603 2
    # timeStamp = np.arange(1, dataInFrame["Time(Iteration)"].values.shape[0] + 1)[20:firstNum+20]  # TODO 0603 2
    # print(timeStamp)

    for fileTotalPath in fileList:
        # 读取
        print("end fileTotalPath", fileTotalPath)
        dataInFrame = pd.read_csv(fileTotalPath)
        # dataInFrame = dataInFrame.drop(0, axis=1)

        # print(dataInFrame)

        for algsName in dataInFrame:

            print(algsName)
            dataNpArray = dataInFrame[algsName].values[:firstNum]  # TODO 0603 2



            dataNpArray_list_dir[algsName].append(dataNpArray)

    # print(dataNpArray_list_dir)
    # budgetPreciseValue = ["budget=0", "budget=2", "budget=8", "budget=17", "budget=25", "budget=33", "budget=42","budget=50", "budget=58", "budget=67"]

    i = -1
    for algsName in dataNpArray_list_dir:
        if DrawList == None:
            pass
        else:
            if algsName in DrawList:
                print(algsName, "绘制")
                pass
            else:
                print(algsName, "不绘制")
                continue
        i += 1

        print("algsName", algsName)

        dataNpArrayAll = np.vstack(dataNpArray_list_dir[algsName])
        dataNpArrayAverage = np.mean(dataNpArrayAll, axis=0)
        """
        dataNpArray_STD = np.std(dataNpArrayAll, ddof=1, axis=0)
        dataNpArray_STE = dataNpArray_STD / sqrtNSequence
        """
        # print("dataNpArray_STD", dataNpArray_STD)
        # print("dataNpArray_STE", dataNpArray_STE)
        # print("dataNpArrayAverage", dataNpArrayAverage)
        # ax = plt.gca()
        # ax.fill_between(timeStamp, dataNpArrayAverage-dataNpArray_STE, dataNpArrayAverage+dataNpArray_STE, facecolor='grey')

        if AlgName_2_NewNameColorLinestyle != None:
            print("!=None")
            if algsName in AlgName_2_NewNameColorLinestyle:
                # print("algsName", algsName, AlgName_2_NewNameColorLinestyle[algsName][0], AlgName_2_NewNameColorLinestyle[algsName][1], AlgName_2_NewNameColorLinestyle[algsName][2])
                plt.plot(timeStamp, dataNpArrayAverage, label=AlgName_2_NewNameColorLinestyle[algsName][0],
                         color=AlgName_2_NewNameColorLinestyle[algsName][1],
                         linestyle=AlgName_2_NewNameColorLinestyle[algsName][2])
        else:
            print("None")
            if algsName == "1train_review_overall" or algsName == "1val_review_overall":
                plt.plot(timeStamp, dataNpArrayAverage, label=algsName, linestyle='--')
            else:
                plt.plot(timeStamp, dataNpArrayAverage, label=algsName)  # ETC12

        """
        if RenameAlgs == None:
            if algsName == "LT-LinUCB":
                plt.plot(timeStamp, dataNpArrayAverage, label=algsName, linestyle='--')
            else:
                plt.plot(timeStamp, dataNpArrayAverage, label=algsName)
        else:
            print("RenameAlgs[i]", i, RenameAlgs[i])  # ETC12
            if algsName == "LT-LinUCB":
                plt.plot(timeStamp, dataNpArrayAverage, label=algsName, linestyle='--')
            else:
                plt.plot(timeStamp, dataNpArrayAverage, label=RenameAlgs[i])  # ETC12
        """
    print("绘制完所有数据")
    plt.ylim(ymax=ymax, ymin=ymin)
    plt.xlim(xmax=xmax, xmin=xmin)
    plt.legend(loc=4)
    if xlabel == None:
        plt.xlabel('Iteration')
    else:
        plt.xlabel(xlabel)

    if ylabel == None:
        plt.ylabel("auc")
    else:
        plt.ylabel(ylabel)

    if Title is not None:
        plt.title(Title)
    print(fileFolderToSave + "/" + drawType + "WithErrorBar.png")
    if issave:
        print("正在存储")
        fileFolderToSavePDF = fileFolderToSave + "/pdf/"
        fileFolderToSavePNG = fileFolderToSave + "/PNG/"
        isExist = os.path.exists(fileFolderToSavePDF)
        if not isExist:
            os.makedirs(fileFolderToSavePDF)
        isExist = os.path.exists(fileFolderToSavePNG)
        if not isExist:
            os.makedirs(fileFolderToSavePNG)



        pp = PdfPages(fileFolderToSavePDF + drawType + ".pdf")
        pp.savefig()
        plt.savefig(fileFolderToSavePNG + drawType + ".png")
        pp.close()
    if isplot:
        print("正在展示")
        plt.show()
    else:
        plt.close("all")


"""
fileFolderForRead = '..\\SimulationResults'
fileFolderToSave = '..\\SimulationResults'

fileFolderForRead = '..\\SimulationResults\\DIY12\\Reward'
fileFolderToSave = '..\\SimulationResults\\DIY12\\Reward'

drawType = "Average"  # "Cumulative" "Average" "Default"

draw_Reward_Folder(fileFolderForRead, fileFolderToSave, drawType=drawType, issave=True, firstNum=10000)
"""
