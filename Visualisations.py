import numpy as np
import sys
import os
import sparse
import itertools
from sklearn import metrics
from scipy.stats import sem
import matplotlib.pyplot as plt

def readMatrix(filename):
    try:
        return sparse.load_npz(filename.replace(".txt", ".npz"))
    except:
        try:
            return np.load(filename)
        except:
            with open(filename, 'r') as outfile:
                dims = outfile.readline().replace("# Array shape: (", "").replace(")", "").replace("\n", "").split(", ")
                for i in range(len(dims)):
                    dims[i] = int(dims[i])

            new_data = np.loadtxt(filename).reshape(dims)
        return new_data

def read_params(folder, features, output, featToClus, nbClus, fold, run=-1):
    s=""
    folderParams = "Output/" + folder + "/"

    txtFin = ""
    if run==-1:
        txtFin = "Final/"

    codeT=""
    for i in featToClus:
        codeT += f"{features[i]}({nbClus[i]})-"
    codeT += f"{output}"+"_fold-"+str(fold)+"of"+str(folds)

    thetas = []
    for i in range(len(features)):
        thetas.append(readMatrix(folderParams+txtFin + "T="+codeT+"_%.0f_" % (run)+s+"theta_"+str(i)+"_Inter_theta.npy"))

    p = readMatrix(folderParams+txtFin + "T="+codeT+"_%.0f_" % (run)+s+"Inter_p.npy")


    codeData = ""
    for i in range(len(features)):
        for _ in range(DS[i]):
            codeData += str(features[i])+"-"
    codeData += f"{output}"
    filenameData = "Data/" + folder + "/" + codeData + "_" + "fold-" + str(fold) + "of" + str(folds) + "_"

    intToOut, intToFeat = {}, []
    with open(filenameData+"outToInt.txt", "r", encoding="utf-8") as f:
        for line in f:
            nom, ind = line.replace("\n", "").split("\t")
            intToOut[int(ind)] = nom
    for i in range(len(features)):
        intToFeat.append({})
        with open(filenameData+"featToInt_"+str(features[i])+".txt", "r", encoding="utf-8") as f:
            for line in f:
                nom, ind = line.replace("\n", "").split("\t")
                intToFeat[-1][int(ind)] = nom

    return thetas, p, intToOut, intToFeat


def plotThetasGraph(thetas, p, intToOut, intToFeat):
    scaleFeat = 2.
    scaleClus = 10.
    scaleTypes = 1.
    scaleAlpha = 1.
    norm = 1


    nbTypes = len(thetas)
    for type in range(len(thetas)):
        maxTheta = np.max(thetas[type])
        shift = (type - nbTypes / 2) * scaleTypes
        nbFeat = thetas[type].shape[0] - 1
        nbClus = thetas[type].shape[-1] - 1
        for k in range(thetas[type].shape[1]):
            plt.plot(0, (k - nbClus / 2) * scaleClus + shift, "or", markersize=15)

        for i in range(len(thetas[type])):
            plt.text(-1, (i - nbFeat / 2) * scaleFeat + shift, intToFeat[type][i].replace("_", " "), ha="right",
                     va="center")
            for k in range(len(thetas[type][i])):
                plt.plot([-1, 0], [(i - nbFeat / 2) * scaleFeat + shift, (k - nbClus / 2) * scaleClus + shift], "k-", linewidth=thetas[type][i][k], alpha=np.exp(-scaleAlpha*(maxTheta**norm-thetas[type][i][k])))

    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.close()


def plotGraph1D(thetas, p, intToOut, intToFeat):
    def findClusters(mat):
        from sklearn.cluster import AgglomerativeClustering
        cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
        cluster.fit_predict(mat)
        return cluster.labels_

    scaleFeat = 2.
    scaleClus = 10.
    scaleOut = 10.
    scalex = 100
    scaleAlpha = 4.

    norm = 1

    nbFeat = thetas[0].shape[0]
    nbClus = thetas[0].shape[-1]
    nbOut = p.shape[-1]

    maxTheta = np.max(thetas[0])
    maxp = np.max(p)

    for k in range(nbClus):
        plt.plot(0, (k - (nbClus-1) / 2) * scaleClus, "or", markersize=15)

    clusLab = findClusters(thetas[0])
    for clus in range(len(clusLab)):
        for i in np.where(clusLab==clus)[0]:
            plt.text(-scalex, (i - (nbFeat-1) / 2) * scaleFeat, intToFeat[0][i].replace("_", " ")+" ", ha="right", va="center")
            for k in range(nbClus):
                plt.plot([-scalex, 0], [(i - (nbFeat-1) / 2) * scaleFeat, (k - (nbClus-1) / 2) * scaleClus], "k-", linewidth=1., alpha=np.exp(-scaleAlpha*(maxTheta**norm-thetas[0][i][k])))

    clusLab = findClusters(p)
    for j in range(nbOut):
        plt.text(scalex, (j - (nbOut-1) / 2) * scaleOut, intToOut[j].replace("_", " ")+" ", ha="left", va="center")
        for clus in range(len(clusLab)):
            for k in np.where(clusLab==clus)[0]:
                plt.plot([0, scalex], [(k - (nbClus-1) / 2) * scaleClus, (j - (nbOut-1) / 2) * scaleOut], "k-", linewidth=1., alpha=np.exp(-scaleAlpha*(maxp**norm-p[k][j])))


    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.close()




try:
    folder = sys.argv[1]
    features = np.array(sys.argv[2].split(","), dtype=int)
    output = int(sys.argv[3])
    DS = np.array(sys.argv[4].split(","), dtype=int)
    nbInterp = np.array(sys.argv[5].split(","), dtype=int)
    nbClus = np.array(sys.argv[6].split(","), dtype=int)
    buildData = bool(int(sys.argv[7]))
    seuil = int(sys.argv[8])
    folds = int(sys.argv[9])
    nbRuns = int(sys.argv[10])
except Exception as e:
    print("Using predefined parameters")
    folder = "Merovingien"
    features = [0]
    output = 1
    DS = [3]
    nbInterp = [1]
    nbClus = [5]
    buildData = True
    seuil = 0
    folds = 5
    nbRuns = 10
list_params = [(features, output, DS, nbInterp, nbClus, buildData, seuil, folds)]

for features, output, DS, nbInterp, nbClus, buildData, seuil, folds in list_params:
    tabDicResAvg = []
    for fold in range(folds):
        featToClus = []
        for iter, interp in enumerate(nbInterp):
            for i in range(interp):
                featToClus.append(iter)
        featToClus = np.array(featToClus, dtype=int)

        run = -1
        thetas, p, intToOut, intToFeat = read_params(folder, features, output, featToClus, nbClus, fold, run=run)
        print(intToFeat)
        print(intToOut)
        nbOut = p.shape[-1]

        #plotThetasGraph(thetas, p, intToOut, intToFeat)
        if nbInterp==[1]:
            pass
            plotGraph1D(thetas, p, intToOut, intToFeat)





