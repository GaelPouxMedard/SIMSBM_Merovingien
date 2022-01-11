import numpy as np
import sys
import os
import sparse
import itertools
from sklearn import metrics
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

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

def findClusters(mat):
    from sklearn.cluster import SpectralClustering
    cluster = SpectralClustering(n_clusters=mat.shape[-1])
    cluster.fit_predict(mat)
    return cluster.labels_

def plotThetasGraph(thetas, intToFeat):
    scaleFeat = 4.
    scaleClus = 30.
    scaleTypes = 1.
    scaleAlpha = 1.

    plt.figure(figsize=(7, 9*len(thetas)))

    nbTypes = len(thetas)
    for type in range(len(thetas)):
        # maxTheta = np.max(thetas[type])
        # thetas[type] = np.exp(-scaleAlpha * (maxTheta - thetas[type]))

        shift = (type - nbTypes / 2) * scaleTypes
        nbFeat = thetas[type].shape[0] - 1
        nbClus = thetas[type].shape[-1] - 1

        clusLab = findClusters(thetas[0])
        pos = 0
        print(clusLab)
        for clus in range(len(clusLab)):
            for i in np.where(clusLab == clus)[0]:
                plt.text(-1, (pos - nbFeat / 2) * scaleFeat + shift, intToFeat[type][i].replace("_", " ").capitalize(), ha="right", va="center", fontsize=11)
                for k in range(len(thetas[type][i])):
                    plt.plot([-1, 0], [(pos - nbFeat / 2) * scaleFeat + shift, (k - nbClus / 2) * scaleClus + shift], "k-", linewidth=thetas[type][i][k], alpha=thetas[type][i][k])
                pos += 1

        plt.text(-1, (nbFeat*1.3 / 2) * scaleFeat + shift, "Inputs", ha="center", va="center", fontsize=12)
        plt.text(0, (nbClus*1.3 / 2)*scaleClus + shift, "Clusters", ha="center", va="center", fontsize=12)
        for k in range(thetas[type].shape[1]):
            plt.plot(0, (k - nbClus / 2) * scaleClus + shift, "or", markersize=25)
            plt.text(0, (k - nbClus / 2) * scaleClus + shift, str(k), ha="center", va="center", fontsize=12)

    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.close()


def plotGraph1D(thetas, p, intToOut, intToFeat):
    scaleFeat = 1.
    scaleClus = 10.
    scaleOut = 10.
    scalex = 10
    scaleAlpha = 4.

    nbFeat = thetas[0].shape[0]
    nbClus = thetas[0].shape[-1]
    nbOut = p.shape[-1]

    # thetas[0] = np.exp(-scaleAlpha*(maxTheta-thetas[0]))
    # p = np.exp(-scaleAlpha*(maxp-p))

    clusLab = findClusters(thetas[0])
    pos = 0
    for clus in range(len(clusLab)):
        for i in np.where(clusLab==clus)[0]:
            plt.text(-scalex, (pos - (nbFeat-1) / 2) * scaleFeat, intToFeat[0][i].replace("_", " ").capitalize()+" ", ha="right", va="center", fontsize=5)
            for k in range(nbClus):
                plt.plot([-scalex, 0], [(pos - (nbFeat-1) / 2) * scaleFeat, (k - (nbClus-1) / 2) * scaleClus], "k-", linewidth=1., alpha=thetas[0][i][k])
            pos += 1

    for j in range(nbOut):
        plt.text(scalex, (j - (nbOut-1) / 2) * scaleOut, intToOut[j].replace("_", " ").capitalize()+" ", ha="left", va="center", fontsize=11)
        for k in range(nbClus):
            plt.plot([0, scalex], [(k - (nbClus-1) / 2) * scaleClus, (j - (nbOut-1) / 2) * scaleOut], "k-", linewidth=1., alpha=p[k][j])
            plt.plot(0, (k - (nbClus - 1) / 2) * scaleClus, "or", markersize=15)


    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.close()


def plotGraph2D(thetas, p, intToOut, intToFeat):
    size = 10
    nbOut = p.shape[-1]
    nbFeat = len(thetas[0])
    nbClus = len(p)
    shift = 2.5
    scaleFeatx = scaleFeaty = 0.5

    plt.figure(figsize=(10, 3))

    for o in range(nbOut):
        plt.subplot(1, nbOut, o+1)
        sns.heatmap(p[:, :, o], linewidths=0.03, cmap="afmhot_r", square=True, cbar_kws={"shrink": 0.55, "label": "Membership"}, vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.title(intToOut[o].capitalize())
        plt.xlabel("Clusters")

    plt.tight_layout()
    plt.show()
    
    
    

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
    DS = [3]
    nbInterp = [2]
    output = 1
    if output==1:
        if nbInterp==[1]:
            nbClus = [3]
        if nbInterp==[2]:
            nbClus = [5]
    else:
        if nbInterp==[1]:
            nbClus = [5]
        if nbInterp==[2]:
            nbClus = [7]
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

        #plotThetasGraph(thetas, intToFeat)
        if nbInterp==[1]:
            plotGraph1D(thetas, p, intToOut, intToFeat)
        if nbInterp==[2]:
            plotGraph2D(thetas, p, intToOut, intToFeat)





