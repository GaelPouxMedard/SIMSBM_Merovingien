import numpy as np
import sys
import os
import sparse
import itertools
from sklearn import metrics
from scipy.stats import sem
import matplotlib.pyplot as plt
from copy import copy
import seaborn as sns

if "Plots" not in os.listdir("."): os.mkdir("./Plots")
if "Merovingien" not in os.listdir("./Plots"): os.mkdir("./Plots/Merovingien")

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

    codeSave=""
    for i in featToClus:
        codeSave += f"{features[i]}({nbClus[i]})-"
    codeSave += f"{output}"+"_fold-"+str(fold)+"of"+str(folds)

    thetas = []
    for i in range(len(features)):
        thetas.append(readMatrix(folderParams+txtFin + "T="+codeSave+"_%.0f_" % (run)+s+"theta_"+str(i)+"_Inter_theta.npy"))

    p = readMatrix(folderParams+txtFin + "T="+codeSave+"_%.0f_" % (run)+s+"Inter_p.npy")


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
            intToOut[int(ind)] = nom.replace("_", " ").capitalize()
    for i in range(len(features)):
        intToFeat.append({})
        with open(filenameData+"featToInt_"+str(features[i])+".txt", "r", encoding="utf-8") as f:
            for line in f:
                nom, ind = line.replace("\n", "").split("\t")
                intToFeat[-1][int(ind)] = nom.replace("_", " ").capitalize()

    return thetas, p, intToOut, intToFeat

def savefig(name, codeSave=""):
    folder = "Plots/Merovingien/"
    plt.savefig(folder+codeSave+name+".pdf", dpi=600)

def findClusters(mat):
    from sklearn.cluster import SpectralClustering
    cluster = SpectralClustering(n_clusters=mat.shape[-1])
    cluster.fit_predict(mat)
    return cluster.labels_

def writeClusters(thetas, codeSave):
    with open(f"Plots/Merovingien/{codeSave}composition_clusters.txt", "w+", encoding="utf-8") as f:
        for num_clus, k in enumerate(thetas[0].T):
            sorted_incides = list(reversed(np.argsort(k)))
            f.write(f"Cluster {num_clus}\n")
            for i in sorted_incides:
                if k[i]>0.05:
                    f.write(f"{np.round(k[i]*100)}% - {intToFeat[0][i]}\n")


def plotThetas(thetas, intToFeat, codeSave):
    scaleFeat = 4.
    scaleClus = 30.
    scaleTypes = 1.
    scaleAlpha = 1.

    plt.figure(figsize=(7, 9*len(thetas)))

    nbTypes = len(thetas)
    for num_theta in range(len(thetas)):
        # maxTheta = np.max(thetas[num_theta])
        # thetas[num_theta] = np.exp(-scaleAlpha * (maxTheta - thetas[num_theta]))

        shift = (num_theta - nbTypes / 2) * scaleTypes
        nbFeat = thetas[num_theta].shape[0] - 1
        nbClus = thetas[num_theta].shape[-1] - 1

        clusLab = findClusters(thetas[0])
        pos = 0
        for clus in range(len(clusLab)):
            for i in np.where(clusLab == clus)[0]:
                plt.text(-1, (pos - nbFeat / 2) * scaleFeat + shift, intToFeat[num_theta][i], ha="right", va="center", fontsize=11)
                for k in range(len(thetas[num_theta][i])):
                    plt.plot([-1, 0], [(pos - nbFeat / 2) * scaleFeat + shift, (k - nbClus / 2) * scaleClus + shift], "k-", linewidth=thetas[num_theta][i][k], alpha=thetas[num_theta][i][k])
                pos += 1

        plt.text(-1, (nbFeat*1.3 / 2) * scaleFeat + shift, "Inputs", ha="center", va="center", fontsize=12)
        plt.text(0, (nbClus*1.3 / 2)*scaleClus + shift, "Clusters", ha="center", va="center", fontsize=12)
        for k in range(thetas[num_theta].shape[1]):
            plt.plot(0, (k - nbClus / 2) * scaleClus + shift, "or", markersize=25)
            plt.text(0, (k - nbClus / 2) * scaleClus + shift, str(k), ha="center", va="center", fontsize=12)

    plt.axis("off")
    plt.tight_layout()

    savefig("Thetas", codeSave)

    plt.close()


def plotGraph1D(thetas, p, intToOut, intToFeat, codeSave):
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
            plt.text(-scalex, (pos - (nbFeat-1) / 2) * scaleFeat, intToFeat[0][i]+" ", ha="right", va="center", fontsize=5)
            for k in range(nbClus):
                plt.plot([-scalex, 0], [(pos - (nbFeat-1) / 2) * scaleFeat, (k - (nbClus-1) / 2) * scaleClus], "k-", linewidth=1., alpha=thetas[0][i][k])
            pos += 1

    for j in range(nbOut):
        plt.text(scalex, (j - (nbOut-1) / 2) * scaleOut, intToOut[j]+" ", ha="left", va="center", fontsize=11)
        for k in range(nbClus):
            plt.plot([0, scalex], [(k - (nbClus-1) / 2) * scaleClus, (j - (nbOut-1) / 2) * scaleOut], "k-", linewidth=1., alpha=p[k][j])
            plt.plot(0, (k - (nbClus - 1) / 2) * scaleClus, "or", markersize=15)


    plt.axis("off")
    plt.tight_layout()

    savefig("Graphe_final", codeSave)

    plt.close()


def plotGraph2D(thetas, p, intToOut, intToFeat, codeSave):
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

    savefig("Graphe_final", codeSave)

    plt.close()


def readRes():
    folder = "Results/Merovingien/"
    files = os.listdir(folder)

    metricsToExclude = ["CovErr", "CovErrNorm"]

    dicAvg, dicStd = {}, {}
    matAvg, matStd = [], []
    for file in files:
        if file[0]!="_": continue
        if "Avg_Results" not in file: continue
        
        params = file.split("_")
        fold, features, output, DS, nbInterp, nbClus = params[1:7]

        if features not in dicAvg: dicAvg[features]={}
        if output not in dicAvg: dicAvg[features][output]={}
        if DS not in dicAvg[features][output]: dicAvg[features][output][DS]={}
        if nbInterp not in dicAvg[features][output][DS]: dicAvg[features][output][DS][nbInterp]={}
        if nbClus not in dicAvg[features][output][DS][nbInterp]: dicAvg[features][output][DS][nbInterp][nbClus]={}
        if fold not in dicAvg[features][output][DS][nbInterp][nbClus]: dicAvg[features][output][DS][nbInterp][nbClus][fold]={}

        if features not in dicStd: dicStd[features]={}
        if output not in dicStd: dicStd[features][output]={}
        if DS not in dicStd[features][output]: dicStd[features][output][DS]={}
        if nbInterp not in dicStd[features][output][DS]: dicStd[features][output][DS][nbInterp]={}
        if nbClus not in dicStd[features][output][DS][nbInterp]: dicStd[features][output][DS][nbInterp][nbClus]={}
        if fold not in dicStd[features][output][DS][nbInterp][nbClus]: dicStd[features][output][DS][nbInterp][nbClus][fold]={}


        with open(folder + file, 'r') as f:
            labels = f.readline().replace("\n", "").split("\t")
            results = f.readline().replace("\n", "").split("\t")

            for lab, res in zip(labels[1:], results[1:]):
                if res == '' or lab in metricsToExclude: continue
                dicAvg[features][output][DS][nbInterp][nbClus][fold][lab] = float(res)
                matAvg.append([fold, features, output, DS, nbInterp, nbClus, lab, float(res)])

        with open(folder + file.replace("Avg", "Std"), 'r') as f:
            labels = f.readline().replace("\n", "").split("\t")
            results = f.readline().replace("\n", "").split("\t")

            for lab, res in zip(labels[1:], results[1:]):
                if res == '' or lab in metricsToExclude: continue
                dicStd[features][output][DS][nbInterp][nbClus][fold][lab] = float(res)
                matStd.append([fold, features, output, DS, nbInterp, nbClus, lab, float(res)])
                

    return np.array(matAvg, dtype=object), np.array(matStd, dtype=object)

    
def plotRes():
    matAvg, matStd = readRes()

    names = ["fold", "features", "output", "DS", "nbInterp", "nbClus"]
    for whatToPlot in [[4, 5], [0], [4], [5]]:
        dicToPlotx = {}
        dicToPlotAvg = {}
        dicToPlotStd = {}
        for val, err in zip(matAvg, matStd):
            valName = copy(val)
            for wt in range(len(whatToPlot)):
                if names[whatToPlot[wt]]=="fold": val[whatToPlot[wt]]=val[whatToPlot[wt]].split("of")[0]
                valName[whatToPlot[wt]] = "All"

            redKey = f"{valName[0]}_{valName[1]}_{valName[2]}_{valName[3]}_{valName[4]}_{valName[5]}_"

            if redKey not in dicToPlotAvg:
                dicToPlotx[redKey] = {}
                dicToPlotAvg[redKey] = {}
                dicToPlotStd[redKey] = {}
            if val[-2] not in dicToPlotx[redKey]:
                dicToPlotx[redKey][val[-2]] = []
                dicToPlotAvg[redKey][val[-2]] = []
                dicToPlotStd[redKey][val[-2]] = []

            coord = tuple([float(val[whatToPlot[wt]].replace("]", "").replace("[", "")) for wt in range(len(whatToPlot))])
            dicToPlotx[redKey][val[-2]].append(coord)
            dicToPlotAvg[redKey][val[-2]].append(val[-1])
            dicToPlotStd[redKey][val[-2]].append(err[-1])

        if len(whatToPlot)==1:
            for redKey in dicToPlotx:
                for label in dicToPlotAvg[redKey]:
                    dicToPlotx[redKey][label] = [v[0] for v in dicToPlotx[redKey][label]]
                    dicToPlotAvg[redKey][label] = np.array([x for _, x in sorted(zip(dicToPlotx[redKey][label], dicToPlotAvg[redKey][label]))])
                    dicToPlotStd[redKey][label] = np.array([x for _, x in sorted(zip(dicToPlotx[redKey][label], dicToPlotStd[redKey][label]))])
                    dicToPlotx[redKey][label] = np.array([x for _, x in sorted(zip(dicToPlotx[redKey][label], dicToPlotx[redKey][label]))])

                    plt.plot(dicToPlotx[redKey][label], dicToPlotAvg[redKey][label], label=label)
                    plt.fill_between(dicToPlotx[redKey][label], dicToPlotAvg[redKey][label]-dicToPlotStd[redKey][label], dicToPlotAvg[redKey][label]+dicToPlotStd[redKey][label], alpha=0.3)

                plt.title(str(list(zip(names, redKey.split("_")))))
                plt.xlabel(names[whatToPlot[0]])
                plt.ylabel("Metrics")
                plt.legend()
                plt.tight_layout()
                nameFig = ""
                if whatToPlot == [0]: nameFig = "Folds"
                if whatToPlot == [4]: nameFig = "Interp"
                if whatToPlot == [5]: nameFig = "Clusters"
                savefig(nameFig, codeSave=redKey)
                plt.close()

        elif len(whatToPlot)==2:
            for redKey in dicToPlotx:
                label = "Acc"
                if label not in dicToPlotAvg[redKey]: continue

                arrx, arry = set(), set()
                for x,y in dicToPlotx[redKey][label]:
                    arrx.add(x)
                    arry.add(y)

                valToIndx = {v: i for i, v in enumerate(sorted(arrx))}
                valToIndy = {v: i for i, v in enumerate(sorted(arry))}

                valToIndx_inv = {i: v for v, i in valToIndx.items()}
                valToIndy_inv = {i: v for v, i in valToIndy.items()}

                coords = []
                for x,y in dicToPlotx[redKey][label]:
                    coords.append((valToIndx[x], valToIndy[y]))

                matHeatAvg = sparse.COO(list(zip(*coords)), data=dicToPlotAvg[redKey][label]).todense()

                sns.heatmap(matHeatAvg, cmap="afmhot_r", linewidth=0.03, vmin=0, vmax=1)

                plt.xlabel(names[whatToPlot[1]])
                plt.ylabel(names[whatToPlot[0]])
                plt.gca().set_xticklabels([int(v) for i, v in sorted(valToIndy_inv.items())])
                plt.gca().set_yticklabels([int(v) for i, v in sorted(valToIndx_inv.items())])
                plt.title(str(list(zip(names, redKey.split("_")))))
                plt.gca().invert_yaxis()
                plt.tight_layout()
                nameFig = ""
                if whatToPlot == [4, 5]:
                    nameFig = "ClusvsInterp"
                savefig(nameFig, codeSave=redKey)
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

plotRes()

for features, output, DS, nbInterp, nbClus, buildData, seuil, folds in list_params:
    tabDicResAvg = []
    for fold in range(folds):
        featToClus = []
        for iter, interp in enumerate(nbInterp):
            for i in range(interp):
                featToClus.append(iter)
        featToClus = np.array(featToClus, dtype=int)

        codeSave = f"{fold}of{folds}_{features}_{output}_{DS}_{nbInterp}_{nbClus}_"

        run = -1
        thetas, p, intToOut, intToFeat = read_params(folder, features, output, featToClus, nbClus, fold, run=run)
        print(intToFeat)
        print(intToOut)
        nbOut = p.shape[-1]

        writeClusters(thetas, codeSave)
        plotThetas(thetas, intToFeat, codeSave)
        if nbInterp==[1]:
            plotGraph1D(thetas, p, intToOut, intToFeat, codeSave)
        if nbInterp==[2]:
            plotGraph2D(thetas, p, intToOut, intToFeat, codeSave)





