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
import pandas as pd

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

def read_params(folder, features, output, featToClus, nbClus, DS, fold, folds, run=-1):
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

def savefig(name, folder_out="", codeSave="", subfolder=""):
    folder = "Plots/"+folder_out+"/"
    if "Plots" not in os.listdir("./"): os.mkdir("./Plots")
    if folder_out not in os.listdir("./Plots/"): os.mkdir("./Plots/"+folder_out+"/")
    if subfolder !="":
        curfol = folder
        for fol in subfolder.split("/"):
            if fol not in os.listdir(curfol) and fol!="":
                os.mkdir(curfol+fol)
            curfol += fol+"/"

    plt.savefig(folder+subfolder+codeSave+name+".pdf", dpi=600)

def findClusters(mat):
    from sklearn.cluster import SpectralClustering
    cluster = SpectralClustering(n_clusters=mat.shape[-1])
    cluster.fit_predict(mat)
    return cluster.labels_

def writeClusters(thetas, codeSave, intToFeat):
    if "Thetas" not in os.listdir("Plots/Merovingien"): os.mkdir("Plots/Merovingien/Thetas/")
    with open(f"Plots/Merovingien/Thetas/{codeSave}composition_clusters.txt", "w+", encoding="utf-8") as f:
        for num_clus, k in enumerate(thetas[0].T):
            sorted_incides = list(reversed(np.argsort(k)))
            f.write(f"Cluster {num_clus}\n")
            for i in sorted_incides:
                if k[i]>0.05:
                    f.write(f"{np.round(k[i]*100)}% - {intToFeat[0][i]}\n")

def plotThetas(thetas, intToFeat, codeSave, folder_out):
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


    savefig("Thetas", folder_out, codeSave, subfolder="Thetas/")

    plt.close()

def plotGraph1D(thetas, p, intToOut, intToFeat, codeSave, folder_out):
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

    savefig("Graphe_final", folder_out, codeSave, subfolder="Inter=1/")

    plt.close()

def plotGraph2D(thetas, p, intToOut, intToFeat, codeSave, folder_out):
    size = 10
    nbOut = p.shape[-1]
    nbFeat = len(thetas[0])
    nbClus = len(p)
    shift = 2.5
    scaleFeatx = scaleFeaty = 0.5

    plt.figure(figsize=(10, 3))

    for o in range(nbOut):
        plt.subplot(1, nbOut, o+1)
        sns.heatmap(np.round(p[:, :, o]*100), linewidths=0.03, cmap="afmhot_r", square=True, cbar_kws={"shrink": 0.55, "label": "Membership"}, fmt='g', vmin=0, vmax=100, annot=True, cbar=False)
        plt.gca().invert_yaxis()
        plt.title(intToOut[o].capitalize())
        plt.xlabel("Clusters")

    plt.tight_layout()

    savefig("Graphe_final", folder_out, codeSave, subfolder="Inter=2/")

    plt.close()

def readRes(folder_out):
    folder = "Results/"+folder_out+"/"
    files = os.listdir(folder)

    metricsToExclude = ["CovErr", "CovErrNorm", "Acc", "P@1", "RankAvgPrec"]

    dicAvg, dicStd, dicSem = {}, {}, {}
    matAvg, matStd, matSem = [], [], []
    for file in files:
        if file[0]!="_": continue
        if "Avg_Results" not in file: continue
        
        params = file.split("_")
        features, output, DS, nbInterp, nbClus = params[1:6]

        if features not in dicAvg: dicAvg[features]={}
        if output not in dicAvg: dicAvg[features][output]={}
        if DS not in dicAvg[features][output]: dicAvg[features][output][DS]={}
        if nbInterp not in dicAvg[features][output][DS]: dicAvg[features][output][DS][nbInterp]={}
        if nbClus not in dicAvg[features][output][DS][nbInterp]: dicAvg[features][output][DS][nbInterp][nbClus]={}

        if features not in dicStd: dicStd[features]={}
        if output not in dicStd: dicStd[features][output]={}
        if DS not in dicStd[features][output]: dicStd[features][output][DS]={}
        if nbInterp not in dicStd[features][output][DS]: dicStd[features][output][DS][nbInterp]={}
        if nbClus not in dicStd[features][output][DS][nbInterp]: dicStd[features][output][DS][nbInterp][nbClus]={}
        
        if features not in dicSem: dicSem[features]={}
        if output not in dicSem: dicSem[features][output]={}
        if DS not in dicSem[features][output]: dicSem[features][output][DS]={}
        if nbInterp not in dicSem[features][output][DS]: dicSem[features][output][DS][nbInterp]={}
        if nbClus not in dicSem[features][output][DS][nbInterp]: dicSem[features][output][DS][nbInterp][nbClus]={}


        with open(folder + file, 'r') as f:
            labels = f.readline().replace("\n", "").split("\t")
            results = f.readline().replace("\n", "").split("\t")

            for lab, res in zip(labels[1:], results[1:]):
                if res == '' or lab in metricsToExclude: continue
                dicAvg[features][output][DS][nbInterp][nbClus][lab] = float(res)
                matAvg.append([features, output, DS, nbInterp, nbClus, lab, float(res)])

        with open(folder + file.replace("Avg", "Std"), 'r') as f:
            labels = f.readline().replace("\n", "").split("\t")
            results = f.readline().replace("\n", "").split("\t")

            for lab, res in zip(labels[1:], results[1:]):
                if res == '' or lab in metricsToExclude: continue
                dicStd[features][output][DS][nbInterp][nbClus][lab] = float(res)
                matStd.append([features, output, DS, nbInterp, nbClus, lab, float(res)])

        with open(folder + file.replace("Avg", "Sem"), 'r') as f:  # Bc déjà changé avant
            labels = f.readline().replace("\n", "").split("\t")
            results = f.readline().replace("\n", "").split("\t")

            for lab, res in zip(labels[1:], results[1:]):
                if res == '' or lab in metricsToExclude: continue
                dicSem[features][output][DS][nbInterp][nbClus][lab] = float(res)
                matSem.append([features, output, DS, nbInterp, nbClus, lab, float(res)])

    return np.array(matAvg, dtype=object), np.array(matStd, dtype=object), np.array(matSem, dtype=object)

def plotRes(folder_out):
    matAvg, matStd, matSem = readRes(folder_out)

    names = ["features", "output", "DS", "nbInterp", "nbClus"]
    for whatToPlot in [[3, 4], [3], [4]]:
        dicToPlotx = {}
        dicToPlotAvg = {}
        dicToPlotStd = {}
        dicToPlotSem = {}

        for val_base, std, sem in zip(matAvg, matStd, matSem):
            valName = val_base.copy()
            val = val_base.copy()
            for wt in range(len(whatToPlot)):
                valName[whatToPlot[wt]] = "All"

            redKey = f"{valName[0]}_{valName[1]}_{valName[2]}_{valName[3]}_{valName[4]}_"

            if redKey not in dicToPlotAvg:
                dicToPlotx[redKey] = {}
                dicToPlotAvg[redKey] = {}
                dicToPlotStd[redKey] = {}
                dicToPlotSem[redKey] = {}
            if val[-2] not in dicToPlotx[redKey]:  # -2 = label
                dicToPlotx[redKey][val[-2]] = []
                dicToPlotAvg[redKey][val[-2]] = []
                dicToPlotStd[redKey][val[-2]] = []
                dicToPlotSem[redKey][val[-2]] = []

            coord = tuple([float(val[whatToPlot[wt]].replace("]", "").replace("[", "")) for wt in range(len(whatToPlot))])
            dicToPlotx[redKey][val[-2]].append(coord)
            dicToPlotAvg[redKey][val[-2]].append(val[-1])
            dicToPlotStd[redKey][val[-2]].append(std[-1])
            dicToPlotSem[redKey][val[-2]].append(sem[-1])

        if len(whatToPlot)==1:
            for redKey in dicToPlotx:
                for label in dicToPlotAvg[redKey]:
                    dicToPlotx[redKey][label] = [v[0] for v in dicToPlotx[redKey][label]]
                    dicToPlotAvg[redKey][label] = np.array([x for _, x in sorted(zip(dicToPlotx[redKey][label], dicToPlotAvg[redKey][label]))])
                    dicToPlotStd[redKey][label] = np.array([x for _, x in sorted(zip(dicToPlotx[redKey][label], dicToPlotStd[redKey][label]))])
                    dicToPlotx[redKey][label] = np.array([x for _, x in sorted(zip(dicToPlotx[redKey][label], dicToPlotx[redKey][label]))])
            for err in ["sem", "std", ""]:
                for redKey in dicToPlotx:
                    for label in dicToPlotAvg[redKey]:
                        plt.plot(dicToPlotx[redKey][label], dicToPlotAvg[redKey][label], label=label)
                        if err =="std":
                            plt.fill_between(dicToPlotx[redKey][label], dicToPlotAvg[redKey][label]-dicToPlotStd[redKey][label], dicToPlotAvg[redKey][label]+dicToPlotStd[redKey][label], alpha=0.3)
                        elif err == "sem":
                            plt.fill_between(dicToPlotx[redKey][label], dicToPlotAvg[redKey][label]-dicToPlotSem[redKey][label], dicToPlotAvg[redKey][label]+dicToPlotSem[redKey][label], alpha=0.3)

                    plt.title(str(list(zip(names, redKey.split("_")))))
                    plt.gca().set_title(str(list(zip(names, redKey.split("_")))), loc='center', wrap=True)
                    plt.ylim([0.5, 1])
                    plt.xlabel(names[whatToPlot[0]])
                    plt.ylabel("Metrics")
                    plt.legend()
                    plt.tight_layout()
                    nameFig, subf = "", ""
                    if whatToPlot == [3]:
                        nameFig = "Interp"
                        subf = "Interps/"
                    if whatToPlot == [4]:
                        nameFig = "Clusters"
                        subf = "Clusters/"
                    nameFig += err
    
                    savefig(nameFig, folder_out, codeSave=redKey, subfolder="Metrics/"+subf)
                    plt.close()

        elif len(whatToPlot)==2:
            for redKey in dicToPlotx:
                label = "Acc@1"
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

                sns.heatmap(np.round(matHeatAvg*100), cmap="afmhot_r", linewidth=0.03, vmin=0, vmax=100, square=True, annot=True, cbar=False, fmt='g')

                plt.xlabel(names[whatToPlot[1]])
                plt.ylabel(names[whatToPlot[0]])
                plt.gca().set_xticklabels([int(v) for i, v in sorted(valToIndy_inv.items())])
                plt.gca().set_yticklabels([int(v) for i, v in sorted(valToIndx_inv.items())])
                plt.title(str(list(zip(names, redKey.split("_")))))
                plt.gca().set_title(str(list(zip(names, redKey.split("_")))), loc='center', wrap=True)
                plt.gca().invert_yaxis()
                plt.tight_layout()
                nameFig = ""
                if whatToPlot == [4, 5]:
                    nameFig = "ClusvsInterp"
                savefig(nameFig, folder_out, codeSave=redKey, subfolder="Metrics/Heatmaps/")
                plt.close()

def plotInput1D(thetas, p, intToOut, intToFeat, codeSave, folder_out):
    def get_label_rotation(angle, offset):
        # Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle + offset)
        if angle <= np.pi:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"
        return rotation, alignment

    def add_labels(angles, values, labels, offset, ax):

        # This is the space between the end of the bar and the label
        padding = 4

        # Iterate over angles, values, and labels, to add all of them.
        for angle, value, label, in zip(angles, values, labels):
            angle = angle

            # Obtain text rotation and alignment
            rotation, alignment = get_label_rotation(angle, offset)

            # And finally add the text
            ax.text(
                x=angle,
                y=value + padding,
                s=label,
                ha=alignment,
                va="center",
                rotation=rotation,
                rotation_mode="anchor"
            )

    names, values, group = [], [], []
    for i in range(len(thetas[0])):
        prob = thetas[0][i].dot(p)
        for o in range(len(prob)):
            if prob[o]>0.05:
                names.append(intToFeat[0][i])
                values.append(prob[o]*100)
                group.append(intToOut[o])

    df = pd.DataFrame({
        "name": names,
        "value": values,
        "group": group
    })
    df_sorted = (
        df
            .groupby(["group"])
            .apply(lambda x: x.sort_values(["value"], ascending = False))
            .reset_index(drop=True)
    )

    VALUES = df_sorted["value"].values
    LABELS = df_sorted["name"].values
    GROUP = df_sorted["group"].values

    PAD = 3
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))

    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)

    offset = 0
    IDXS = []
    GROUP_LAB, GROUPS_SIZE = np.unique(GROUP, return_counts=True)
    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    OFFSET = np.pi / 2

    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})

    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-100, 100)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    COLORS = []
    for i, size in enumerate(GROUPS_SIZE):
        for _ in range(size):
            c = f"C{i}"
            if GROUP_LAB[i]=="Masculin": c="blue"
            if GROUP_LAB[i]=="Féminin": c="pink"
            COLORS.append(c)

    # Add bars to represent ...
    ax.bar(
        ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
        edgecolor="white", linewidth=2
    )

    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

    offset = 0
    for group, size in zip(GROUP_LAB, GROUPS_SIZE):
        # Add line below bars
        x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
        ax.plot(x1, [-5] * 50, color="#333333")

        # Add text to indicate group
        ax.text(
            np.mean(x1), -20, group, color="#333333", fontsize=14,
            fontweight="bold", ha="center", va="center", rotation=np.mean(x1)*180/np.pi
        )

        # Add reference lines at 20, 40, 60, and 80
        x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
        ax.plot(x2, [20] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [40] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [60] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [80] * 50, color="#bebebe", lw=0.8)

        offset += size + PAD

    savefig("CircBarPlot_input_to_output", folder_out, codeSave, subfolder="Inter=1/InputToOutput/")
    plt.close()

def plotInput2D(thetas, p, intToOut, intToFeat, codeSave, folder_out):
    import scipy.cluster.hierarchy as sch
    matHeatAvg = thetas[0].dot(thetas[0].dot(p))
    xlabels = ylabels = np.array([intToFeat[0][i] for i in range(len(thetas[0]))])

    plt.figure(figsize=(5*p.shape[-1], 5))

    Y = sch.linkage(np.max(matHeatAvg, axis=-1), method='centroid')
    Z = sch.dendrogram(Y, orientation='right', no_plot=True)
    index = Z['leaves']

    for o in range(p.shape[-1]):
        xlabels_new = xlabels[index]
        ylabels_new = ylabels[index]

        matHeatAvg[:, :, o] = matHeatAvg[index,:, o]
        matHeatAvg[:, :, o] = matHeatAvg[:,index, o]

        plt.subplot(1, p.shape[-1], o+1)
        sns.heatmap(np.round(matHeatAvg[:, :, o]*100), cmap="PuOr", linewidth=0.03, vmin=-35, vmax=135, square=True, annot=False, cbar=True, fmt='g')

        plt.xticks(0.5+np.array(list(range(len(xlabels)))), xlabels_new, fontsize=3, rotation=90)
        plt.yticks(0.5+np.array(list(range(len(ylabels)))), ylabels_new, fontsize=3)
        plt.gca().invert_yaxis()
        plt.title(intToOut[o])

    plt.tight_layout()
    savefig("Heatmap_input_to_output", folder_out, codeSave=codeSave, subfolder="Inter=2/InputToOutput/")
    plt.close()

def writeInput3D(thetas, p, intToOut, intToFeat, codeSave, folder_out):
    pio = thetas[0].dot(thetas[0].dot(thetas[0].dot(p)))
    pio = np.round(pio*100)
    txt = ""
    for o in range(pio.shape[-1]):
        txt += intToOut[o].upper()+"\n"
        for i in [i_tmp for _, i_tmp in reversed(sorted(zip(pio[:, :, :, o].sum(axis=(1,2)), list(range(len(pio))))))]:
            if np.mean(pio[i, :, :, o])<5: continue
            txt += intToFeat[0][i].upper()+f"({np.round(np.mean(pio[i, :, :, o]))}%)"+"\n"
            for j in [i_tmp for _, i_tmp in reversed(sorted(zip(pio[i, :, :, o].sum(axis=(1)), list(range(len(pio))))))]:
                if np.mean(pio[i, j, :, o])<5: continue
                txt += "-- "+intToFeat[0][j].upper()+f"({np.round(np.mean(pio[i, j, :, o]))}%)"+"\n"
                for k in [i_tmp for _, i_tmp in reversed(sorted(zip(pio[i, j, :, o], list(range(len(pio))))))]:
                    if pio[i, j, k, o]<5: continue
                    txt += "---- "+intToFeat[0][k].upper()+f"({pio[i, j, k, o]}%)"+"\n"
        txt += "\n\n"

    if "Inter=3" not in os.listdir("Plots/Merovingien"): os.mkdir("Plots/Merovingien/Inter=3/")
    with open("Plots/"+folder_out+"/Inter=3/"+codeSave+"readmap.txt", "w+", encoding="utf-8") as f:
        f.write(txt)

def visualize_all():
    from run_all import folder, folder_out, features, DS, folds, nbRuns, list_output, list_nbInterp, list_nbClus, prec, maxCnt, lim, seuil, propTrainingSet, num_processes
    list_params = []

    for nbInterp in list_nbInterp:
        for nbClus in list_nbClus:
            for output in list_output:
                list_params.append((features, output, DS, nbInterp, nbClus, seuil, folds))


    plotRes(folder_out)
    for features, output, DS, nbInterp, nbClus, seuil, folds in list_params:
        for fold in range(folds):
            featToClus = []
            for iter, interp in enumerate(nbInterp):
                for i in range(interp):
                    featToClus.append(iter)
            featToClus = np.array(featToClus, dtype=int)

            codeSave = f"{features}_{output}_{DS}_{nbInterp}_{nbClus}_{fold}_"
            print(codeSave)

            run = -1  # Only final run
            thetas, p, intToOut, intToFeat = read_params(folder, features, output, featToClus, nbClus, DS, fold, folds, run=run)
            # Valeur hors limite erreur informatique last digits
            p[p>1]=1.;p[p<0]=0.
            for i in range(len(thetas)):
                thetas[i][thetas[i]>1]=1.
                thetas[i][thetas[i]<0]=0.
            print(intToFeat)
            print(intToOut)

            if nbInterp==[1]:
                plotInput1D(thetas, p, intToOut, intToFeat, codeSave, folder_out)
                plotGraph1D(thetas, p, intToOut, intToFeat, codeSave, folder_out)
            if nbInterp==[2]:
                plotInput2D(thetas, p, intToOut, intToFeat, codeSave, folder_out)
                plotGraph2D(thetas, p, intToOut, intToFeat, codeSave, folder_out)
            if nbInterp==[3]:
                writeInput3D(thetas, p, intToOut, intToFeat, codeSave, folder_out)

            writeClusters(thetas, codeSave, intToFeat)
            plotThetas(thetas, intToFeat, codeSave, folder_out)
            # except Exception as e:
            #     print("Run manquant -", e)


if __name__ == "__main__":
    try:
        folder = sys.argv[1]
        features = np.array(sys.argv[2].split(","), dtype=int)
        output = int(sys.argv[3])
        DS = np.array(sys.argv[4].split(","), dtype=int)
        nbInterp = np.array(sys.argv[5].split(","), dtype=int)
        nbClus = np.array(sys.argv[6].split(","), dtype=int)
        seuil = int(sys.argv[7])
        folds = int(sys.argv[8])
        nbRuns = int(sys.argv[9])
        folder_out = folder
    except Exception as e:
        print("Using predefined parameters")
        folder = "Merovingien"
        folder_out = folder
        features = [0]
        DS = [3]
        nbInterp = [1]
        output = 2
        if output==1:
            if nbInterp==[1]:
                nbClus = [3]
            if nbInterp==[2]:
                nbClus = [5]
            if nbInterp==[3]:
                nbClus = [7]
        else:
            if nbInterp==[1]:
                nbClus = [5]
            if nbInterp==[2]:
                nbClus = [7]
            if nbInterp==[3]:
                nbClus = [9]
        seuil = 0
        folds = 5
        nbRuns = 10
    list_params = [(features, output, DS, nbInterp, nbClus, seuil, folds)]

    for output in [1, 2]:
        for nbInterp in [[1], [2], [3]]:
            for nbClus in [[3], [4], [5], [6], [7], [8], [9], [10]]:
                pass
                list_params.append((features, output, DS, nbInterp, nbClus, seuil, folds))


    plotRes(folder_out)
    for features, output, DS, nbInterp, nbClus, seuil, folds in list_params:
        for fold in range(folds):
            featToClus = []
            for iter, interp in enumerate(nbInterp):
                for i in range(interp):
                    featToClus.append(iter)
            featToClus = np.array(featToClus, dtype=int)

            codeSave = f"{fold}of{folds}_{features}_{output}_{DS}_{nbInterp}_{nbClus}_"

            try:
                run = -1
                thetas, p, intToOut, intToFeat = read_params(folder, features, output, featToClus, nbClus, DS, fold, folds, run=run)
                # Valeur hors limite erreur informatique last digits
                p[p>1]=1.;p[p<0]=0.
                for i in range(len(thetas)):
                    thetas[i][thetas[i]>1]=1.
                    thetas[i][thetas[i]<0]=0.
                print(intToFeat)
                print(intToOut)
                nbOut = p.shape[-1]

                if nbInterp==[1]:
                    plotInput1D(thetas, p, intToOut, intToFeat, codeSave, folder_out)
                    plotGraph1D(thetas, p, intToOut, intToFeat, codeSave, folder_out)
                if nbInterp==[2]:
                    plotInput2D(thetas, p, intToOut, intToFeat, codeSave, folder_out)
                    plotGraph2D(thetas, p, intToOut, intToFeat, codeSave, folder_out)
                if nbInterp==[3]:
                    writeInput3D(thetas, p, intToOut, intToFeat, codeSave, folder_out)

                writeClusters(thetas, codeSave, intToFeat)
                plotThetas(thetas, intToFeat, codeSave, folder_out)
            except:
                print("Run manquant")





