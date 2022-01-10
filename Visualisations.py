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
    return thetas, p


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

        thetas, p = read_params(folder, features, output, featToClus, nbClus, fold, run=run)
        nbOut = p.shape[-1]





