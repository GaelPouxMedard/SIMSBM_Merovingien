import numpy as np
import sys
import os
import sparse
import itertools
from sklearn import metrics
from scipy.stats import sem

# Generic function to read matrices of dim=2 or 3
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

def read_params(folder, features, output, featToClus, nbClus, fold, folds, run=-1):
    s=""
    folderParams = "Output/" + folder + "/"
    if run == -1:
        folderParams += "Final/"

    codeT=""
    for i in featToClus:
        codeT += f"{features[i]}({nbClus[i]})-"
    codeT += f"{output}"+"_fold-"+str(fold)+"of"+str(folds)

    thetas = []
    for i in range(len(features)):
        thetas.append(readMatrix(folderParams + "T="+codeT+"_%.0f_" % (run)+s+"theta_"+str(i)+"_Inter_theta.npy"))

    p = readMatrix(folderParams + "T="+codeT+"_%.0f_" % (run)+s+"Inter_p.npy")
    return thetas, p

def getDataTe(folder, featuresData, outputData, DS, fold, folds):
    folderName = "Data/" + folder + "/"

    outToInt = {}
    featToInt = [{} for _ in range(len(featuresData))]

    codeSave = ""
    for i in range(len(featuresData)):
        for _ in range(DS[i]):
            codeSave += str(featuresData[i])+"-"
    codeSave += f"{outputData}"
    filename = "Data/"+folder+"/"+codeSave+"_"+"fold-"+str(fold)+"of"+str(folds)+"_"

    with open(filename +"IDTe.txt") as f:
        IDsTe = f.read().replace("[", "").replace("]", "").split(", ")
        IDsTe = np.array(IDsTe, dtype=int)

    with open(filename +"outToInt.txt", encoding="utf-8") as f:
        for line in f:
            lab, num = line.replace("\n", "").split("\t")
            num=int(num)
            outToInt[lab]=num

    for i in range(len(featuresData)):
        with open(filename + "featToInt_%.0f.txt" % featuresData[i], "r", encoding="utf-8") as f:
            for line in f:
                lab, num = line.replace("\n", "").split("\t")
                num = int(num)
                featToInt[i][lab] = num

    outcome = {}
    listOuts = set()
    lg=len(IDsTe)
    with open(folderName + "feature_%.0f.txt" %outputData, "r", encoding="utf-8") as f:
        j=0
        for line in f:
            num, out = line.replace("\n", "").split("\t")
            num = int(num)
            if num not in IDsTe: continue
            #if j%(lg//10)==0: print("Outcomes:", j*100/lg, "%")
            j+=1
            if j==len(IDsTe): break
            out = out.split(" ")
            outcome[num]=[]
            for o in out:
                listOuts.add(o)
                if o not in outToInt:
                    continue
                outcome[num].append(outToInt[o])

    features = []
    listFeatures = []
    for i in range(len(featuresData)):
        features.append({})
        listFeatures.append(set())
        lg=len(IDsTe)
        with open(folderName + "/feature_%.0f.txt" % featuresData[i], "r", encoding="utf-8") as f:
            j=0
            for line in f:
                num, feat = line.replace("\n", "").split("\t")
                num = int(num)
                if num not in IDsTe: continue
                #if j%(lg//10)==0: print(f"Features {featuresData[i]}:", j*100/lg, "%")
                j+=1
                if j==len(IDsTe): break
                feat = feat.split(" ")
                features[i][num] = []
                for f in feat:
                    listFeatures[i].add(f)
                    if f not in featToInt[i]:
                        continue
                    features[i][num].append(featToInt[i][f])

    return features, outcome, featToInt, outToInt, IDsTe

def dicsToList(tabK, *a):
    nba = len(a)
    lists = [[] for _ in range(nba)]
    for k in tabK:
        for i in range(nba):
            lists[i].append(a[i][k])

    for i in range(nba):
        lists[i]=np.array(lists[i])

    return lists

def getIndsMod(DS, nbInterp):
    indsMod = []
    ind = 0
    for i in range(len(DS)):
        for j in range(nbInterp[i]):
            indsMod.append(ind+j)
        ind += DS[i]

    return np.array(indsMod)

def getElemProb(c, thetas, p, featToClus):
    nbFeat = len(featToClus)

    probs = p
    for i in range(nbFeat):
        tet = thetas[featToClus[i]][c[i]]  # k
        probs = np.tensordot(tet, probs, axes=1)
    v = probs

    return v

def buildArraysProbs(folder, featuresCons, outputCons, DS, thetas, p, featToClus, nbInterp, fold, folds):
    features, outcome, featToInt, outToInt, IDsTe = getDataTe(folder, featuresCons, outputCons, DS, fold, folds)

    inds = getIndsMod(DS, nbInterp)

    nbOut = p.shape[-1]

    lg = len(IDsTe)
    nb=0
    index_obs = 0
    tabTrue, tabProbs = [], []
    for j, id in enumerate(IDsTe):
        #if j % (lg//10) == 0: print("Build list probs", j * 100. / lg, f"% ({j}/{lg})")

        if id not in outcome: continue

        toProd = []
        for i in range(len(features)):
            for _ in range(nbInterp[i]):
                toProd.append(list(features[i][id]))
        listKeys = list(itertools.product(*toProd))

        a = np.zeros((nbOut))
        for o in outcome[id]:
            a[o] = 1

        tempProb = []
        for k in listKeys:
            karray = np.array(k)

            # [inds] important car réduit le DS au modèle considéré
            prob = getElemProb(karray[inds], thetas, p, featToClus)
            tempProb.append(prob)

            index_obs += 1
            nb+=1

        prob = np.mean(tempProb, axis=0)

        tabTrue.append(a)
        tabProbs.append(prob)

    print("NOMBRE D'EVALUATIONS :", len(tabTrue))

    return tabTrue, tabProbs



def scores(listTrue, listProbs, label, tabMetricsAll, nbOut, fold):
    labels_considered = np.where(np.sum(listTrue, axis=0)!=0)[0]
    nanmask = np.isnan(listProbs)
    if np.any(nanmask):
        print(f"CAREFUL !!!!! {np.sum(nanmask.astype(int))} NANs IN PROBA !!!")
        pause()
    if label not in tabMetricsAll: tabMetricsAll[label]={}
    if fold not in tabMetricsAll[label]: tabMetricsAll[label][fold]={}

    tabMetricsAll[label][fold]["F1"], tabMetricsAll[label][fold]["Acc"] = 0, 0
    for thres in np.linspace(0, 1, 1001):
        F1 = metrics.f1_score(listTrue, (listProbs>thres).astype(int), labels=labels_considered, average="micro")

        listTrue_flat = np.array(listTrue).flatten()
        listProbs_flat = (np.array(listProbs).flatten()>thres).astype(int)
        acc = np.sum((listTrue_flat==listProbs_flat).astype(int))/len(listTrue_flat)
        if F1 > tabMetricsAll[label][fold]["F1"]:
            tabMetricsAll[label][fold]["F1"] = F1
        if acc > tabMetricsAll[label][fold]["Acc"]:
            tabMetricsAll[label][fold]["Acc"] = acc

    k = 1  # Si k=1, sklearn considère les 0 et 1 comme des classes, mais de fait on prédit jamais 0 dans un P@k...
    topk = np.argpartition(listProbs, -k, axis=1)[:, -k:]
    probsTopK = np.zeros(np.array(listProbs).shape)
    for row in range(len(probsTopK)):
        probsTopK[row][topk[row]] = 1.
    trueTopK = listTrue

    tabMetricsAll[label][fold][f"P@{k}"] = metrics.precision_score(trueTopK, probsTopK, labels=labels_considered, average="micro")

    listTrue_flat = np.argpartition(listTrue, -1, axis=1)[:, -1]
    listProbs_flat = np.argpartition(listProbs, -1, axis=1)[:, -1]
    tabMetricsAll[label][fold][f"Acc@1"] = np.sum((listTrue_flat==listProbs_flat).astype(int))/len(listTrue_flat)

    tabMetricsAll[label][fold]["AUCROC"] = metrics.roc_auc_score(listTrue, listProbs, labels=labels_considered, average="micro")
    tabMetricsAll[label][fold]["AUCPR"] = metrics.average_precision_score(listTrue, listProbs, average="micro")
    tabMetricsAll[label][fold]["RankAvgPrec"] = metrics.label_ranking_average_precision_score(listTrue, listProbs)
    c=metrics.coverage_error(listTrue, listProbs)
    tabMetricsAll[label][fold]["CovErr"] = c-1
    tabMetricsAll[label][fold]["CovErrNorm"] = (c-1)/nbOut
    tabMetricsAll[label][fold]["Freq"] = np.max(np.mean(listTrue, axis=0))

    print("\t".join(map(str, tabMetricsAll[label][fold].keys())).expandtabs(20))
    print("\t".join(map(str, np.round(list(tabMetricsAll[label][fold].values()), 4))).expandtabs(20))

    return tabMetricsAll

def saveResults(tabMetricsAll, folder, features, output, DS, nbInterp, nbClus, fold, folds, printRes=True, final=False, averaged=True):

    if "Results" not in os.listdir("./"): os.mkdir("./Results")
    if folder not in os.listdir("./Results"): os.mkdir("./Results/"+folder)

    dicResAvg = {}
    if averaged:
        fAvg = open("Results/" + folder + f"/_{features}_{output}_{DS}_{nbInterp}_{nbClus}_Avg_Results.txt", "w+")
        fStd = open("Results/" + folder + f"/_{features}_{output}_{DS}_{nbInterp}_{nbClus}_Std_Results.txt", "w+")
        fSem = open("Results/" + folder + f"/_{features}_{output}_{DS}_{nbInterp}_{nbClus}_Sem_Results.txt", "w+")
        fstPass = True
        for label in tabMetricsAll:
            dicResAllFolds = {}
            dicResAvg = {}
            dicResStd = {}
            dicResSem = {}
            for fold in tabMetricsAll[label]:
                for metric in tabMetricsAll[label][fold]:
                    if metric not in dicResAllFolds: dicResAllFolds[metric] = []

                    dicResAllFolds[metric].append(tabMetricsAll[label][fold][metric])

            if fstPass:
                fAvg.write("\t")
                fStd.write("\t")
                fSem.write("\t")
                for metric in dicResAllFolds:
                    fAvg.write(metric+"\t")
                    fStd.write(metric+"\t")
                    fSem.write(metric+"\t")
                fAvg.write("\n")
                fStd.write("\n")
                fSem.write("\n")
                fstPass = False

            fAvg.write(label+"\t")
            fStd.write(label+"\t")
            fSem.write(label+"\t")
            for metric in dicResAllFolds:
                dicResAvg[metric] = np.mean(dicResAllFolds[metric])
                dicResStd[metric] = np.std(dicResAllFolds[metric])
                dicResSem[metric] = sem(dicResAllFolds[metric])

                fAvg.write("%.4f\t" % (dicResAvg[metric]))
                fStd.write("%.4f\t" % (dicResStd[metric]))
                fSem.write("%.4f\t" % (dicResSem[metric]))
            fAvg.write("\n")
            fStd.write("\n")
            fSem.write("\n")


    txtFin = ""

    if not os.path.exists("Results/" + folder + "/"):
        os.makedirs("Results/" + folder + "/")
    with open("Results/" + folder + f"/{txtFin}{features}_{output}_{DS}_{nbInterp}_{nbClus}_Results.txt", "w+") as f:
        firstPassage = True
        for label in sorted(list(tabMetricsAll.keys()), key=lambda x: "".join(list(reversed(x)))):
            if firstPassage:
                f.write("\tfold\t")
                for fold in tabMetricsAll[label]:
                    for metric in tabMetricsAll[label][fold]:
                        f.write(metric+"\t")
                    f.write("\n")
                    firstPassage = False
                    break

            for fold in tabMetricsAll[label]:
                f.write(label+"\t"+str(fold)+"\t")
                for metric in tabMetricsAll[label][fold]:
                    f.write("%.4f\t" % (tabMetricsAll[label][fold][metric]))
                f.write("\n")
                if printRes:
                    print(label + " " + str(tabMetricsAll[label][fold]))

    return dicResAvg

def evaluate(args):
    folder, folder_out, features, output, DS, nbInterp, nbClus, seuil, folds, nbRuns = args

    tabDicResAvg = []
    tabMetricsAll = {}
    run = -1
    for fold in range(folds):
        featToClus = []
        for iter, interp in enumerate(nbInterp):
            for i in range(interp):
                featToClus.append(iter)
        featToClus = np.array(featToClus, dtype=int)

        print(f"Scores SIMSBM_{nbInterp} - Fold {fold} - Run {run}")
        try:
            thetas, p = read_params(folder, features, output, featToClus, nbClus, fold, folds, run=run)
        except Exception as e:
            print("Run not found", e)
            continue
        nbOut = p.shape[-1]

        listTrue, listProbs = buildArraysProbs(folder, features, output, DS, thetas, p, featToClus, nbInterp, fold, folds)
        tabMetricsAll = scores(listTrue, listProbs, f"SIMSBM_{nbInterp}", tabMetricsAll, nbOut, fold)

        dicResAvg = saveResults(tabMetricsAll, folder_out, features, output, DS, nbInterp, nbClus, fold, folds, printRes=False, final=False, averaged=True)
        tabDicResAvg.append(dicResAvg)

    print()
    arrRes = []
    for i, d in enumerate(tabDicResAvg):
        if i==0: print("\t".join(map(str, d.keys())).expandtabs(20))
        print("\t".join(map(str, np.round(list(d.values()), 4))).expandtabs(20))
        if len(d.values())!=0:
            arrRes.append(list(d.values()))
    print()
    if len(arrRes)!=0:
        print("\t".join(map(str, np.round(np.mean(arrRes, axis=0), 4))).expandtabs(20))

def evaluate_all():
    from run_all import folder, folder_out, features, DS, folds, nbRuns, list_output, list_nbInterp, list_nbClus, prec, maxCnt, lim, seuil, propTrainingSet, num_processes
    list_params = []

    for nbInterp in list_nbInterp:
        for nbClus in list_nbClus:
            for output in list_output:
                list_params.append((features, output, DS, nbInterp, nbClus, seuil, folds))

    for features, output, DS, nbInterp, nbClus, seuil, folds in list_params:
        args = (folder, folder_out, features, output, DS, nbInterp, nbClus, seuil, folds, nbRuns)
        evaluate(args)

if __name__=="__main__":
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
            elif nbInterp==[2]:
                nbClus = [5]
            elif nbInterp==[3]:
                nbClus = [3]
        else:
            if nbInterp==[1]:
                nbClus = [5]
            elif nbInterp==[2]:
                nbClus = [7]
            elif nbInterp==[2]:
                nbClus = [5]
        seuil = 0
        folds = 5
        nbRuns = 100
        list_params = [(features, output, DS, nbInterp, nbClus, seuil, folds)]

        for features, output, DS, nbInterp, nbClus, seuil, folds in list_params:
            args = (folder, features, output, DS, nbInterp, nbClus, seuil, folds, nbRuns)
            evaluate(args)

