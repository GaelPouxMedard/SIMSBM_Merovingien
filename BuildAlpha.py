import numpy as np
import itertools
import sparse

np.random.seed(111)


def getData(folder, featuresData, outputData, lim):
    folderName = "Data/" + folder

    outToInt = {}
    featToInt = [{} for _ in featuresData]

    if lim <= 0: lim = 1e20

    outcome = {}
    listOuts = set()
    ind=0
    lg = open(folderName + f"/feature_{outputData}.txt", "r", encoding="utf-8").read().count("\n")
    with open(folderName + f"/feature_{outputData}.txt", "r", encoding="utf-8") as f:
        for j, line in enumerate(f):
            if j%(lg//5)==0:
                print("Outcomes:", j*100/lg, "%")
            num, out = line.replace("\n", "").split("\t")
            num = int(num)
            out = out.split(" ")
            if out==[""]: continue
            outcome[num]=[]  # set()  # Choix de ne pas inclure les répétitions
            for o in out:
                listOuts.add(o)
                if o not in outToInt:
                    outToInt[o] = ind
                    ind+=1
                outcome[num].append(outToInt[o])
            outcome[num] = list(outcome[num])
            if j>lim: break

    features = []
    listFeatures = []
    for i in range(len(featuresData)):
        features.append({})
        listFeatures.append(set())
        ind=0
        lg=10
        lg = open(folderName + "/feature_%.0f.txt" %i, "r", encoding="utf-8").read().count("\n")
        with open(folderName + "/feature_%.0f.txt" % featuresData[i], "r", encoding="utf-8") as f:
            for j, line in enumerate(f):
                if j%(lg//5)==0:
                    print(f"Features {featuresData[i]}:", j*100/lg, "%")
                try:
                    num, feat = line.replace("\n", "").split("\t")
                except:
                    continue
                num = int(num)
                feat = feat.split(" ")
                if feat==[""]: continue
                features[i][num] = []

                if len(feat)>15: continue

                for f in feat:
                    listFeatures[i].add(f)
                    if f not in featToInt[i]:
                        featToInt[i][f]=ind
                        ind += 1
                    features[i][num].append(featToInt[i][f])
                features[i][num] = list(features[i][num])

                if j > lim: break

    return features, outcome, featToInt, outToInt

def splitDS(outcome, features, propTrainingSet, folds):
    # On prend que les entrées pour lesquelles on a eu moins une feature de chaque type
    IDs = set(outcome.keys())
    for i in range(len(features)):
        IDs = set(features[i].keys()).intersection(IDs)

    if folds != 1:
        IDs = np.array(list(IDs))
        allIDsTest = list(np.array_split(IDs, folds))

    else:
        allIDsTest = [np.random.choice(list(IDs), size=int((1.-propTrainingSet)*len(IDs)), replace=False)]

    allIDsTraining = []
    for i in range(len(allIDsTest)):
        IDsTraining = set(IDs)-set(allIDsTest[i])
        allIDsTraining.append(list(IDsTraining))
        allIDsTest[i] = list(allIDsTest[i])

    return allIDsTraining, allIDsTest

def getAlpha(features, outcome, nbInterp, IDsTraining, IDsTest, featToInt, outToInt):
    # ================================ ALPHA TRAINING
    dicAlpha = {}
    keysSeen = set()

    lgIDs = len(IDsTraining)
    for iter, num in enumerate(IDsTraining):
        if iter%(lgIDs//5)==0:
            print(f"Alpha training : {iter}/{lgIDs}")

        toProd = []
        for i in range(len(features)):
            for interp in range(nbInterp[i]):
                toProd.append(features[i][num])
        toProd.append(outcome[num])
        listKeys = list(itertools.product(*toProd))

        for k in listKeys:
            if k not in dicAlpha: dicAlpha[k] = 0
            keysSeen.add(k)
            dicAlpha[k] += 1

    print("Build Alpha_Tr")
    alphaTr = sparse.COO(list(zip(*dicAlpha.keys())), list(dicAlpha.values()))
    #print(alphaTr, "# triples:", alphaTr.sum())
    del dicAlpha

    # ================================ ALPHA TEST
    dicAlpha = {}

    lgIDs = len(IDsTest)
    for iter, num in enumerate(IDsTest):
        if iter%(lgIDs//5)==0:
            print(f"Alpha test : {iter}/{lgIDs}")


        toProd = []
        for i in range(len(features)):
            for interp in range(nbInterp[i]):
                toProd.append(features[i][num])
        toProd.append(outcome[num])
        listKeys = list(itertools.product(*toProd))

        for k in listKeys:
            if k in keysSeen or True:
                if k not in dicAlpha: dicAlpha[k] = 0
                dicAlpha[k] += 1

    print("Build Alpha_Te")
    alphaTe = sparse.COO(list(zip(*dicAlpha.keys())), list(dicAlpha.values()))
    #print(alphaTe, "# triples:", alphaTe.sum())

    shape = tuple([len(featToInt[i]) for i in range(len(featToInt)) for _ in range(nbInterp[i])]+[len(outToInt)])
    alphaTe.shape=shape
    alphaTr.shape=shape

    # Symmetry alphaTr (necessary for algorithmic simplification in maximizeTheta)
    prev = 0
    print("Enforcing symmetry training")
    for num, i in enumerate(nbInterp):
        permuts = list(itertools.permutations(list(range(prev, prev+int(i))), int(i)))
        alphaTr2 = alphaTr.copy()
        for per in permuts[1:]:
            arrTot = np.array(list(range(len(alphaTr.shape))))
            arrTot[prev:prev+i] = np.array(per)
            #print(arrTot)
            alphaTr2 = alphaTr2 + alphaTr.transpose(arrTot)
        alphaTr = alphaTr2 / len(permuts)  # somme permutations = 1 obs
        prev += i

    # Symmetry alphaTe (necessary for algorithmic simplification in maximizeTheta)
    print("Enforcing symmetry test")
    prev = 0
    for num, i in enumerate(nbInterp):
        permuts = list(itertools.permutations(list(range(prev, prev+int(i))), int(i)))
        alphaTe2 = alphaTe.copy()
        for per in permuts[1:]:
            arrTot = np.array(list(range(len(alphaTe.shape))))
            arrTot[prev:prev+i] = np.array(per)
            alphaTe2 += alphaTe.transpose(arrTot)
        alphaTe = alphaTe2 / len(permuts)
        prev += i

    return alphaTr, alphaTe, IDsTraining, IDsTest

def saveData(alphaTr, alphaTe, folder, nbInterp, featToInt, outToInt, IDsTraining, IDsTest, featuresData, outputData, fold, folds):
    codeSave = ""
    for i in range(len(featuresData)):
        for _ in range(nbInterp[i]):
            codeSave += str(featuresData[i])+"-"
    codeSave += f"{outputData}"
    filename = "Data/"+folder+"/"+codeSave+"_"+"fold-"+str(fold)+"of"+str(folds)+"_"
    print("Save AlphaTr")
    sparse.save_npz(filename+"AlphaTr.npz", alphaTr)
    print("Save AlphaTe")
    sparse.save_npz(filename+"AlphaTe.npz", alphaTe)
    print("Save keys")
    with open(filename+"outToInt.txt", "w+", encoding="utf-8") as f:
        for o in outToInt:
            f.write(str(o)+"\t"+str(outToInt[o])+"\n")
    for i in range(len(featuresData)):
        with open(filename+"featToInt_"+str(featuresData[i])+".txt", "w+", encoding="utf-8") as f:
            for feat in featToInt[i]:
                f.write(str(feat)+"\t"+str(featToInt[i][feat])+"\n")

    with open(filename+"IDTr.txt", "w+", encoding="utf-8") as f:
        f.write(str(list(IDsTraining)))
    with open(filename+"IDTe.txt", "w+", encoding="utf-8") as f:
        f.write(str(list(IDsTest)))

def reduceAlpha(seuil, alphaTr, alphaTe, nbInterp, featToInt, outToInt):

    alphaTr *= (alphaTr>seuil).astype(int)

    nnz = alphaTr.nonzero()
    shapeObj = []
    prev = 0
    for num, i in enumerate(nbInterp):
        obj=set()
        for j in nnz[prev:prev+i]:
            obj|=set(j)
            shapeObj.append(len(obj))
        prev += i
    shapeObj.append(len(set(nnz[-1])))
    shapeObj = tuple(shapeObj)
    print("End shape :", shapeObj)

    if shapeObj == alphaTr.shape or sum(shapeObj)>0.9*sum(alphaTr.shape):
        return alphaTr, alphaTe, featToInt, outToInt

    nnzAtr = alphaTr.nonzero()
    featToIntNew, outToIntNew = [{} for _ in range(sum(nbInterp))], {}
    newCoords, newVals = [], []

    print("Compress AlphaTr")
    coordsTemp = []
    num = np.zeros((sum(nbInterp)+1))
    data = alphaTr.data
    lg = len(data)
    for j, c in enumerate(zip(*nnzAtr)):
        if j%(lg//5)==0:
            print("Compress ATr", j*100./lg, "%")
        vn = 0
        vf = -1
        tmp = []
        bk=0
        for i in range(len(c)):
            if i == vn:
                vf += 1
                try:
                    vn += nbInterp[vf]
                except:
                    pass
                    vn += 1

            if i == len(c)-1:
                #if np.count_nonzero(nnzAtr[i] == c[i]) < seuil[vf]:
                if np.count_nonzero(nnzAtr[i] == c[i]) == 0:
                    bk=1
                    break
                if c[i] not in outToIntNew:
                    outToIntNew[c[i]] = num[vf]
                    num[vf] += 1
                tmp.append(outToIntNew[c[i]])

            else:
                if np.count_nonzero(nnzAtr[i] == c[i]) == 0:
                    bk=1
                    break
                if c[i] not in featToIntNew[vf]:
                    featToIntNew[vf][c[i]] = num[vf]
                    num[vf] += 1
                tmp.append(featToIntNew[vf][c[i]])

        if bk==0:
            newVals.append(data[j])
            tmp = np.array(tmp, dtype=int)
            coordsTemp.append(tmp)


    newVals = np.array(newVals)
    newCoords = list(zip(*coordsTemp))
    shape = []
    for i in range(len(newCoords)):
        shape.append(max(newCoords[i])+1)
    shape = np.array(shape)
    prev = 0
    for i in nbInterp:
        shape[prev:prev+i] = max(shape[prev:prev+i])
    shape = tuple(shape)

    newAlphaTr = sparse.COO(newCoords, newVals, shape=shape)


    print("Compress AlphaTe")
    nnzAte = alphaTe.nonzero()
    newCoords, newVals = [], []
    coordsTemp = []
    data = alphaTe.data
    lg = len(data)
    for j, c in enumerate(zip(*nnzAte)):
        if j%1000==0:
            print("Compress ATe", j*100./lg, "%")
        vn = 0
        vf = -1
        tmp = []
        bk=0
        for i in range(len(c)):
            if i == vn:
                vf += 1
                try:
                    vn += nbInterp[vf]
                except:
                    pass
                    vn += 1

            if i == len(c)-1:
                if c[i] not in outToIntNew:
                    bk=1
                    break
                tmp.append(outToIntNew[c[i]])

            else:
                if c[i] not in featToIntNew[vf]:
                    bk=1
                    break
                tmp.append(featToIntNew[vf][c[i]])

        if bk==0:
            newVals.append(data[j])
            tmp = np.array(tmp, dtype=int)
            coordsTemp.append(tmp)


    newVals = np.array(newVals)
    newCoords = list(zip(*coordsTemp))

    newAlphaTe = sparse.COO(newCoords, newVals, shape=shape)

    newNewFeatToInt, newNewOutToInt = [{} for _ in range(sum(nbInterp))], {}
    for i in range(len(featToInt)):
        for f in featToInt[i]:
            try:
                if f not in newNewFeatToInt[i]: newNewFeatToInt[i][f]=int(featToIntNew[i][featToInt[i][f]])
            except:
                pass
    for f in outToInt:
        try:
            if f not in newNewOutToInt: newNewOutToInt[f]=int(outToIntNew[outToInt[f]])
        except:
            pass

    print("New alpha:", newAlphaTr)
    return newAlphaTr, newAlphaTe, newNewFeatToInt, newNewOutToInt

def run(folder, nbInterp, featuresData, outputData, propTrainingSet, folds, lim=0, seuil=0):
    features, outcome, featToInt, outToInt = getData(folder, featuresData, outputData, lim)

    allIDsTraining, allIDsTest = splitDS(outcome, features, propTrainingSet, folds)

    for fold in range(folds):
        print("FOLD", fold)
        alphaTr, alphaTe, IDsTraining, IDsTest = getAlpha(features, outcome, nbInterp, allIDsTraining[fold], allIDsTest[fold], featToInt, outToInt)
        print("Original shape :", alphaTr.shape)
        alphaTr, alphaTe, currentFeatToInt, currentOutToInt = reduceAlpha(seuil, alphaTr, alphaTe, nbInterp, featToInt, outToInt)
        print("Final shape :", alphaTr.shape)
        saveData(alphaTr, alphaTe, folder, nbInterp, currentFeatToInt, currentOutToInt, IDsTraining, IDsTest, featuresData, outputData, fold, folds)


'''
folder = "PubMedReduit"
nbInterp = [2]
run(folder, nbInterp, [0], 0.9, lim=10000, seuil=0)
'''











