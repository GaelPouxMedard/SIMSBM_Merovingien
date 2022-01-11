import os
os.environ['OPENBLAS_NUM_THREADS'] = '5'
os.environ['MKL_NUM_THREADS'] = '5'
import numpy as np
import random
import sparse
import itertools
import sys

'''
import pprofile
profiler = pprofile.Profile()
with profiler:
    RhovsT(g, 0.1, 0.28999, 1, 1000)
profiler.print_stats()
profiler.dump_stats("Benchmark.txt")
pause()
'''

'''
from memory_profiler import profile
import gc
fp = open("memory_profiler_Norm.log", "a")
@profile(stream=fp, precision=5)
'''

seed = 111
np.random.seed(seed)
random.seed(seed)


#// region Manipulates the data files

# Generic function to save matrices of dim=2 or 3
def writeMatrix(arr, filename):
    try:
        sparse.save_npz(filename.replace(".txt", ""), arr)
    except:
        try:
            np.save(filename, arr)
        except:
            with open(filename, 'a') as outfile:
                outfile.truncate(0)
                outfile.write('# Array shape: {0}\n'.format(arr.shape))
                for slice_2d in arr:
                    np.savetxt(outfile, slice_2d)
                    outfile.write("# New slice\n")

    # np.savetxt(filename, arr)

# Generic function to read matrices of dim=2 or 3
def readMatrix(filename):
    try:
        return sparse.load_npz(filename.replace(".txt", ".npz"))
    except:
        with open(filename, 'r') as outfile:
            dims = outfile.readline().replace("# Array shape: (", "").replace(")", "").replace("\n", "").split(", ")
            for i in range(len(dims)):
                dims[i] = int(dims[i])

        new_data = np.loadtxt(filename).reshape(dims)
        return new_data

    # return sparse.csr_matrix(new_data)

# Saves the model's parameters theta, p
def writeToFile_params(folder, thetas, p, maxL, features, output, featToClus, popFeat, nbClus, fold, folds, run=-1):
    while True:
        try:
            s=""
            folderParams = "Output/" + folder + "/"
            curfol = "./"
            for fol in folderParams.split("/"):
                if fol not in os.listdir(curfol) and fol!="":
                    os.mkdir(curfol+fol)
                curfol += fol+"/"

            codeT=""
            for i in featToClus:
                codeT += f"{features[i]}({nbClus[i]})-"
            codeT += f"{output}"+"_fold-"+str(fold)+"of"+str(folds)

            for i in range(len(thetas)):
                writeMatrix(thetas[i], folderParams + "/T="+codeT+"_%.0f_" % (run)+s+"theta_"+str(i)+"_Inter_theta")

            writeMatrix(p, folderParams + "/T="+codeT+"_%.0f_" % (run)+s+"Inter_p")

            f = open(folderParams + "/T="+codeT+"_%.0f_" % (run)+s+"Inter_L.txt", "w")
            f.write(str(maxL) + "\n")
            f.close()

            f = open(folderParams + "/T="+codeT+"_%.0f_" % (run)+s+"Inter_FeatToClus.txt", "w")
            for i in range(len(featToClus)):
                f.write(str(i)+"\t" + str(featToClus[i]) + "\t" + str(popFeat[i]) + "\n")
            f.close()

            break

        except Exception as e:
            print("Retrying to write file -", e)

#// endregion


#// region Toolbox
# Makes alpha fit the wanted number of interactions for each nature
def reduceAlphaInter(alpha, DS, nbInterp):
    toRem, ind = [], 0
    for i in range(len(DS)):
        if DS[i] != nbInterp[i]:
            for t in range(ind, ind+DS[i]-nbInterp[i]):
                toRem.append(t)
        ind += DS[i]
    if len(toRem)!=0:
        alpha = alpha.sum(toRem)
    return alpha

# Recursive function to build the dict whose keys are nonzero values of alpha
def getDicNonZeros(alpha):
    dicnnz = {}
    coords = alpha.nonzero()

    def buildDicCoords(f, dic):
        if len(f)==1:
            dic[f[0]]=1
        else:
            if f[0] not in dic: dic[f[0]]={}
            dic[f[0]] = buildDicCoords(f[1:], dic[f[0]])

        return dic

    for f in zip(*coords):
        dicnnz = buildDicCoords(f, dicnnz)
    return dicnnz

def normalized(a, axis=-1):
    l2 = np.sum(a, axis=axis)
    l2[l2==0]=1
    return a / np.expand_dims(l2, axis)

#// endregion


#// region Fit tools


# Recursive function to build an array P_{f,o} for given coordinates stored in dic (the denominator of omega, see main paper) as a sparse array
def getAllProbs(dic, vals, prob, thetas, featToClus, feat, nbFeat):
    if feat == nbFeat:
        for k in dic:
            vals.append(prob[k])
    else:
        for k in dic:
            tet = thetas[featToClus[feat]][k]
            probk = np.moveaxis(prob, 1, -1).dot(tet)  # Move the axis in 1 to the end bc with recurrency the indices "slide" back to 1
            vals = getAllProbs(dic[k], vals, probk, thetas, featToClus, feat+1, nbFeat)

    return vals


def likelihood(alpha, Pfo):
    return np.sum(alpha*np.log(Pfo+1e-20))

# Normalization term used in maximizationTheta, can be computed once for all the dataset
def getCm(alpha, nbNatures, featToClus, popFeat):
    Cm = []
    for nature in range(nbNatures):
        arrFeat = []
        for feat in range(len(featToClus)):
            if nature==featToClus[feat]: arrFeat.append(feat)

        Cm.append(np.zeros((popFeat[arrFeat[-1]])))
        dataa = alpha.data
        for i, ialpha in enumerate(list(zip(*alpha.nonzero()))):
            indAlpha = tuple(list(np.array(ialpha)[arrFeat]))
            for m in set(indAlpha):
                im = np.where(np.array(indAlpha)==m)[0]
                cm = list(np.array(indAlpha)[im]).count(m)
                Cm[nature][m] += cm*dataa[i]
    return Cm

# EM step for p
def maximization_p(alpha, featToClus, popFeat, nbClus, theta, pPrev, Pfo):
    nbFeat = len(featToClus)

    alphadivided = alpha / (Pfo + 1e-20)  # features, o
    for t in range(nbFeat):
        sizeAfterOperation = np.prod(alphadivided.shape)*theta[featToClus[nbFeat - t - 1]].shape[1]*8/alphadivided.shape[-2]
        if sizeAfterOperation < 2e9 and sizeAfterOperation > 0:  # As soon as we can use non-sparse we do it (2Gb)
            #print("P dense")
            tet = theta[featToClus[nbFeat - t - 1]].T  # K F1
            alphadivided = np.dot(tet, alphadivided)
        else:
            #print("P sparse")
            tet = sparse.COO(theta[featToClus[nbFeat - t - 1]].T)  # K F1
            alphadivided = sparse.COO(np.dot(tet, alphadivided))

    # print(alphadivided.shape)  # clusters, o
    if np.prod(alphadivided.shape)*8 < 2e9 and type(alphadivided) is not type(np.array([])):
        alphadivided = alphadivided.todense()

    grandDiv = np.sum(pPrev*alphadivided, -1)
    grandDiv = np.expand_dims(grandDiv, -1)
    p = pPrev*alphadivided / (grandDiv+1e-20)

    '''  Explicit computation for 2 feature natures with 2 interactions each
    omegatop = np.moveaxis(pPrev, -1, 0)
    print(omegatop.shape)
    omegatop = omegatop[:, :, :, :, :, None]*theta[0].T[None, :, None, None, None, :]
    print(omegatop.shape)
    omegatop = omegatop[:, :, :, :, :, :, None]*theta[0].T[None, None, :, None, None, None, :]
    print(omegatop.shape)
    omegatop = omegatop[:, :, :, :, :, :, :, None]*theta[1].T[None, None, None, :, None, None, None, :]
    print(omegatop.shape)
    omegatop = omegatop[:, :, :, :, :, :, :, :, None]*theta[1].T[None, None, None, None, :, None, None, None, :]
    print(omegatop.shape)
    omegatop = np.moveaxis(omegatop, 0, -1)
    print(omegatop.shape)

    omega = omegatop/omegatop.sum(axis=(0, 1, 2, 3))[None, None, None, None, :, :, :, :, :]
    p2 = omega[:, :, :, :, :, :, :, :, :]*alpha[None, None, None, None, :, :, :, :, :]
    p2 = p2.sum(axis=(4, 5, 6, 7))
    p2 = p2/p2.sum(axis=-1)[:, :, :, :, None]

    '''

    return p

# EM step for theta
def maximization_Theta(alpha, featToClus, nbClus, thetaPrev, p, Cm, Pfo):
    nbFeat = len(featToClus)
    nbNatures = len(nbClus)
    thetas = []

    for nature in range(nbNatures):
        theta_base = thetaPrev[nature]

        alphadivided = alpha / (Pfo + 1e-20)  # f1 f2 g r

        arrFeat = []
        for feat in range(len(featToClus)):
            if nature==featToClus[feat]: arrFeat.append(feat)
        nbInter = len(arrFeat)

        if len(arrFeat) == 0:
            continue

        # Sum over all other natures' permutations (they are identical for this nature)
        omega = np.moveaxis(p, arrFeat, range(len(arrFeat)))
        for t in reversed(range(nbFeat)):
            if t not in arrFeat:
                omega = thetaPrev[featToClus[t]].dot(omega)

        for i in range(1, nbInter):  #Keep theta_mn out from omega
            omega = np.dot(theta_base, omega)

        omega = omega * nbInter

        idxalpha = tuple(arrFeat[1:]+[i for i in range(nbFeat) if i not in arrFeat]+[-1])
        idxomega = tuple([i for i in range(len(arrFeat)-1)]+[i for i in range(len(arrFeat)-1, nbFeat-1)]+[-1])
        omegalpha = np.tensordot(omega, alphadivided, axes=(idxomega, idxalpha)).T

        thetaNatureNew = omegalpha * theta_base / (Cm[nature][:, None]+1e-20)
        thetas.append(thetaNatureNew)

        '''  Explicit summation
        for m in range(I):
            for n in range(K):
                # Explicit
                for ia, ialpha in enumerate(list(zip(*alpha.nonzero()))):
                    indAlpha = tuple(ialpha)[:-1]
                    o = ialpha[-1]
                    im = np.where(np.array(indAlpha)==m)[0]
        
                    cm = list(np.array(indAlpha)[im]).count(m)
        
                    permutOmega = list(itertools.product(list(range(K)), repeat=nbInter))
                    for indOmega in permutOmega:
        
                        if m not in np.array(indAlpha)[im]: continue
                        if n not in np.array(indOmega)[im]: continue
        
                        indOmega = tuple(indOmega)
                        cn = list(np.array(indOmega)[im]).count(n)
        
                        tupInd = list(indAlpha)+list(indOmega)+[o]
                        tupInd = tuple(tupInd)
        
                        theta[m, n] += omega[tupInd]*dataa[ia]*cn
        
            theta[m] = theta[m]/Cm[m]
        '''


    return thetas

# Random initialisation of p, theta
def initVars(featToClus, popFeat, nbOutputs, nbNatures, nbClus, nbInterp):
    nbFeat = len(featToClus)
    thetas = []
    for i in range(nbNatures):
        pop = 0
        for j in range(len(featToClus)):
            if featToClus[j]==i:
                pop = popFeat[j]
                break

        t = np.random.random((pop, nbClus[i]))
        t = t / np.sum(t, axis=1)[:, None]
        thetas.append(t)

    shape = [nbClus[featToClus[i]] for i in range(nbFeat)]+[nbOutputs]
    p = np.random.random(tuple(shape))  # clusters, o

    # Important to make symmetric initialization for each cluster nature, otherwise assumptions made in the algorithm do not hold (maximizationp).
    prev = 0
    for num, i in enumerate(nbInterp):
        permuts = list(itertools.permutations(list(range(prev, prev+int(i))), int(i)))
        p2 = p.copy()
        for per in permuts[1:]:
            arrTot = np.array(list(range(len(p.shape))))
            arrTot[prev:prev+i] = np.array(per)
            p2 = p2 + p.transpose(arrTot)
        p = p2 / len(permuts)  # somme permutations = 1 obs
        prev += i

    p = normalized(p, axis=-1)

    return thetas, p

# Main loop of the EM algorithm, for 1 run
def EMLoop(alpha, featToClus, popFeat, nbOutputs, nbNatures, nbClus, maxCnt, prec, folder, run, Cm, dicnnz, nbInterp, features, output, fold, folds):
    nbFeat = len(featToClus)
    thetas, p = initVars(featToClus, popFeat, nbOutputs, nbNatures, nbClus, nbInterp)
    maskedProbs = getAllProbs(dicnnz, [], np.moveaxis(p, -1, 0), thetas, featToClus, 0, nbFeat)
    Pfo = sparse.COO(alpha.nonzero(), np.array(maskedProbs), shape=alpha.shape)
    maxThetas, maxP = initVars(featToClus, popFeat, nbOutputs, nbNatures, nbClus, nbInterp)
    prevL, L, maxL = -1e20, -1e20, -1e20
    cnt = 0
    num_iterations = 0
    prec_iteration = 0
    while num_iterations < 1e10:  # 1000000 iterations top ; prevents infinite loops but never reached in practice
        #print(i)
        if num_iterations%10==0:  # Computes the likelihood and possibly save the results every 10 iterations
            L = likelihood(alpha, Pfo)
            print(f"Run {run} - Iter {num_iterations} - Feat {features} - Interps {nbInterp} - L={L}")

            if ((L - prevL) / abs(L)) < prec:
                cnt += num_iterations-prec_iteration
                if cnt > maxCnt:
                        break
            else:
                cnt = 0

            prec_iteration=num_iterations

            if L > maxL:
                maxThetas, maxP = thetas, p
                maxL = L
                writeToFile_params(folder, maxThetas, maxP, maxL, features, output, featToClus, popFeat, nbClus, fold, folds, run)
                print("Saved")
            prevL = L

        maskedProbs = getAllProbs(dicnnz, [], np.moveaxis(p, -1, 0), thetas, featToClus, 0, nbFeat)
        Pfo = sparse.COO(alpha.nonzero(), np.array(maskedProbs), shape=alpha.shape) # Sparse matrix of every probability for observed entries (normalization term of omega)

        pNew = maximization_p(alpha, featToClus, popFeat, nbClus, thetas, p, Pfo)
        thetasNew = maximization_Theta(alpha, featToClus, nbClus, thetas, p, Cm, Pfo)

        p = pNew
        thetas = thetasNew

        num_iterations += 1

    return maxThetas, maxP, maxL


#// endregion


def runFit(folder, alpha, nbClus, nbInterp, DS, prec, nbRuns, maxCnt, features, output, fold):
    nbOutputs = alpha.shape[-1]
    popFeat = [l for l in alpha.shape[:-1]]
    nbNatures = len(nbClus)
    featToClus = []
    nbClus = np.array(nbClus)
    for iter, interp in enumerate(nbInterp):
        for i in range(interp):
            featToClus.append(iter)
    featToClus = np.array(featToClus, dtype=int)

    print("Natures (which input features have been chosen)")
    print(features)
    print("Pop feats (for each layer, how many different nodes)")
    print(popFeat)
    print("Feat to clus (for each layer, which nature it belongs to)")
    print(featToClus)
    print("Nb clus (for each nature, how many clusters to use)")
    print(nbClus)
    print("Dataset (for each nature, how many interactions are considered)")
    print(DS)
    print("Nb interactions (for each nature, how many interactions will the model consider)")
    print(nbInterp)
    print("Final shape of observation tensor alpha :", alpha.shape)
    print("Number of training nplets:", int(alpha.sum()))

    dicnnz = getDicNonZeros(alpha)

    Cm = getCm(alpha, nbNatures, featToClus, popFeat)

    maxL = -1e100
    for i in range(nbRuns):
        print("RUN", i)
        theta, p, L = EMLoop(alpha, featToClus, popFeat, nbOutputs, nbNatures, nbClus, maxCnt, prec, folder, i, Cm, dicnnz, nbInterp, features, output, fold, folds)
        if L > maxL:
            maxL = L
            writeToFile_params(folder + "/Final/", theta, p, L, features, output, featToClus, popFeat, nbClus, fold, folds, -1)
            print("######saved####### MAX L =", L)
        print("=============================== END EM ==========================")

def runForOneDS(args):
    folder, DS, features, output, nbInterp, nbClus, buildData, seuil, lim, propTrainingSet, folds, prec, nbRuns, maxCnt = args
    if buildData:
        print("Build alpha training and alpha test (matrix of observations)")
        import BuildAlpha
        BuildAlpha.run(folder, DS, features, output, propTrainingSet, folds, lim, seuil=seuil)

    for fold in range(folds):
        print("Get alpha training")
        codeSave = ""
        for i in range(len(features)):
            for j in range(DS[i]):
                codeSave += str(features[i]) + "-"
        codeSave += f"{output}"
        fname = "Data/"+folder+"/"+codeSave+"_fold-"+str(fold)+"of"+str(folds)
        alpha = readMatrix(fname+"_AlphaTr.npz")

        alpha = reduceAlphaInter(alpha, DS, nbInterp)

        runFit(folder, alpha, nbClus, nbInterp, DS, prec, nbRuns, maxCnt, features, output, fold)



if __name__ == "__main__":
    # Just for them to be defined
    features = []  # Which features to consider (see key)
    DS = []  # Which dataset use (some are already built for interactions)
    nbInterp = []  # How many interactions consider for each dataset (reduces DS to this number by summing)
    nbClus = []
    buildData = bool

    seuil=0  # If retreatEverything=True : choose the threshold for the number of apparitions of an nplet.
    # If an nplet appears stricly less than "seuil" times, it's not included in the dataset

    prec = 1e-5  # Stopping threshold : when relative variation of the likelihood over 10 steps is < to prec
    maxCnt = 30  # Number of consecutive times the relative variation is lesser than prec for the algorithm to stop
    saveToFile = True
    propTrainingSet = 0.7
    lim = -1

    try:
        folder=sys.argv[1]
        features = np.array(sys.argv[2].split(","), dtype=int)
        output = int(sys.argv[3])
        DS=np.array(sys.argv[4].split(","), dtype=int)
        nbInterp=np.array(sys.argv[5].split(","), dtype=int)
        nbClus=np.array(sys.argv[6].split(","), dtype=int)
        buildData = bool(int(sys.argv[7]))
        seuil = int(sys.argv[8])
        folds = int(sys.argv[9])
        nbRuns = int(sys.argv[10])
    except Exception as e:
        print("Using predefined parameters")
        folder = "Merovingien"
        features = [0]
        output = 1
        DS=[3]
        nbInterp=[2]
        nbClus=[5]
        buildData = True
        seuil = 0
        folds = 5
        nbRuns = 100
    list_params = [(features, output, DS, nbInterp, nbClus, buildData, seuil, folds)]


    for features, output, DS, nbInterp, nbClus, buildData, seuil, folds in list_params:
        args = folder, DS, features, output, nbInterp, nbClus, buildData, seuil, lim, propTrainingSet, folds, prec, nbRuns, maxCnt
        runForOneDS(args)

    sys.exit(0)



