from SIMSBM import runForOneDS
import multiprocessing
import tqdm

if __name__ == "__main__":
    folder = "Merovingien"
    prec = 1e-5  # Stopping threshold : when relative variation of the likelihood over 10 steps is < to prec
    maxCnt = 30  # Number of consecutive times the relative variation is lesser than prec for the algorithm to stop
    saveToFile = True
    propTrainingSet = 0.7
    lim = -1
    seuil = 0

    features = [0]
    output = 1
    DS = [3]
    nbInterp = [2]
    nbClus = [5]
    buildData = True
    folds = 5
    nbRuns = 100
    list_params = []

    for output in [1, 2]:
        for nbInterp in [1, 2, 3]:
            for nbClus in [3, 4, 5, 6, 7, 8, 9, 10]:
                list_params.append((features, output, DS, nbInterp, nbClus, buildData, seuil, folds))

    with multiprocessing.Pool(processes=15) as p:
        with tqdm.tqdm(total=len(list_params)) as progress:
            args = [(folder, DS, features, output, nbInterp, nbClus, buildData, seuil, lim, propTrainingSet, folds, prec, nbRuns, maxCnt) for features, output, DS, nbInterp, nbClus, buildData, seuil, folds in list_params]
            for i, res in enumerate(p.imap(runForOneDS, args)):
                progress.update()

