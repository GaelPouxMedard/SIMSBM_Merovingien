from SIMSBM import runForOneDS
from Evaluate import evaluate
import multiprocessing
import tqdm


def run_all_XP():
    from run_all import folder, features, DS, folds, nbRuns, list_output, list_nbInterp, list_nbClus, prec, maxCnt, lim, seuil, propTrainingSet, num_processes
    buildData = False
    list_params = []

    for nbInterp in list_nbInterp:
        for nbClus in list_nbClus:
            for output in list_output:
                list_params.append((features, output, DS, nbInterp, nbClus, buildData, seuil, folds))

    with multiprocessing.Pool(processes=num_processes) as p:
        with tqdm.tqdm(total=len(list_params)) as progress:
            args = [(folder, DS, features, output, nbInterp, nbClus, buildData, seuil, lim, propTrainingSet, folds, prec, nbRuns, maxCnt) for features, output, DS, nbInterp, nbClus, buildData, seuil, folds in list_params]
            for i, res in enumerate(p.imap(runForOneDS, args)):
                progress.update()


if __name__ == "__main__":
    run_all_XP()