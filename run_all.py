import os
from treat_data import treat_all
from Evaluate import evaluate_all
from Visualisations import visualize_all
import BuildAlpha

folder = "Merovingien"
folder_out = folder

features = [0]
DS = [3]
folds = 10
nbRuns = 20
list_output = [1, 2]
list_nbInterp = [[1], [2], [3]]
list_nbClus = [[2], [3], [4], [5], [6], [7], [8]]

prec = 1e-5  # Stopping threshold : when relative variation of the likelihood over 10 steps is < to prec
maxCnt = 30  # Number of consecutive times the relative variation is lesser than prec for the algorithm to stop
propTrainingSet = 1.
lim = -1
seuil = 0
num_processes = 7

if __name__ == "__main__":
    treat_all()
    for output in list_output:
        BuildAlpha.run(folder, DS, features, output, propTrainingSet, folds, lim, seuil=seuil)
    os.system("python run_XP.py")  # For multiprocessing
    evaluate_all()
    visualize_all()

