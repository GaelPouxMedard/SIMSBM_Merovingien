import numpy as np
import os


dicData = {}
columns_considered = ["Sepulture", "Statut", "Objets", "Nombre_indiv", "Sexe", "Classe_age"]
setObj, setAge, setSexe = [], [], []
with open("Data/data.csv", "r") as f:
    columns = f.readline()
    print("\t".join(columns.split(";")))
    for line in f:
        data_line = list(map(str.lower, line.split(";")))

        data_line[3] = data_line[3].replace(" ", "_")
        data_line[7] = data_line[7].replace(" ?", "")
        data_line[10] = data_line[10].replace("non_identifié", "indéterminé")

        if data_line[1] not in dicData:
            dicData[data_line[1]] = {}
            dicData[data_line[1]]["Statut"] = "-1"
            dicData[data_line[1]]["Objets"] = []
            dicData[data_line[1]]["Nombre_indiv"] = "-1"
            dicData[data_line[1]]["Sexe"] = "-1"
            dicData[data_line[1]]["Classe_age"] = "-1"

        if data_line[3] in ["non_identifié", "indéterminé", "objet_décontextualisé"]: continue

        dicData[data_line[1]]["Statut"] = data_line[2]
        dicData[data_line[1]]["Objets"].append(data_line[3])
        dicData[data_line[1]]["Nombre_indiv"] = data_line[6]
        dicData[data_line[1]]["Sexe"] = data_line[7]
        dicData[data_line[1]]["Classe_age"] = data_line[10]

        setObj.append(data_line[3].replace(" ", "_"))
        setSexe.append(data_line[7])
        setAge.append(data_line[10])


print(np.unique(setObj, return_counts=True))
print(np.unique(setAge, return_counts=True))
print(np.unique(setSexe, return_counts=True))

tabNumObj = []
for s in dicData:
    tabNumObj.append(len(dicData[s]["Objets"]))

if "Data" not in os.listdir("."):
    os.mkdir("Data/")

featureObj = open("Data/Merovingien/feature_0.txt", "w+", encoding="utf-8")
featureSexe = open("Data/Merovingien/feature_1.txt", "w+", encoding="utf-8")
featureAge = open("Data/Merovingien/feature_2.txt", "w+", encoding="utf-8")
for s in dicData:
    featureObj.write(f"{s}\t{' '.join(dicData[s]['Objets'])}\n")
    if dicData[s]['Sexe'] != "indéterminé":
        featureSexe.write(f"{s}\t{dicData[s]['Sexe']}\n")
    if not dicData[s]['Classe_age'] in ["nd", "-1"]:
        featureAge.write(f"{s}\t{dicData[s]['Classe_age']}\n")

featureObj.close()
featureSexe.close()
featureAge.close()