import numpy as np
import os

def treat_all():
    dicData = {}
    columns_considered = ["Sepulture", "Statut", "Objets", "Nombre_indiv", "Sexe", "Classe_age"]
    setObj, setAge, setSexe = [], [], []
    if "Merovingien" not in os.listdir("Data"): os.mkdir("Data\\Merovingien\\")
    with open("Data/data.csv", "r") as f:
        columns = f.readline()
        print("\t".join(columns.split(";")))
        for line in f:
            data_line = list(map(str.lower, line.split(";")))

            data_line[3] = data_line[3].replace(" ", "_")
            data_line[7] = data_line[7].replace(" ?", "")
            data_line[10] = data_line[10].replace("non_identifié", "indéterminé")

            data_line[8] = data_line[8].replace(" ?", "")
            data_line[9] = data_line[9].replace(" ?", "")
            if data_line[7] not in ["féminin", "masculin"]:
                if data_line[8] not in ["", "indéterminé"]:
                    data_line[7] = data_line[8]
                if data_line[9] not in ["", "indéterminé"]:
                    data_line[7] = data_line[9]

            data_line[10] = data_line[10].replace("adolescent", "adulte")  # APPROXIMATION
            data_line[10] = data_line[10].replace("adulte mature-âgé", "adulte")  # APPROXIMATION


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


    print(len(set(setObj)), np.unique(setObj, return_counts=True))
    print(len(setAge), np.unique(setAge, return_counts=True))
    print(len(setSexe), np.unique(setSexe, return_counts=True))

    tabNumObj = []
    for s in dicData:
        tabNumObj.append(len(dicData[s]["Objets"]))

    if "Data" not in os.listdir("."):
        os.mkdir("Data/")

    featureObj = open("Data/Merovingien/feature_0.txt", "w+", encoding="utf-8")
    featureSexe = open("Data/Merovingien/feature_1.txt", "w+", encoding="utf-8")
    featureAge = open("Data/Merovingien/feature_2.txt", "w+", encoding="utf-8")
    for s in dicData:
        if dicData[s]['Objets'] != []:
            featureObj.write(f"{s}\t{' '.join(dicData[s]['Objets'])}\n")
        if dicData[s]['Sexe'] not in ["indéterminé", "-1", ""]:
            featureSexe.write(f"{s}\t{dicData[s]['Sexe']}\n")
        if not dicData[s]['Classe_age'] in ["nd", "-1", ""]:
            featureAge.write(f"{s}\t{dicData[s]['Classe_age']}\n")

    featureObj.close()
    featureSexe.close()
    featureAge.close()


if __name__ == "__main__":
    dicData = {}
    columns_considered = ["Sepulture", "Statut", "Objets", "Nombre_indiv", "Sexe", "Classe_age"]
    setObj, setAge, setSexe = [], [], []
    setPlaces = set()
    with open("Data/data.csv", "r") as f:
        columns = f.readline()
        print("\t".join(columns.split(";")))
        for line in f:
            data_line = list(map(str.lower, line.split(";")))

            data_line[3] = data_line[3].replace(" ", "_")
            data_line[7] = data_line[7].replace(" ?", "")
            data_line[10] = data_line[10].replace("non_identifié", "indéterminé")

            data_line[8] = data_line[8].replace(" ?", "")
            data_line[9] = data_line[9].replace(" ?", "")
            if data_line[7] not in ["féminin", "masculin"]:
                if data_line[8] not in ["", "indéterminé"]:
                    data_line[7] = data_line[8]
                if data_line[9] not in ["", "indéterminé"]:
                    data_line[7] = data_line[9]

            data_line[10] = data_line[10].replace("adolescent", "adulte")  # APPROXIMATION
            data_line[10] = data_line[10].replace("adulte mature-âgé", "adulte")  # APPROXIMATION

            setPlaces.add(data_line[0])

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

            setObj.append(data_line[3].replace(" ", "_"))
            if data_line[7] not in ["indéterminé", "-1", ""]:  # Sexe
                setSexe.append(data_line[7])
                dicData[data_line[1]]["Sexe"] = data_line[7]
            if data_line[10] not in ["nd", "-1", ""]:  # Age
                setAge.append(data_line[10])
                dicData[data_line[1]]["Classe_age"] = data_line[10]


    print(len(dicData))
    print("Nombre objets", np.unique(setSexe, return_counts=True))
    print("Nombre objets", np.unique(setAge, return_counts=True))
    print()
    print("Tombes avec sexe :", len([1 for k in dicData if dicData[k]["Sexe"]!="-1"]))
    print("Tombes féminin :", len([1 for k in dicData if dicData[k]["Sexe"]=="féminin"]))
    print("Tombes masculin :", len([1 for k in dicData if dicData[k]["Sexe"]=="masculin"]))
    print()
    print("Tombes avec age :", len([1 for k in dicData if dicData[k]["Classe_age"]!="-1"]))
    print("Tombes adolescent :", len([1 for k in dicData if dicData[k]["Classe_age"]=="adolescent"]))
    print("Tombes adulte :", len([1 for k in dicData if dicData[k]["Classe_age"]=="adulte"]))
    print("Tombes adulte mature-âgé :", len([1 for k in dicData if dicData[k]["Classe_age"]=="adulte mature-âgé"]))
    print("Tombes immature :", len([1 for k in dicData if dicData[k]["Classe_age"]=="immature"]))

    for place in setPlaces:
        print(place)


