import matplotlib.pyplot as plt
import numpy as np
import os

def treat_all():
    dicData = {}
    columns_considered = ["Sepulture", "Statut", "Objets", "Nombre_indiv", "Sexe", "Classe_age"]
    setObj, setAge, setSexe = [], [], []

    cntobj = {}
    with open("Data/data.csv", "r") as f:
        f.readline()
        for line in f:
            data_line = list(map(str.lower, line.split(";")))
            if data_line[3] not in cntobj: cntobj[data_line[3]] = 0
            cntobj[data_line[3]] += 1

    if "Merovingien" not in os.listdir("Data"): os.mkdir("Data\\Merovingien\\")
    id_to_cnt = {}
    cnt = 0
    with open("Data/data.csv", "r") as f:
        columns = f.readline()
        print("\t".join(columns.split(";")))
        for line in f:
            data_line = list(map(str.lower, line.split(";")))
            if cntobj[data_line[3]]<10: continue  # Considère pas objets apparaissant moins de 10 fois au total

            data_line[3] = data_line[3].replace(" ", "_")
            data_line[7] = data_line[7].replace(" ?", "")
            data_line[10] = data_line[10].replace("non_identifié", "indéterminé")

            data_line[8] = data_line[8].replace(" ?", "")
            data_line[9] = data_line[9].replace(" ?", "")
            if data_line[7] not in ["féminin", "masculin"]:
                if data_line[8] not in ["", "indéterminé"]:
                    data_line[7] = data_line[8]
                # if data_line[9] not in ["", "indéterminé"]:  # Sexe archéo
                #     data_line[7] = data_line[9]

            data_line[10] = data_line[10].replace("adolescent", "adulte")  # APPROXIMATION
            data_line[10] = data_line[10].replace("adulte mature-âgé", "adulte")  # APPROXIMATION

            id_tombe = data_line[0]+" - "+data_line[1]
            if id_tombe not in id_to_cnt:
                id_to_cnt[id_tombe] = cnt
                cnt += 1
            if id_to_cnt[id_tombe] not in dicData:
                dicData[id_to_cnt[id_tombe]] = {}
                dicData[id_to_cnt[id_tombe]]["Statut"] = "-1"
                dicData[id_to_cnt[id_tombe]]["Objets"] = []
                dicData[id_to_cnt[id_tombe]]["Nombre_indiv"] = "-1"
                dicData[id_to_cnt[id_tombe]]["Sexe"] = "-1"
                dicData[id_to_cnt[id_tombe]]["Classe_age"] = "-1"

            if data_line[3] in ["non_identifié", "indéterminé", "objet_décontextualisé"]: continue

            dicData[id_to_cnt[id_tombe]]["Statut"] = data_line[2]
            dicData[id_to_cnt[id_tombe]]["Objets"].append(data_line[3])
            dicData[id_to_cnt[id_tombe]]["Nombre_indiv"] = data_line[6]
            dicData[id_to_cnt[id_tombe]]["Sexe"] = data_line[7]
            dicData[id_to_cnt[id_tombe]]["Classe_age"] = data_line[10]

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

    id_to_sep = open("Data/Merovingien/id2sep.txt", "w+", encoding="utf-8")
    featureObj = open("Data/Merovingien/feature_0.txt", "w+", encoding="utf-8")
    featureSexe = open("Data/Merovingien/feature_1.txt", "w+", encoding="utf-8")
    featureAge = open("Data/Merovingien/feature_2.txt", "w+", encoding="utf-8")
    for s in id_to_cnt:
        id_to_sep.write(f"{s}\t{id_to_cnt[s]}\n")
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


    cntobj = {}
    with open("Data/data.csv", "r") as f:
        f.readline()
        for line in f:
            data_line = list(map(str.lower, line.split(";")))
            if data_line[3] not in cntobj: cntobj[data_line[3]] = 0
            cntobj[data_line[3]] += 1


    id_to_cnt = {}
    cnt_to_id = {}
    cnt = 0
    with open("Data/data.csv", "r") as f:
        columns = f.readline()
        print("\t".join(columns.split(";")))
        for line in f:
            data_line = list(map(str.lower, line.split(";")))
            if cntobj[data_line[3]]<10: continue  # Considère pas objets apparaissant moins de 10 fois au total

            data_line[3] = data_line[3].replace("contenant du défunt/chambre funéraire", "contenant du défunt")
            data_line[3] = data_line[3].replace(" ", "_")
            data_line[7] = data_line[7].replace(" ?", "")
            data_line[10] = data_line[10].replace("non_identifié", "indéterminé")

            data_line[8] = data_line[8].replace(" ?", "")
            data_line[9] = data_line[9].replace(" ?", "")
            if data_line[7] not in ["féminin", "masculin"]:
                if data_line[8] not in ["", "indéterminé"]:
                    data_line[7] = data_line[8]
                # if data_line[9] not in ["", "indéterminé"]:  # Sexe archéo
                #     data_line[7] = data_line[9]

            data_line[10] = data_line[10].replace("adolescent", "adulte")  # APPROXIMATION
            data_line[10] = data_line[10].replace("adulte mature-âgé", "adulte")  # APPROXIMATION

            setPlaces.add(data_line[0])

            id_tombe = data_line[0]+" - "+data_line[1]
            if id_tombe not in id_to_cnt:
                id_to_cnt[id_tombe] = cnt
                cnt_to_id[cnt] = id_tombe
                cnt += 1
            if id_to_cnt[id_tombe] not in dicData:
                dicData[id_to_cnt[id_tombe]] = {}
                dicData[id_to_cnt[id_tombe]]["Statut"] = "-1"
                dicData[id_to_cnt[id_tombe]]["Objets"] = []
                dicData[id_to_cnt[id_tombe]]["Nombre_indiv"] = "-1"
                dicData[id_to_cnt[id_tombe]]["Sexe"] = "-1"
                dicData[id_to_cnt[id_tombe]]["Classe_age"] = "-1"

            if data_line[3] in ["non_identifié", "indéterminé", "objet_décontextualisé"]: continue

            dicData[id_to_cnt[id_tombe]]["Statut"] = data_line[2]
            dicData[id_to_cnt[id_tombe]]["Objets"].append(data_line[3])
            dicData[id_to_cnt[id_tombe]]["Nombre_indiv"] = data_line[6]

            setObj.append(data_line[3].replace(" ", "_"))
            if data_line[7] not in ["indéterminé", "-1", ""]:  # Sexe
                setSexe.append(data_line[7])
                dicData[id_to_cnt[id_tombe]]["Sexe"] = data_line[7]
            if data_line[10] not in ["nd", "-1", ""]:  # Age
                setAge.append(data_line[10])
                dicData[id_to_cnt[id_tombe]]["Classe_age"] = data_line[10]

    i = 1
    txt = ""
    txt_corr = ""
    if False:
        txt += "\\begin{table}\n" \
               "    \\centering\n " \
               "    \\begin{tabularx}{\linewidth}{|L|L|L|L|} \n \\hline \n"
        for k in dicData:
            txt += f"                Sépulture {k} & {', '.join(dicData[k]['Objets']).replace('_', ' ').capitalize()} & & \\\\ \\hline \n"
            if i%20==0:
                txt += "    \end{tabularx}" \
                       "\end{table}\n\n\n"
                txt += "\\begin{table}\n" \
                       "    \\centering\n " \
                       "    \\begin{tabularx}{\linewidth}{|L|L|L|L|} \n \\hline \n"
            i+=1
        txt += "    \end{tabularx}" \
               "\end{table}\n\n\n"
    else:
        coche = "☒"
        i = 1
        for k in dicData:
            if dicData[k]['Objets'] != [] and dicData[k]['Sexe'] not in ["indéterminé", "-1", ""]:
                txt += f"{cnt_to_id[k].capitalize()}\t{', '.join(dicData[k]['Objets']).replace('_', ' ').capitalize()}\t☐ Féminin - ☐ Masculin\n"
                if dicData[k]['Sexe']=="féminin":
                    txt_corr += f"{cnt_to_id[k].capitalize()}\t{', '.join(dicData[k]['Objets']).replace('_', ' ').capitalize()}\t"+coche+" Féminin - ☐ Masculin\n"
                if dicData[k]['Sexe']=="masculin":
                    txt_corr += f"{cnt_to_id[k].capitalize()}\t{', '.join(dicData[k]['Objets']).replace('_', ' ').capitalize()}\t☐ Féminin - "+coche+" Masculin\n"
                i += 1
    print(txt_corr)
    #pause()


    nombre_objets = 2158
    sexe, nombre_objets_sexe = np.unique(setSexe, return_counts=True)
    age, nombre_objets_age = np.unique(setAge, return_counts=True)

    tombes_avec_sexe = [1 for k in dicData if dicData[k]["Sexe"]!="-1"]
    tombes_feminin = [1 for k in dicData if dicData[k]["Sexe"]=="féminin"]
    tombes_masculin = [1 for k in dicData if dicData[k]["Sexe"]=="masculin"]
    tombes_avec_age = [1 for k in dicData if dicData[k]["Classe_age"]!="-1"]
    tombes_adolescent = [1 for k in dicData if dicData[k]["Classe_age"]=="adolescent"]
    tombes_adulte = [1 for k in dicData if dicData[k]["Classe_age"]=="adulte"]
    tombes_immature = [1 for k in dicData if dicData[k]["Classe_age"]=="immature"]
    tombes_mature = [1 for k in dicData if dicData[k]["Classe_age"]=="adulte mature-âgé"]

    objets = []
    for k in dicData:
        objets += dicData[k]["Objets"]
    u, c = np.unique(objets, return_counts=True)
    u = np.array([u_i for _, u_i in sorted(zip(c, u))])
    c = np.array([c_i for c_i, u_i in sorted(zip(c, u))])
    print(list(zip(u,c)))
    print(len(c), np.sum(c))
    print(len(c[c>10]), np.sum(c[c>10]))

    print(len(dicData))
    print("Nombre objets sexe", nombre_objets_sexe, sexe)
    print("Nombre objets age", nombre_objets_age, age)
    print()
    print("Tombes avec sexe :", len(tombes_avec_sexe))
    print("Tombes féminin :", len(tombes_feminin))
    print("Tombes masculin :", len(tombes_masculin))
    print("Ratio féminin/masculin :", len(tombes_feminin)/(len(tombes_masculin)+len(tombes_feminin)))
    print()
    print("Tombes avec age :", len(tombes_avec_age))
    print("Tombes adolescent :", len(tombes_adolescent))
    print("Tombes adulte :", len(tombes_adulte))
    print("Tombes adulte mature-âgé :", len(tombes_mature))
    print("Tombes immature :", len(tombes_immature))
    print("Ratio adulte/immature :", len(tombes_adulte)/(len(tombes_immature)+len(tombes_adulte)))

    for place in setPlaces:
        pass
        # print(place)

    plt.figure(figsize=(11,7))
    plt.subplot(221)
    plt.bar(["Total", "Age identifié", "Sexe identifié"], [nombre_objets, sum(nombre_objets_age), sum(nombre_objets_sexe)], color=["darkred", "g", "gold"])
    plt.bar(["Age identifié"], [nombre_objets_age[0]], color="orange")
    plt.bar(["Sexe identifié"], [nombre_objets_sexe[0]], color="purple")
    plt.ylabel("Nombre d'objets")


    plt.subplot(222)
    plt.bar(["Total", "Age identifié", "Sexe identifié"], [len(dicData), len(tombes_avec_age), len(tombes_avec_sexe)], color=["darkred", "g", "gold"])
    plt.bar(["Age identifié"], [len(tombes_adulte)], color="orange")
    plt.bar(["Sexe identifié"], [len(tombes_feminin)], color="purple")
    plt.ylabel("Nombre de sépultures")

    plt.subplot(212)
    obj_lab, obj_cnt = np.unique(objets, return_counts=True)
    print([(a,b) for a,b in zip(obj_lab, obj_cnt)])
    obj_lab = [l.replace("_", " ").capitalize() for c, l in sorted(zip(obj_cnt, obj_lab), reverse=True)]
    obj_cnt = [c for c, l in sorted(zip(obj_cnt, obj_lab), reverse=True)]
    plt.bar(obj_lab, obj_cnt, color=["darkred"])
    num_moyen_obj = sum(obj_cnt)/len(dicData)
    plt.text(len(obj_lab)//2, max(obj_cnt)*2/3, f"Nombre moyen d'objets par sépulture : {np.round(num_moyen_obj, 1)}")
    plt.xticks(rotation=90)
    plt.ylabel("Nombre d'objets")


    plt.tight_layout()
    # plt.savefig("Figures_article\\Stats_DS.pdf")


