import os
import pandas as pd
import openpyxl

myseed = 42
mybackbone = "resnet34"
subset = 0
subset_seed = 42

weights = 'best'

drive = "d:"
pathroot = os.path.join(drive, 'solardnn')

sites = ["Cal_Fresno", "Cal_Stockton", "France_ign", "France_google", "Germany", "NYC"]  # "Cal_Oxnard"- too few files
models = sites + ["combo_dataset"]
site_names = ["CA-F", "CA-S", "FR-I", "FR-G", "DE-G", "NY-Q"]
model_names = site_names + ["COMB"]

loss = pd.DataFrame([], index=model_names, columns=site_names)
iou_score = loss.copy(deep=True)
precision = loss.copy(deep=True)
recall = loss.copy(deep=True)
f1_score = loss.copy(deep=True)

for site in sites:
    for model in models:
        i = sites.index(site)
        j = models.index(model)

        resultroot = os.path.join(pathroot, "results")
        myresultfile = os.path.join(resultroot, f"results_set{subset}_{site}_predby{model}_{mybackbone}_{myseed}\\results_set{subset}_{site}_predby{model}_{mybackbone}_{myseed}.csv")
        dat = pd.read_csv(myresultfile)

        loss.iloc[j,i] = dat['loss'][0]
        iou_score.iloc[j, i] = dat['iou_score'][0]
        precision.iloc[j, i] = dat['precision'][0]
        recall.iloc[j, i] = dat['recall'][0]
        f1_score.iloc[j, i] = dat['f1-score'][0]

with pd.ExcelWriter(os.path.join(resultroot, "generalization_summary.xlsx")) as w:
    loss.to_excel(w, sheet_name="loss")
    iou_score.to_excel(w, sheet_name="iou_score")
    precision.to_excel(w, sheet_name="precision")
    recall.to_excel(w, sheet_name="recall")
    f1_score.to_excel(w, sheet_name="f1_score")