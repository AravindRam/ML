import pandas as pd
import matplotlib.pyplot as plt

df_train=pd.DataFrame.from_csv("train.csv",index_col=False,parse_dates=True)

dist_count = df_train.groupby("PdDistrict").count()
plt.figure()
dist_count.sort(columns="Dates",ascending=1)["Dates"].plot(kind="barh")
plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
plt.tight_layout()
plt.show()

dist_count = df_train.groupby("DayOfWeek").count()
plt.figure()
dist_count.sort(columns="Dates",ascending=1)["Dates"].plot(kind="barh")
plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
plt.tight_layout()
plt.show()
