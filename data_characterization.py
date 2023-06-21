import numpy as np
import pandas as pd
import sklearn as sci
import matplotlib.pyplot as plt

df = pd.read_csv("filename.txt", sep="\t")
time_encoded = df["time_encoded"]
toff = df["time_new"]

quant_25, quant_33, quant_50, quant_66, quant_75 = (
    toff.quantile(0.25),
    toff.quantile(1 / 3),
    toff.quantile(0.5),
    toff.quantile(2 / 3),
    toff.quantile(0.75),
)
df.corr

# plot histogram with quantiles labelled
fig, ax = plt.subplots(figsize=(6, 4))
toff.plot(kind="hist", bins=15)
ax.set_xlabel("toff value")
ax.set_ylabel("Frequency")
quants = [quant_25, 0.85], [quant_50, 1], [quant_75, 0.77]
for i in quants:
    ax.axvline(i[0], ymax=i[1], linestyle=":", color="black")
ax.text(quant_25 - 0.1, 26.5, round(quant_25, 2), size=10)
ax.text(quant_50, 31, round(quant_50, 2), size=10)
ax.text(quant_75, 24, round(quant_75, 2), size=10)
plt.show()

print(
    "toff 25th percentile: ",
    quant_25,
    "\n",
    "toff 33th percentile: ",
    quant_33,
    "\n",
    "toff 50th percentile: ",
    quant_50,
    "\n",
    "toff 66th percentile: ",
    quant_66,
    "\n",
    "toff 75th percentile: ",
    quant_75,
)
