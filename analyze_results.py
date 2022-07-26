import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df_Idle = pd.read_csv("outputs/Idle.csv")
df_Charging = pd.read_csv("outputs/Charging.csv")
df_Discharging = pd.read_csv("outputs/Discharging.csv")
df_BangBang = pd.read_csv("outputs/BangBang.csv")

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)

ax1.plot(df_Idle["t"],        df_Idle["v"],        ls = "--", label = "Idle")
ax1.plot(df_Charging["t"],    df_Charging["v"],    ls = "--", label = "Charging")
ax1.plot(df_Discharging["t"], df_Discharging["v"], ls = "--", label = "Discharging")
ax1.plot(df_BangBang["t"],    df_BangBang["v"],               label = "BangBang")

ax1.axhline(y = 0.985, color = "gray", ls = ":")
ax1.axhline(y = 0.975, color = "gray", ls = ":")

ax1.legend(loc = "upper right")
ax1.set_xlabel("Time (hr)")
ax1.set_ylabel("Voltage (pu)")
ax1.set_title("Voltage in Node 652.1")

plt.show()
