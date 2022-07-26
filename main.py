import dss as dss_interface # Because is a little faster than opendssdirect
import os
import pathlib # Paths become platform agnostic
from dss_reader import Reader
from dss_viewer import Viewer
import pandas as pd
import numpy as np

sim_mode = "24hr"

dss = dss_interface.DSS
dss.Start(0)

path_proj = os.path.dirname(os.path.abspath(__file__))
#file_13Bus = pathlib.Path(path_proj).joinpath("13Bus", "IEEE13Nodeckt.dss")
file_Circuit = "13Bus/IEEE13Nodeckt.dss"

dss_text = dss.Text
dss_circuit = dss.ActiveCircuit


# Run a snapshot simmulation
dss_text.Command = "clear"
dss_text.Command = f"compile {file_Circuit}"
dss_text.Command = "New EnergyMeter.Feeder Line.650632 1" # Used to calculate distance of busses

# Create Storage
mode = "Idling"
dss_text.Command = "New Storage.N98 Bus1=652  phases=1 kV=4.16 kWRated=200 kWhRated=200 kWhStored=100"
dss_text.Command = f"~ State={mode}"

dss_text.Command = "set mode=snapshot"
dss_circuit.Solution.Solve()


circuit_reader = Reader(dss_circuit)
#circuit_controller.read_propsLine(do_print = True)

os.chdir(path_proj)
df_circuit = circuit_reader.read_circuit()
df_circuit.to_csv("outputs/df_circuit.csv", index=False)
# print(df_circuit.info())

circuit_viewer = Viewer(df_circuit)
#circuit_viewer.view_plot()
df_circuit_orig = df_circuit.copy()
df_Loads_orig = df_circuit_orig[df_circuit_orig["type"] == "load"].reset_index(drop = True)


# Main loop, for every hour in a day
if sim_mode == "24hr":
    t = np.linspace(0, 23, 24)
    df_Loads_TS = pd.read_csv("outputs/df_LoadsTS.csv")
    print(df_Loads_TS.info())
    for i in t:
        print("--", i, "/", len(t-1))
        for j, row in df_Loads_orig.iterrows():
            load_row = df_Loads_TS[(df_Loads_TS["name"] == row["name"]) & (df_Loads_TS["t"] == i)].iloc[0]
            #print(load_row["t"], load_row["name"], load_row["kW"], load_row["kvar"])
            loadchangecommand = "Edit {} kW = {:.4f} kvar = {:.4f}".format(load_row["name"], load_row["kW"], load_row["kvar"])
            #print(loadchangecommand)
            dss_text.Command = loadchangecommand

        dss_text.Command = 'set mode=snapshot'
        dss_text.Command = 'solve'

        circuit_reader.read_NodeT("652.1", 1)
        print(mode)
        if circuit_reader.node_ts[-1] > 0.985:
            mode = "charging"
        elif circuit_reader.node_ts[-1] < 0.975:
            mode = "discharging"
        else:
            mode = "idling"

        storagechangecommand = f"Edit Storage.N98 state={mode}"
        dss_text.Command = storagechangecommand

        circuit_viewer.t = i
        circuit_viewer.df_circuit = circuit_reader.read_circuit()
        circuit_viewer.view_plot(do_show = False, do_save = True)

    print(circuit_viewer.v_min, circuit_viewer.v_max)
    circuit_viewer.plot_NodeTs(circuit_reader.node_ts)

elif sim_mode == "once":
    dss_text.Command = 'set mode=snapshot'
    dss_text.Command = 'solve'
    dictNodesPU = circuit_reader.read_nodesPU()
    circuit_viewer.view_profile(dictNodesPU)
    

    # dss_text.Command = 'Plot Profile Phases = All'
    # dss_text.Command = "show eventlog"
    # circuit_viewer.t = 0
    # circuit_viewer.df_circuit = circuit_reader.read_circuit()
    # circuit_viewer.view_plot(do_show)


    #input()

# To create gif
# convert -delay 20 -loop 0 *.png animation.gif
# Or more efficient with ffmpeg
# ffmpeg -f image2 -framerate 2 -i bus37_%003d.png video.avi
# Less compression
# ffmpeg -f image2 -framerate 4 -i bus37_%003d.png -c:v libx264 -qp 0 -f mp4 video.mp4
# mpv animation.gif # In mpv type "L"
