import dss as dss_interface # Because is a little faster than opendssdirect
import os
import pathlib # Paths become platform agnostic
from dss_reader import Reader
from dss_viewer import Viewer
from dss_controller import BatteryController
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

# Global params
start_date = datetime.datetime(2019, 6, 1, 5, 0, 0) #, tzinfo=datetime.timezone.utc)
end_date = datetime.datetime(2019, 6, 3, 0, 0, 0) #, tzinfo=datetime.timezone.utc)
# Some missing columns on 2019-07-09

# Setup
dss = dss_interface.DSS
dss.Start(0)

path_proj = os.path.dirname(os.path.abspath(__file__))
file_Circuit = "37Bus/ieee37.dss"

dss_text = dss.Text
dss_circuit = dss.ActiveCircuit

# Run a snapshot simulation
dss_text.Command = "clear"
dss_text.Command = f"compile {file_Circuit}"
dss_text.Command = "New EnergyMeter.Feeder Line.L1 1" # Used to calculate distance of busses

# Use loads for simple power flow (battery discharge is negative load)
dss_text.Command = "New Load.B725 Bus1 = 725.2.3 Phases=1, Conn=Delta Model=1 kV=4.8 kW = 0 kVAR = 0 status=variable"
bat_controller = BatteryController(bus = "725.2.3", ts = 0.25)

dss_text.Command = "set mode=snapshot"
dss_circuit.Solution.Solve()

os.chdir(path_proj)

# Helper objects
circuit_reader = Reader(dss_circuit)
df_circuit = circuit_reader.read_circuit()
df_circuit.to_csv("data/df_circuit37.csv", index = False)
circuit_viewer = Viewer(df_circuit)

# Data
df_load_ts_all = pd.read_csv("data/df_timeseries_ieee.csv", parse_dates = ["DateTimeUTC"])
df_load_orig = pd.read_csv("data/37bus_loads.csv")

interval_mask = (start_date <= df_load_ts_all["DateTimeUTC"]) &\
        (df_load_ts_all["DateTimeUTC"] <= end_date)

df_load_ts = df_load_ts_all[interval_mask].reset_index(drop = True)

control = False

# Loop
n_ts = df_load_ts.shape[0]
for i, time_row in tqdm(df_load_ts.iterrows(), total=len(df_load_ts)):
    #print(i+1, '/', n_ts)

    # Change values at each timestep
    for j, load_row in df_load_orig.iterrows():
        loadchangecommand = "Edit {} kW = {:.4f} kvar = {:.4f}".format(load_row.Load,
                                                                       time_row[load_row.Load + "_kW"],
                                                                       time_row[load_row.Load + "_kVAR"],)
        dss_text.Command = loadchangecommand


    if control:
        if i > 0:
            battery_power = bat_controller.step(circuit_reader.node_ts[-1])
        else:
            battery_power = bat_controller.step(None)

        batterychangecommand = "Edit Load.B725 kW = {:.4f}, kvar = 0".format(battery_power)
        dss_text.Command = batterychangecommand
    else:
        _ = bat_controller.step(None)

    dss_text.Command = 'set mode=snapshot'
    dss_text.Command = 'solve'
    circuit_viewer.t = i
    circuit_viewer.dt = time_row["DateTimeUTC"]
    circuit_viewer.df_circuit = circuit_reader.read_circuit()
    circuit_reader.read_NodeT("725.2", 2)
    bus_puvoltsabs = circuit_viewer.df_circuit[(df_circuit["type"] == 'bus') & (df_circuit["name"]=="725")].iloc[0].puvoltsabs

    #print("Bus puvolts: ", bus_puvoltsabs, circuit_reader.node_ts[-1])
    #break
    #circuit_viewer.view_plot(do_show = False, do_save = True, fig_name = "bus37")
    #circuit_viewer.view_plot(do_show = True, do_save = False, fig_name = "bus37")
    #break

#print("Pu voltage min max", circuit_viewer.v_min, circuit_viewer.v_max)

#circuit_viewer.plot_NodeTs(circuit_reader.node_ts, df_load_ts["DateTimeUTC"].to_numpy())
#df_out = pd.DataFrame({'DateTimeUTC': df_load_ts["DateTimeUTC"].to_numpy(), 
#                       'puvoltsabs': circuit_reader.node_ts,
#                       'soc': bat_controller.battery.soc})
#df_out.to_csv("outputs/df_uncontrolled_37.csv", index = False)
print(np.mean(circuit_reader.node_ts))

