import dss as dss_interface # Because is a little faster than opendssdirect
import os
import pathlib # Paths become platform agnostic
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as text
from matplotlib.collections import LineCollection
from matplotlib import cm
import re


# Port of dssvplot.py for l00nix
# Ported by Javier Sales-Ortiz
# Plots the line voltages, lines and nodes of a circuit.
# The code was supposed to be somewhat "agnostic" to the file, but it has some hardcoded names (see EnergyMeter)
# Need a *.dss file for circuit definition and a Buscoords *.dat or *.csv

# Original one has Classes, this is implemented in a less fancy way.
# Just using a change of structure to think more about the code
# Possibly in the future, use class structure as the original

def main():

    np.set_printoptions(precision=3)

    dss = dss_interface.DSS
    dss.Start(0)

    # "Robust" way of handling paths
    path_proj = os.path.dirname(os.path.abspath(__file__))
    file_Circuit = pathlib.Path(path_proj).joinpath("13Bus", "IEEE13Nodeckt.dss")
    file_Coords = pathlib.Path(path_proj).joinpath("13Bus", "IEEE13Node_BusXY.csv")

    # Common shortcuts
    dss_text = dss.Text
    dss_circuit = dss.ActiveCircuit

    # Run a snapshot simmulation
    dss_text.Command = "clear"
    dss_text.Command = f"compile {file_Circuit}"
    dss_text.Command = "New EnergyMeter.Feeder Line.650632 1" # Used to calculate distance of busses
    dss_circuit.Solution.Solve()

    # This plot has two data sources, here dictionaries: Bus and Branch

    # self.bus = Bus(self.circuit)
    print("\n--- Filling in Buses ---")

    # Obviously, there are better ways to do this
    n = dss_circuit.NumBuses
    x = np.zeros(n)
    y = np.zeros(n)
    distance = np.zeros(n)
    V = np.zeros((n,3), dtype=complex)
    name = np.empty(n, dtype=object)

    Bus = {}

    # print("name, x, y, distance, V")
    for i in range(n): # There must be more pythonic ways to iterate
        bus = dss_circuit.Buses(i)
        name[i] = bus.Name
        x[i] = bus.x
        y[i] = bus.y
        distance[i] = bus.Distance
        v = np.array(bus.Voltages)
        nodes = np.array(bus.Nodes)
        
        # We are only interested in the first three nodes in each bus ...apparently
        if nodes.size > 3:
            print(f"\t Bus {name[i]} has more than three buses")
            nodes = nodes[0:3]
        cidx = 2 * np.array(range(0, min(v.size // 2, 3))) # Voltages come as a complex array
        V[i, nodes-1] = v[cidx] + 1j * v[cidx + 1]

        # print(name[i], x[i], y[i], distance[i], V[i, nodes-1], sep=', ')

    Bus["name"] = name 
    Bus["x"] = x
    Bus["y"] = y
    Bus["distance"] = distance
    Bus["V"] = V


    # self.branch = Branch(self.circuit)
    print("\n--- Filling in Branches ---") # Aka Lines

    # I'm sure there are better ways to do this
    n = dss_circuit.NumCktElements
    name = np.empty(n, dtype = object)
    busname = np.empty(n, dtype = object)
    busnameto = np.empty(n, dtype = object)
    x = np.zeros(n)
    y = np.zeros(n)
    xto = np.zeros(n)
    yto = np.zeros(n)
    distance = np.zeros(n)
    nphases = np.zeros(n)
    kvbase = np.zeros(n)
    I = np.zeros((n,3), dtype=complex)
    V = np.zeros((n,3), dtype=complex)
    Vto = np.zeros((n,3), dtype=complex)
    i = 0

    Branch = {}

    # print("name, busname, busnameto, nphases, kvbase, x, y, xto, yto, distance, V, Vto, I")
    for j in range(n):
        el = dss_circuit.CktElements(j)
        if not re.search("^Line", el.Name): # Check if element is of type line. There must be better ways to do this
            continue
        name[i] = el.Name

        bus2 = dss_circuit.Buses(re.sub(r"\..*", "", el.BusNames[-1])) # Gotcha! This passes a reference, not a copy
        # So we cant have bus1 and bus2 at the same time

        busnameto[i] = bus2.Name
        xto[i] = bus2.x
        yto[i] = bus2.y

        if bus2.x == 0 or bus2.y == 0: # Although having buses at 0 should be allowed, don't blame me
            # For now, manually changed all buses correctly defined at 0 to 1
            print(f"{j}: Skipping line {name[i]} without proper bus coordinates")
            continue 

        distance[i] = bus2.Distance # Distance to energymeter
        v = np.array(bus2.Voltages)
        nodes = np.array(bus2.Nodes)
        kvbase[i] = bus2.kVBase # 
        nphases[i] = nodes.size # Number of phases
        
        if nodes.size > 3: 
            nodes = nodes[0:3]
        cidx = 2 * np.array(range(0, min(v.size // 2, 3)))

        bus1 = dss_circuit.Buses(re.sub(r"\..*", "", el.BusNames[0]))

        if bus1.x == 0 or bus1.y == 0: # Same as before
            print(f"{j}: Skipping line {name[i]} without proper bus coordinates")
            continue 
        busname[i]  = bus1.Name

        Vto[i, nodes-1] = v[cidx] + 1j * v[cidx+1]
        x[i] = bus1.x
        y[i] = bus1.y
        v = np.array(bus1.Voltages)
        V[i, nodes-1] = v[cidx] + 1j * v[cidx+1]
        current = np.array(el.Currents)
        I[i, nodes-1] = current[cidx] + 1j * current[cidx + 1]
        # print(name[i], busname[i], busnameto[i], nphases[i], round(kvbase[i],4), x[i], y[i], xto[i], yto[i], distance[i], sep=', ')
        # V[i, nodes-1], Vto[i, nodes-1], I[i, nodes-1], sep=', ')
        i += 1

    Branch["name"] = name[0:i]
    Branch["busname"] = busname[0:i]
    Branch["busnameto"] = busnameto[0:i]
    Branch["nphases"] = nphases[0:i]
    Branch["kvbase"] = kvbase[0:i]
    Branch["x"] = x[0:i]
    Branch["y"] = y[0:i]
    Branch["xto"] = xto[0:i]
    Branch["yto"] = yto[0:i]
    Branch["distance"] = distance[0:i]

    Branch["V"] = V[0:i]
    Branch["Vto"] = Vto[0:i]
    Branch["I"] = I[0:i]

    # First figure: Line Voltages
    # plot_voltage
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    scalefactor = 1 / Branch["kvbase"] * 120 / 1000

    # Distance from energymeter to somewhat differentiatie the lines
    # Although still some get on top of each other
    x = Branch["distance"]

    y = (abs(Branch["Vto"].transpose()) * scalefactor).transpose()
    ax1.plot(x, y, '*', markersize = 5, picker=5)

    # On click actions
    plt.connect("pick_event", lambda e: hilight_voltage_plot(e, Bus, Branch, selected, fig))
    plt.connect("pick_event", lambda e: hilight_map(e, Bus, Branch, mapselected, mapfig))
    plt.connect("pick_event", lambda e: print_branch_info(e, Bus, Branch, selected, fig))

    # Draws nice circle around the selected line voltages and around the lines in the map
    selected, = ax1.plot(x[0:3], y[0], 'o', ms=12, alpha=0.7, color='yellow', visible=False)

    ax1.set_xlabel('distance (km)')
    ax1.set_ylabel('Voltage (120V base)')
    ax1.set_title('Primary Voltages by phase')

    # Set all the font sizes, nice method. I didn't know this
    for o in fig.findobj(text.Text):
        o.set_fontsize(12)

    ax1.set_xlim(x.min() - 0.1, x.max()+ 0.1)
    ax1.set_ylim(y.min()-20, y.max()+20)

    # Second figure: Line map
    # plot_map
    mapfig = plt.figure()
    x1 = Branch["x"]
    y1 = Branch["y"]
    x2 = Branch["xto"]
    y2 = Branch["yto"]

    plt.xticks([])
    plt.yticks([])
    plt.xlim(x1.min()-20, x1.max()+20)
    plt.ylim(y1.min()-20, y1.max()+20)

    # There are more elegant ways to do this
    segments = [( (thisx1, thisy1), (thisx2, thisy2)) for thisx1, thisy1, thisx2, thisy2 in zip(x1,y1,x2,y2)]

    # Colors
    voltsAvg = []
    for voltsBranch in Branch["Vto"]:
        voltsAvg.append(np.average([x for x in abs(voltsBranch) if x > 0]))
    v_max = np.max(voltsAvg)
    v_min = np.min(voltsAvg)

    voltsColor = [(x - v_min)*(0.4/(v_max-v_min))+0.3 for x in voltsAvg]
    voltsRGB = [cm.get_cmap('plasma')(vC) for vC in voltsColor]
    
    line_segments =  LineCollection(segments, colors = voltsRGB, linewidths = Branch["nphases"] * 4, linestyle='solid', picker = 5)
    line_segmentsk =  LineCollection(segments, colors = ["k"] * len(Branch["name"]), linewidths = Branch["nphases"] , linestyle='solid', picker = 5)
    
    mapax = mapfig.add_subplot(111)
    mapax.add_collection(line_segments)
    mapax.add_collection(line_segmentsk)
    mapax.set_title("Map of IEEE13Bus\nColor indicates voltage level")

    # Now link the second figure to the same events. Both figures can modify each other
    plt.connect("pick_event", lambda e: hilight_voltage_plot(e, Bus, Branch, selected, fig))
    plt.connect("pick_event", lambda e: hilight_map(e, Bus, Branch, mapselected, mapfig))
    plt.connect("pick_event", lambda e: print_branch_info(e, Bus, Branch, selected, fig))

    # Show where the buses lie, although they are not clickable...
    mapax.plot(Bus["x"], Bus["y"], 's', ms=6, color='k')

    # Hilights around the line. This could be better to highlight the entire line
    mapselected, = mapax.plot([x2[0]], [y2[0]], 'o', ms=12, alpha=0.4, color='yellow', visible=False)
    mapax.set_aspect('equal', 'datalim')

    plt.show()

    print("\nAll Line Names")
    print(type(dss_circuit.Lines.AllNames))
    print(len(dss_circuit.Lines.AllNames))
    print(dss_circuit.Lines.AllNames)

    print("\nAll Bus Names")
    print(type(dss_circuit.AllBusNames))
    print(len(dss_circuit.AllBusNames))
    print(dss_circuit.AllBusNames)
    
    print("\nAll Bus Volts")
    print(type(dss_circuit.AllBusVolts))
    print(dss_circuit.AllBusVolts.shape)
    print(dss_circuit.AllBusVolts)


def hilight_voltage_plot(event, Bus, Branch, selected, fig):
    #axis = event.artist.get_figure().axes
    ind = event.ind[0]
    x = Branch["distance"][ind].repeat(3)
    y = abs(Branch["Vto"][ind]) / Branch["kvbase"][ind] * 120 / 1000
    selected.set_visible(True)
    selected.set_data(x,y)
    fig.canvas.draw()

def hilight_map(event, Bus, Branch, mapselected, mapfig):
    #axis = event.artist.get_figure().axes
    ind = event.ind[0]
    x = Branch["x"][ind]
    y = Branch["y"][ind]
    xto = Branch["xto"][ind]
    yto = Branch["yto"][ind]
    mapselected.set_visible(True)
    mapselected.set_data(xto, yto)
    mapfig.canvas.draw()

def print_branch_info(event, Bus, Branch, selected, fig):
    ind = event.ind[0]
    print(" ")
    print("line: ", Branch["name"][ind])
    print("number of phases: ", Branch["nphases"][ind])
    print("voltages: ", np.around(abs(Branch["Vto"][ind]), 1))
    print("voltages (120-V base): ", np.around(abs(Branch["Vto"][ind]) / Branch["kvbase"][ind] * 120 / 1000, 1))

    print("currents: ", np.around(abs(Branch["I"][ind])))
    S = Branch["V"][ind] * Branch["I"][ind].conj()
    print("kilowatts: ", np.around(S.real / 1000))
    print("kilovars: ", np.around(S.imag / 1000))
    print("pf: ", np.around(S.real / abs(S), 2))

    print(" ")

if __name__ == '__main__':
    main()
