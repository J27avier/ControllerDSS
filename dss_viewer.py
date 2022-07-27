import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib import markers
import matplotlib.text as mpl_text
import pandas as pd
import datetime
from multiprocessing import Pool
import itertools
import pickle

import numpy as np


class Viewer:
    def __init__(self, df_circuit):
        self.df_circuit = df_circuit
        np.set_printoptions(precision=3)
        self.cmap_max = 0.9
        self.cmap_min = 0.1
        self.cmap = "plasma"
        self.t = 0
        self.dt = datetime.datetime(2000, 1, 1, 0, 0, 0)

        self.v_min = 100000
        self.v_max = 0

        # Cache vars
        self.dist_to_bus = None
        self.df_Buses_Agg = None

    def set_circuit(self, df_circuit):
        self.df_circuit = df_circuit

    def view_plot(self, do_show = True, do_save = False, fig_name = "bus" ):
        #breakpoint()
        fig1 = plt.figure(figsize=(10,10))
        ax1 = fig1.add_subplot(1,1,1)

        df_Buses = self.df_circuit[self.df_circuit["type"] == "bus"].reset_index(drop = True)

        x_min = df_Buses["x"].min()
        x_max = df_Buses["x"].max()

        y_min = df_Buses["y"].min()
        y_max = df_Buses["y"].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min

        ax1.set_aspect('equal', 'datalim')
        ax1.set_xlim(x_min - 0.1*x_range, x_max + 0.1*x_range) 
        ax1.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range) 

        # Colormap 
        ax_xmin, ax_xmax = ax1.get_xlim()
        ax_ymin, ax_ymax = ax1.get_ylim()
        x_coords, y_coords, c = self._get_colormesh(df_Buses, [ax_xmin, ax_xmax, ax_ymin, ax_ymax])
        im = ax1.pcolormesh(x_coords, y_coords, c, shading='gouraud', cmap='rainbow_r', vmin = 0.995, vmax = 1.0368)
        fig1.colorbar(im, ax=ax1)

        self._get_segments()

        line_segmentsRGB = LineCollection(self.df_Lines["segments"],
                                          colors = self.df_Lines["segmentsRGB"],
                                          linewidths = self.df_Lines["phases"] * 4,
                                          linestyle = "solid",
                                          picker = 5
                                         )
        line_segmentsBLK = LineCollection(self.df_Lines["segments"],
                                          colors = "k",
                                          linewidths = self.df_Lines["phases"],
                                          linestyle = "solid",
                                          label = "line",
                                          picker = 5
                                         )

        #ax1.add_collection(line_segmentsRGB)
        ax1.add_collection(line_segmentsBLK)

        ax1.set_title("Map")

        self.mapselected, = ax1.plot([self.df_Lines["x1"][0], self.df_Lines["x2"][0]],
                                     [self.df_Lines["y1"][0], self.df_Lines["y2"][0]],
                                     lw = 5,
                                     # 'o', ms= 12, 
                                     alpha = 0.4,
                                     color="yellow", visible = False)


        # Buses
        ax1.plot(df_Buses["x"].to_numpy(), df_Buses["y"].to_numpy(), 's', ms = 6, label = 'bus', color = 'k')
        
        # Bus text
        df_Buses["x_quant"] = df_Buses["x"].apply(lambda x: (x*100 // 3) * 3 / 100)
        df_Buses["y_quant"] = df_Buses["y"].apply(lambda y: (y*100 // 3) * 3 / 100)

        if self.df_Buses_Agg is None:
            self.df_Buses_Agg = df_Buses.groupby(["x_quant","y_quant"]).agg(x = ("x", min), 
                                                                            y = ("y", min),
                                                                            name = ("name", lambda x: (', ').join(x)),
                                                                           ).reset_index(drop = True)

        for i, row in self.df_Buses_Agg.iterrows():
            ax1.text(row["x"]+0.02*x_range, row["y"]+0.005*y_range, row["name"],) #backgroundcolor = 'white')
        # This doesn't work with connect because pick event needs a collection

        # Capacitors
        df_Capas = self.df_circuit[self.df_circuit["type"] == "capacitor"].reset_index(drop = True)
        if not df_Capas.empty:
            df_Capas["xy"] = df_Capas.apply(lambda row: (row["x"], row["y"]), axis=1)
            df_Capas["incremental"] = df_Capas.groupby(["xy"]).cumcount()
            df_Capas["ms"] = df_Capas["incremental"]*500 + 500
            ax1.scatter(df_Capas["x"], df_Capas["y"], s = df_Capas["ms"], color = 'green', facecolor = 'none', label = "capacitor", marker = markers.MarkerStyle(marker = 'D', fillstyle = 'none'))
        # Transformer
        df_Trans = self.df_circuit[self.df_circuit["type"] == "transformer" ].reset_index(drop = True)
        df_Trans["xy"] = df_Trans.apply(lambda row: (row["x"], row["y"]), axis=1)
        df_Trans["incremental"] = df_Trans.groupby(["xy"]).cumcount() 
        df_Trans["ms"] = df_Trans["incremental"]*500 + 800
        ax1.scatter(df_Trans["x"], df_Trans["y"], s = df_Trans["ms"], color = 'blue', facecolor = 'none', label = "transformer", marker = markers.MarkerStyle(marker = 'o', fillstyle = 'none'))
    

        # Loads 
        df_Loads = self.df_circuit[self.df_circuit["type"] == "load" ].reset_index(drop = True)
        df_Loads["xy"] = df_Loads.apply(lambda row: (row["x"], row["y"]), axis=1)
        df_Loads["incremental"] = df_Loads.groupby(["xy"]).cumcount() 
        df_Loads["ms"] = df_Loads["incremental"]*500 + 500
        ax1.scatter(df_Loads["x"], df_Loads["y"], s = df_Loads["ms"], color = 'red', facecolor = 'none', label = "load", marker = markers.MarkerStyle(marker = 's', fillstyle = 'none'))

        ax1.legend(loc="upper right", markerscale = 0.2)
        plt.connect("pick_event", self.hilight_map)


        time_text = f"t: {self.t} "
        if self.dt >= datetime.datetime(2000, 1, 1, 0, 0, 0):
            time_text += self.dt.strftime("%b %d, %H:%M:%S")
        ax1.text(x_min + 0.1*x_range, y_max - 0.1*y_range, time_text, ha = 'right', va = 'bottom')

        self.fig1 = fig1

        if do_save:
            fig1.savefig("outputs/animations/{}_{:03.0f}.jpg".format(fig_name, self.t), dpi=100)


        if do_show:
            plt.show()
        else:
            plt.close(fig1)

    def _calc_dist(self, yx_p, yx_b):
        # Simple function to calculate euclidean distance
        return np.sqrt((yx_p[0]-yx_b[0])**2 + (yx_p[1]-yx_b[1])**2)

    def _get_dist_to_buses(self, y_coords, x_coords, df_Buses_mesh):
        # Create 3d array to store distances from each point to each bus
        self.dist_to_bus = np.zeros((len(y_coords), len(x_coords), df_Buses_mesh.shape[0]))

        # Bus coordiantes
        bus_yx = zip(df_Buses_mesh["y"].to_numpy(), df_Buses_mesh["x"].to_numpy())

        with Pool(8) as p:
            # Use pooling to get all combinations of all coords /w each bus
            l_dist_to_bus = p.starmap(self._calc_dist, itertools.product(itertools.product(y_coords, x_coords), bus_yx))

        # Reshape list into 3d array
        self.dist_to_bus = np.reshape(l_dist_to_bus, (len(y_coords),
                                                      len(x_coords),
                                                      df_Buses_mesh.shape[0]  ))

        # Epsilon, if point is close enough to bus, show the true voltage /wout considering rbf
        self.bool_to_bus = self.dist_to_bus > 0.01

        # RBF from each distance to each point
        self.rbf_to_bus = np.exp(-(self.dist_to_bus*4)**2)

    def _get_colormesh(self, df_Buses, lims):
        x_min, x_max, y_min, y_max = lims
        x_coords = np.linspace(x_min, x_max, 32)
        y_coords = np.linspace(y_min, y_max, 32)

        # Order buses by x and y s.t. the order is not affected
        df_Buses_mesh = df_Buses.copy()
        df_Buses_mesh = df_Buses.sort_values(by=["x", "y"])
        bus_volts = np.array([np.mean(row) for row in df_Buses_mesh["puvoltsabs"].to_numpy()])
        bus_volts

        # Only compute distances adn rbf distance once since buses don't move
        if self.dist_to_bus is None:
            self._get_dist_to_buses(y_coords, x_coords, df_Buses_mesh)

        # Value of rbf controller of pmesh
        rbf_val = (1-bus_volts) * self.rbf_to_bus

        # Combined rbf value for all buses, masked by bool (ignore points very close to the buses)
        rbf_comb = (1.03* np.ones(self.dist_to_bus.shape[:1]) - np.sum(rbf_val, axis=2)) * self.bool_to_bus.all(axis=2)

        # Combined value of true voltages (only for points very close to buses)
        volt_comb = np.max(bus_volts * np.ones(self.dist_to_bus.shape) * ~self.bool_to_bus, axis= 2) # If two buses share the same xy coords, show max value

        # Sum the  two components
        c = rbf_comb + volt_comb

        return x_coords, y_coords, c
        

    def _get_segments(self):
        df_Lines = self.df_circuit[self.df_circuit["type"] == "line"].reset_index(drop = True)
        df_Lines["segments"] = df_Lines.apply(lambda row: ((row["x1"], row["y1"] ), (row["x2"], row["y2"])) , axis=1)

        # Colors
        df_Lines["puvoltsavg"] = df_Lines["puvoltsabs"].apply(lambda x: np.average(x))

        self.v_min = min(df_Lines["puvoltsavg"].min(), self.v_min)
        self.v_max = max(df_Lines["puvoltsavg"].max(), self.v_max)
        
        print(self.v_min, self.v_max)

        v_min = 0.995
        v_max = 1.04

        cmap_min = self.cmap_min 
        cmap_max = self.cmap_max
        cmap_delta = cmap_max - cmap_min

        df_Lines["segmentsColor"] = df_Lines["puvoltsavg"].apply(lambda x: (x - v_min)*(cmap_delta/(v_max - v_min)) + cmap_min)

        df_Lines["segmentsRGB"] = df_Lines["segmentsColor"].apply(lambda x: cm.get_cmap(self.cmap)(x))
        self.df_Lines = df_Lines 

        #print(df_Lines[["x1", "y1", "x2", "y2", "segments", "voltsavg", "segmentsColor", "segmentsRGB"]])

        
       
    def hilight_map(self, event):
        ind = event.ind[0]
        #print(f"event: artist: {event.artist}, canvas: {event.canvas}, guiEvent: {event.guiEvent}, ind: {event.ind}, mouseevent: {event.mouseevent}, name: {event.name} \n")

        row = self.df_Lines.iloc[ind]
        #print(row)
        self.mapselected.set_visible(True)        
        self.mapselected.set_data([row.x1, row.x2], [row.y1, row.y2])        
        self.fig1.canvas.draw()

    def view_profile(self, dictNodesPU):    
        fig2 = plt.figure()
        pax1 = fig2.add_subplot(1,1,1)

        df_Node1, segments1 = self._node_DF(dictNodesPU, 1)
        df_Node2, segments2 = self._node_DF(dictNodesPU, 2)
        df_Node3, segments3 = self._node_DF(dictNodesPU, 3)

        pax1.scatter(df_Node1["d_nodes"], df_Node1["vpu_nodes"], alpha = 0.5, color = "k", label = "Phase A")
        pax1.scatter(df_Node2["d_nodes"], df_Node2["vpu_nodes"], alpha = 0.5, color = "red", label = "Phase B")
        pax1.scatter(df_Node3["d_nodes"], df_Node3["vpu_nodes"], alpha = 0.5, color = "blue", label = "Phase C")
        
        #line_segments1 = LineCollection(segments1, linestyle = "solid", picker = 5, colors = "k")
        #line_segments2 = LineCollection(segments2, linestyle = "solid", picker = 5, colors = "red")
        #line_segments3 = LineCollection(segments3, linestyle = "solid", picker = 5, colors = "blue")

        #pax1.add_collection(line_segments1)
        #pax1.add_collection(line_segments2)
        #pax1.add_collection(line_segments3)

        pax1.legend(loc = "upper right")
        pax1.set_ylim([0.9, 1.1])
        pax1.set_xlabel("Distance from Energymeter")
        pax1.set_ylabel("Volts (pu)")
        pax1.set_title("Voltage Profile Plot")
        
        #plt.show()

    def _node_DF(self, dictNodesPU, nphase):
        dictNode = {}
        dictNode["vpu_nodes"]  = dictNodesPU[f"vpu_nodes{nphase}"]
        dictNode["d_nodes"]    = dictNodesPU[f"d_nodes{nphase}"]
        dictNode["name_nodes"] = dictNodesPU[f"name_nodes{nphase}"]
        df_Node = pd.DataFrame.from_dict(dictNode) 
        df_Node = df_Node.sort_values(by= ["d_nodes"], ascending = True).reset_index(drop = True)
        df_Node_cpy = df_Node.copy()

        segments = []
        print(df_Node.head())
        #for i, row in df_Node.iterrows():
        #    if row.vpu_nodes > 0:
        #        try: 
        #            distanceNode = df_Node[df_Node["d_nodes"] < row.d_nodes].iloc[0].d_nodes
        #            for j, row_after in df_Node[df_Node["d_nodes"] == distanceNode].iterrows():
        #                segments.append([[row_after.d_nodes, row_after.vpu_nodes], [row.d_nodes, row.vpu_nodes]])
        #        except Exception:
        #            pass
        print(self.df_circuit[self.df_circuit["type"] == "line"][["name", "buses"]].head)

        return df_Node, segments

    def plot_NodeTs(self, node_ts, t= None):
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        if t is None:
            t = np.linspace(0, 23, len(node_ts))
        print(t)
        print(node_ts)
        ax3.plot(t, node_ts)
        ax3.set_ylim(0.96, 1.04)

        dictNode = {"t": [], "v": []}
        dictNode["t"] = t
        dictNode["v"] = node_ts
        df_Node = pd.DataFrame.from_dict(dictNode)
        #df_Node.to_csv("outputs/37_BangBang.csv", index = False)

        plt.show()




