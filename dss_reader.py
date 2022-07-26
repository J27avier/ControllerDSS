import dss as dss_interface
import numpy as np 
import pandas as pd
import re


class Reader:
    def __init__(self, dss_circuit):
        self.dss_circuit = dss_circuit
        self.node_ts = []

    def read_propsBus(self, do_print = False, km=False):
        # Add pu's?
        dictBuses = {"idx": [],
                     "type": [], 
                     "name": [],
                     "nodes": [],
                     "kv": [],
                     "volts": [],
                     "puvolts": [],
                     "voltsabs": [],
                     "puvoltsabs": [],
                     "x": [],
                     "y": [],
                     "distance": [],
                    }

        busNames  = self.dss_circuit.AllBusNames
        n  = len(busNames)
        for i in range(n):
            ubus = self.dss_circuit.Buses(i)
            bName = ubus.Name
            bNodes = ubus.NumNodes
            bkv = ubus.kVBase

            bVoltsRaw = ubus.Voltages
            bPuVoltsRaw = ubus.puVoltages

            bVolts = np.array([complex(real, im) for real, im in zip(bVoltsRaw[::2], bVoltsRaw[1::2])])
            bPuVolts = np.array([complex(real, im) for real, im in zip(bPuVoltsRaw[::2], bPuVoltsRaw[1::2])])

            bVoltsAbs   = abs(bVolts)
            bPuVoltsAbs = abs(bPuVolts)

            if do_print:
                print(f"{i+1} {bName}; Phases: {bPhases}")
                print(f"\tProps \tkVBase {bkv}")
                print(f"\tVolts \t{np.around(bVolts,2)}\t{np.around(bVoltsAbs, 2)}")
                print()

            dictBuses["idx"].append(i)
            dictBuses["type"].append("bus")
            dictBuses["name"].append(bName)
            dictBuses["nodes"].append(bNodes)
            dictBuses["kv"].append(bkv)
            dictBuses["volts"].append(bVolts)
            dictBuses["puvolts"].append(bPuVolts)
            dictBuses["voltsabs"].append(bVoltsAbs)
            dictBuses["puvoltsabs"].append(bPuVoltsAbs)


            dictBuses["x"].append(ubus.x)
            dictBuses["y"].append(ubus.y)
            dictBuses["distance"].append(ubus.Distance)

        self.df_Buses = pd.DataFrame.from_dict(dictBuses)
        return self.df_Buses

    def read_propsCapa(self, do_print = False, km=False):
        capaIdx = self.dss_circuit.Capacitors.First
        
        dictCapa = {"idx": [],
                     "name": [],
                     "type": [],
                     "phases": [],
                     "kv": [],
                     # "kw": [],
                     # "kvar": [],
                     "volts": [],
                     "current": [],
                     "power": [],
                     "voltsabs": [],
                     "currentabs": [],
                     "powerabs": [],
                     "buses": [],
                     "x": [],
                     "y": [],
                     "distance": [],
                    }

        while capaIdx != 0:

            cName = self.dss_circuit.ActiveCktElement.Name
            cPhases = self.dss_circuit.ActiveCktElement.NumPhases
            cBus = self.dss_circuit.ActiveCktElement.BusNames

            # lkv = self.dss_circuit.Loads.kV
            # lkw = self.dss_circuit.Loads.kW
            # lkvar = self.dss_circuit.Loads.kvar
            ckv = self.dss_circuit.Capacitors.kV


            cVoltsRaw = self.dss_circuit.ActiveCktElement.Voltages
            cCurrentRaw = self.dss_circuit.ActiveCktElement.Currents
            cPowerRaw = self.dss_circuit.ActiveCktElement.Powers

            clampPhases = min(3, cPhases)
            cVolts = np.array([complex(real, im) for real, im in zip(cVoltsRaw[:2*clampPhases:2], cVoltsRaw[1:2*clampPhases:2])])
            cCurrent = np.array([complex(real, im) for real, im in zip(cCurrentRaw[:2*clampPhases:2], cCurrentRaw[1:2*clampPhases:2])])
            cPower = np.array([complex(real, im) for real, im in zip(cPowerRaw[:2*clampPhases:2], cPowerRaw[1:2*clampPhases:2])])

            cVoltsAbs   = abs(cVolts)
            cCurrentAbs = abs(cCurrent)
            cPowerAbs   = abs(cPower)

            #Properties from bus
            ubus = self.dss_circuit.Buses(re.sub(r"\..*", "", cBus[0]))

            
            if do_print:
                print(f"{capaIdx} {cName}; Phases: {cPhases}")
                print(f"\tProps \tkV {ckv}")
                print(f"\tVolts \t{np.around(cVolts,2)}\t{np.around(cVoltsAbs, 2)}")
                print(f"\tCurrent\t{np.around(cCurrent,2)}\t{np.around(cCurrentAbs, 2)}")
                print(f"\tPower \t{np.around(cPower,2)}\t{np.around(cPowerAbs, 2)}")
                print(f"\tBuses \t{cBus}")
                print()
            
            dictCapa["idx"].append(capaIdx)
            dictCapa["type"].append("capacitor")
            dictCapa["name"].append(cName)
            dictCapa["phases"].append(cPhases)

            dictCapa["kv"].append(ckv)

            dictCapa["volts"].append(cVolts)
            dictCapa["current"].append(cCurrent)
            dictCapa["power"].append(cPower)

            dictCapa["voltsabs"].append(cVoltsAbs)
            dictCapa["currentabs"].append(cCurrentAbs)
            dictCapa["powerabs"].append(cPowerAbs)
            dictCapa["buses"].append(cBus)

            dictCapa["x"].append(ubus.x)
            dictCapa["y"].append(ubus.y)
            dictCapa["distance"].append(ubus.Distance)

            capaIdx = self.dss_circuit.Capacitors.Next

        self.df_Capas = pd.DataFrame.from_dict(dictCapa)
        return self.df_Capas

    def read_propsTran(self, do_print = False, km=False):
        tranIdx = self.dss_circuit.Transformers.First
        
        dictTrans = {"idx": [],
                     "name": [],
                     "type": [],
                     "phases": [],
                     "kv": [],
                     # "kw": [],
                     # "kvar": [],
                     "volts": [],
                     "current": [],
                     "power": [],
                     "voltsabs": [],
                     "currentabs": [],
                     "powerabs": [],
                     "buses": [],
                     "x": [],
                     "y": [],
                     "distance": [],
                    }

        while tranIdx != 0:

            tName = self.dss_circuit.ActiveCktElement.Name
            tPhases = self.dss_circuit.ActiveCktElement.NumPhases
            tBus = self.dss_circuit.ActiveCktElement.BusNames

            # lkv = self.dss_circuit.Loads.kV
            # lkw = self.dss_circuit.Loads.kW
            # lkvar = self.dss_circuit.Loads.kvar
            tkv = self.dss_circuit.Transformers.kV


            tVoltsRaw = self.dss_circuit.ActiveCktElement.Voltages
            tCurrentRaw = self.dss_circuit.ActiveCktElement.Currents
            tPowerRaw = self.dss_circuit.ActiveCktElement.Powers

            clampPhases = min(3, tPhases)
            tVolts = np.array([complex(real, im) for real, im in zip(tVoltsRaw[:2*clampPhases:2], tVoltsRaw[1:2*clampPhases:2])])
            tCurrent = np.array([complex(real, im) for real, im in zip(tCurrentRaw[:2*clampPhases:2], tCurrentRaw[1:2*clampPhases:2])])
            tPower = np.array([complex(real, im) for real, im in zip(tPowerRaw[:2*clampPhases:2], tPowerRaw[1:2*clampPhases:2])])

            tVoltsAbs   = abs(tVolts)
            tCurrentAbs = abs(tCurrent)
            tPowerAbs   = abs(tPower)

            #Properties from bus
            ubus = self.dss_circuit.Buses(re.sub(r"\..*", "", tBus[0]))

            
            if do_print:
                print(f"{tranIdx} {tName}; Phases: {tPhases}")
                print(f"\tProps \tkV {tkv}")
                print(f"\tVolts \t{np.around(tVolts,2)}\t{np.around(tVoltsAbs, 2)}")
                print(f"\tCurrent\t{np.around(tCurrent,2)}\t{np.around(tCurrentAbs, 2)}")
                print(f"\tPower \t{np.around(tPower,2)}\t{np.around(tPowerAbs, 2)}")
                print(f"\tBuses \t{tBus}")
                print()
            
            dictTrans["idx"].append(tranIdx)
            dictTrans["type"].append("transformer")
            dictTrans["name"].append(tName)
            dictTrans["phases"].append(tPhases)

            dictTrans["kv"].append(tkv)

            dictTrans["volts"].append(tVolts)
            dictTrans["current"].append(tCurrent)
            dictTrans["power"].append(tPower)

            dictTrans["voltsabs"].append(tVoltsAbs)
            dictTrans["currentabs"].append(tCurrentAbs)
            dictTrans["powerabs"].append(tPowerAbs)
            dictTrans["buses"].append(tBus)

            dictTrans["x"].append(ubus.x)
            dictTrans["y"].append(ubus.y)
            dictTrans["distance"].append(ubus.Distance)

            tranIdx = self.dss_circuit.Transformers.Next

        self.df_Trans = pd.DataFrame.from_dict(dictTrans)
        return self.df_Trans

    def read_propsLoad(self, do_print = False, km=False):
        loadIdx = self.dss_circuit.Loads.First
        dictLoads = {"idx": [],
                     "name": [],
                     "type": [],
                     "phases": [],
                     "kv": [],
                     "kw": [],
                     "kvar": [],
                     "volts": [],
                     "current": [],
                     "power": [],
                     "voltsabs": [],
                     "currentabs": [],
                     "powerabs": [],
                     "buses": [],
                     "x": [],
                     "y": [],
                     "distance": [],
                    }

        while loadIdx != 0:
            lName = self.dss_circuit.ActiveCktElement.Name
            lPhases = self.dss_circuit.ActiveCktElement.NumPhases
            lBus = self.dss_circuit.ActiveCktElement.BusNames

            lkv = self.dss_circuit.Loads.kV
            lkw = self.dss_circuit.Loads.kW
            lkvar = self.dss_circuit.Loads.kvar


            lVoltsRaw = self.dss_circuit.ActiveCktElement.Voltages
            lCurrentRaw = self.dss_circuit.ActiveCktElement.Currents
            lPowerRaw = self.dss_circuit.ActiveCktElement.Powers

            clampPhases = min(3, lPhases)
            lVolts = np.array([complex(real, im) for real, im in zip(lVoltsRaw[:2*clampPhases:2], lVoltsRaw[1:2*clampPhases:2])])
            lCurrent = np.array([complex(real, im) for real, im in zip(lCurrentRaw[:2*clampPhases:2], lCurrentRaw[1:2*clampPhases:2])])
            lPower = np.array([complex(real, im) for real, im in zip(lPowerRaw[:2*clampPhases:2], lPowerRaw[1:2*clampPhases:2])])

            lVoltsAbs   = abs(lVolts)
            lCurrentAbs = abs(lCurrent)
            lPowerAbs   = abs(lPower)

            #Properties from bus
            ubus = self.dss_circuit.Buses(re.sub(r"\..*", "", lBus[0]))

            
            if do_print:
                print(f"{loadIdx} {lName}; Phases: {lPhases}")
                print(f"\tProps \tkV {lkv}\tkW {lkw}\tkvar {lkvar}")
                print(f"\tVolts \t{np.around(lVolts,2)}\t{np.around(lVoltsAbs, 2)}")
                print(f"\tCurrent\t{np.around(lCurrent,2)}\t{np.around(lCurrentAbs, 2)}")
                print(f"\tPower \t{np.around(lPower,2)}\t{np.around(lPowerAbs, 2)}")
                print(f"\tBuses \t{lBus}")
                print()
            
            dictLoads["idx"].append(loadIdx)
            dictLoads["type"].append("load")
            dictLoads["name"].append(lName)
            dictLoads["phases"].append(lPhases)

            dictLoads["kv"].append(lkv)
            dictLoads["kw"].append(lkw)
            dictLoads["kvar"].append(lkvar)

            dictLoads["volts"].append(lVolts)
            dictLoads["current"].append(lCurrent)
            dictLoads["power"].append(lPower)

            dictLoads["voltsabs"].append(lVoltsAbs)
            dictLoads["currentabs"].append(lCurrentAbs)
            dictLoads["powerabs"].append(lPowerAbs)
            dictLoads["buses"].append(lBus)

            dictLoads["x"].append(ubus.x)
            dictLoads["y"].append(ubus.y)
            dictLoads["distance"].append(ubus.Distance)

            loadIdx = self.dss_circuit.Loads.Next

        self.df_Loads = pd.DataFrame.from_dict(dictLoads)
        return self.df_Loads

    def read_propsLine(self, do_print = False, km=False):
        lineIdx = self.dss_circuit.Lines.First
        dictLines = {"idx": [],
                     "name": [],
                     "type": [],
                     "phases": [],
                     "kv": [],
                     "kw": [],
                     "kvar": [],
                     "volts": [],
                     "puvolts": [],
                     "current": [],
                     "power": [],
                     "voltsabs": [],
                     "puvoltsabs": [],
                     "currentabs": [],
                     "powerabs": [],
                     "buses": [],
                     "x1": [],
                     "y1": [],
                     "x2": [],
                     "y2": [],
                     "distance": [],
                    }

        while lineIdx != 0:
            lName = self.dss_circuit.ActiveCktElement.Name
            lPhases = self.dss_circuit.ActiveCktElement.NumPhases
            

            lBus = self.dss_circuit.ActiveCktElement.BusNames

            lkv = self.dss_circuit.Loads.kV
            lkw = self.dss_circuit.Loads.kW
            lkvar = self.dss_circuit.Loads.kvar


            lVoltsRaw = self.dss_circuit.ActiveCktElement.Voltages
            lCurrentRaw = self.dss_circuit.ActiveCktElement.Currents
            lPowerRaw = self.dss_circuit.ActiveCktElement.Powers

            clampPhases = min(3, lPhases)
            lVolts = np.array([complex(real, im) for real, im in zip(lVoltsRaw[:2*clampPhases:2], lVoltsRaw[1:2*clampPhases:2])])
            lCurrent = np.array([complex(real, im) for real, im in zip(lCurrentRaw[:2*clampPhases:2], lCurrentRaw[1:2*clampPhases:2])])
            lPower = np.array([complex(real, im) for real, im in zip(lPowerRaw[:2*clampPhases:2], lPowerRaw[1:2*clampPhases:2])])
            lPuVolts = np.array([complex(real, im) for real, im in zip()])

            lVoltsAbs   = abs(lVolts)
            lCurrentAbs = abs(lCurrent)
            lPowerAbs   = abs(lPower)

            #Properties from bus
            ubus1 = self.dss_circuit.Buses(re.sub(r"\..*", "", lBus[0]))
            lx1 = ubus1.x
            ly1 = ubus1.y

            ubus2 = self.dss_circuit.Buses(re.sub(r"\..*", "", lBus[1]))
            lx2 = ubus2.x
            ly2 = ubus2.y
            ldistance = ubus2.Distance

            # Pu voltages
            lPuVolts = lVolts / (1000 * ubus2.kVBase)
            lPuVoltsAbs = abs(lPuVolts)

            if do_print:
                print(f"{lineIdx} {lName}; Phases: {lPhases}")
                print(f"\tProps \tkV {lkv}\tkW {lkw}\tkvar {lkvar}")
                print(f"\tVolts \t{np.around(lVolts,2)}\t{np.around(lVoltsAbs, 2)}")
                print(f"\tCurrent\t{np.around(lCurrent,2)}\t{np.around(lCurrentAbs, 2)}")
                print(f"\tPower \t{np.around(lPower,2)}\t{np.around(lPowerAbs, 2)}")
                print(f"\tBuses \t{lBus}")
                print()
            

            dictLines["idx"].append(lineIdx)
            dictLines["name"].append(lName)
            dictLines["type"].append("line")
            dictLines["phases"].append(lPhases)
            dictLines["kv"].append(lkv)
            dictLines["kw"].append(lkw)
            dictLines["kvar"].append(lkvar)
            dictLines["volts"].append(lVolts)
            dictLines["puvolts"].append(lPuVolts)
            dictLines["current"].append(lCurrent)
            dictLines["power"].append(lPower)
            dictLines["voltsabs"].append(lVoltsAbs)
            dictLines["puvoltsabs"].append(lPuVoltsAbs)
            dictLines["currentabs"].append(lCurrentAbs)
            dictLines["powerabs"].append(lPowerAbs)
            dictLines["buses"].append(lBus)
            dictLines["x1"].append(lx1)
            dictLines["y1"].append(ly1)
            dictLines["x2"].append(lx2)
            dictLines["y2"].append(ly2)
            dictLines["distance"].append(ldistance)

            lineIdx = self.dss_circuit.Lines.Next

        self.df_Lines = pd.DataFrame.from_dict(dictLines)
        return self.df_Lines

    def read_circuit(self):
        df_Buses = self.read_propsBus()
        df_Loads = self.read_propsLoad()
        df_Lines = self.read_propsLine()
        df_Trans = self.read_propsTran()
        df_Capas = self.read_propsCapa()
        self.df_circuit = pd.concat([df_Buses, df_Loads, df_Lines, df_Trans, df_Capas], ignore_index = True)
        return self.df_circuit

    def read_nodesPU(self):
        vpu_nodes1 = self.dss_circuit.AllNodeVmagPUByPhase(1)
        vpu_nodes2 = self.dss_circuit.AllNodeVmagPUByPhase(2)
        vpu_nodes3 = self.dss_circuit.AllNodeVmagPUByPhase(3)

        d_nodes1 = self.dss_circuit.AllNodeDistancesByPhase(1)
        d_nodes2 = self.dss_circuit.AllNodeDistancesByPhase(2)
        d_nodes3 = self.dss_circuit.AllNodeDistancesByPhase(3)

        name_nodes1 = self.dss_circuit.AllNodeNamesByPhase(1)
        name_nodes2 = self.dss_circuit.AllNodeNamesByPhase(2)
        name_nodes3 = self.dss_circuit.AllNodeNamesByPhase(3)
        
        dictNodesPU = {}
        dictNodesPU["vpu_nodes1"] = vpu_nodes1
        dictNodesPU["vpu_nodes2"] = vpu_nodes2
        dictNodesPU["vpu_nodes3"] = vpu_nodes3

        dictNodesPU["d_nodes1"] = d_nodes1
        dictNodesPU["d_nodes2"] = d_nodes2
        dictNodesPU["d_nodes3"] = d_nodes3

        dictNodesPU["name_nodes1"] = name_nodes1
        dictNodesPU["name_nodes2"] = name_nodes2
        dictNodesPU["name_nodes3"] = name_nodes3

        return dictNodesPU

    def read_NodeT(self, node, phase):
        vpu_node = self.dss_circuit.AllNodeVmagPUByPhase(phase)
        name_nodes = self.dss_circuit.AllNodeNamesByPhase(phase)
        #print(vpu_node, len(vpu_node))
        #print(name_nodes, len(name_nodes))

        idx = name_nodes.index(node)
        val = vpu_node[idx]
        self.node_ts.append(val)
        
        
if __name__ == '__main__':
    
    # Secret code for YCM
    if False:
        import os
        import pathlib
        dss = dss_interface.DSS
        dss.Start(0)

        path_proj = os.path.dirname(os.path.abspath(__file__))
        file_13Bus = pathlib.Path(path_proj).joinpath("13Bus", "IEEE13Nodeckt.dss")

        dss_text = dss.Text
        dss_circuit = dss.ActiveCircuit
