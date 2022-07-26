import dss as dss_interface
from citylearn_energymodels import Battery
import numpy as np 
import pandas as pd

class BatteryController:
    def __init__(self, bus=None, ts=1):
        self.battery = Battery(capacity = 1820,
                               nominal_power= 910,
                               ts = 0.25,
                               capacity_loss_coef =1e-5,
                               efficiency=0.9,
                               loss_coef=0,
                               power_efficiency_curve=[[0,0.83],[0.3,0.83],[0.7,0.9],[0.8,0.9],[1,0.85]],
                               capacity_power_curve=[[0.0 , 1],[0.8, 1],[1.0 ,0.2]],
                               save_memory = False) 
        self.bus = bus
        self.energy_balance = 0
        self.ts = ts

    def step(self, observation):
        set_point = 1.018
        epsilon = 0.001
        if observation is None:
            observation = set_point
        if set_point - epsilon <= observation <= set_point + epsilon:
            energy = 0 # Positive for energy entering storage
        elif observation < set_point - epsilon:
            energy = -250
        elif set_point + epsilon < observation:
            energy = 250

        energy_balance = self.battery.charge(energy)
        
        power = energy_balance / self.ts

        return power

if __name__ == "__main__":
    battery = Battery(capacity = 2000,
                      nominal_power= 1000,
                      ts = 0.25,
                      capacity_loss_coef =1e-5,
                      efficiency=0.9,
                      loss_coef=0,
                      power_efficiency_curve=[[0,0.83],[0.3,0.83],[0.7,0.9],[0.8,0.9],[1,0.85]],
                      capacity_power_curve=[[0.0 , 1],[0.8, 1],[1.0 ,0.2]],
                      save_memory = False) 
    print(battery.str_state())
    battery.charge(100, trying=True)
    print(battery.str_state())
    battery.charge(100)
    print(battery.str_state())
    battery.charge(-60)
    print(battery.str_state())
    battery.charge(-200)
    print(battery.str_state())
    battery.charge(-200)
    print(battery.str_state())
    battery.charge(250)
    print(battery.str_state())
    battery.charge(250)
    print(battery.str_state())
    battery.charge(250)
    print(battery.str_state())
    battery.charge(250)
    print(battery.str_state())
    battery.charge(250)
    battery.charge(250)
    battery.charge(250)
    battery.charge(250)
    battery.charge(250)
    battery.charge(250)
    battery.charge(250)
    print(battery.str_state())
