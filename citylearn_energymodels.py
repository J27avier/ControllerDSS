from gym import spaces
import numpy as np


class Battery:
    def __init__(self, capacity, nominal_power = None, ts=1, capacity_loss_coef = None, power_efficiency_curve = None, capacity_power_curve = None, efficiency = None, loss_coef = 0, save_memory = True):
        """
        Generic energy storage class. It can be used as a chilled water storage tank or a DHW storage tank
        Args:
            capacity (float): Maximum amount of energy that the storage unit is able to store (kWh)
            max_power_charging (float): Maximum amount of power that the storage unit can use to charge (kW)
            efficiency (float): Efficiency factor of charging and discharging the storage unit (from 0 to 1)
            loss_coef (float): Loss coefficient used to calculate the amount of energy lost every hour (from 0 to 1)
            power_efficiency_curve (float): Charging/Discharging efficiency as a function of the power released or consumed
            capacity_power_curve (float): Max. power of the battery as a function of its current state of charge
            capacity_loss_coef (float): Battery degradation. Storage capacity lost in each charge and discharge cycle (as a fraction of the total capacity)
        """
            
        self.ts = ts
        self.capacity = capacity
        self.c0 = capacity
        self.nominal_power = nominal_power
        self.capacity_loss_coef = capacity_loss_coef*self.ts
        
        if power_efficiency_curve is not None:
            self.power_efficiency_curve = np.array(power_efficiency_curve).T
        else:
            self.power_efficiency_curve = power_efficiency_curve
            
        if capacity_power_curve is not None:
            self.capacity_power_curve = np.array(capacity_power_curve).T
        else:
            self.capacity_power_curve = capacity_power_curve
            
        self.efficiency = efficiency**0.5
        self.loss_coef = loss_coef*self.ts
        self.max_power = None
        self._eff = []
        self._energy = []
        self._max_power = []
        self.soc = []
        self._soc = 0 # State of Charge
        self.energy_balance = []
        self._energy_balance = 0
        self.save_memory = save_memory
        
    def terminate(self):
        if self.save_memory == False:
            self.energy_balance = np.array(self.energy_balance)
            self.soc =  np.array(self.soc)
        
    def charge(self, energy, trying = False):
        """Method that controls both the energy CHARGE and DISCHARGE of the energy storage device
        energy < 0 -> Discharge
        energy > 0 -> Charge
        Args:
            energy (float): Amount of energy stored in that time-step (Wh)
        Return:
            energy_balance (float): 
        """
        
        #The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        soc_init = self._soc*(1-self.loss_coef)
        if self.capacity_power_curve is not None:
            soc_normalized = soc_init/self.capacity
            # Calculating the maximum power rate at which the battery can be charged or discharged
            idx = max(0, np.argmax(soc_normalized <= self.capacity_power_curve[0]) - 1)
            
            # Linear picewise interpolation?
            self.max_power = self.nominal_power*(self.capacity_power_curve[1][idx] + (self.capacity_power_curve[1][idx+1] - self.capacity_power_curve[1][idx]) * (soc_normalized - self.capacity_power_curve[0][idx])/(self.capacity_power_curve[0][idx+1] - self.capacity_power_curve[0][idx]))
        
        else:
            self.max_power = self.nominal_power
          
        #Charging    
        if energy >= 0:
            if self.nominal_power is not None:
                
                energy =  min(energy, self.max_power*self.ts)
                if self.power_efficiency_curve is not None:
                    # Calculating the maximum power rate at which the battery can be charged or discharged
                    energy_normalized = np.abs(energy)/self.nominal_power
                    idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
                    self.efficiency = self.power_efficiency_curve[1][idx] + (energy_normalized - self.power_efficiency_curve[0][idx])*(self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx])/(self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])
                    self.efficiency = self.efficiency**0.5
                 
            temp_soc = soc_init + energy*self.efficiency
            
        #Discharging
        else:
            if self.nominal_power is not None:
                energy = max(-self.max_power, energy)
                
            if self.power_efficiency_curve is not None:
                
                # Calculating the maximum power rate at which the battery can be charged or discharged
                energy_normalized = np.abs(energy)/self.nominal_power
                idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
                self.efficiency = self.power_efficiency_curve[1][idx] + (energy_normalized - self.power_efficiency_curve[0][idx])*(self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx])/(self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])
                self.efficiency = self.efficiency**0.5
                    
            temp_soc = max(0, soc_init + energy/self.efficiency)
            
        if self.capacity is not None:
            temp_soc = min(temp_soc, self.capacity)
          
        if not trying:
            self._soc = temp_soc
        # Calculating the energy balance with its external environment (amount of energy taken from or relseased to the environment)
        
        #Charging    
        if energy >= 0:
            temp_energy_balance = (temp_soc - soc_init)/self.efficiency
            
        #Discharging
        else:
            temp_energy_balance = (temp_soc - soc_init)*self.efficiency
            
        # Calculating the degradation of the battery: new max. capacity of the battery after charge/discharge       
        if not trying:
            self.capacity -= self.capacity_loss_coef*self.c0*np.abs(temp_energy_balance)/(2*self.capacity)

        
        if self.save_memory == False and not trying:
            self.energy_balance.append(np.float32(self._energy_balance))
            self.soc.append(np.float32(self._soc))

        if not trying:
            self._energy_balance = temp_energy_balance
            
        return temp_energy_balance
    
    def reset(self):
        self.soc = []
        self._soc = 0 #State of charge
        self.energy_balance = [] #Positive for energy entering the storage
        self._energy_balance = 0
        self.time_step = 0

    def str_state(self):
        return f"soc={self._soc}, energy_balance={self._energy_balance}, capacity={self.capacity}"

if __name__ == "__main__":
    battery = Battery(capacity = 1820,
                      nominal_power= 910,
                      ts = 0.25,
                      capacity_loss_coef =1e-5,
                      efficiency=0.9,
                      loss_coef=0,
                      power_efficiency_curve=[[0,0.83],[0.3,0.83],[0.7,0.9],[0.8,0.9],[1,0.85]],
                      capacity_power_curve=[[0.0 , 1],[0.8, 1],[1.0 ,0.2]],
                      save_memory = False) 
    print(battery.capacity)

