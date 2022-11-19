# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:03:27 2022

@author: Publico
"""
import visa 
import time
import datetime
import numpy as np 
from scipy import optimize
import matplotlib.pyplot as plt
import os
from IPython import get_ipython
from scipy.optimize import curve_fit
import pandas as pd
rm = visa.ResourceManager()


resource_name = 'GPIB0::24::INSTR'

mult = rm.open_resource(resource_name)
print(mult.query('*IDN?'))
dc = mult.query('MEASURE:RESistance?')
print(dc)


# otras cosas que se pueden medir
#  MEASure
#   :VOLTage:DC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :VOLTage:DC:RATio? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :VOLTage:AC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :CURRent:DC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :CURRent:AC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :RESistance? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :FRESistance? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :FREQuency? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :PERiod? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :CONTinuity?
#   :DIODe?



#%%

DEBUG = True

medirP = False
medirR = True
#%%
meds = []
i = 0
try:
    while True:
        if medirP:
            p = man.GetPressure()
            tP = time.time() #Tiempo de presión
        if medirR:
            value = mult.query('MEASURE:RESistance?')
            tR = time.time() #Tiempo de resistencia

        if medirP and medirR: #Medir las dos al mismo tiempo.
            meds.append({"i": i,"t_P": tP, "P": p, "t_R": tR,"R": value})
            if DEBUG: print(f"i: {i}, t_P: {tP}, P: {p}, t_R: {tR}, R: {value}")
        elif medirP: #Medir solamente P
            meds.append({"i": i,"t_P": tP, "P": p})
            if DEBUG: print(f"i: {i}, t_P: {tP}, P: {p}")
        elif medirR: #Medir solamente R.
            meds.append({"i": i, "t_v": tR,"Resistencia(Ohms)": value})
            if DEBUG: print(f"i: {i}, t_v: {tR}, R(Kohm): {value}")

except KeyboardInterrupt:
    df = pd.DataFrame(meds)
    filename = input("Ingrese el nombre del archivo")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    df.to_csv(filename + current_time + ".csv", index = False)
    

#%%
mult.close()
# otras cosas que se pueden medir
#  MEASure
#   :VOLTage:DC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :VOLTage:DC:RATio? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :VOLTage:AC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :CURRent:DC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :CURRent:AC? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :RESistance? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :FRESistance? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :FREQuency? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :PERiod? {<range>|MIN|MAX|DEF},{<resolution>|MIN|MAX|DEF}
#   :CONTinuity?
#   :DIODe?