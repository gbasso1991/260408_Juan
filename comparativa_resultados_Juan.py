#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from clase_resultados import ResultadosESAR
#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% Obtengo ciclos y resultados para cada concentracion - Todo a 300 kHz
label_0='1209_0'
ciclos_0 = glob("1_1209_0/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_0.sort()
resultados_0 = glob("1_1209_0/**/*resultados.txt",recursive=True)
resultados_0.sort()
conc_0 = 4.6 #g/L

label_10='1209_10'
ciclos_10 = glob("2_1209_10/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_10.sort()
resultados_10 = glob("2_1209_10/**/*resultados.txt",recursive=True)
resultados_10.sort()
conc_10 = 6.2 #g/L

label_60='1209_60'
ciclos_60 = glob("3_1209_60/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_60.sort()
resultados_60 = glob("3_1209_60/**/*resultados.txt",recursive=True)
resultados_60.sort()
conc_60 = 6.6 #g/L

label_120='1209_120'
ciclos_120 = glob("4_1209_120/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_120.sort()
resultados_120 = glob("4_1209_120/**/*resultados.txt",recursive=True)
resultados_120.sort()
conc_120 = 6.2 #g/L

label_360='1209_360'
ciclos_360 = glob("5_1209_360/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_360.sort()
resultados_360 = glob("5_1209_360/**/*resultados.txt",recursive=True)
resultados_360.sort()
conc_360 = 8.2 #g/L

#%% Ploteo Ciclos Promedio 
# 00
fig00, ((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
for i,e in enumerate(ciclos_0):
    if '060dA' in e:
        _,_,_, H_00,M_00,_ = lector_ciclos(ciclos_0[i])
        a.plot(H_00/1000,M_00,'-',label=f'NF{i}')

a2.set_title('34.7 kA/m',loc='left')
for i,e in enumerate(ciclos_0):
    if '090dA' in e:
        _,_,_, H_01,M_01,_ = lector_ciclos(ciclos_0[i])
        a2.plot(H_01/1000,M_01,'-',label=f'NF{i}')

a3.set_title('45.9 kA/m',loc='left')
for i,e in enumerate(ciclos_0):
    if '120dA' in e:
        _,_,_, H_02,M_02,_ = lector_ciclos(ciclos_0[i])
        a3.plot(H_02/1000,M_02,'-',label=f'NF{i}')

a4.set_title('58 kA/m',loc='left')
for i,e in enumerate(ciclos_0):
    if '152dA' in e:
        _,_,_, H_03,M_03,_ = lector_ciclos(ciclos_0[i])
        a4.plot(H_03/1000,M_03,'-',label=f'NF{i}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='upper left')

a.set_ylabel('M (A/m)')
a3.set_ylabel('M (A/m)')
a3.set_xlabel('H (kA/m)')
a4.set_xlabel('H (kA/m)')

plt.suptitle(f'Comparativa ciclos promedio {label_0}\n300 kHz')
plt.savefig('0_ciclos_promedio_120900_24_35_46_58.png',dpi=300)
#%% 10
fig01, ((a,a2),(a3,a4)) = plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)
a.set_title('24.6 kA/m',loc='left')
for i,e in enumerate(ciclos_10):
    if '060dA' in e:
        _,_,_, H_10,M_10,_ = lector_ciclos(ciclos_10[i])
        a.plot(H_10/1000,M_10,'-',label=f'NF{i}')

a2.set_title('34.7 kA/m',loc='left')
for i,e in enumerate(ciclos_10):
    if '090dA' in e:
        _,_,_, H_11,M_11,_ = lector_ciclos(ciclos_10[i])
        a2.plot(H_11/1000,M_11,'-',label=f'NF{i}')

a3.set_title('45.9 kA/m',loc='left')
for i,e in enumerate(ciclos_10):
    if '120dA' in e:
        _,_,_, H_12,M_12,_ = lector_ciclos(ciclos_10[i])
        a3.plot(H_12/1000,M_12,'-',label=f'NF{i}')

a4.set_title('58 kA/m',loc='left')
for i,e in enumerate(ciclos_10):
    if '152dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_10[i])
        a4.plot(H_13/1000,M_13,'-',label=f'NF{i}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='upper left')

a.set_ylabel('M (A/m)')
a3.set_ylabel('M (A/m)')
a3.set_xlabel('H (kA/m)')
a4.set_xlabel('H (kA/m)')

plt.suptitle(f'Comparativa ciclos promedio {label_10}\n300 kHz')
plt.savefig('0_ciclos_promedio_120910_24_35_46_58.png',dpi=300)

#%% 60
fig02, ((a,a2),(a3,a4)) = plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
for i,e in enumerate(ciclos_60):
    if '060dA' in e:
        _,_,_, H_20,M_20,_ = lector_ciclos(ciclos_60[i])
        a.plot(H_20/1000,M_20,'-',label=f'NF{i}')

a2.set_title('34.7 kA/m',loc='left')
for i,e in enumerate(ciclos_60):
    if '090dA' in e:
        _,_,_, H_21,M_21,_ = lector_ciclos(ciclos_60[i])
        a2.plot(H_21/1000,M_21,'-',label=f'NF{i}')

a3.set_title('45.9 kA/m',loc='left')
for i,e in enumerate(ciclos_60):
    if '120dA' in e:
        _,_,_, H_22,M_22,_ = lector_ciclos(ciclos_60[i])
        a3.plot(H_22/1000,M_22,'-',label=f'NF{i}')

a4.set_title('58 kA/m',loc='left')
for i,e in enumerate(ciclos_60):
    if '152dA' in e:
        _,_,_, H_23,M_23,_ = lector_ciclos(ciclos_60[i])
        a4.plot(H_23/1000,M_23,'-',label=f'NF{i}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='upper left')

a.set_ylabel('M (A/m)')
a3.set_ylabel('M (A/m)')
a3.set_xlabel('H (kA/m)')
a4.set_xlabel('H (kA/m)')

plt.suptitle(f'Comparativa ciclos promedio {label_60}\n300 kHz')
plt.savefig('0_ciclos_promedio_120960_24_35_46_58.png',dpi=300)
#%% 120
fig03, ((a,a2),(a3,a4)) = plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
for i,e in enumerate(ciclos_120):
    if '060dA' in e:
        _,_,_, H_30,M_30,_ = lector_ciclos(ciclos_120[i])
        a.plot(H_30/1000,M_30,'-',label=f'NF{i}')

a2.set_title('34.7 kA/m',loc='left')
for i,e in enumerate(ciclos_120):
    if '090dA' in e:
        _,_,_, H_31,M_31,_ = lector_ciclos(ciclos_120[i])
        a2.plot(H_31/1000,M_31,'-',label=f'NF{i}')

a3.set_title('45.9 kA/m',loc='left')
for i,e in enumerate(ciclos_120):
    if '120dA' in e:
        _,_,_, H_32,M_32,_ = lector_ciclos(ciclos_120[i])
        a3.plot(H_32/1000,M_32,'-',label=f'NF{i}')

a4.set_title('58 kA/m',loc='left')
for i,e in enumerate(ciclos_120):
    if '152dA' in e:
        _,_,_, H_33,M_33,_ = lector_ciclos(ciclos_120[i])
        a4.plot(H_33/1000,M_33,'-',label=f'NF{i}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='upper left')

a.set_ylabel('M (A/m)')
a3.set_ylabel('M (A/m)')
a3.set_xlabel('H (kA/m)')
a4.set_xlabel('H (kA/m)')

plt.suptitle(f'Comparativa ciclos promedio {label_120}\n300 kHz')
plt.savefig('0_ciclos_promedio_1209120_24_35_46_58.png',dpi=300)
#%% 360
fig04, ((a,a2),(a3,a4)) = plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
for i,e in enumerate(ciclos_360):
    if '060dA' in e:
        _,_,_, H_40,M_40,_ = lector_ciclos(ciclos_360[i])
        a.plot(H_40/1000,M_40,'-',label=f'NF{i}')

a2.set_title('34.7 kA/m',loc='left')
for i,e in enumerate(ciclos_360):
    if '090dA' in e:
        _,_,_, H_41,M_41,_ = lector_ciclos(ciclos_360[i])
        a2.plot(H_41/1000,M_41,'-',label=f'NF{i}')

a3.set_title('45.9 kA/m',loc='left')
for i,e in enumerate(ciclos_360):
    if '120dA' in e:
        _,_,_, H_42,M_42,_ = lector_ciclos(ciclos_360[i])
        a3.plot(H_42/1000,M_42,'-',label=f'NF{i}')

a4.set_title('58 kA/m',loc='left')
for i,e in enumerate(ciclos_360):
    if '152dA' in e:
        _,_,_, H_43,M_43,_ = lector_ciclos(ciclos_360[i])
        a4.plot(H_43/1000,M_43,'-',label=f'NF{i}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='upper left')

a.set_ylabel('M (A/m)')
a3.set_ylabel('M (A/m)')
a3.set_xlabel('H (kA/m)')
a4.set_xlabel('H (kA/m)')

plt.suptitle(f'Comparativa ciclos promedio {label_360}\n300 kHz')
plt.savefig('0_ciclos_promedio_1209360_24_35_46_58.png',dpi=300)

#%% Listas con Resultados

res_00, res_10,res_60, res_120, res_360=[],[],[],[],[]
print('Resultados', '='*80,'\n')
for r in resultados_0:
    res_00.append(ResultadosESAR(os.path.dirname(r)))
for r in resultados_10:
    res_10.append(ResultadosESAR(os.path.dirname(r)))
for r in resultados_60:
    res_60.append(ResultadosESAR(os.path.dirname(r)))
for r in resultados_120:
    res_120.append(ResultadosESAR(os.path.dirname(r)))
for r in resultados_360:
    res_360.append(ResultadosESAR(os.path.dirname(r)))


#%% 2 - Tau vs time / Temp
#% 13 hs

fig210, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(12,8),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_13):
    if '38_100' in r.directorio:
        ax.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '47_125' in r.directorio:
        ax2.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '57_150' in r.directorio:
        ax3.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('τ (ns)')
ax.set_xlim(0,)

ax3.set_xlabel('t (s)')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  
plt.suptitle(f'tau vs time\nNF 13 hs - {conc_13:0.1f} g/L' )

fig211, (ax,ax2,ax3) = plt.subplots(3,1,figsize=(12,8),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_13):
    if '38_100' in r.directorio:
        ax.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '47_125' in r.directorio:
        ax2.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '57_150' in r.directorio:
        ax3.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('τ (ns)')
ax3.set_xlabel('T (°C)')
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  

plt.suptitle(f'tau vs Temperatura\nNF 13 hs - {conc_13:0.1f} g/L' )
plt.savefig('2_tau_NF13h_38_47_57.png',dpi=300) 

#%% 3 - ESAR vs time / Temp
#% 13 hs
ESAR_13_100,ESAR_13_125,ESAR_13_150=[],[],[]
fig310, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_13):
    if '38_100' in r.directorio:
        ax.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')
        ESAR_13_100.append(r.SAR)
for i,r in enumerate(res_13):
    if '47_125' in r.directorio:
        ax2.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')
        ESAR_13_125.append(r.SAR)
for i,r in enumerate(res_13):
    if '57_150' in r.directorio:
        ax3.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')
        ESAR_13_150.append(r.SAR)
for a in ax,ax2,ax3:
    a.grid()
    #a.legend(loc='best')
    a.set_ylabel('ESAR (W/g)')
ax.set_xlim(0,)
ax3.set_xlabel('t (s)')
ESAR_13_100 = ufloat(np.mean(np.concatenate(ESAR_13_100)),np.std(np.concatenate(ESAR_13_100)))
ESAR_13_125 = ufloat(np.mean(np.concatenate(ESAR_13_125)),np.std(np.concatenate(ESAR_13_125)))
ESAR_13_150 = ufloat(np.mean(np.concatenate(ESAR_13_150)),np.std(np.concatenate(ESAR_13_150)))

ax.text(0.98,0.2,f'ESAR = {ESAR_13_100:.2uS} W/g',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax.transAxes)

ax2.text(0.98,0.2,f'ESAR = {ESAR_13_125:.2uS} W/g',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax2.transAxes)

ax3.text(0.98,0.2,f'ESAR = {ESAR_13_150:.2uS} W/g',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax3.transAxes)    
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  
plt.suptitle(f'ESAR vs time\nNF 13 hs - {conc_13:0.1f} g/L' )
plt.savefig('3_ESAR_vs_time_NF13h_38_47_57.png',dpi=300)

fig311, (ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_13):
    if '38_100' in r.directorio:
        ax.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '47_125' in r.directorio:
        ax2.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '57_150' in r.directorio:
        ax3.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('ESAR (W/g)')
ax3.set_xlabel('T (°C)')
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  
ax3.set_xlim(20,100)
plt.suptitle(f'ESAR vs Temperatura\nNF 13 hs - {conc_13:0.1f} g/L' )
plt.savefig('3_ESAR_vs_Temp_NF13h_38_47_57.png',dpi=300) 

#%% 3 - Hc vs time / Temp
# 13 hs
HC_13_100,HC_13_125,HC_13_150=[],[],[]
fig410, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_13):
    if '38_100' in r.directorio:
        ax.plot(r.time,r.Hc,'.-',label=f'NF{str(i).zfill(2)}')
        HC_13_100.append(r.Hc)

for i,r in enumerate(res_13):
    if '47_125' in r.directorio:
        ax2.plot(r.time,r.Hc,'.-',label=f'NF{str(i).zfill(2)}')
        HC_13_125.append(r.Hc)

for i,r in enumerate(res_13):
    if '57_150' in r.directorio:
        ax3.plot(r.time,r.Hc,'.-',label=f'NF{str(i).zfill(2)}')
        HC_13_150.append(r.Hc)

for a in ax,ax2,ax3:
    a.grid()
    a.set_ylabel('Hc (kA/m)')

ax.set_xlim(0,)
ax3.set_xlabel('t (s)')

HC_13_100 = ufloat(np.mean(np.concatenate(HC_13_100)),np.std(np.concatenate(HC_13_100)))
HC_13_125 = ufloat(np.mean(np.concatenate(HC_13_125)),np.std(np.concatenate(HC_13_125)))
HC_13_150 = ufloat(np.mean(np.concatenate(HC_13_150)),np.std(np.concatenate(HC_13_150)))

ax.text(0.98,0.2,f'H$_c$ = {HC_13_100:.2uS}',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax.transAxes)

ax2.text(0.98,0.2,f'H$_c$ = {HC_13_125:.2uS}',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax2.transAxes)

ax3.text(0.98,0.2,f'H$_c$ = {HC_13_150:.2uS}',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax3.transAxes)

ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')

plt.suptitle(f'Hc vs time\nNF 13 hs - {conc_13:0.1f} g/L')
plt.savefig('4_Hc_vs_time_NF13h_38_47_57.png',dpi=300)

#%% 5 - Mag Remanente vs time/Temp

Mr_13_100,Mr_13_125,Mr_13_150=[],[],[]
fig410, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_13):
    if '38_100' in r.directorio:
        ax.plot(r.time,r.Mr,'.-',label=f'NF{str(i).zfill(2)}')
        Mr_13_100.append(r.Mr)

for i,r in enumerate(res_13):
    if '47_125' in r.directorio:
        ax2.plot(r.time,r.Mr,'.-',label=f'NF{str(i).zfill(2)}')
        Mr_13_125.append(r.Mr)

for i,r in enumerate(res_13):
    if '57_150' in r.directorio:
        ax3.plot(r.time,r.Mr,'.-',label=f'NF{str(i).zfill(2)}')
        Mr_13_150.append(r.Mr)

for a in ax,ax2,ax3:
    a.grid()
    a.set_ylabel('Mr (A/m)')

ax.set_xlim(0,)
ax3.set_xlabel('t (s)')

Mr_13_100 = ufloat(np.mean(np.concatenate(Mr_13_100)),np.std(np.concatenate(Mr_13_100)))
Mr_13_125 = ufloat(np.mean(np.concatenate(Mr_13_125)),np.std(np.concatenate(Mr_13_125)))
Mr_13_150 = ufloat(np.mean(np.concatenate(Mr_13_150)),np.std(np.concatenate(Mr_13_150)))

ax.text(0.98,0.2,f'H$_c$ = {Mr_13_100:.2uS}',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax.transAxes)

ax2.text(0.98,0.2,f'H$_c$ = {Mr_13_125:.2uS}',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax2.transAxes)

ax3.text(0.98,0.2,f'H$_c$ = {Mr_13_150:.2uS}',
        bbox=dict(boxstyle="round", fc='C3',alpha=0.6,lw=1),
        ha='right',va='top',
        transform=ax3.transAxes)

ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')

plt.suptitle(f'Mr vs time\nNF 13 hs - {conc_13:0.1f} g/L')
plt.savefig('4_Mr_vs_time_NF13h_38_47_57.png',dpi=300)

#%% Ciclos todos
_,_,_, H_13_100,M_13_100,_ = lector_ciclos(ciclos_13[1])
_,_,_, H_13_125,M_13_125,_ = lector_ciclos(ciclos_13[4])
_,_,_, H_13_150,M_13_150,_ = lector_ciclos(ciclos_13[8])

fig40, ax2 =plt.subplots(figsize=(7,6),constrained_layout=True,sharey=True,sharex=False)

for i,e in enumerate(ciclos_13):
    if '100dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13[i])
        ax2.plot(H_13/1000,M_13,'-',c='C0',label=f'38 {i}',alpha=0.8)
 
    if '125dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13[i])
        ax2.plot(H_13/1000,M_13,'-',c='C1',label=f'47 {i}',alpha=0.8)

    if '150dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13[i])
        ax2.plot(H_13/1000,M_13,'-',c='C2',label=f'57 {i}',alpha=0.8)
# ax2.plot(H_13_125/1000,M_13_125,'-',label='47')
# ax2.plot(H_13_150/1000,M_13_150,'-',label='57')

        
ax.set_ylabel('M (A/m)')
ax2.set_title(f'13 hs   C={conc_13:0.1f} g/L',loc='left')

ax.set_xticks([-57,-47,-38,0,38,47,57])
ax2.set_xticks([-57,-47,-38,0,38,47,57])
ax3.set_xticks([-57,-45,-38,0,38,45,57])
for a in ax,ax2,ax3:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(title='H$_0$ (kA/m)',loc='upper left',ncol=3)
plt.suptitle('Comparativa ciclos promedio NF@cit\n300 kHz')
plt.savefig('0_comparativa_ciclos_internos_08_13_18_hs.png',dpi=300)


