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
        a.plot(H_00/1000,M_00,'-',label=f'{i}')

a2.set_title('34.7 kA/m',loc='left')
for i,e in enumerate(ciclos_0):
    if '090dA' in e:
        _,_,_, H_01,M_01,_ = lector_ciclos(ciclos_0[i])
        a2.plot(H_01/1000,M_01,'-',label=f'{i}')

a3.set_title('45.9 kA/m',loc='left')
for i,e in enumerate(ciclos_0):
    if '120dA' in e:
        _,_,_, H_02,M_02,_ = lector_ciclos(ciclos_0[i])
        a3.plot(H_02/1000,M_02,'-',label=f'{i}')

a4.set_title('58 kA/m',loc='left')
for i,e in enumerate(ciclos_0):
    if '152dA' in e:
        _,_,_, H_03,M_03,_ = lector_ciclos(ciclos_0[i])
        a4.plot(H_03/1000,M_03,'-',label=f'{i}')

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
        a.plot(H_10/1000,M_10,'-',label=f'{i}')

a2.set_title('34.7 kA/m',loc='left')
for i,e in enumerate(ciclos_10):
    if '090dA' in e:
        _,_,_, H_11,M_11,_ = lector_ciclos(ciclos_10[i])
        a2.plot(H_11/1000,M_11,'-',label=f'{i}')

a3.set_title('45.9 kA/m',loc='left')
for i,e in enumerate(ciclos_10):
    if '120dA' in e:
        _,_,_, H_12,M_12,_ = lector_ciclos(ciclos_10[i])
        a3.plot(H_12/1000,M_12,'-',label=f'{i}')

a4.set_title('58 kA/m',loc='left')
for i,e in enumerate(ciclos_10):
    if '152dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_10[i])
        a4.plot(H_13/1000,M_13,'-',label=f'{i}')

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
        a.plot(H_20/1000,M_20,'-',label=f'{i}')

a2.set_title('34.7 kA/m',loc='left')
for i,e in enumerate(ciclos_60):
    if '090dA' in e:
        _,_,_, H_21,M_21,_ = lector_ciclos(ciclos_60[i])
        a2.plot(H_21/1000,M_21,'-',label=f'{i}')

a3.set_title('45.9 kA/m',loc='left')
for i,e in enumerate(ciclos_60):
    if '120dA' in e:
        _,_,_, H_22,M_22,_ = lector_ciclos(ciclos_60[i])
        a3.plot(H_22/1000,M_22,'-',label=f'{i}')

a4.set_title('58 kA/m',loc='left')
for i,e in enumerate(ciclos_60):
    if '152dA' in e:
        _,_,_, H_23,M_23,_ = lector_ciclos(ciclos_60[i])
        a4.plot(H_23/1000,M_23,'-',label=f'{i}')

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
        a.plot(H_30/1000,M_30,'-',label=f'{i}')

a2.set_title('34.7 kA/m',loc='left')
for i,e in enumerate(ciclos_120):
    if '090dA' in e:
        _,_,_, H_31,M_31,_ = lector_ciclos(ciclos_120[i])
        a2.plot(H_31/1000,M_31,'-',label=f'{i}')

a3.set_title('45.9 kA/m',loc='left')
for i,e in enumerate(ciclos_120):
    if '120dA' in e:
        _,_,_, H_32,M_32,_ = lector_ciclos(ciclos_120[i])
        a3.plot(H_32/1000,M_32,'-',label=f'{i}')

a4.set_title('58 kA/m',loc='left')
for i,e in enumerate(ciclos_120):
    if '152dA' in e:
        _,_,_, H_33,M_33,_ = lector_ciclos(ciclos_120[i])
        a4.plot(H_33/1000,M_33,'-',label=f'{i}')

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
        a.plot(H_40/1000,M_40,'-',label=f'{i}')

a2.set_title('34.7 kA/m',loc='left')
for i,e in enumerate(ciclos_360):
    if '090dA' in e:
        _,_,_, H_41,M_41,_ = lector_ciclos(ciclos_360[i])
        a2.plot(H_41/1000,M_41,'-',label=f'{i}')

a3.set_title('45.9 kA/m',loc='left')
for i,e in enumerate(ciclos_360):
    if '120dA' in e:
        _,_,_, H_42,M_42,_ = lector_ciclos(ciclos_360[i])
        a3.plot(H_42/1000,M_42,'-',label=f'{i}')

a4.set_title('58 kA/m',loc='left')
for i,e in enumerate(ciclos_360):
    if '152dA' in e:
        _,_,_, H_43,M_43,_ = lector_ciclos(ciclos_360[i])
        a4.plot(H_43/1000,M_43,'-',label=f'{i}')

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


#%% 2 - Tau 00
fig200, ((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
a2.set_title('34.7 kA/m',loc='left')
a3.set_title('45.9 kA/m',loc='left')
a4.set_title('58.0 kA/m',loc='left')

for i,e in enumerate(res_00):
    if '060dA' in e.directorio:
        a.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '090dA' in e.directorio:
        a2.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '120dA' in e.directorio:
        a3.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '152dA' in e.directorio:
        a4.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='best')

a.set_ylabel('τ (ns)')
a3.set_ylabel('τ (ns)')
a3.set_xlabel('indx')
a4.set_xlabel('indx')

plt.suptitle(f'tau vs indx {label_0}\n300 kHz')
plt.savefig('1_tau_120900_24_35_46_58.png',dpi=300)
plt.show()

fig201,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,e in enumerate(res_00):
    if '060dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C0',label='24.6 kA/m' if i==0 else None)
        
    elif '090dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C1',label='34.7 kA/m' if i==3 else None)
        
    elif '120dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C2',label='45.9 kA/m' if i==7 else None)
        
    elif '152dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C3',label='58.0 kA/m' if i==10 else None)

ax.grid()
ax.legend(loc='best',ncol=4)
ax.set_ylabel('τ (ns)')
ax.set_xlabel('indx')

plt.suptitle(f'tau vs indx {label_0}\n300 kHz')
plt.savefig('1_tau_120900_24_35_46_58_all.png',dpi=300)
#%% Tau 10
fig202,((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
a2.set_title('34.7 kA/m',loc='left')
a3.set_title('45.9 kA/m',loc='left')
a4.set_title('58.0 kA/m',loc='left')

for i,e in enumerate(res_10):
    if '060dA' in e.directorio:
        a.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '090dA' in e.directorio:
        a2.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '120dA' in e.directorio:
        a3.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '152dA' in e.directorio:
        a4.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='best')

a.set_ylabel('τ (ns)')
a3.set_ylabel('τ (ns)')
a3.set_xlabel('indx')
a4.set_xlabel('indx')

plt.suptitle(f'tau vs indx {label_10}\n300 kHz')
plt.savefig('1_tau_120910_24_35_46_58.png',dpi=300)
plt.show()

fig203,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,e in enumerate(res_10):
    if '060dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C0',label='24.6 kA/m' if i==0 else None)
        
    elif '090dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C1',label='34.7 kA/m' if i==3 else None)
        
    elif '120dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C2',label='45.9 kA/m' if i==7 else None)
        
    elif '152dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C3',label='58.0 kA/m' if i==10 else None)

ax.grid()
ax.legend(loc='best',ncol=4)
ax.set_ylabel('τ (ns)')
ax.set_xlabel('indx')

plt.suptitle(f'tau vs indx {label_10}\n300 kHz')
plt.savefig('1_tau_120910_24_35_46_58_all.png',dpi=300)
#%% tau 60
fig204,((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
a2.set_title('34.7 kA/m',loc='left')
a3.set_title('45.9 kA/m',loc='left') 
a4.set_title('58.0 kA/m',loc='left')

for i,e in enumerate(res_60):
    if '060dA' in e.directorio:
        a.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '090dA' in e.directorio:
        a2.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '120dA' in e.directorio:
        a3.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '152dA' in e.directorio:
        a4.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='best')

a.set_ylabel('τ (ns)')
a3.set_ylabel('τ (ns)')
a3.set_xlabel('indx')
a4.set_xlabel('indx')

plt.suptitle(f'tau vs indx {label_60}\n300 kHz')
plt.savefig('1_tau_120960_24_35_46_58.png',dpi=300)
plt.show()

fig205,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,e in enumerate(res_60):
    if '060dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C0',label='24.6 kA/m' if i==0 else None)
        
    elif '090dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C1',label='34.7 kA/m' if i==3 else None)
        
    elif '120dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C2',label='45.9 kA/m' if i==7 else None)
        
    elif '152dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C3',label='58.0 kA/m' if i==10 else None)

ax.grid()
ax.legend(loc='best',ncol=4)
ax.set_ylabel('τ (ns)')
ax.set_xlabel('indx')
plt.suptitle(f'tau vs indx {label_60}\n300 kHz')
plt.savefig('1_tau_120960_24_35_46_58_all.png',dpi=300) 
#%% tau 120
fig206,((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
a2.set_title('34.7 kA/m',loc='left')
a3.set_title('45.9 kA/m',loc='left') 
a4.set_title('58.0 kA/m',loc='left')

for i,e in enumerate(res_120):
    if '060dA' in e.directorio:
        a.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '090dA' in e.directorio:
        a2.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '120dA' in e.directorio:
        a3.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '152dA' in e.directorio:
        a4.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='best')

a.set_ylabel('τ (ns)')
a3.set_ylabel('τ (ns)')
a3.set_xlabel('indx')
a4.set_xlabel('indx')

plt.suptitle(f'tau vs indx {label_120}\n300 kHz')
plt.savefig('1_tau_1209120_24_35_46_58.png',dpi=300)
plt.show()

fig207,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,e in enumerate(res_120):
    if '060dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C0',label='24.6 kA/m' if i==0 else None)
        
    elif '090dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C1',label='34.7 kA/m' if i==3 else None)
        
    elif '120dA' in e.directorio:        
        ax.plot(e.tau,'.-',c='C2',label='45.9 kA/m' if i==7 else None)
        
    elif '152dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C3',label='58.0 kA/m' if i==10 else None)

ax.grid()
ax.legend(loc='best',ncol=4)
ax.set_ylabel('τ (ns)')
ax.set_xlabel('indx')
plt.suptitle(f'tau vs indx {label_120}\n300 kHz')    
plt.savefig('1_tau_1209120_24_35_46_58_all.png',dpi=300)    
#%% Tau 360
fig208,((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
a2.set_title('34.7 kA/m',loc='left')
a3.set_title('45.9 kA/m',loc='left') 
a4.set_title('58.0 kA/m',loc='left')

for i,e in enumerate(res_360):
    if '060dA' in e.directorio:
        a.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '090dA' in e.directorio:
        a2.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '120dA' in e.directorio:
        a3.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')
    elif '152dA' in e.directorio:
        a4.plot(e.tau,'.-',label=f'{str(i).zfill(2)}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='best')

a.set_ylabel('τ (ns)')
a3.set_ylabel('τ (ns)')
a3.set_xlabel('indx')
a4.set_xlabel('indx')

plt.suptitle(f'tau vs indx {label_360}\n300 kHz')
plt.savefig('1_tau_3609120_24_35_46_58.png',dpi=300)
plt.show()

fig209,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,e in enumerate(res_360):    
    if '060dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C0',label='24.6 kA/m' if i==0 else None)
        
    elif '090dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C1',label='34.7 kA/m' if i==3 else None)
        
    elif '120dA' in e.directorio:        
        ax.plot(e.tau,'.-',c='C2',label='45.9 kA/m' if i==7 else None)
        
    elif '152dA' in e.directorio:
        ax.plot(e.tau,'.-',c='C3',label='58.0 kA/m' if i==10 else None)

ax.grid()
ax.legend(loc='best',ncol=4)
ax.set_ylabel('τ (ns)')
ax.set_xlabel('indx')
plt.suptitle(f'tau vs indx {label_360}\n300 kHz')    
plt.savefig('1_tau_3609120_24_35_46_58_all.png',dpi=300)    
#%% 3 - Coercitivo 00
fig300, ((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
a2.set_title('34.7 kA/m',loc='left')
a3.set_title('45.9 kA/m',loc='left')
a4.set_title('58.0 kA/m',loc='left')

for i,e in enumerate(res_00):
    if '060dA' in e.directorio:
        a.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '090dA' in e.directorio:
        a2.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '120dA' in e.directorio:
        a3.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '152dA' in e.directorio:
        a4.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='best')

a.set_ylabel('H$_c$ (kA/m)')
a3.set_ylabel('H$_c$ (kA/m)')
a3.set_xlabel('indx')
a4.set_xlabel('indx')

plt.suptitle(f'Hc vs indx {label_0}\n300 kHz')
plt.savefig('2_Hc_120900_24_35_46_58.png',dpi=300)
plt.show()

fig301,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,e in enumerate(res_00):
    if '060dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C0',label='24.6 kA/m' if i==0 else None)
        
    elif '090dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C1',label='34.7 kA/m' if i==3 else None)
        
    elif '120dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C2',label='45.9 kA/m' if i==7 else None)
        
    elif '152dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C3',label='58.0 kA/m' if i==10 else None)

ax.grid()
ax.legend(loc='best',ncol=4)
ax.set_ylabel('τ (ns)')
ax.set_xlabel('indx')

plt.suptitle(f'Hc vs indx {label_0}\n300 kHz')
plt.savefig('2_Hc_120900_24_35_46_58_all.png',dpi=300)
#%% Hc 10
fig302,((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
a2.set_title('34.7 kA/m',loc='left')
a3.set_title('45.9 kA/m',loc='left')
a4.set_title('58.0 kA/m',loc='left')

for i,e in enumerate(res_10):
    if '060dA' in e.directorio:
        a.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '090dA' in e.directorio:
        a2.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '120dA' in e.directorio:
        a3.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '152dA' in e.directorio:
        a4.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='best')

a.set_ylabel('τ (ns)')
a3.set_ylabel('τ (ns)')
a3.set_xlabel('indx')
a4.set_xlabel('indx')

plt.suptitle(f'Hc vs indx {label_10}\n300 kHz')
plt.savefig('2_Hc_120910_24_35_46_58.png',dpi=300)
plt.show()

fig303,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,e in enumerate(res_10):
    if '060dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C0',label='24.6 kA/m' if i==0 else None)
        
    elif '090dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C1',label='34.7 kA/m' if i==3 else None)
        
    elif '120dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C2',label='45.9 kA/m' if i==7 else None)
        
    elif '152dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C3',label='58.0 kA/m' if i==10 else None)

ax.grid()
ax.legend(loc='best',ncol=4)
ax.set_ylabel('τ (ns)')
ax.set_xlabel('indx')

plt.suptitle(f'Hc vs indx {label_10}\n300 kHz')
plt.savefig('2_Hc_120910_24_35_46_58_all.png',dpi=300)
#%% Hc 60
fig304,((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
a2.set_title('34.7 kA/m',loc='left')
a3.set_title('45.9 kA/m',loc='left') 
a4.set_title('58.0 kA/m',loc='left')

for i,e in enumerate(res_60):
    if '060dA' in e.directorio:
        a.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '090dA' in e.directorio:
        a2.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '120dA' in e.directorio:
        a3.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '152dA' in e.directorio:
        a4.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='best')

a.set_ylabel('τ (ns)')
a3.set_ylabel('τ (ns)')
a3.set_xlabel('indx')
a4.set_xlabel('indx')

plt.suptitle(f'Hc vs indx {label_60}\n300 kHz')
plt.savefig('2_Hc_120960_24_35_46_58.png',dpi=300)
plt.show()

fig305,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,e in enumerate(res_60):
    if '060dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C0',label='24.6 kA/m' if i==0 else None)
        
    elif '090dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C1',label='34.7 kA/m' if i==3 else None)
        
    elif '120dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C2',label='45.9 kA/m' if i==7 else None)
        
    elif '152dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C3',label='58.0 kA/m' if i==10 else None)

ax.grid()
ax.legend(loc='best',ncol=4)
ax.set_ylabel('τ (ns)')
ax.set_xlabel('indx')
plt.suptitle(f'Hc vs indx {label_60}\n300 kHz')
plt.savefig('2_Hc_120960_24_35_46_58_all.png',dpi=300) 
#%% Hc 120
fig306,((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
a2.set_title('34.7 kA/m',loc='left')
a3.set_title('45.9 kA/m',loc='left') 
a4.set_title('58.0 kA/m',loc='left')

for i,e in enumerate(res_120):
    if '060dA' in e.directorio:
        a.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '090dA' in e.directorio:
        a2.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '120dA' in e.directorio:
        a3.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '152dA' in e.directorio:
        a4.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='best')

a.set_ylabel('τ (ns)')
a3.set_ylabel('τ (ns)')
a3.set_xlabel('indx')
a4.set_xlabel('indx')

plt.suptitle(f'Hc vs indx {label_120}\n300 kHz')
plt.savefig('2_Hc_1209120_24_35_46_58.png',dpi=300)
plt.show()

fig307,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,e in enumerate(res_120):
    if '060dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C0',label='24.6 kA/m' if i==0 else None)
        
    elif '090dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C1',label='34.7 kA/m' if i==3 else None)
        
    elif '120dA' in e.directorio:        
        ax.plot(e.Hc,'.-',c='C2',label='45.9 kA/m' if i==7 else None)
        
    elif '152dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C3',label='58.0 kA/m' if i==10 else None)

ax.grid()
ax.legend(loc='best',ncol=4)
ax.set_ylabel('τ (ns)')
ax.set_xlabel('indx')
plt.suptitle(f'Hc vs indx {label_120}\n300 kHz')    
plt.savefig('2_Hc_1209120_24_35_46_58_all.png',dpi=300)    
#%% Hc 360
fig208,((a,a2),(a3,a4)) =plt.subplots(2,2,figsize=(10,8),constrained_layout=True,sharey=True,sharex=True)

a.set_title('24.6 kA/m',loc='left')
a2.set_title('34.7 kA/m',loc='left')
a3.set_title('45.9 kA/m',loc='left') 
a4.set_title('58.0 kA/m',loc='left')

for i,e in enumerate(res_360):
    if '060dA' in e.directorio:
        a.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '090dA' in e.directorio:
        a2.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '120dA' in e.directorio:
        a3.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')
    elif '152dA' in e.directorio:
        a4.plot(e.Hc,'.-',label=f'{str(i).zfill(2)}')

for i in [a,a2,a3,a4]:
    i.grid()
    i.legend(loc='best')

a.set_ylabel('τ (ns)')
a3.set_ylabel('τ (ns)')
a3.set_xlabel('indx')
a4.set_xlabel('indx')

plt.suptitle(f'Hc vs indx {label_360}\n300 kHz')
plt.savefig('2_Hc_3609120_24_35_46_58.png',dpi=300)
plt.show()

fig209,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,e in enumerate(res_360):    
    if '060dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C0',label='24.6 kA/m' if i==0 else None)
        
    elif '090dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C1',label='34.7 kA/m' if i==3 else None)
        
    elif '120dA' in e.directorio:        
        ax.plot(e.Hc,'.-',c='C2',label='45.9 kA/m' if i==7 else None)
        
    elif '152dA' in e.directorio:
        ax.plot(e.Hc,'.-',c='C3',label='58.0 kA/m' if i==10 else None)

ax.grid()
ax.legend(loc='best',ncol=4)
ax.set_ylabel('τ (ns)')
ax.set_xlabel('indx')
plt.suptitle(f'Hc vs indx {label_360}\n300 kHz')    
plt.savefig('2_Hc_3609120_24_35_46_58_all.png',dpi=300)    
# %%
