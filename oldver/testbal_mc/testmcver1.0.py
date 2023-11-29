# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:45:15 2023


"""

import os
import shutil
import matplotlib.pyplot as plt
import flopy
import math
import csv
import pandas as pd
import numpy as np
import scipy.stats as stats
from copy import deepcopy

def smooth(zones_vec):
    
    """
    This function can smooth the zones.
    
    :param zones_vec: array [n_cells, n_cells]  
    
    :return: smoothened mod_zones fields<np.array[n_cells_x, n_cells_y]>
        with zone distribution (ints from 1 to n_zones)
    """
    mod_zones = deepcopy(zones_vec)

    for i in range(0, zones_vec.shape[0]):
        for j in range(0, zones_vec.shape[1]):
            # Get adjacent indexes
            n = len(mod_zones)-1
            m = len(mod_zones[0])-1

            # Initialising a vector array where adjacent elements will be stored
            v = []

            # Checking for adjacent elements and adding them to array
            # Deviation of row that gets adjusted according to the provided position

            #
            for dx in range(-1 if (i > 0) else 0, 2 if (i < n) else 1):
                # Deviation of the column that gets adjusted according to the provided position
                for dy in range(-1 if (j > 0) else 0, 2 if (j < m) else 1):
                    if dx != 0 or dy != 0:
                        v.append(mod_zones[i + dx][j + dy])
            #
            if np.count_nonzero(v == mod_zones[i, j]) < np.count_nonzero(v != mod_zones[i, j]):
                mod_zones[i, j] = stats.mode(v)[0]

    return mod_zones 


def hkfields(zones_vec, parameter_sets, n_zones):
    """
    Function assigns corresponding log(K) to each cells.

    :param zones_vec: array [n_cells, n_cells]
    
    :param parameter_values: array [n_mc, n_parameters]
        with parameter sets that need to be transformed into log(K) fields
    :param n_zones: <int>
        number of zones       
    :return: log(K) fields<np.array[n_cells_x, n_cells_y]>
        with zone distribution (ints from 1 to n_zones)
    
    """
    zones_vec = np.array(zones_vec, dtype=float)
    log_k_fields = zones_vec
    for i in range(n_zones):
        log_k_fields[np.where(log_k_fields == i+1)] = parameter_sets[0, i]

    return log_k_fields


import time

t0 = time.time()









# # read the zone distribution

n_zones = 6 
n_reali =  100000


gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant
parameter_sets = np.random.normal(loc=gm_mean, 
                                      scale=gm_var,
                                      size=(n_reali, n_zones)
                                      )

    
zones_vec50x50 = pd.read_csv('C:/Users/tian/Desktop/gwf/gwf50x50/input/zone_distribution.csv', dtype=np.float64)
zones_vec = np.array(zones_vec50x50, dtype = float)


df = pd.DataFrame(parameter_sets)        
df.to_csv('C:/Users/tian/Desktop/gp/testbal_mc/outputMC/parameter_sets.csv')



count = 0
model_name = "50"  #Name maximum length of 16
exe_name = 'C:/Users/tian/Desktop/mf6.4.2/bin/mf6.exe'  #remamber to change this path.
waterhead = []
concentration  = []

for count in range(n_reali):
    # tmpdir = tempfile.mkdtemp()
    
    sim_name = f'{model_name}_{count}'
    sim_ws = os.path.join('model',sim_name)
    
    # ws = os.path.join(tmpdir,name)          
    k = hkfields(zones_vec = zones_vec,parameter_sets = parameter_sets[count:,],n_zones = n_zones)
        
    k11 = np.exp(k)/100.0 * 86400  # cm^2/s -> m^2/d
    hk = k11
    k33 = hk  # Vertical hydraulic conductivity ($m/d$)
    







    
    
    gwfname = "f_" + sim_name # a name you can change for groundwater flow model
    gwtname = "t_" + sim_name # a name you can change for groundwater transport model
    
    sim = flopy.mf6.MFSimulation(
        sim_name= sim_name,
        sim_ws=sim_ws,
        exe_name=exe_name
        )
    
    
    length_units = "meters"
    time_units = "days"
    Lx = 50 # x derection length ($m$)
    Ly = 50 # y derection length ($m$)
    top = 0  # Top of the model ($m$)
    botm= -1 # bottom of the model ($m$)
    nlay = 1  # Number of layers
    nrow = 50  # Number of rows
    ncol = 50  # Number of columns
    delr = Lx/ncol  # Column width ($m$)
    delc = Ly/nrow  # Row width ($m$)
    delz = top-botm  # Layer thickness ($m$)

    # Transport parameters
    prsity = 0.35  # Porosity
    alpha_l = 2.5  # Longitudinal dispersivity ($m$)
    alpha_th = 0.25  # Transverse horizontal dispersivity ($m$)
    dmcoef = 1.10e-9 / 100 / 100 * 86400  # cm^2/s -> m^2/d
    idomain = np.ones((nlay, nrow, ncol), dtype=int) # 1 means a activate cell



    # This is the water head and concentration boundary condition.
    chdspd = []
    H1 = 1.0  #left water head
    H2 = 0.0  #right water head
    #     [(layer,row, column), head, concentration]
    chdspd += [[0, i, 0, H1, 0.0] for i in range(nrow)]
    for j in np.arange(nrow):
        chdspd.append([0, j, ncol-1, H2, 0.0])
    chdspd = {0: chdspd}


    # Setup constant concentration information
    sconc = 0.0 # this is the satrting Concentration or background Concentration normaly is 0
    cncspd = []
    cnc = 1.0
    for row in np.arange(20, 31):  # this is the concentration on left side (for 10 meters)
        cncspd.append([(0, row, 0), cnc])
    cncspd = {0: cncspd}

    #Solver settings
    nper = 1  # Number of periods
    nstp = 50  # Number of time steps
    perlen = 100000  # Simulation time length ($d$)
    icelltype = 0
    mixelm = 0
    
    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-6, 1e-6, 1.0
    percel = 1.0  # HMOC parameters
    itrack = 3
    wd = 0.5
    dceps = 1.0e-5
    nplane = 0
    npl = 0
    nph = 16
    npmin = 4
    npmax = 32
    dchmoc = 1.0e-3
    nlsink = nplane
    npsink = nph



    """"
    # MODFLOW 6 gwf pakage for  simulation waterhead, hydraulic conductivity, wells and etc..
    """
    # Instantiating MODFLOW 6 time discretization
    tdis_rc = []
    tdis_rc.append((perlen, nstp, 1.0))
    flopy.mf6.ModflowTdis(
        sim,
        nper=nper,
        perioddata=tdis_rc,
        time_units=time_units
    )

    # Instantiating MODFLOW 6 groundwater flow model
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfname,
        save_flows=True,
        model_nam_file="{}.nam".format(gwfname),
    )

    # Instantiating MODFLOW 6 solver for flow model
    imsgwf = flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        complexity="complex",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="dbd",
        linear_acceleration="BICGSTAB",
        under_relaxation_theta=0.7,
        under_relaxation_kappa=0.08,
        under_relaxation_gamma=0.05,
        under_relaxation_momentum=0.0,
        backtracking_number=20,
        backtracking_tolerance=2.0,
        backtracking_reduction_factor=0.2,
        backtracking_residual_limit=5.0e-4,
        inner_dvclose=hclose,
        rcloserecord="0.0001 relative_rclose",
        inner_maximum=ninner,
        relaxation_factor=relax,
        number_orthogonalizations=2,
        preconditioner_levels=8,
        preconditioner_drop_tolerance=0.001,
        filename="{}.ims".format(gwfname),
    )
    sim.register_ims_package(imsgwf, [gwf.name])

    # Instantiating MODFLOW 6 discretization package
    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idomain,
        filename="{}.dis".format(gwfname),
    )

    # Instantiating MODFLOW 6 node-property flow package
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=False,
        icelltype=icelltype,
        k=hk,
        k33=hk, # Vertical hydraulic conductivity ($m/d$)
        save_specific_discharge=True,
        filename="{}.npf".format(gwfname),
    )

    # Instantiate storage package
    flopy.mf6.ModflowGwfsto(
        gwf,
        ss=0,
        sy=0
    )

    # Instantiating MODFLOW 6 storage package (steady flow conditions, so no actual storage, using to print values in .lst file)
    flopy.mf6.ModflowGwfsto(
        gwf,
        ss=0,
        sy=0,
        filename="{}.sto".format(gwfname)
    )

    # Instantiating MODFLOW 6 initial conditions package for flow model
    flopy.mf6.ModflowGwfic(
        gwf,
        strt=0, #starting simulation water head. i set 0 here.
        filename="{}.ic".format(gwfname)
    )

    # Instantiating MODFLOW 6 constant head package
    flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=False,
        auxiliary="CONCENTRATION",
        pname="CHD-1",
        filename="{}.chd".format(gwfname)
    )

    # Instantiating MODFLOW 6 output control package for flow model
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="{}.hds".format(gwfname),
        budget_filerecord="{}.bud".format(gwfname),
        headprintrecord=[
            ("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")
        ],
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    # Instantiating MODFLOW 6 obsevation package for flow model
    obsdict = {}
    obslist = [
        #[  name of the obspoint    head(if want to observe water head)    (layer, row, column)
        ["loc_1", "head", (0, 25, 25)],
        ["loc_2", "head", (0, 26, 31)],
        ["loc_3", "head", (0, 5, 18)],
        ["loc_4", "head", (0, 22, 7)],
        ["loc_5", "head", (0, 18, 27)],
        ["loc_6", "head", (0, 42, 36)],
        ["loc_7", "head", (0, 43, 10)],
        ["loc_8", "head", (0, 32, 20)],
        ["loc_9", "head", (0, 22, 32)],
        ["loc_10", "head", (0, 11, 41)],
    ]
    
    # obslist = pd.read_csv('C:/Users/tian/Desktop/test/obslist.csv').values.tolist()
    
    obsdict["{}.obs.head.csv".format(gwfname)] = obslist

    obs = flopy.mf6.ModflowUtlobs(
        gwf,
        print_input=False,
        continuous=obsdict
    )


    """
    Instantiating MODFLOW 6 groundwater transport package for transport problem.
    """
    #
    
    
    gwt = flopy.mf6.MFModel(
        sim,
        model_type="gwt6",
        modelname=gwtname,
        model_nam_file="{}.nam".format(gwtname),
    )
    gwt.name_file.save_flows = True

    # create iterative model solution and register the gwt model with it
    imsgwt = flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        complexity="complex",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="dbd",
        linear_acceleration="BICGSTAB",
        under_relaxation_theta=0.7,
        under_relaxation_kappa=0.08,
        under_relaxation_gamma=0.05,
        under_relaxation_momentum=0.0,
        backtracking_number=20,
        backtracking_tolerance=2.0,
        backtracking_reduction_factor=0.2,
        backtracking_residual_limit=5.0e-4,
        inner_dvclose=hclose,
        rcloserecord="0.0001 relative_rclose",
        inner_maximum=ninner,
        relaxation_factor=relax,
        number_orthogonalizations=2,
        preconditioner_levels=8,
        preconditioner_drop_tolerance=0.001,
        filename="{}.ims".format(gwtname),
    )
    sim.register_ims_package(imsgwt, [gwt.name])


    # Instantiating MODFLOW 6 transport discretization package
    flopy.mf6.ModflowGwtdis(
        gwt,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idomain,
        filename="{}.dis".format(gwtname),
    )

    # Instantiating MODFLOW 6 transport initial concentrations
    flopy.mf6.ModflowGwtic(
        gwt,
        strt=sconc,
        filename="{}.ic".format(gwtname)
    )

    # Instantiating MODFLOW 6 transport advection package
    if mixelm >= 0:
        scheme = "UPSTREAM"
    elif mixelm == -1:
        scheme = "TVD"
    else:
        raise Exception()

    flopy.mf6.ModflowGwtadv(
        gwt,
        scheme=scheme,
        filename="{}.adv".format(gwtname)
    )

    # Instantiating MODFLOW 6 transport dispersion package
    flopy.mf6.ModflowGwtdsp(
        gwt,
        diffc=dmcoef,
        alh=alpha_l,
        ath1=alpha_th,
        filename="{}.dsp".format(gwtname),
    )

    # Instantiating MODFLOW 6 transport mass storage package
    flopy.mf6.ModflowGwtmst(
        gwt,
        porosity=prsity,
        first_order_decay=False,
        decay=None,
        decay_sorbed=None,
        sorption=None,
        bulk_density=None,
        distcoef=None,
        filename="{}.mst".format(gwtname),
    )

    # Instantiating MODFLOW 6 transport source-sink mixing package
    sourcerecarray = [("CHD-1", "AUX", "CONCENTRATION")]
    flopy.mf6.ModflowGwtssm(
        gwt,
        sources=sourcerecarray,
        filename="{}.ssm".format(gwtname)
    )

    # Instantiating MODFLOW 6 transport output control package
    flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord="{}.cbc".format(gwtname),
        concentration_filerecord="{}.ucn".format(gwtname),
        concentrationprintrecord=[
                    ("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")
        ],
        saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],

    )

    # Instantiate constant concentration at left boundary.
    flopy.mf6.ModflowGwtcnc(
        gwt,
        print_flows=True,
        stress_period_data=cncspd,
        pname="CNC-1",
        filename="{}.cnc".format(gwtname),
    )

    # Instantiate observation package (for transport)
    obslist = [
        ["loc_1", "concentration", (0, 25, 25)],
        ["loc_2", "concentration", (0, 26, 31)],
        ["loc_3", "concentration", (0, 5, 18)],
        ["loc_4", "concentration", (0, 22, 7)],
        ["loc_5", "concentration", (0, 18, 27)],
        ["loc_6", "concentration", (0, 42, 36)],
        ["loc_7", "concentration", (0, 43, 10)],
        ["loc_8", "concentration", (0, 32, 20)],
        ["loc_9", "concentration", (0, 22, 32)],
        ["loc_10", "concentration", (0, 11, 41)],
    ]
    obsdict = {"{}.obs.conc.csv".format(gwtname): obslist}
    obs = flopy.mf6.ModflowUtlobs(
        gwt,
        print_input=False,
        continuous=obsdict,
    )

    # Instantiating MODFLOW 6 flow-transport exchange mechanism
    flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=gwfname,
        exgmnameb=gwtname,
        filename="{}.gwfgwt".format(sim_name),
    )



    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation()
    
    waterhead1 = pd.read_csv(f'C:/Users/tian/Desktop/gp/testbal_mc/model/{sim_name}/{gwfname}.obs.head.csv', dtype=np.float64, header = 0, usecols=range(1,11),skiprows=1)[-1:]
    waterhead1 = np.array(waterhead1)
    waterhead.append(waterhead1)
    
    concentration1 = pd.read_csv(f'C:/Users/tian/Desktop/gp/testbal_mc/model/{sim_name}/{gwtname}.obs.conc.csv', dtype=np.float64, header = 0, usecols=range(1,11),skiprows=1)[-1:]
    concentration1 = np.array(concentration1)
    concentration.append(concentration1)
    count = count + 1 
    
   
waterhead=np.array(waterhead).reshape(n_reali,10)
df = pd.DataFrame(waterhead)        
df.to_csv('C:/Users/tian/Desktop/gp/testbal_mc/outputMC/waterhead.csv')

concentration=np.array(concentration).reshape(n_reali,10)
df = pd.DataFrame(concentration)        
df.to_csv('C:/Users/tian/Desktop/gp/testbal_mc/outputMC/concentration.csv')


# shutil.rmtree('C:/Users/tian/Desktop/gp/testbal_mc/model')



print("--- %s seconds ---" % (time.time() - t0))



    
            