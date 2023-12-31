# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:45:15 2023


"""



import matplotlib.pyplot as plt
import flopy
import numpy as np
import scipy.stats as stats
from copy import deepcopy
import arviz as az

'''
Modflow6 is used in this model. 
        https://water.usgs.gov/water-resources/software/MODFLOW-6/
More example questions please check this website:
        https://modflow6-examples.readthedocs.io/en/latest/notebook_examples.html
Need flopy packege
        pip install flopy
        https://flopy.readthedocs.io/en/3.3.2/introduction.html#flopy-installation
Modflow6 gwf(Ground water Flow) and gwt(Ground water Transport) are used in this case.
After running this, the observation (10 locations in layer 0[(25, 25),
                                                            (26, 31),
                                                            (5, 18),
                                                            (22, 7),
                                                            (18, 27),
                                                            (42, 36),
                                                            (43, 10),
                                                            (32, 20),
                                                            (22, 32),
                                                            (11, 41)]) 
data will be saved as a csv file. 
And those observation need to be manuly change in (obsevation package for flow model (line 267)),
                                                  (obsevation package for transport (line 423)) and
                                                  (plot part (line 460))
'''










def runmodelflow(sim_name, sim_ws,exe_name, gwfname,gwtname,hk, plot = False):            
    """
    This function is the main body of modflow6.
    :param sim_name: string
        simulation name of a model 
    :param sim_ws: string
        worksapce for model
    :param exe_name: string
        Path where modflow6 executable is
    :param gwfname:string
        a name for groundwater flow model. Name maximum length of 16.
    :param gwtname: string
        a name for groundwater transport model. Name maximum length of 16.
    :param hydraulic conductivity: array [n_reali, n_zones]

    """
    

    
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
    

     
    
    # return 

    """
    Ploting part for water head, concentration and hk.
    """
    
    if plot is True:
        gwf = sim.get_model(gwfname)
        # az.style.use("arviz-doc")

        gwt = sim.get_model(gwtname)
        cmap = plt.get_cmap('turbo')
        # gwf = sim.get_model(gwfname)
        # gwt = sim.get_model(gwtname)
        point_loc = np.array(
            [(25, 25),
            (26, 31),
            (5, 18),
            (22, 7),
            (18, 27),
            (42, 36),
            (43, 10),
            (32, 20),
            (22, 32),
            (11, 41)]
            ) # this mark the location of obs points on plot
      
        H = gwf.output.head().get_data()
        #.get_data()
        fig = plt.figure()
    
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        cmap = plt.get_cmap(
            # 'GnBu'
            )
        mm = flopy.plot.PlotMapView(model=gwf)
        plt.rcParams["lines.dashed_pattern"] = [5.0, 5.0]
        im1 = mm.plot_array(H,
                            cmap=cmap
                            # vmin=0,
                            # vmax=1.1
                            )
        # lc = mm.plot_grid()
        h=mm.contour_array(H,
                            # levels=np.linspace(0, 1.1),
                            colors="black",
                            linestyles="--"
                            )
        plt.scatter(point_loc[:, 1], point_loc[:, 0], marker='o',edgecolor = 'k',c='#FFFFFF', s=80) 
        cbar1 = plt.colorbar(im1,
                             # ax=ax1
                            shrink=0.8
                            )
        plt.title("Simulated Hydraulic Heads")
        # letter = chr(ord("@"))
        # fs.heading(letter=letter, heading=title)
        plt.clabel(h, fmt="%2.1f")
        
        
        
        cmap = plt.get_cmap('turbo')
        conct = gwt.output.concentration()
        times = conct.get_times() # simulation time
        # times1 = times[round(len(times)/4.)] # 1/4 simulation time
        # times2 = times[round(len(times)/2.)] # 1/2 simulation time
        times3 = times[-1] # the last simulation time
        # conc1 =conct.get_data(totim=times1)
        # conc2 =conct.get_data(totim=times2)
        conc3 =conct.get_data(totim=times3)
        fig= plt.figure()
        plt.rcParams["lines.dashed_pattern"] = [5.0, 5.0]
        mm = flopy.plot.PlotMapView(model=gwt)
        pa = mm.plot_array(conc3,
                            vmin=0,
                            vmax=1.1,
                            cmap=cmap)
        # lc = mm.plot_grid() # grid
        
        cs = mm.contour_array(conc3,
                              levels=np.linspace(0, 1.1, 5),
                              colors="black",
                              linestyles="--")
        cbar2 = plt.colorbar(pa,
                     # ax=ax2
                      shrink=0.8
                     )
        plt.scatter(point_loc[:, 1], point_loc[:, 0], marker='o',edgecolor = 'k',c='#FFFFFF', s=80) 
        plt.title("Simulated Concentration")
        plt.clabel(cs, fontsize=20, fmt='%1.1f', zorder=1)
        
        
        
        # ax1.set_title("Zone Distribution")
        # bounds = [1, 2, 3, 4, 5, 6]
        # cbar = fig.colorbar(im3, ax=ax1,  ticks=bounds, spacing='proportional')
        cmap = plt.get_cmap(
            # 'GnBu',6
            ) 
        c  =  hk
        fig, (ax3)= plt.subplots()
        mod_zones01 = c#[0 : nrow, 0 : ncol]
        im3 = plt.imshow(mod_zones01,cmap=cmap)
        plt.title("HK Field")
        cbar3 = fig.colorbar(im3, 
                            ax=ax3,
                            # shrink=0.8
                            )
        point_loc = np.array(
            [(25, 25),
            (24, 31),
            (45, 18),
            (28, 7),
            (32, 27),
            (8, 36),
            (7, 10),
            (18, 20),
            (28, 32),
            (39, 41)]
            ) # this mark the location of obs points on plot
        plt.scatter(point_loc[:, 1], point_loc[:, 0], marker='o',edgecolor = 'k',c='#FFFFFF', s=80)         
        plt.show()
        
    return

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


def add_noise(true_meas, error_vals, n_sizes, seed, ret_error=True):   
    """
    Function adds noise to synthetically-generated observation values.
    Args:
        true_meas: np.array[n_reali, n_obs], with observed values
        error_vals: list[2, 1] with [absolute error, relative error], 
            #####in this case, the conc and waterhead max value are both 1.0.#######
        ret_error: bool, True to return error value and observations with noise, False to only return observation values
        with noise
    Returns: np.array[n_reali, n_obs] with observations with noise and (opt) np.array[n_reali,n_obs] with measurement error
    associated to each observation
    """
    if ret_error is True:
        total_error = true_meas * error_vals[1] + error_vals[0]
    else:
        total_error = error_vals
    np.random.seed(seed)    
    rnd_noise = np.random.normal(size=(1, n_sizes)) * total_error
    data_with_noise =  np.array(true_meas + rnd_noise)
    data_with_noise[np.where(data_with_noise > 1)] = 1.0 #only apply in this case!!!
    data_with_noise[np.where(data_with_noise < 0)] = 0.0
    
    return data_with_noise, total_error

            