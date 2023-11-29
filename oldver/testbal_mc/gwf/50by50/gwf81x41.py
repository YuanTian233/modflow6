import os
import matplotlib.pyplot as plt
import flopy
import numpy as np
import scipy.stats as stats
from copy import deepcopy

'''
Modflow6 is used in this model. 
        https://water.usgs.gov/water-resources/software/MODFLOW-6/
Modflow6 gwf(Ground water Flow) and gwt(Ground water Transport) are used in this case.
After running this, the data will be saved as a csv file.
'''

name = "gwf81x41"  # a name you like.
ws = os.path.join('model',name)  # crate a folder name ".../model/gwf20x20"
exe_name = 'C:/Users/tian/Desktop/mf6.4.2/bin/mf6.exe'  #remamber to change this path.

length_units = "meters"
time_units = "days"

gwfname = "gwf_" + name
gwtname = "gwt_" + name
sim = flopy.mf6.MFSimulation(
    sim_name=name,
    sim_ws=ws,
    exe_name=exe_name
    )




"""
Here is the basic setting.
This case is 1 layer 81 by 41 grids. Every grid is 1 by 2 meter.
The contamination is on the cell (0,16,24) with c1 = 1.0 mg/l and the  # Volumetric injection rate
qwell = 0.01 m^3/d.
Water head on left is 12.0 m. right side is 11.0m
"""
Lx = 20
Ly = 10
top = 0  # Top of the model ($m$)
botm= -1 # bottom of the model ($m$)
nlay = 1  # Number of layers
nrow = 41  # Number of rows
ncol = 81  # Number of columns
delr = Lx/ncol  # Column width ($m$)
delc = Ly/nrow  # Row width ($m$)
delz = top-botm  # Layer thickness ($m$)

# Transport parameters
prsity = 0.35  # Porosity
alpha_l = 2.5  # Longitudinal dispersivity ($m$)
alpha_th = 0.25  # Transverse horizontal dispersivity ($m$)
dmcoef = 1.10e-9 / 100 / 100 * 86400  # cm^2/s -> m^2/d


"""
Here is the Flow boundary condition.

"""


# Flow boundary condition

# rech = 10.0
# idomain0 = np.loadtxt( 'E:/kurs/masterThesis/code/test/input/idomain.txt', dtype=np.int32)
# idomain = nlay * [idomain0]
idomain = np.ones((nlay, nrow, ncol), dtype=int)
# idomain[0, 0, :] = -1
# idomain[0, 49, :] = -1

chdspd = []
H1 = 12.0  #left water head
H2 = 11.0 #right water head
#            [(layer,row, column), head, conc]
chdspd += [[0, i, 0, H1, 0.0] for i in range(nrow)]
for j in np.arange(nrow):
    chdspd.append([0, j, 80, H2, 0.0])
chdspd = {0: chdspd}



# Setup constant concentration information for MF6
sconc=0  # this is the satrting Concentration or background Concentration normaly is 0
qwell = 0.01  # Volumetric injection rate ($m^3/d$)
c1 = 1.0  # Concentration of injected water ($mg/L$)
c0 = 0.0 # Concentration of feild ($mg/L$)
# [   (layer, row, column), conc
# cncspd = [[(0, 4, 6), c1]]
#{0:        [[(layer, row, column),  flow,   conc]]}
spd = {0: [[(0, 16, 24), qwell, c1]]}  #coordinate of the well.

#Solver settings

nper = 1  # Number of periods
nstp = 20  # Number of time steps
perlen = 1500  # Simulation time length ($d$)
icelltype = 0
mixelm = 0



def rungwfmodle(hk):
    #Solver settings
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
        k33=hk,
        save_specific_discharge=True,
        filename="{}.npf".format(gwfname),
    )

    # Instantiate storage package
    flopy.mf6.ModflowGwfsto(gwf, ss=0, sy=0)

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
        strt=11,
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

    # Instantiate the wel package
    flopy.mf6.ModflowGwfwel(
        gwf,
        print_input=True,
        print_flows=True,
        stress_period_data=spd,
        save_flows=False,
        auxiliary="CONCENTRATION",
        pname="WEL-1",
        filename="{}.wel".format(gwfname),
    )


    # # Instantiate recharge package
    # flopy.mf6.ModflowGwfrcha(
    #     gwf,
    #     print_flows=True,
    #     recharge=rech,
    #     pname="RCH-1",
    #     filename="{}.rch".format(gwfname),
    # )

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

    obsdict = {}
    #loc = np.loadtxt('E:/kurs/masterThesis/code/test/input/point_loc.txt',dtype=int)
    obslist = [
        #[  name of the obspoint    head(if want to observe water head)    (layer, row, column)
        ["loc_1", "head", (0, 16, 24)],
        ["loc_2", "head", (0, 28, 24)],
        ["loc_3", "head", (0, 8, 32)],
        ["loc_4", "head", (0, 16, 32)],
        ["loc_5", "head", (0, 32, 32)],
        ["loc_6", "head", (0, 36, 32)],
        ["loc_7", "head", (0, 16, 40)],
        ["loc_8", "head", (0, 32, 40)],
    ]
    obsdict["{}.obs.head.csv".format(gwfname)] = obslist

    obs = flopy.mf6.ModflowUtlobs(
        gwf,
        print_input=False,
        continuous=obsdict
    )

    """
    Instantiating MODFLOW 6 groundwater transport package for transport problem.
    """
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
                gwt, scheme=scheme, filename="{}.adv".format(gwtname)
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

    # Instantiate constant concentration
    # flopy.mf6.ModflowGwtcnc(
    #     gwt,
    #     print_flows=True,
    #     stress_period_data=cncspd,
    #     pname="CNC-1",
    #     filename="{}.cnc".format(gwtname),
    # )

    # Instantiating MODFLOW 6 transport source-sink mixing package
    sourcerecarray = [("WEL-1", "AUX", "CONCENTRATION")]
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


    # Instantiate observation package (for transport)
    obslist = [
        ["loc_1", "concentration", (0, 16, 24)],
        ["loc_2", "concentration", (0, 28, 24)],
        ["loc_3", "concentration", (0, 8, 32)],
        ["loc_4", "concentration", (0, 16, 32)],
        ["loc_5", "concentration", (0, 32, 32)],
        ["loc_6", "concentration", (0, 36, 32)],
        ["loc_7", "concentration", (0, 16, 40)],
        ["loc_8", "concentration", (0, 32, 40)],
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
        filename="{}.gwfgwt".format(name),
    )



    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation()
    return

"""
Ploting part for water head, concentration and hk.
"""
def plotgwm(): 
    gwf = sim.get_model(gwfname)
    H = gwf.output.head().get_data()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    mm = flopy.plot.PlotMapView(model=gwf)
    plt.rcParams["lines.dashed_pattern"] = [5.0, 5.0]
    pa = mm.plot_array(H,
                       # vmin=0,
                       # vmax=1.1
                       )
    # lc = mm.plot_grid()
    h=mm.contour_array(H,
                       # levels=np.linspace(0, 1.1),
                       colors="black",
                       linestyles="--"
                       )
    cbar = plt.colorbar(pa, shrink=0.25)
    plt.title("Simulated Hydraulic Heads")
    # letter = chr(ord("@"))
    # fs.heading(letter=letter, heading=title)
    plt.clabel(h, fmt="%2.1f")
    
    
    
    gwt = sim.get_model(gwtname)
    conct = gwt.output.concentration()
    times = conct.get_times() # simulation time
    times1 = times[round(len(times)/4.)] # 1/4 simulation time
    times2 = times[round(len(times)/2.)] # 1/2 simulation time
    times3 = times[-1] # the last simulation time
    conc1 =conct.get_data(totim=times1)
    conc2 =conct.get_data(totim=times2)
    conc3 =conct.get_data(totim=times3)
    fig = plt.figure(figsize=(6, 6))
    x = 1000
    y = 1000
    plt.rcParams["lines.dashed_pattern"] = [5.0, 5.0]
    mm = flopy.plot.PlotMapView(model=gwt)
    pa = mm.plot_array(conc3,
                       vmin=0,
                       vmax=1.1)
    # lc = mm.plot_grid() # grid
    
    cs = mm.contour_array(conc3,
                          levels=np.linspace(0, 1.1, 5),
                          colors="black",
                          linestyles="--")
    cbar = plt.colorbar(pa, shrink=0.25)
    plt.title("Simulated Concentration")
    plt.clabel(cs, fontsize=20, fmt='%1.1f', zorder=1)
    
    
    plt.show()
    return