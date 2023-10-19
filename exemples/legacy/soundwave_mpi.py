import shamrock
import numpy as np
import matplotlib.pyplot as plt 
import os




ctx = shamrock.Context()
ctx.pdata_layout_new()

ctx.pdata_layout_add_field("xyz",1,"f32_3")
ctx.pdata_layout_add_field("hpart",1,"f32")

#field for leapfrog integrator
ctx.pdata_layout_add_field("vxyz",1,"f32_3")
ctx.pdata_layout_add_field("axyz",1,"f32_3")
ctx.pdata_layout_add_field("axyz_old",1,"f32_3")


#start the scheduler
ctx.init_sched(int(1e4),1)






setup = shamrock.SetupSPH_M4_single()
setup.init(ctx)


rho_g = 1
dv0 = 1e-2
nmode = 2
cs = 1


pmass = -1
hfac = 1.2

bsz = 1
dr = 0.02




aspect_rat = 1/4



#set box size
bdim = setup.get_ideal_box(dr,((-bsz*aspect_rat,-bsz*aspect_rat,-bsz),(bsz*aspect_rat,bsz*aspect_rat,bsz)))
dimm,dimM = bdim
xm,ym,zm = dimm
xM,yM,zM = dimM
vol_b = (xM-xm)*(yM-ym)*(zM-zm)
print("box resized to :",bdim,"| volume :",vol_b)
ctx.set_box_size(bdim)

#set BC & add particles
setup.set_boundaries(True)
setup.add_cube_fcc(ctx,dr, bdim)

#set particle mass
totmass = rho_g*vol_b
print("Total mass :", totmass)
setup.set_total_mass(totmass)
pmass = setup.get_part_mass()
print("Current part mass :", pmass)

#update particle smoothing length
for it in range(5):
    setup.update_smoothing_length(ctx)


#compute physical mode
zs = zM - zm
k = nmode*2.*np.pi/zs
omega = 2*np.pi*(2/zs)*cs

#pertub mode
setup.pertub_eigenmode_wave(ctx,(0,dv0),(0,0,k),0)

#clean setup
setup.clear()


def print_dist(dic, cname : str,fname : str,tval):

   

    #import os
    #if not os.path.exists(report_dir):
    #    os.mkdir(report_dir)

    

    plt.figure()

    xyz = np.array(dic["xyz"])
    vxyz = np.array(dic["vxyz"])


    anal_vz = dv0*np.sin(nmode*2.*np.pi*(xyz[:,2])/(zM-zm))*np.cos(omega*tval)


    plt.title(cname)
    plt.scatter(xyz[:,2], vxyz[:,2],label = "simulation")
    plt.scatter(xyz[:,2], anal_vz,label = "analytic")
    plt.legend()
    plt.savefig(fname)

    




dic_setup = ctx.collect_data()

model = shamrock.BasicSPHGas_M4_single()
model.init()
model.set_cfl_cour(0.3)
model.set_cfl_force(0.3)
model.set_particle_mass(pmass)


twant = 2*np.pi/omega

t_end = model.simulate_until(ctx, 0,twant ,1,1,"dump_")
model.clear()

dic_final = ctx.collect_data()


if(ctx.get_world_rank() == 0):
        

    #plt.style.use("~/Documents/cosmicshine.mplstyle")

    print_dist(dic_setup,"$\omega t = 0$","setup.pdf",0)
    print_dist(dic_final,r"$\omega t = 2 \pi$ ","final.pdf",t_end)


    def get_L2_err(dic, tval):
        xyz = np.array(dic["xyz"])
        vxyz = np.array(dic["vxyz"])


        anal_vz = dv0*np.sin(nmode*2.*np.pi*(xyz[:,2])/(zM-zm))*np.cos(omega*tval)


        err_v = (vxyz[:,2] - anal_vz)**2 + vxyz[:,1]**2 + vxyz[:,0]**2

        return np.average(err_v)/np.max(np.abs(vxyz[:,2]))











    Tex_template = r"""

    \documentclass{article}

    \usepackage[a4paper,total={170mm,260mm},left=20mm,top=20mm,]{geometry}

    \usepackage{fancyhdr} % entêtes et pieds de pages personnalisés

    \pagestyle{fancy}
    \fancyhead[L]{\scriptsize \textsc{Soundwave test}} % À changer
    \fancyhead[R]{\scriptsize \textsc{\textsc{SHAMROCK}}} % À changer
    \fancyfoot[C]{ \thepage}

    \usepackage{graphicx}

    \usepackage{titling}

    \setlength{\droptitle}{-4\baselineskip} % Move the title up

    \pretitle{\begin{center}\Huge\bfseries} % Article title formatting
    \posttitle{\end{center}} % Article title closing formatting
    \title{\textsc{SHAMROCK} Soundwave test} % Article title
    \author{%
    \textsc{Timothée David--Cléris}\thanks{timothee.david--cleris@ens-lyon.fr} \\[1ex] % Your name
    \normalsize CRAL ENS de Lyon \\ % Your institution
    }
    \date{\today}

    \usepackage{xcolor}
    \definecolor{linkcolor}{rgb}{0,0,0.6}


    \usepackage[ pdftex,colorlinks=true,
    pdfstartview=ajustementV,
    linkcolor= linkcolor,
    citecolor= linkcolor,
    urlcolor= linkcolor,
    hyperindex=true,
    hyperfigures=false]
    {hyperref}

    \begin{document}
    \maketitle


    %%content%%
    \end{document}
    """


    fig_tex = r"""

    \begin{figure}[ht!]
    \includegraphics[width=0.5\textwidth]{%%%filename%%%}
    \caption{%%%figcap%%%}
    \end{figure}

    """

    def make_tex_report():

        L2 = get_L2_err(dic_final, t_end)

        ctn = fig_tex.replace("%%%filename%%%", "setup.pdf").replace("%%%figcap%%%", "state after setup")

        ctn += fig_tex.replace("%%%filename%%%", "final.pdf").replace("%%%figcap%%%", "state after a wave period")

        ctn += "L2 error : $L_2 = "+str(L2)+"$"

        out_tex = Tex_template.replace(r"%%content%%",ctn)

        out_file = open("soundwave_test.tex", "w")
        out_file.write(out_tex)
        out_file.close()

    make_tex_report()
