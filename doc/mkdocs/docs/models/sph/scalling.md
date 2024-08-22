# SPH scalling tests

## Sedov blast


![Sedov scalling CPU](../../assets/figures/sedov_scalling_cpu.svg)
![Sedov scalling CPU](../../assets/figures/sedov_scalling_div_cpu.svg)
![Sedov scalling CPU](../../assets/figures/sedov_scalling_eff_cpu.svg)


![Sedov scalling](../../assets/figures/sedov_scalling_GPU.svg)
![Sedov scalling](../../assets/figures/sedov_scalling_div_GPU.svg)
![Sedov scalling](../../assets/figures/sedov_scalling_eff_GPU.svg)


## Scripts for the scalling tests

```bash
oarsub -q production -p grvingt -l nodes=1,walltime=0:30 --stdout=n2_test1.log --stderr=n2_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=2,walltime=0:30 --stdout=n4_test1.log --stderr=n4_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=3,walltime=0:30 --stdout=n6_test1.log --stderr=n6_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=4,walltime=0:30 --stdout=n8_test1.log --stderr=n8_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=5,walltime=0:30 --stdout=n10_test1.log --stderr=n10_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=6,walltime=0:30 --stdout=n12_test1.log --stderr=n12_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=7,walltime=0:30 --stdout=n14_test1.log --stderr=n14_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=8,walltime=0:30 --stdout=n16_test1.log --stderr=n16_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=10,walltime=0:30 --stdout=n20_test1.log --stderr=n20_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=12,walltime=0:30 --stdout=n24_test1.log --stderr=n24_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=14,walltime=0:30 --stdout=n28_test1.log --stderr=n28_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=16,walltime=0:30 --stdout=n32_test1.log --stderr=n32_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=20,walltime=0:30 --stdout=n40_test1.log --stderr=n40_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=26,walltime=0:30 --stdout=n52_test1.log --stderr=n52_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=32,walltime=0:30 --stdout=n64_test1.log --stderr=n64_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=40,walltime=0:30 --stdout=n80_test1.log --stderr=n80_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=50,walltime=0:30 --stdout=n100_test1.log --stderr=n100_test1.logerr "sh mpicall.sh"
oarsub -q production -p grvingt -l nodes=60,walltime=0:30 --stdout=n120_test1.log --stderr=n120_test1.logerr "sh mpicall.sh"
```

```sh linenums="1" title="mpicall.sh"
mpirun -machinefile $OAR_NODEFILE --bind-to socket -npernode 2 sh runscript.sh
```

```sh linenums="1" title="runscript.sh"
export ACPP_DEBUG_LEVEL=0
export LD_LIBRARY_PATH=/grid5000/spack/v1/opt/spack/linux-debian11-x86_64_v2/gcc-10.4.0/llvm-13.0.1-i53qugtbmlvnfi6tppnc7bresushxg2j/lib:$LD_LIBRARY_PATH
export OMP_SCHEDULE="dynamic"
export OMP_NUM_THREADS=32

./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./sedov_scale_weak.py
```

```py linenums="1" title="sedov_scale_weak.py"
import shamrock


gamma = 5./3.
rho_g = 1
target_tot_u = 1

bmin = (-0.6,-0.6,-0.6)
bmax = ( 0.6, 0.6, 0.6)

N_target_base = 4e6
compute_multiplier = shamrock.sys.world_size()
scheduler_split_val = int(2e6)
scheduler_merge_val = int(1)

N_target = N_target_base*compute_multiplier
xm,ym,zm = bmin
xM,yM,zM = bmax
vol_b = (xM - xm)*(yM - ym)*(zM - zm)

part_vol = vol_b/N_target

#lattice volume
part_vol_lattice = 0.74*part_vol

dr = (part_vol_lattice / ((4./3.)*3.1416))**(1./3.)

pmass = -1




ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")
model.init_scheduler(scheduler_split_val,scheduler_merge_val)
bmin,bmax = model.get_ideal_fcc_box(dr,bmin,bmax)
xm,ym,zm = bmin
xM,yM,zM = bmax
model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)
xc,yc,zc = model.get_closest_part_to((0,0,0))
ctx.close_sched()
del model
del ctx


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e6),1)


bmin = (xm - xc,ym - yc, zm - zc)
bmax = (xM - xc,yM - yc, zM - zc)
xm,ym,zm = bmin
xM,yM,zM = bmax

model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho_g*vol_b)
#print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint","f64", 0 , bmin,bmax)

rinj = 0.008909042924642563*2/2
#rinj = 0.008909042924642563*2*2
#rinj = 0.01718181
u_inj = 1
model.add_kernel_value("uint","f64", u_inj,(0,0,0),rinj)

model.set_particle_mass(pmass)

model.set_cfl_cour(0.01)
model.set_cfl_force(0.01)

t_sum = 0
t_target = 0.1
current_dt = 1e-7
i = 0
i_dump = 0
while t_sum < t_target:

    #print("step : t=",t_sum)

    next_dt = model.evolve(t_sum,current_dt, False, "dump_"+str(i_dump)+".vtk", False)

    if i % 1 == 0:
        i_dump += 1

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum

    i+= 1

    if i > 5:
        break

res_rate,res_cnt = model.solver_logs_last_rate(), model.solver_logs_last_obj_count()

if shamrock.sys.world_rank() == 0:
    print("result rate :",res_rate)
    print("result cnt :",res_cnt)
```
