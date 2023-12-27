# Shearing box in SPH

Following [Stone & Gardiner 2010](https://ui.adsabs.harvard.edu/abs/2010ApJS..189..142S/abstract):

\[
    \mathbf{f} = 
        2\Omega_0 \left(  q \Omega_0 x +  v_y \right) \mathbf{e}_{x} -2\Omega_0
	v_x \mathbf{e}_{y} - \Omega_0^2 z \mathbf{e}_{z}
\]

Shear speed :

$$
\omega = q \Omega_0 L_x 
$$

To add it : 

```py linenums="1"
xm,ym,zm = bmin
xM,yM,zM = bmax

Omega_0 = 1
eta = 0.00
q = 3./2.

shear_speed = -q*Omega_0*(xM - xm)

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_boundary_shearing_periodic((1,0,0),(0,1,0),shear_speed)
cfg.set_eos_adiabatic(gamma)
cfg.add_ext_force_shearing_box(
    Omega_0  = Omega_0,
    eta      = eta,
    q        = q
)
cfg.set_units(shamrock.UnitSystem())
cfg.print_status()
```