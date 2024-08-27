import shamrock


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_AMRGodunov(
    context = ctx,
    vector_type = "f64_3",
    grid_repr = "i64_3")


model.init_scheduler(int(1e7),1)

multx = 1
multy = 1
multz = 1

sz = 1 << 4
base = 4
model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

cfg = model.gen_default_config()
scale_fact = 2/(sz*base*multx)
cfg.set_scale_factor(scale_fact)
model.set_config(cfg)

model.evolve_once(0,0)
