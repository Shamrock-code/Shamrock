import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

dump = shamrock.load_phantom_dump("reference-files/blast_00010")

cfg = model.gen_config_from_phantom_dump(dump)
cfg.set_boundary_periodic()
cfg.print_status()

model.set_solver_config(cfg)
model.init_scheduler(int(1e5),1)

model.init_from_phantom_dump(dump)
