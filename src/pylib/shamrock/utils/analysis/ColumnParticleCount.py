from .StandardPlotHelper import StandardPlotHelper


def ColumnParticleCount(model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix):
    def compute_particle_count(helper):
        return helper.column_integ_render("inv_hpart", "f64")

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_particle_count,
    )
