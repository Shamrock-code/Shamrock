from .StandardPlotHelper import StandardPlotHelper


def ColumnDensityPlot(model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix):
    def compute_rho_integ(helper):
        return helper.column_integ_render("rho", "f64")

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
        compute_function=compute_rho_integ,
    )
