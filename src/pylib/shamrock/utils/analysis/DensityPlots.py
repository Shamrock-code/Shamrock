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


def SliceDensityPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_rho_slice(helper):
        return helper.slice_render("rho", "f64", do_normalization, min_normalization)

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
        compute_function=compute_rho_slice,
    )
