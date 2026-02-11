from .StandardPlotHelper import StandardPlotHelper


def SliceDtPart(
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
    def compute_dt_part_slice(helper):
        if not model.get_current_config().should_save_dt_to_fields():
            raise ValueError("dt_part is not saved to fields")
        return helper.slice_render("dt_part", "f64", do_normalization, min_normalization)

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
        compute_function=compute_dt_part_slice,
    )
