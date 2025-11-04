"""
Testing reload of old dump
==========================

CI test for reload of old dump
"""

import os

import shamrock

os.environ["SHAMDUMP_OFFSET_MODE_OLD"] = "1"


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

model.load_from_dump("reference-files/old_sham_dump/santa_barbara_100K_0000.sham")
