from math import *

import matplotlib.pyplot as plt
import numpy as np

import shamrock


def run_sim(times, vg_num, vd_num):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    multx = 1
    multy = 1
    multz = 1

    sz = 1 << 1
    base = 2

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    cfg.set_Csafe(
        0.44
    )  #  TODO : remember to add possibility of different CFL for fluids(e.g Csafe_gas and Csafe_dust ...)
    cfg.set_eos_gamma(1.4)  # set adiabatic index gamma , here adiabatic EOS
    cfg.set_dust_mode_dhll(20)  # enable dust config
    # cfg.set_drag_mode_irk1(True)  # enable drag config
    cfg.set_drag_mode_expo(True)
    cfg.set_face_time_interpolation(True)

    # ======= set drag coefficients for test B ========
    cfg.set_alpha_values(1.0 / 1.58489319e-03)  # ts :=1.58489319e-03
    cfg.set_alpha_values(1.0 / 2.51188643e-03)  # ts :=2.51188643e-03
    cfg.set_alpha_values(1.0 / 3.98107171e-03)  # ts :=3.98107171e-03
    cfg.set_alpha_values(1.0 / 6.30957344e-03)  # ts :=6.30957344e-03
    cfg.set_alpha_values(1.0 / 1.00000000e-02)  # ts :=1.00000000e-02
    cfg.set_alpha_values(1.0 / 1.58489319e-02)  # ts :=1.58489319e-02
    cfg.set_alpha_values(1.0 / 2.51188643e-02)  # ts :=2.51188643e-02
    cfg.set_alpha_values(1.0 / 3.98107171e-02)  # ts :=3.98107171e-02
    cfg.set_alpha_values(1.0 / 6.30957344e-02)  # ts :=6.30957344e-02
    cfg.set_alpha_values(1.0 / 1.00000000e-01)  # ts :=1.00000000e-01
    cfg.set_alpha_values(1.0 / 1.58489319e-01)  # ts :=1.58489319e-01
    cfg.set_alpha_values(1.0 / 2.51188643e-01)  # ts :=2.51188643e-01
    cfg.set_alpha_values(1.0 / 3.98107171e-01)  # ts :=3.98107171e-01
    cfg.set_alpha_values(1.0 / 6.30957344e-01)  # ts :=6.30957344e-01
    cfg.set_alpha_values(1.0 / 1.00000000e00)  # ts :=1.00000000e+00
    cfg.set_alpha_values(1.0 / 1.58489319e00)  # ts :=1.58489319e+00
    cfg.set_alpha_values(1.0 / 2.51188643e00)  # ts :=2.51188643e+00
    cfg.set_alpha_values(1.0 / 3.98107171e00)  # ts :=3.98107171e+00
    cfg.set_alpha_values(1.0 / 6.30957344e00)  # ts :=6.30957344e+00
    cfg.set_alpha_values(1.0 / 1.00000000e01)  # ts :=1.00000000e+01

    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    # ============= Fileds maps for gas ==============

    def rho_map(rmin, rmax):
        return 1  # 1 is the initial density

    def rhovel_map(rmin, rmax):
        return (1, 0, 0)  # vg_x:=1, vg_y:=0, vg_z:=0

    def rhoe_map(rmin, rmax):
        cs_0 = 1.4
        gamma = 1.4
        rho_0 = 1
        press = (cs_0 * rho_0) / gamma
        rhoeint = press / (gamma - 1.0)
        rhoekin = 0.5 * rho_0  # vg_x:=1, vg_y:=0, vg_z:=0
        return rhoeint + rhoekin

    # =========== Fields maps for dust in test B ============
    def b_rho_d_1_map(rmin, rmax):
        return 0.0026154081999410863

    def b_rho_d_2_map(rmin, rmax):
        return 0.003292603845120664

    def b_rho_d_3_map(rmin, rmax):
        return 0.004145142651593601

    def b_rho_d_4_map(rmin, rmax):
        return 0.00521842541960303

    def b_rho_d_5_map(rmin, rmax):
        return 0.006569608370290907

    def b_rho_d_6_map(rmin, rmax):
        return 0.00827064692289489

    def b_rho_d_7_map(rmin, rmax):
        return 0.010412127583209595

    def b_rho_d_8_map(rmin, rmax):
        return 0.013108092005345549

    def b_rho_d_9_map(rmin, rmax):
        return 0.016502110125665475

    def b_rho_d_10_map(rmin, rmax):
        return 0.020774925785426113

    def b_rho_d_11_map(rmin, rmax):
        return 0.026154081999410862

    def b_rho_d_12_map(rmin, rmax):
        return 0.032926038451206716

    def b_rho_d_13_map(rmin, rmax):
        return 0.04145142651593591

    def b_rho_d_14_map(rmin, rmax):
        return 0.05218425419603042

    def b_rho_d_15_map(rmin, rmax):
        return 0.06569608370290894

    def b_rho_d_16_map(rmin, rmax):
        return 0.08270646922894889

    def b_rho_d_17_map(rmin, rmax):
        return 0.10412127583209614

    def b_rho_d_18_map(rmin, rmax):
        return 0.1310809200534553

    def b_rho_d_19_map(rmin, rmax):
        return 0.16502110125665503

    def b_rho_d_20_map(rmin, rmax):
        return 0.20774925785426088

    def b_rhovel_d_1_map(rmin, rmax):
        return (0.5 * 0.0026154081999410863, 0, 0)

    def b_rhovel_d_2_map(rmin, rmax):
        return (0.5 * 0.003292603845120664, 0, 0)

    def b_rhovel_d_3_map(rmin, rmax):
        return (0.5 * 0.004145142651593601, 0, 0)

    def b_rhovel_d_4_map(rmin, rmax):
        return (0.5 * 0.00521842541960303, 0, 0)

    def b_rhovel_d_5_map(rmin, rmax):
        return (0.5 * 0.006569608370290907, 0, 0)

    def b_rhovel_d_6_map(rmin, rmax):
        return (0.5 * 0.00827064692289489, 0, 0)

    def b_rhovel_d_7_map(rmin, rmax):
        return (0.5 * 0.010412127583209595, 0, 0)

    def b_rhovel_d_8_map(rmin, rmax):
        return (0.5 * 0.013108092005345549, 0, 0)

    def b_rhovel_d_9_map(rmin, rmax):
        return (0.5 * 0.016502110125665475, 0, 0)

    def b_rhovel_d_10_map(rmin, rmax):
        return (0.5 * 0.020774925785426113, 0, 0)

    def b_rhovel_d_11_map(rmin, rmax):
        return (0.5 * 0.026154081999410862, 0, 0)

    def b_rhovel_d_12_map(rmin, rmax):
        return (0.5 * 0.032926038451206716, 0, 0)

    def b_rhovel_d_13_map(rmin, rmax):
        return (0.5 * 0.04145142651593591, 0, 0)

    def b_rhovel_d_14_map(rmin, rmax):
        return (0.5 * 0.05218425419603042, 0, 0)

    def b_rhovel_d_15_map(rmin, rmax):
        return (0.5 * 0.06569608370290894, 0, 0)

    def b_rhovel_d_16_map(rmin, rmax):
        return (0.5 * 0.08270646922894889, 0, 0)

    def b_rhovel_d_17_map(rmin, rmax):
        return (0.5 * 0.10412127583209614, 0, 0)

    def b_rhovel_d_18_map(rmin, rmax):
        return (0.5 * 0.1310809200534553, 0, 0)

    def b_rhovel_d_19_map(rmin, rmax):
        return (0.5 * 0.16502110125665503, 0, 0)

    def b_rhovel_d_20_map(rmin, rmax):
        return (0.5 * 0.20774925785426088, 0, 0)

    # ============ set init fields values for gas =============

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    # ============ set init fields values for dusts in test B ==========
    model.set_field_value_lambda_f64("rho_dust", b_rho_d_1_map, 0)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_1_map, 0)
    model.set_field_value_lambda_f64("rho_dust", b_rho_d_2_map, 1)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_2_map, 1)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_3_map, 2)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_3_map, 2)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_4_map, 3)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_4_map, 3)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_5_map, 4)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_5_map, 4)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_6_map, 5)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_6_map, 5)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_7_map, 6)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_7_map, 6)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_8_map, 7)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_8_map, 7)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_9_map, 8)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_9_map, 8)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_10_map, 9)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_10_map, 9)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_11_map, 10)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_11_map, 10)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_12_map, 11)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_12_map, 11)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_13_map, 12)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_13_map, 12)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_14_map, 13)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_14_map, 13)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_15_map, 14)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_15_map, 14)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_16_map, 15)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_16_map, 15)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_17_map, 16)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_17_map, 16)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_18_map, 17)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_18_map, 17)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_19_map, 18)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_19_map, 18)

    model.set_field_value_lambda_f64("rho_dust", b_rho_d_20_map, 19)
    model.set_field_value_lambda_f64_3("rhovel_dust", b_rhovel_d_20_map, 19)

    def convert_to_cell_coords(dic):

        cmin = dic["cell_min"]
        cmax = dic["cell_max"]

        xmin = []
        ymin = []
        zmin = []
        xmax = []
        ymax = []
        zmax = []

        for i in range(len(cmin)):

            m, M = cmin[i], cmax[i]

            mx, my, mz = m
            Mx, My, Mz = M

            for j in range(8):
                a, b = model.get_cell_coords(((mx, my, mz), (Mx, My, Mz)), j)

                x, y, z = a
                xmin.append(x)
                ymin.append(y)
                zmin.append(z)

                x, y, z = b
                xmax.append(x)
                ymax.append(y)
                zmax.append(z)

        dic["xmin"] = np.array(xmin)
        dic["ymin"] = np.array(ymin)
        dic["zmin"] = np.array(zmin)
        dic["xmax"] = np.array(xmax)
        dic["ymax"] = np.array(ymax)
        dic["zmax"] = np.array(zmax)

        return dic

    dt = 0.05
    t = 0
    tend = 2.5

    for i in range(10000000):
        dic_i = convert_to_cell_coords(ctx.collect_data())
        vg_num.append(dic_i["rhovel"][0][0] / dic_i["rho"][0])
        vd_num[0].append(dic_i["rhovel_dust"][0][0] / dic_i["rho_dust"][0])
        vd_num[1].append(dic_i["rhovel_dust"][1][0] / dic_i["rho_dust"][1])
        vd_num[2].append(dic_i["rhovel_dust"][2][0] / dic_i["rho_dust"][2])
        vd_num[3].append(dic_i["rhovel_dust"][3][0] / dic_i["rho_dust"][3])
        vd_num[4].append(dic_i["rhovel_dust"][4][0] / dic_i["rho_dust"][4])
        vd_num[5].append(dic_i["rhovel_dust"][5][0] / dic_i["rho_dust"][5])
        vd_num[6].append(dic_i["rhovel_dust"][6][0] / dic_i["rho_dust"][6])
        vd_num[7].append(dic_i["rhovel_dust"][7][0] / dic_i["rho_dust"][7])
        vd_num[8].append(dic_i["rhovel_dust"][8][0] / dic_i["rho_dust"][8])
        vd_num[9].append(dic_i["rhovel_dust"][9][0] / dic_i["rho_dust"][9])
        vd_num[10].append(dic_i["rhovel_dust"][10][0] / dic_i["rho_dust"][10])
        vd_num[11].append(dic_i["rhovel_dust"][11][0] / dic_i["rho_dust"][11])
        vd_num[12].append(dic_i["rhovel_dust"][12][0] / dic_i["rho_dust"][12])
        vd_num[13].append(dic_i["rhovel_dust"][13][0] / dic_i["rho_dust"][13])
        vd_num[14].append(dic_i["rhovel_dust"][14][0] / dic_i["rho_dust"][14])
        vd_num[15].append(dic_i["rhovel_dust"][15][0] / dic_i["rho_dust"][15])
        vd_num[16].append(dic_i["rhovel_dust"][16][0] / dic_i["rho_dust"][16])
        vd_num[17].append(dic_i["rhovel_dust"][17][0] / dic_i["rho_dust"][17])
        vd_num[18].append(dic_i["rhovel_dust"][18][0] / dic_i["rho_dust"][18])
        vd_num[19].append(dic_i["rhovel_dust"][19][0] / dic_i["rho_dust"][19])

        model.evolve_once_override_time(dt * float(i), dt)
        t = dt * i
        times.append(t)

        if t > tend:
            break


## ========= analytical function for velocity =====
def analytical_velocity(t, Vcom, C, lambdas):
    res = Vcom
    for i in range(len(C)):
        res += C[i] * exp(lambdas[i] * t)
    return res


## set params
Vcom = 0.75
lambdas = [
    -6.32633296e02,
    -3.99434092e02,
    -2.52241238e02,
    -1.59324677e02,
    -1.00662986e02,
    -6.36218168e01,
    -4.02279880e01,
    -2.54495852e01,
    -1.61107420e01,
    -1.02068680e01,
    -6.47255286e00,
    -4.10890184e00,
    -2.61147691e00,
    -1.66171944e00,
    -1.05843168e00,
    -6.74531852e-01,
    -4.29747651e-01,
    -2.73354931e-01,
    -1.08926355e-01,
    -1.73243440e-01,
]

cg = [
    1.34514410e-03,
    1.68115333e-03,
    2.10874242e-03,
    2.64673470e-03,
    3.32109295e-03,
    4.16368597e-03,
    5.21225036e-03,
    6.50963687e-03,
    8.10117619e-03,
    1.00284109e-02,
    1.23165595e-02,
    1.49524861e-02,
    0.01785116,
    0.02081455,
    0.02350039,
    0.02543454,
    0.02609805,
    0.02506717,
    0.01676212,
    0.02208495,
]

cd1 = [
    -5.06416084e-01,
    4.58155295e-03,
    3.51325569e-03,
    3.54084187e-03,
    3.95151854e-03,
    4.63060766e-03,
    5.56719860e-03,
    6.78323792e-03,
    8.31345021e-03,
    1.01933059e-02,
    1.24442161e-02,
    1.50504975e-02,
    0.01792535,
    0.02086951,
    0.02353988,
    0.02546176,
    0.02611584,
    0.02507804,
    0.01676501,
    0.02209102,
]

cd2 = [
    -2.28337679e-03,
    -5.04384867e-01,
    5.75532246e-03,
    4.41273582e-03,
    4.44503871e-03,
    4.95565268e-03,
    5.79814178e-03,
    6.95419393e-03,
    8.44284420e-03,
    1.02922897e-02,
    1.25201156e-02,
    1.51084216e-02,
    0.01796903,
    0.0209018,
    0.02356303,
    0.02547771,
    0.02612625,
    0.02508439,
    0.01676671,
    0.02209456,
]

cd3 = [
    8.85803269e-04,
    -2.84856382e-03,
    -5.03225053e-01,
    7.23711075e-03,
    5.54205074e-03,
    5.57598937e-03,
    6.20617193e-03,
    7.24352653e-03,
    8.65637921e-03,
    1.04531678e-02,
    1.26423231e-02,
    1.52011439e-02,
    0.0180387,
    0.02095317,
    0.02359983,
    0.02550303,
    0.02614278,
    0.02509448,
    0.01676939,
    0.02210019,
]

cd4 = [
    -4.49633410e-04,
    -1.10583369e-03,
    -3.56486733e-03,
    -5.02155417e-01,
    9.10238859e-03,
    6.95601484e-03,
    6.98525885e-03,
    7.75488559e-03,
    9.01785876e-03,
    1.07187061e-02,
    1.28409726e-02,
    1.53504530e-02,
    0.01815023,
    0.0210351,
    0.02365838,
    0.02554325,
    0.02616901,
    0.02511048,
    0.01677365,
    0.02210912,
]

cd5 = [
    -2.52546003e-04,
    -5.61443528e-04,
    -1.38513221e-03,
    -4.46143974e-03,
    -5.00929944e-01,
    1.14455578e-02,
    8.72021902e-03,
    8.73185870e-03,
    9.65698873e-03,
    1.11683496e-02,
    1.31689252e-02,
    1.55931952e-02,
    0.01832984,
    0.02116628,
    0.02375178,
    0.02560727,
    0.02621069,
    0.02513588,
    0.0167804,
    0.02212328,
]

cd6 = [
    -1.49020646e-04,
    -3.15377660e-04,
    -7.03440737e-04,
    -1.73542041e-03,
    -5.57791145e-03,
    -4.99372121e-01,
    1.43814243e-02,
    1.09102877e-02,
    1.08789985e-02,
    1.19637647e-02,
    1.37244561e-02,
    1.59940456e-02,
    0.0186219,
    0.02137756,
    0.02390133,
    0.02570939,
    0.02627702,
    0.02517624,
    0.01679111,
    0.02214576,
]


cd7 = [
    -9.03325094e-05,
    -1.86105588e-04,
    -3.95190614e-04,
    -8.81640997e-04,
    -2.17272249e-03,
    -6.96143084e-03,
    -4.97287047e-01,
    1.80454657e-02,
    1.36081845e-02,
    1.34860244e-02,
    1.47078005e-02,
    1.66733593e-02,
    0.01910435,
    0.02172121,
    0.02414225,
    0.02587292,
    0.02638285,
    0.02524048,
    0.01680811,
    0.02218148,
]

cd8 = [
    -5.56175958e-05,
    -1.12815775e-04,
    -2.33218832e-04,
    -4.95380704e-04,
    -1.10428292e-03,
    -2.71633877e-03,
    -8.66534763e-03,
    -4.94418878e-01,
    2.25898737e-02,
    1.68925936e-02,
    1.65919136e-02,
    1.78767321e-02,
    0.01992239,
    0.02228907,
    0.02453418,
    0.0261364,
    0.02655232,
    0.02534296,
    0.01683512,
    0.02223833,
]

cd9 = [
    -3.45649120e-05,
    -6.94617193e-05,
    -1.41380723e-04,
    -2.92369405e-04,
    -6.20602057e-04,
    -1.38132699e-03,
    -3.38850696e-03,
    -1.07462262e-02,
    -4.90412769e-01,
    2.81704718e-02,
    2.08187310e-02,
    2.01857178e-02,
    0.02137282,
    0.02325252,
    0.02518211,
    0.02656516,
    0.02682543,
    0.02550711,
    0.01687812,
    0.02232903,
]

cd10 = [
    -2.16041145e-05,
    -4.31691360e-05,
    -8.70513395e-05,
    -1.77246974e-04,
    -3.66311889e-04,
    -7.76491028e-04,
    -1.72431270e-03,
    -4.21347031e-03,
    -1.32572709e-02,
    -4.84773310e-01,
    3.49163546e-02,
    2.53814921e-02,
    0.02416066,
    0.02496264,
    0.02628218,
    0.02727428,
    0.02726997,
    0.02577165,
    0.01694671,
    0.0224743,
]

cd11 = [
    -1.35509565e-05,
    -2.69821899e-05,
    -5.41014727e-05,
    -1.09138031e-04,
    -2.22087145e-04,
    -4.58385171e-04,
    -9.69593565e-04,
    -2.14592512e-03,
    -5.21519104e-03,
    -1.62356203e-02,
    -4.76822438e-01,
    4.28704623e-02,
    0.03045707,
    0.02825625,
    0.02823718,
    0.02847913,
    0.02800552,
    0.02620236,
    0.01705658,
    0.02270846,
]

cd12 = [
    -8.51840638e-06,
    -1.69243668e-05,
    -3.38155530e-05,
    -6.78292527e-05,
    -1.36752663e-04,
    -2.77929667e-04,
    -5.72471938e-04,
    -1.20713207e-03,
    -2.65888211e-03,
    -6.41264518e-03,
    -1.96803041e-02,
    -4.65672043e-01,
    0.05188888,
    0.03572731,
    0.03201103,
    0.03062317,
    0.02925619,
    0.02691527,
    0.01723365,
    0.02308974,
]

cd13 = [
    -5.36221934e-06,
    -1.06390274e-05,
    -2.12106484e-05,
    -4.23963949e-05,
    -8.49935293e-05,
    -1.71145773e-04,
    -3.47135053e-04,
    -7.12864383e-04,
    -1.49639317e-03,
    -3.27359187e-03,
    -7.81126088e-03,
    -2.35182119e-02,
    -0.45024469,
    0.06149826,
    0.04061379,
    0.0347721,
    0.03148461,
    0.02812821,
    0.01752195,
    0.02372097,
]

cd14 = [
    -3.37836166e-06,
    -6.69713167e-06,
    -1.33335205e-05,
    -2.65931126e-05,
    -5.31255446e-05,
    -1.06372095e-04,
    -2.13773254e-04,
    -4.32315621e-04,
    -8.83906971e-04,
    -1.84342456e-03,
    -3.99381968e-03,
    -9.38906984e-03,
    -0.02755955,
    -0.42939546,
    0.07074706,
    0.04428025,
    0.03580728,
    0.03029175,
    0.01799916,
    0.02479531,
]

cd15 = [
    -2.12962823e-06,
    -4.21940131e-06,
    8.39329738e-06,
    -1.67171332e-05,
    -3.33232336e-05,
    -6.64893831e-05,
    -1.32870703e-04,
    -2.66247333e-04,
    -5.36120344e-04,
    -1.08923152e-03,
    2.25060586e-03,
    -4.80957163e-03,
    -0.01107751,
    -0.03145525,
    -0.4021857,
    0.07814756,
    0.04576579,
    0.03449713,
    0.01881115,
    0.02671276,
]


cd16 = [
    -1.34291995e-06,
    -2.65979866e-06,
    -5.28804514e-06,
    -1.05232690e-05,
    -2.09479707e-05,
    -4.17061819e-05,
    -8.30544007e-05,
    -1.65492759e-04,
    -3.30204634e-04,
    -6.60772611e-04,
    -1.33032554e-03,
    -2.71263126e-03,
    -0.00568705,
    -0.01274115,
    -0.03468686,
    -0.36829126,
    0.08183878,
    0.04422877,
    0.02025968,
    0.03044404,
]

cd17 = [
    -8.47013139e-07,
    -1.67724020e-06,
    -3.33344601e-06,
    -6.63000572e-06,
    -1.31865808e-05,
    -2.62179054e-05,
    -5.20974254e-05,
    -1.03448312e-04,
    -2.05257512e-04,
    -4.07022084e-04,
    -8.07202981e-04,
    -1.60415557e-03,
    -0.00321079,
    -0.00655773,
    -0.0141683,
    -0.03663086,
    -0.32837113,
    0.07999392,
    0.02307594,
    0.03910002,
]

cd18 = [
    -5.34304999e-07,
    -1.05787753e-06,
    -2.10203597e-06,
    -4.17938719e-06,
    -8.30799625e-06,
    -1.65040281e-05,
    -3.27504481e-05,
    -6.48909087e-05,
    -1.28309002e-04,
    -2.53023709e-04,
    -4.97283195e-04,
    -9.73606512e-04,
    -0.00189977,
    -0.00370667,
    -0.00731258,
    -0.01509146,
    -0.03671354,
    -0.28406149,
    0.02959642,
    0.07117165,
]

cd19 = [
    -3.37074253e-07,
    -6.67320631e-07,
    -1.32580731e-06,
    -2.63547923e-06,
    -5.23715636e-06,
    -1.03981273e-05,
    -2.06163301e-05,
    -4.07933259e-05,
    -8.04870590e-05,
    -1.58174374e-04,
    -3.09157981e-04,
    -5.99889083e-04,
    -0.00115338,
    -0.00219453,
    -0.00413867,
    -0.00781157,
    -0.01524842,
    -0.03458719,
    0.05360085,
    -0.23723737,
]

cd20 = [
    -2.12659809e-07,
    -4.20989183e-07,
    -8.36333811e-07,
    -1.66226414e-06,
    -3.30250035e-06,
    -6.55473375e-06,
    -1.29890648e-05,
    -2.56794611e-05,
    -5.05983808e-05,
    -9.92237244e-05,
    -1.93275125e-04,
    -3.72982096e-04,
    -0.00071078,
    -0.0013328,
    -0.00245196,
    -0.004427,
    -0.00791455,
    -0.01446003,
    -0.18778233,
    -0.0301528,
]

## ===== get numerical results ==================
times = []
vg_num = []
vd_num = [
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
]
vd_anal = []
vg_anal = []
run_sim(times, vg_num, vd_num)

## =========== get analytical results =========
vg_anal = [analytical_velocity(t, Vcom, cg, lambdas) for t in times]
Cd = [
    cd1,
    cd2,
    cd3,
    cd4,
    cd5,
    cd6,
    cd7,
    cd8,
    cd9,
    cd10,
    cd11,
    cd12,
    cd13,
    cd14,
    cd15,
    cd16,
    cd17,
    cd18,
    cd19,
    cd20,
]
for i in range(20):
    vd_i_ana = [analytical_velocity(t, Vcom, Cd[i], lambdas) for t in times]
    vd_anal.append(vd_i_ana)

# ============ plots ======================
plt.figure(figsize=(25, 15))
plt.plot(times, vg_anal, "k--", lw=1.5, label="analytic")
plt.scatter(times, vg_num, marker="^", label="$v_{g}$")
colors = [
    "aqua",
    "blue",
    "chartreuse",
    "chocolate",
    "coral",
    "crimson",
    "cyan",
    "fuchsia",
    "gold",
    "green",
    "lime",
    "magenta",
    "navy",
    "pink",
    "khaki",
    "purple",
    "orange",
    "indigo",
    "tomato",
    "sienna",
]

_ls = "-."
markers = [
    "o",
    "v",
    ">",
    "<",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
]

for i in range(20):
    id = i + 1
    plt.scatter(times, vd_num[i], marker=markers[i], c=colors[i], label="$vd$" + "_" + f"${id}$")
    plt.plot(times, vd_anal[i], "k--", lw=1.5)
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("DustyCollision - 20 dust species with EXPO solver/AdaptiveCPP")
plt.legend(ncol=6)

# plt.savefig("acpp_dusty_collision_test_rk1_20f.png")
plt.savefig("acpp_dusty_collision_test_exp_20f.png")
