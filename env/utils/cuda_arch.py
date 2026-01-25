NVIDIA_ARCH_DESC = {
    "sm_50": "Nvidia Maxwell: Tesla/Quadro M series.",
    "sm_52": "Nvidia Maxwell: Quadro M6000 , GeForce 900, GTX-970, GTX-980, GTX Titan X",
    "sm_53": "Nvidia Maxwell: Tegra (Jetson) TX1 / Tegra X1, Drive CX, Drive PX, Jetson Nano",
    "sm_60": "Nvidia Pascal: Quadro GP100, Tesla P100, DGX-1",
    "sm_61": "Nvidia Pascal: GTX 1030->1080, GT 1010, Titan Xp, Tesla P40, Tesla P4, Drive PX2",
    "sm_62": "Nvidia Pascal: iGPU on Drive PX2, Tegra (Jetson) TX2",
    "sm_70": "Nvidia Volta: DGX-1 with Volta, Tesla V100, GTX 1180, Titan V, Quadro GV100",
    "sm_72": "Nvidia Volta: Jetson AGX Xavier, Drive AGX Pegasus, Xavier NX ",
    "sm_75": "Nvidia Turing: GTX 1660 Ti, RTX 2060->2080, Titan RTX, Quadro RTX 4000->8000, Quadro T1000/T2000, Tesla T4",
    "sm_80": "Nvidia Ampere: A100",
    "sm_86": "Nvidia Ampere: RTX 3050->3090, RTX A2000->A6000, RTX A10->A40, A2 Tensor Core GPU, A800 40GB",
    "sm_87": "Nvidia Ampere: Jetson AGX Orin and Drive AGX Orin only",
    "sm_89": "Nvidia Ada Lovelace: NVIDIA GeForce RTX 4090, RTX 4080, RTX 6000 Ada, Tesla L40, L40s Ada, L4 Ada",
    "sm_90": "Nvidia Hopper: NVIDIA H100 (GH100), NVIDIA H200",
    "sm_90a": "Nvidia Hopper: NVIDIA H100 (GH100), NVIDIA H200 (for PTX ISA version 8.0)",
    "sm_95": "Nvidia Blackwell: NVIDIA B100 (GB100), B200",
}

NVIDIA_ARCH_LIST = [k for k in NVIDIA_ARCH_DESC.keys()]


def print_description(arch_code):
    if not (arch_code in NVIDIA_ARCH_LIST):
        print("-- unknown cuda arch code possible list :")
        for k in NVIDIA_ARCH_DESC.keys():
            print("      ", k, NVIDIA_ARCH_DESC[k])
        raise ValueError(f"unknown cuda arch code: {arch_code}")

    print("-- arch :", arch_code)
    print("   ->", NVIDIA_ARCH_DESC[arch_code])
