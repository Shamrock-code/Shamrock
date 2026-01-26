AMD_ARCH_DESC = {
    "gfx803": "AMD GCN 3.0 & 4.0: MI8, MI6",
    "gfx900": "AMD GCN 5.0: MI25",
    "gfx906": "AMD GCN 5.1: MI50, MI60",
    "gfx908": "AMD CDNA: MI100",
    "gfx90a": "AMD CDNA2: MI250X, MI250, MI210",
    "gfx940": "AMD CDNA3: MI300A",
    "gfx941": "AMD CDNA3: MI300X",
    "gfx942": "AMD CDNA3: MI300X, MI300A",
}

AMD_ARCH_LIST = [k for k in AMD_ARCH_DESC.keys()]


def print_description(arch_code):
    if not (arch_code in AMD_ARCH_LIST):
        print("-- unknown AMD arch code possible list :")
        for k in AMD_ARCH_DESC.keys():
            print("      ", k, AMD_ARCH_DESC[k])
        raise ValueError(f"unknown AMD arch code: {arch_code}")

    print("-- arch :", arch_code)
    print("   ->", AMD_ARCH_DESC[arch_code])
