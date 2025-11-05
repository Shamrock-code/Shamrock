"""
Symbolic SPH kernels & c++ tests
================================

"""

from __future__ import division

import matplotlib.pyplot as plt
import mpmath
import numpy as np
from sympy import *

import shamrock

# Set precision to 32 digits
mpmath.mp.dps = 32

q = symbols("q")

# %%
# Utilities
# ^^^^^^^^^^^^^^^^^^^


def getnorm(w, R):
    c1D = sympify(1) / (2 * integrate(w, (q, 0, R)))
    c2D = sympify(1) / (integrate(2 * pi * q * w, (q, 0, R)))
    c3D = sympify(1) / (integrate(4 * pi * q * q * w, (q, 0, R)))
    return (c1D, c2D, c3D)


def replace_pi_constants(cpp_code):
    """Replace M_PI and M_1_PI with shambase::constants::pi<Tscal>"""
    cpp_code = cpp_code.replace("M_1_PI", "1/shambase::constants::pi<Tscal>")
    cpp_code = cpp_code.replace("M_PI", "shambase::constants::pi<Tscal>")
    return cpp_code


def sympy_to_cpp(expr):
    """Convert a sympy expression to C++ code with proper formatting"""

    import re

    # Expand the expression first
    # expr = expand(expr)
    cpp_code = ccode(expr)

    # Replace pow(q, n) with q*q*q... for powers up to 11
    # for power in range(11, 1, -1):  # From 11 down to 2
    #     q_mult = '*'.join(['q'] * power)
    #     cpp_code = cpp_code.replace(f'pow(q, {power})', q_mult)

    # Replace pow(base, n) with sham::pow_constexpr<n>(base)
    # Use regex to match pow(anything, number) pattern
    def replace_pow_with_constexpr(match):
        base = match.group(1)
        exponent = match.group(2)
        return f"sham::pow_constexpr<{exponent}>({base})"

    cpp_code = re.sub(r"pow\(([^,]+),\s*(\d+)\)", replace_pow_with_constexpr, cpp_code)

    # Wrap divisions in parentheses for clarity
    # Pattern: number/number followed by * (but not already in parentheses)
    # Match something like: 1155.0/512.0*q  -> (1155.0/512.0)*q
    cpp_code = re.sub(r"(\d+\.?\d*)/(\d+\.?\d*)(\*)", r"(\1/\2)\3", cpp_code)

    # Replace pi constants
    cpp_code = replace_pi_constants(cpp_code)

    return cpp_code


def process_power_variables(cpp_exprs, indent=8):
    """
    Self-contained power variable processing:
    1. Extracts sham::pow_constexpr calls from expressions
    2. Replaces them with temporary variable names (t1, t2, ...)
    3. Generates header declarations

    Returns: (modified_exprs, power_header_string)
    """
    spaces = " " * indent

    # Find all sham::pow_constexpr<n>(...) calls
    # Need to handle nested parentheses properly
    def find_pow_constexpr_calls(text):
        """Find all sham::pow_constexpr calls with proper parenthesis matching"""
        calls = []
        pattern = "sham::pow_constexpr<"
        pos = 0

        while True:
            pos = text.find(pattern, pos)
            if pos == -1:
                break

            # Find the power number
            power_start = pos + len(pattern)
            power_end = text.find(">", power_start)
            power = text[power_start:power_end]

            # Find the matching closing parenthesis
            paren_start = text.find("(", power_end)
            if paren_start == -1:
                break

            # Count parentheses to find the matching closing one
            paren_count = 1
            i = paren_start + 1
            while i < len(text) and paren_count > 0:
                if text[i] == "(":
                    paren_count += 1
                elif text[i] == ")":
                    paren_count -= 1
                i += 1

            if paren_count == 0:
                # Found matching closing parenthesis
                full_match = text[pos:i]
                base = text[paren_start + 1 : i - 1]
                calls.append((full_match, power, base))
                pos = i
            else:
                pos += 1

        return calls

    # Dictionary to store unique pow_constexpr calls
    # Key: full match string, Value: (var_name, power, base)
    pow_calls = {}
    var_counter = 1

    vars = {}

    for expr in cpp_exprs:
        for full_match, power, base in find_pow_constexpr_calls(expr):
            vars[full_match] = (power, base)

    # print(vars)

    # sort them lexicographically by match (from the end of the key)
    sorted_vars = sorted(vars.items(), key=lambda x: x[0])
    for full_match, (power, base) in sorted_vars:
        if full_match not in pow_calls:
            var_name = f"t{var_counter}"
            pow_calls[full_match] = (var_name, power, base)
            var_counter += 1

    # Replace pow_constexpr calls with variable names in expressions
    # Sort by length (longest first) to avoid partial replacements
    sorted_pow_calls = sorted(pow_calls.items(), key=lambda x: len(x[0]), reverse=True)
    for full_match, (var_name, power, base) in sorted_pow_calls:
        cpp_exprs = [expr.replace(full_match, var_name) for expr in cpp_exprs]

    # Generate header declarations
    power_header = ""
    for full_match, (var_name, power, base) in pow_calls.items():
        power_header += f"{spaces}    Tscal {var_name} = sham::pow_constexpr<{power}>({base});\n"

    return cpp_exprs, power_header


def generate_function_body(pieces, cpp_exprs, indent=8):
    """Generate the body section with if-else conditions and return statements"""
    spaces = " " * indent
    body = ""

    for i, ((expr, cond), cpp_expr) in enumerate(zip(pieces, cpp_exprs)):
        if cond == True:
            body += f"{spaces}    else\n"
            body += f"{spaces}        return {cpp_expr};\n"
        else:
            if_keyword = "if" if i == 0 else "else if"
            # Convert condition to string
            cond_str = str(cond)

            body += f"{spaces}    {if_keyword} ({cond_str}) {{\n"
            body += f"{spaces}        return {cpp_expr};\n"
            body += f"{spaces}    }}"
            if i < len(pieces) - 1 and pieces[i + 1][1] != True:
                body += " "
            else:
                body += "\n"

    return body


def generate_piecewise_function(func_expr, func_name, indent=8):
    """
    Generate C++ code for a piecewise function.
    Works with separate header (declarations) and body (logic) sections.
    """
    spaces = " " * indent

    # Extract pieces from Piecewise object
    pieces = []
    for arg in func_expr.args:
        # Each arg is an ExprCondPair (expr, cond) tuple
        if hasattr(arg, "expr") and hasattr(arg, "cond"):
            pieces.append((arg.expr, arg.cond))
        elif isinstance(arg, tuple) and len(arg) == 2:
            pieces.append(arg)

    # Phase 1: Generate raw C++ expressions for the body
    cpp_exprs = []
    for expr, _ in pieces:
        cpp_exprs.append(sympy_to_cpp(expr))

    # Phase 2: Process power variables (self-contained: detect, replace, generate header)
    cpp_exprs, power_header = process_power_variables(cpp_exprs, indent)

    # Phase 3: Extract constants and replace in body
    # constants = extract_constants(cpp_exprs)
    # cpp_exprs = replace_constants_in_body(cpp_exprs, constants)

    # Phase 4: Generate constant declarations header
    # const_header = generate_constant_header(constants, indent)

    # Phase 5: Generate body (if-else structure)
    body = generate_function_body(pieces, cpp_exprs, indent)

    # Phase 6: Combine everything (constants first, then powers, then body)
    code = f"{spaces}inline static Tscal {func_name}(Tscal q) {{\n"
    # if const_header:
    #    code += const_header
    if power_header:
        code += power_header
    code += body
    code += f"{spaces}}}\n"

    return code


def kernel_to_shamrock(kernel_gen):
    """
    Generate complete C++ kernel definition from SymPy expression

    Args:
        f_func: Function that returns (R, f_expr, name) tuple
        generate_phi: Whether to generate phi_tilde_3d (requires manual derivation)
    """
    R, f_expr, name = kernel_gen()

    # Compute normalization constants
    c_norm_1d, c_norm_2d, c_norm_3d = getnorm(f_expr, R)

    class text_body:
        def __init__(self):
            self.text = ""

        def __call__(self, text=""):
            self.text += text + "\n"

    text = text_body()

    # Generate class header
    text("template<class Tscal>")
    text(f"class KernelDef{name} {{")
    text("    public:")
    text(
        f"    inline static constexpr Tscal Rkern = {ccode(R)}; ///< Compact support radius of the kernel"
    )
    text("    /// default hfact to be used for this kernel")
    text("    inline static constexpr Tscal hfactd = 1.2;")
    text()

    # Normalize constants with pi handling
    norm_1d_str = replace_pi_constants(ccode(c_norm_1d))
    norm_2d_str = replace_pi_constants(ccode(c_norm_2d))
    norm_3d_str = replace_pi_constants(ccode(c_norm_3d))

    text("    /// 1D norm of the kernel")
    text(f"    inline static constexpr Tscal norm_1d = {norm_1d_str};")
    text("    /// 2D norm of the kernel")
    text(f"    inline static constexpr Tscal norm_2d = {norm_2d_str};")
    text("    /// 3D norm of the kernel")
    text(f"    inline static constexpr Tscal norm_3d = {norm_3d_str};")
    text()

    from sympy.polys.polyfuncs import horner

    # Generate f function
    text(generate_piecewise_function(f_expr, "f", indent=4))
    text()

    # Generate df function
    df_expr = diff(f_expr, q)
    text(generate_piecewise_function(df_expr, "df", indent=4))

    # Generate ddf function
    ddf_expr = diff(df_expr, q)
    text(generate_piecewise_function(ddf_expr, "ddf", indent=4))

    text("};")
    text()

    return {
        "R": R,
        "name": name,
        "norm_1d": c_norm_1d.evalf(),
        "norm_2d": c_norm_2d.evalf(),
        "norm_3d": c_norm_3d.evalf(),
        "Rkern": R,
        "f": f_expr,
        "df": df_expr,
        "ddf": ddf_expr,
        "text": text.text,
    }


# %%
# Testing the kernels
# ^^^^^^^^^^^^^^^^^^^^


def test_kernel(ret, tolerance=1e-12):

    R = ret["R"]
    name = ret["name"]
    norm_1d = ret["norm_1d"]
    norm_2d = ret["norm_2d"]
    norm_3d = ret["norm_3d"]
    Rkern = ret["Rkern"]
    f_expr = ret["f"]
    df_expr = ret["df"]
    ddf_expr = ret["ddf"]
    text = ret["text"]

    print("------------------------------------------")
    print(f"Testing kernel {name} matching with Shamrock code")
    print("------------------------------------------")

    f = lambdify((q), f_expr, modules=["mpmath"])
    df = lambdify((q), df_expr, modules=["mpmath"])
    ddf = lambdify((q), ddf_expr, modules=["mpmath"])

    print("Testing norms:")
    shamrock_norm_1d = getattr(shamrock.math.sphkernel, f"{name}_norm_1d")()
    shamrock_norm_2d = getattr(shamrock.math.sphkernel, f"{name}_norm_2d")()
    shamrock_norm_3d = getattr(shamrock.math.sphkernel, f"{name}_norm_3d")()

    print(f"Shamrock norm_1d = {shamrock_norm_1d}, delta={abs(shamrock_norm_1d - norm_1d)}")
    print(f"Shamrock norm_2d = {shamrock_norm_2d}, delta={abs(shamrock_norm_2d - norm_2d)}")
    print(f"Shamrock norm_3d = {shamrock_norm_3d}, delta={abs(shamrock_norm_3d - norm_3d)}")

    # test the norm constants down to 1e-12
    assert abs(shamrock_norm_1d - norm_1d) < tolerance
    assert abs(shamrock_norm_2d - norm_2d) < tolerance
    assert abs(shamrock_norm_3d - norm_3d) < tolerance

    print()
    print("Testing kernel radius:")
    shamrock_Rkern = getattr(shamrock.math.sphkernel, f"{name}_Rkern")()

    print(f"Shamrock Rkern = {shamrock_Rkern}, delta={abs(shamrock_Rkern - Rkern)}")
    assert abs(shamrock_Rkern - Rkern) < tolerance

    print()
    print("Testing kernel functions:")
    shamrock_f = getattr(shamrock.math.sphkernel, f"{name}_f")
    shamrock_df = getattr(shamrock.math.sphkernel, f"{name}_df")
    #shamrock_ddf = getattr(shamrock.math.sphkernel, f"{name}_ddf")

    print(f"Shamrock f(q) = {shamrock_f}")
    print(f"Shamrock df(q) = {shamrock_df}")
    #print(f"Shamrock ddf(q) = {shamrock_ddf}")

    q_arr = np.linspace(0, 1.1 * float(Rkern), 1000)
    shamrock_f = [shamrock_f(x) for x in q_arr]
    shamrock_df = [shamrock_df(x) for x in q_arr]
    #shamrock_ddf = [shamrock_ddf(x) for x in q_arr]

    sympy_f = [f(x) for x in q_arr]
    sympy_df = [df(x) for x in q_arr]
    sympy_ddf = [ddf(x) for x in q_arr]

    # compute the absolute error
    abs_err_f = np.max(np.abs(np.array(shamrock_f) - np.array(sympy_f)))
    abs_err_df = np.max(np.abs(np.array(shamrock_df) - np.array(sympy_df)))
    #abs_err_ddf = np.max(np.abs(np.array(shamrock_ddf) - np.array(sympy_ddf)))

    print(f"Absolute error f(q) = {abs_err_f}")
    print(f"Absolute error df(q) = {abs_err_df}")
    #print(f"Absolute error ddf(q) = {abs_err_ddf}")

    assert abs_err_f < tolerance
    assert abs_err_df < tolerance*10
    #assert abs_err_ddf < tolerance*100

    print("------------------------------------------")
    print("")

    return {
        "q_arr": q_arr,
        "shamrock_Cf": np.array(shamrock_f) * norm_3d,
        "shamrock_Cdf": np.array(shamrock_df) * norm_3d,
    }


def print_kernel_info(ret):
    print("------------------------------------------")
    print(f"Sympy expression for kernel {ret['name']}")
    print("------------------------------------------")
    print(f"f(q)   = {ret['f']}")
    print(f"df(q)  = {ret['df']}")
    print(f"ddf(q) = {ret['ddf']}")
    print("------------------------------------------")
    print("")


def print_kernel_cpp_code(ret):
    print("------------------------------------------")
    print(f"C++ generated code for kernel {ret['name']}")
    print("------------------------------------------")
    print(ret["text"])
    print("------------------------------------------")
    print("")


# %%
# All the SPH kernels
# ^^^^^^^^^^^^^^^^^^^^


# %%
# Cubic splines kernels (M-series)
def m4():
    Rkern = 2
    R = sympify(Rkern)
    f = Piecewise(
        (sympify(1) / 4 * (R - q) ** 3 - (R / 2 - q) ** 3, q < R / 2),
        (sympify(1) / 4 * (R - q) ** 3, q < R),
        (0, True),
    )
    return (R, f, "M4")


def m5():
    Rkern = sympify(5) / 2
    R = Rkern
    term1 = (R - q) ** 4
    term2 = -5 * (sympify(3) / 5 * R - q) ** 4
    term3 = 10 * (sympify(1) / 5 * R - q) ** 4
    f = Piecewise(
        (term1 + term2 + term3, q < sympify(1) / 5 * R),
        (term1 + term2, q < sympify(3) / 5 * R),
        (term1, q < R),
        (0, True),
    )
    return (R, f, "M5")


def m6():
    Rkern = 3
    R = sympify(Rkern)
    term1 = (R - q) ** 5
    term2 = -6 * (sympify(2) / 3 * R - q) ** 5
    term3 = 15 * (sympify(1) / 3 * R - q) ** 5
    f = Piecewise(
        (term1 + term2 + term3, q < sympify(1) / 3 * R),
        (term1 + term2, q < sympify(2) / 3 * R),
        (term1, q < R),
        (0, True),
    )
    return (R, f, "M6")


def m7():
    Rkern = sympify(7) / 2
    R = Rkern
    term1 = (R - q) ** 6
    term2 = -7 * (sympify(5) / 7 * R - q) ** 6
    term3 = 21 * (sympify(3) / 7 * R - q) ** 6
    term4 = -35 * (sympify(1) / 7 * R - q) ** 6
    f = Piecewise(
        (term1 + term2 + term3 + term4, q < sympify(1) / 7 * R),
        (term1 + term2 + term3, q < sympify(3) / 7 * R),
        (term1 + term2, q < sympify(5) / 7 * R),
        (term1, q < R),
        (0, True),
    )
    return (R, f, "M7")


def m8():
    Rkern = 4
    R = sympify(Rkern)
    term1 = (4 - q) ** 7
    term2 = -8 * (3 - q) ** 7
    term3 = 28 * (2 - q) ** 7
    term4 = -56 * (1 - q) ** 7
    f = Piecewise(
        (term1 + term2 + term3 + term4, q < 1),
        (term1 + term2 + term3, q < 2),
        (term1 + term2, q < 3),
        (term1, q < R),
        (0, True),
    )
    return (R, f, "M8")


def m9():
    Rkern = sympify(9) / 2
    R = Rkern
    term1 = (R - q) ** 8
    term2 = -9 * (sympify(7) / 9 * R - q) ** 8
    term3 = 36 * (sympify(5) / 9 * R - q) ** 8
    term4 = -84 * (sympify(3) / 9 * R - q) ** 8
    term5 = 126 * (sympify(1) / 9 * R - q) ** 8
    f = Piecewise(
        (term1 + term2 + term3 + term4 + term5, q < sympify(1) / 9 * R),
        (term1 + term2 + term3 + term4, q < sympify(3) / 9 * R),
        (term1 + term2 + term3, q < sympify(5) / 9 * R),
        (term1 + term2, q < sympify(7) / 9 * R),
        (term1, q < R),
        (0, True),
    )
    return (R, f, "M9")


def m10():
    Rkern = 5
    R = sympify(Rkern)
    term1 = (R - q) ** 9
    term2 = -10 * (sympify(4) / 5 * R - q) ** 9
    term3 = 45 * (sympify(3) / 5 * R - q) ** 9
    term4 = -120 * (sympify(2) / 5 * R - q) ** 9
    term5 = 210 * (sympify(1) / 5 * R - q) ** 9
    f = Piecewise(
        (term1 + term2 + term3 + term4 + term5, q < sympify(1) / 5 * R),
        (term1 + term2 + term3 + term4, q < sympify(2) / 5 * R),
        (term1 + term2 + term3, q < sympify(3) / 5 * R),
        (term1 + term2, q < sympify(4) / 5 * R),
        (term1, q < R),
        (0, True),
    )
    return (R, f, "M10")


# %%
# Wendland kernels (C-series)
def c2():
    Rkern = 2
    R = sympify(Rkern)
    f = Piecewise(((1 - q / 2) ** 4 * (1 + 2 * q), q < R), (0, True))
    return (R, f, "C2")


def c4():
    Rkern = 2
    R = sympify(Rkern)
    f = Piecewise(((1 - q / 2) ** 6 * (1 + 3 * q + sympify(35) / 12 * q**2), q < R), (0, True))
    return (R, f, "C4")


def c6():
    Rkern = 2
    R = sympify(Rkern)
    f = Piecewise(
        ((1 - q / 2) ** 8 * (1 + 4 * q + sympify(25) / 4 * q**2 + 4 * q**3), q < R), (0, True)
    )
    return (R, f, "C6")

# %%
# Double hump
def m4dh():
    Rkern = 2
    R = sympify(Rkern)
    f = q*q*Piecewise(
        (sympify(1) / 4 * (R - q) ** 3 - (R / 2 - q) ** 3, q < R / 2),
        (sympify(1) / 4 * (R - q) ** 3, q < R),
        (0, True),
    )
    return (R, f, "M4DH")

# %%
# M4 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m4)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m4 = test_kernel(ret)


# %%
# M5 Kernel
# ^^^^^^^^^^

# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m5)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m5 = test_kernel(ret)


# %%
# M6 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m6)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m6 = test_kernel(ret)


# %%
# M7 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m7)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m7 = test_kernel(ret, tolerance=1e-11)

# %%
# M8 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m8)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m8 = test_kernel(ret, tolerance=1e-10)

# %%
# M9 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m9)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m9 = test_kernel(ret, tolerance=1e-9)

# %%
# M10 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m10)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m10 = test_kernel(ret, tolerance=1e-8)

# %%
# C2 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(c2)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_c2 = test_kernel(ret)

# %%
# C4 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(c4)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_c4 = test_kernel(ret)

# %%
# C6 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(c6)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_c6 = test_kernel(ret)


# %%
# M4DH Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m4dh)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m4dh = test_kernel(ret)

# %%
# Plot the kernels
# ^^^^^^^^^^^^^^^^

# %%
# Cubic splines kernels (M-series)

plt.figure()
ax_f = plt.subplot(2, 1, 1)
ax_df = plt.subplot(2, 1, 2)

ax_f.plot(test_result_m4["q_arr"], test_result_m4["shamrock_Cf"], label="Shamrock C_3d f_m4(q)")
ax_df.plot(test_result_m4["q_arr"], test_result_m4["shamrock_Cdf"], label="Shamrock C_3d df_m4(q)")

ax_f.plot(test_result_m5["q_arr"], test_result_m5["shamrock_Cf"], label="Shamrock C_3d f_m5(q)")
ax_df.plot(test_result_m5["q_arr"], test_result_m5["shamrock_Cdf"], label="Shamrock C_3d df_m5(q)")

ax_f.plot(test_result_m6["q_arr"], test_result_m6["shamrock_Cf"], label="Shamrock C_3d f_m6(q)")
ax_df.plot(test_result_m6["q_arr"], test_result_m6["shamrock_Cdf"], label="Shamrock C_3d df_m6(q)")

ax_f.plot(test_result_m7["q_arr"], test_result_m7["shamrock_Cf"], label="Shamrock C_3d f_m7(q)")
ax_df.plot(test_result_m7["q_arr"], test_result_m7["shamrock_Cdf"], label="Shamrock C_3d df_m7(q)")

ax_f.plot(test_result_m8["q_arr"], test_result_m8["shamrock_Cf"], label="Shamrock C_3d f_m8(q)")
ax_df.plot(test_result_m8["q_arr"], test_result_m8["shamrock_Cdf"], label="Shamrock C_3d df_m8(q)")

ax_f.plot(test_result_m9["q_arr"], test_result_m9["shamrock_Cf"], label="Shamrock C_3d f_m9(q)")
ax_df.plot(test_result_m9["q_arr"], test_result_m9["shamrock_Cdf"], label="Shamrock C_3d df_m9(q)")

ax_f.plot(test_result_m10["q_arr"], test_result_m10["shamrock_Cf"], label="Shamrock C_3d f_m10(q)")
ax_df.plot(
    test_result_m10["q_arr"], test_result_m10["shamrock_Cdf"], label="Shamrock C_3d df_m10(q)"
)

ax_f.set_title("C_3d f(q)")
ax_df.set_title("C_3d df(q)")
ax_f.set_xlabel("q")
ax_df.set_xlabel("q")
ax_f.legend()
ax_df.legend()
plt.tight_layout()
plt.show()

# %%
# Wendland kernels (C-series)

plt.figure()
ax_f = plt.subplot(2, 1, 1)
ax_df = plt.subplot(2, 1, 2)

ax_f.plot(test_result_c2["q_arr"], test_result_c2["shamrock_Cf"], label="Shamrock C_3d f_c2(q)")
ax_df.plot(test_result_c2["q_arr"], test_result_c2["shamrock_Cdf"], label="Shamrock C_3d df_c2(q)")

ax_f.plot(test_result_c4["q_arr"], test_result_c4["shamrock_Cf"], label="Shamrock C_3d f_c4(q)")
ax_df.plot(test_result_c4["q_arr"], test_result_c4["shamrock_Cdf"], label="Shamrock C_3d df_c4(q)")

ax_f.plot(test_result_c6["q_arr"], test_result_c6["shamrock_Cf"], label="Shamrock C_3d f_c6(q)")
ax_df.plot(test_result_c6["q_arr"], test_result_c6["shamrock_Cdf"], label="Shamrock C_3d df_c6(q)")

ax_f.set_title("C_3d f(q)")
ax_df.set_title("C_3d df(q)")
ax_f.set_xlabel("q")
ax_df.set_xlabel("q")
ax_f.legend()
ax_df.legend()
plt.tight_layout()
plt.show()
