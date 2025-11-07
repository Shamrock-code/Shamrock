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
            cond_str = ccode(cond)

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

    # phi_expr
    phi_tilde_3d_expr = integrate(integrate(f_expr * 4 * pi * q * q, q) * (1 / q**2), q).simplify()

    # correct phi to have 0 at infinity
    filter_cond = next(expr for expr, cond in phi_tilde_3d_expr.args if cond == True)
    lim_phi = limit(filter_cond, q, oo)
    phi_tilde_3d_expr -= lim_phi
    phi_tilde_3d_expr = phi_tilde_3d_expr.simplify()

    text(generate_piecewise_function(phi_tilde_3d_expr, "phi_tilde_3d", indent=4))

    # phi_tilde_3d_prime
    phi_tilde_3d_prime_expr = diff(phi_tilde_3d_expr, q)

    text(generate_piecewise_function(phi_tilde_3d_prime_expr, "phi_tilde_3d_prime", indent=4))

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
        "phi_tilde_3d": phi_tilde_3d_expr,
        "phi_tilde_3d_prime": phi_tilde_3d_prime_expr,
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
    phi_tilde_3d_expr = ret["phi_tilde_3d"]
    phi_tilde_3d_prime_expr = ret["phi_tilde_3d_prime"]
    text = ret["text"]

    print("------------------------------------------")
    print(f"Testing kernel {name} matching with Shamrock code")
    print("------------------------------------------")

    f = lambdify((q), f_expr, modules=["mpmath"])
    df = lambdify((q), df_expr, modules=["mpmath"])
    ddf = lambdify((q), ddf_expr, modules=["mpmath"])
    phi_tilde_3d = lambdify((q), phi_tilde_3d_expr, modules=["mpmath"])
    phi_tilde_3d_prime = lambdify((q), phi_tilde_3d_prime_expr, modules=["mpmath"])

    print("Testing norms:")
    shamrock_norm_1d = getattr(shamrock.math.sphkernel, f"{name}_norm_1d")()
    shamrock_norm_2d = getattr(shamrock.math.sphkernel, f"{name}_norm_2d")()
    shamrock_norm_3d = getattr(shamrock.math.sphkernel, f"{name}_norm_3d")()

    print(
        f"Shamrock norm_1d = {shamrock_norm_1d}, expected={norm_1d}, delta={abs(shamrock_norm_1d - norm_1d)}"
    )
    print(
        f"Shamrock norm_2d = {shamrock_norm_2d}, expected={norm_2d}, delta={abs(shamrock_norm_2d - norm_2d)}"
    )
    print(
        f"Shamrock norm_3d = {shamrock_norm_3d}, expected={norm_3d}, delta={abs(shamrock_norm_3d - norm_3d)}"
    )

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
    shamrock_ddf = getattr(shamrock.math.sphkernel, f"{name}_ddf")
    shamrock_phi_tilde_3d = getattr(shamrock.math.sphkernel, f"{name}_phi_tilde_3d")
    shamrock_phi_tilde_3d_prime = getattr(shamrock.math.sphkernel, f"{name}_phi_tilde_3d_prime")

    print(f"Shamrock f(q) = {shamrock_f}")
    print(f"Shamrock df(q) = {shamrock_df}")
    print(f"Shamrock ddf(q) = {shamrock_ddf}")
    print(f"Shamrock phi_tilde_3d(q) = {shamrock_phi_tilde_3d}")
    print(f"Shamrock phi_tilde_3d_prime(q) = {shamrock_phi_tilde_3d_prime}")

    q_arr = np.linspace(0, max(1.1 * float(Rkern), 5.0), 1000)
    shamrock_f = [shamrock_f(x) for x in q_arr]
    shamrock_df = [shamrock_df(x) for x in q_arr]
    shamrock_ddf = [shamrock_ddf(x) for x in q_arr]
    shamrock_phi_tilde_3d = [shamrock_phi_tilde_3d(x) for x in q_arr]
    shamrock_phi_tilde_3d_prime = [shamrock_phi_tilde_3d_prime(x) for x in q_arr]

    sympy_f = [f(x) for x in q_arr]
    sympy_df = [df(x) for x in q_arr]
    sympy_ddf = [ddf(x) for x in q_arr]
    sympy_phi_tilde_3d = [phi_tilde_3d(x) for x in q_arr]
    sympy_phi_tilde_3d_prime = [phi_tilde_3d_prime(x) for x in q_arr]

    # compute the absolute error
    abs_err_f = np.max(np.abs(np.array(shamrock_f) - np.array(sympy_f)))
    abs_err_df = np.max(np.abs(np.array(shamrock_df) - np.array(sympy_df)))
    abs_err_ddf = np.max(np.abs(np.array(shamrock_ddf) - np.array(sympy_ddf)))
    abs_err_phi_tilde_3d = np.max(
        np.abs(np.array(shamrock_phi_tilde_3d) - np.array(sympy_phi_tilde_3d))
    )
    abs_err_phi_tilde_3d_prime = np.max(
        np.abs(np.array(shamrock_phi_tilde_3d_prime) - np.array(sympy_phi_tilde_3d_prime))
    )

    print(f"Absolute error f(q) = {abs_err_f}")
    print(f"Absolute error df(q) = {abs_err_df}")
    print(f"Absolute error ddf(q) = {abs_err_ddf}")
    print(f"Absolute error phi_tilde_3d(q) = {abs_err_phi_tilde_3d}")
    print(f"Absolute error phi_tilde_3d_prime(q) = {abs_err_phi_tilde_3d_prime}")

    assert abs_err_f < tolerance
    assert abs_err_df < tolerance * 10
    assert abs_err_ddf < tolerance * 100
    assert abs_err_phi_tilde_3d < tolerance * 100
    assert abs_err_phi_tilde_3d_prime < tolerance * 1000

    print("------------------------------------------")
    print("")

    return {
        "q_arr": q_arr,
        "shamrock_Cf": np.array(shamrock_f) * norm_3d,
        "shamrock_Cdf": np.array(shamrock_df) * norm_3d,
        "shamrock_Cddf": np.array(shamrock_ddf) * norm_3d,
        "shamrock_Cphi_tilde_3d": np.array(shamrock_phi_tilde_3d) * norm_3d,
        "shamrock_Cphi_tilde_3d_prime": np.array(shamrock_phi_tilde_3d_prime) * norm_3d,
    }


def print_kernel_info(ret):
    print("------------------------------------------")
    print(f"Sympy expression for kernel {ret['name']}")
    print("------------------------------------------")
    print(f"f(q)   = {ret['f']}")
    print(f"df(q)  = {ret['df']}")
    print(f"ddf(q) = {ret['ddf']}")
    print(f"phi_tilde_3d(q) = {ret['phi_tilde_3d']}")
    print(f"phi_tilde_3d_prime(q) = {ret['phi_tilde_3d_prime']}")
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
# Helper function to multiply into piecewise without expanding
def multiply_piecewise(piecewise_expr, factor):
    """Multiply a factor into each piece of a Piecewise expression without expanding"""
    new_args = []
    for arg in piecewise_expr.args:
        if hasattr(arg, "expr") and hasattr(arg, "cond"):
            # Multiply the expression part, keep condition unchanged
            new_expr = factor * arg.expr
            new_args.append((new_expr, arg.cond))
        elif isinstance(arg, tuple) and len(arg) == 2:
            new_expr = factor * arg[0]
            new_args.append((new_expr, arg[1]))

    return Piecewise(*new_args)


# %%
# Helper function to shift and scale a kernel
def shift_scale_kernel(piecewise_expr, shift_val, scale_val):
    return piecewise_fold(
        Piecewise(
            (1, q < shift_val),
            (piecewise_expr.subs({q: (q - shift_val) * scale_val}), q >= shift_val),
        )
    )


# %%
# Double hump
def m4dh():
    R, f, _ = m4()
    f = multiply_piecewise(f, q * q)
    return (R, f, "M4DH")


def m4dh3():
    R, f, _ = m4()
    f = multiply_piecewise(f, q * q * q)
    return (R, f, "M4DH3")


def m4dh5():
    R, f, _ = m4()
    f = multiply_piecewise(f, q * q * q * q * q)
    return (R, f, "M4DH5")


def m4dh7():
    R, f, _ = m4()
    f = multiply_piecewise(f, q * q * q * q * q * q * q)
    return (R, f, "M4DH7")


# %%
# M4Shift kernels
def m4shift2():
    R, f, _ = m4()
    # For q < 1: return 1
    # For q >= 1: return M4((q - 1) * 2)
    f_shifted = shift_scale_kernel(f, shift_val=1, scale_val=2)
    return (R, f_shifted, "M4Shift2")


def m4shift4():
    R, f, _ = m4()
    # For q < 1.5: return 1
    # For q >= 1.5: return M4((q - 1.5) * 4)
    f_shifted = shift_scale_kernel(f, shift_val=sympify(3) / 2, scale_val=4)
    return (R, f_shifted, "M4Shift4")


def m4shift8():
    R, f, _ = m4()
    # For q < 1.75: return 1
    # For q >= 1.75: return M4((q - 1.75) * 8)
    f_shifted = shift_scale_kernel(f, shift_val=sympify(7) / 4, scale_val=8)
    return (R, f_shifted, "M4Shift8")


def m4shift16():
    R, f, _ = m4()
    # For q < 1.875: return 1
    # For q >= 1.875: return M4((q - 1.875) * 16)
    f_shifted = shift_scale_kernel(f, shift_val=sympify(15) / 8, scale_val=16)
    return (R, f_shifted, "M4Shift16")


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
# M4DH3 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m4dh3)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m4dh3 = test_kernel(ret)


# %%
# M4DH5 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m4dh5)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m4dh5 = test_kernel(ret)


# %%
# M4DH7 Kernel
# ^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m4dh7)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m4dh7 = test_kernel(ret)


# %%
# M4Shift2 Kernel
# ^^^^^^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m4shift2)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m4shift2 = test_kernel(ret)


# %%
# M4Shift4 Kernel
# ^^^^^^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m4shift4)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m4shift4 = test_kernel(ret)


# %%
# M4Shift8 Kernel
# ^^^^^^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m4shift8)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m4shift8 = test_kernel(ret)


# %%
# M4Shift16 Kernel
# ^^^^^^^^^^^^^^^^


# %%
# Generate c++ code for the kernel
ret = kernel_to_shamrock(m4shift16)

# %%
print_kernel_info(ret)

# %%
print_kernel_cpp_code(ret)

# %%
# Test the kernel
test_result_m4shift16 = test_kernel(ret)


# %%
# Plot the kernels
# ^^^^^^^^^^^^^^^^

fig_sz = (6.4, 12)


# %%
# Plotting helper functions
def create_kernel_plot_figure(fig_sz):
    """Create a figure with 5 subplots for kernel plotting"""
    plt.figure(figsize=fig_sz)
    ax_f = plt.subplot(5, 1, 1)
    ax_df = plt.subplot(5, 1, 2)
    ax_ddf = plt.subplot(5, 1, 3)
    ax_phi_tilde_3d = plt.subplot(5, 1, 4)
    ax_phi_tilde_3d_prime = plt.subplot(5, 1, 5)
    return ax_f, ax_df, ax_ddf, ax_phi_tilde_3d, ax_phi_tilde_3d_prime


def plot_kernel_result(axes, test_result, kernel_label):
    """Plot a single kernel result on the given axes"""
    ax_f, ax_df, ax_ddf, ax_phi_tilde_3d, ax_phi_tilde_3d_prime = axes
    q_arr = test_result["q_arr"]

    ax_f.plot(q_arr, test_result["shamrock_Cf"], label=f"C_3d f_{kernel_label}(q)")
    ax_df.plot(q_arr, test_result["shamrock_Cdf"], label=f"C_3d df_{kernel_label}(q)")
    ax_ddf.plot(q_arr, test_result["shamrock_Cddf"], label=f"C_3d ddf_{kernel_label}(q)")
    ax_phi_tilde_3d.plot(
        q_arr, test_result["shamrock_Cphi_tilde_3d"], label=f"C_3d phi_tilde_3d_{kernel_label}(q)"
    )
    ax_phi_tilde_3d_prime.plot(
        q_arr,
        test_result["shamrock_Cphi_tilde_3d_prime"],
        label=f"C_3d phi_tilde_3d_prime_{kernel_label}(q)",
    )


def finalize_kernel_plot(axes):
    """Add titles, labels, and legends to the kernel plot"""
    ax_f, ax_df, ax_ddf, ax_phi_tilde_3d, ax_phi_tilde_3d_prime = axes

    # Get current axis limits before adding reference line
    xlim = ax_phi_tilde_3d.get_xlim()
    ylim = ax_phi_tilde_3d.get_ylim()

    ylim = (1.2 * ylim[0], ylim[1])

    # Add -1/r reference line (only within current x range)
    q = np.linspace(max(1e-6, xlim[0]), xlim[1], 1000)
    one_over_r = -1 / q
    ax_phi_tilde_3d.plot(q, one_over_r, "--", color="grey", label="-1/r")

    # Restore original limits (ignore reference line for autoscaling)
    ax_phi_tilde_3d.set_xlim(xlim)
    ax_phi_tilde_3d.set_ylim(ylim)

    # Get current axis limits before adding reference line
    xlim = ax_phi_tilde_3d_prime.get_xlim()
    ylim = ax_phi_tilde_3d_prime.get_ylim()

    ylim = (ylim[0], 1.5 * ylim[1])

    # Add 1/r^2 reference line (only within current x range)
    one_over_r_squared = 1 / q**2
    ax_phi_tilde_3d_prime.plot(q, one_over_r_squared, "--", color="grey", label="1/r^2")

    # Restore original limits (ignore reference line for autoscaling)
    ax_phi_tilde_3d_prime.set_xlim(xlim)
    ax_phi_tilde_3d_prime.set_ylim(ylim)

    ax_f.set_title("C_3d f(q)", fontsize=10)
    ax_df.set_title("C_3d df(q)", fontsize=10)
    ax_ddf.set_title("C_3d ddf(q)", fontsize=10)
    ax_phi_tilde_3d.set_title("C_3d phi_tilde_3d(q)", fontsize=10)
    ax_phi_tilde_3d_prime.set_title("C_3d phi_tilde_3d_prime(q)", fontsize=10)

    ax_f.set_xlabel("q")
    ax_df.set_xlabel("q")
    ax_ddf.set_xlabel("q")
    ax_phi_tilde_3d.set_xlabel("q")
    ax_phi_tilde_3d_prime.set_xlabel("q")

    ax_f.legend(loc="right", fontsize=8)
    ax_df.legend(loc="right", fontsize=8)
    ax_ddf.legend(loc="right", fontsize=8)
    ax_phi_tilde_3d.legend(loc="right", fontsize=8)
    ax_phi_tilde_3d_prime.legend(loc="right", fontsize=8)

    plt.tight_layout()
    plt.show()


# %%
# Cubic splines kernels (M-series)

axes = create_kernel_plot_figure(fig_sz)
plot_kernel_result(axes, test_result_m4, "m4")
plot_kernel_result(axes, test_result_m5, "m5")
plot_kernel_result(axes, test_result_m6, "m6")
plot_kernel_result(axes, test_result_m7, "m7")
plot_kernel_result(axes, test_result_m8, "m8")
plot_kernel_result(axes, test_result_m9, "m9")
plot_kernel_result(axes, test_result_m10, "m10")
finalize_kernel_plot(axes)
plt.show()

# %%
# Wendland kernels (C-series)

axes = create_kernel_plot_figure(fig_sz)
plot_kernel_result(axes, test_result_c2, "c2")
plot_kernel_result(axes, test_result_c4, "c4")
plot_kernel_result(axes, test_result_c6, "c6")
finalize_kernel_plot(axes)
plt.show()

# %%
# Double hump kernels

axes = create_kernel_plot_figure(fig_sz)
plot_kernel_result(axes, test_result_m4dh, "m4dh")
plot_kernel_result(axes, test_result_m4dh3, "m4dh3")
plot_kernel_result(axes, test_result_m4dh5, "m4dh5")
plot_kernel_result(axes, test_result_m4dh7, "m4dh7")
finalize_kernel_plot(axes)
plt.show()


# %%
# M4Shift kernels

axes = create_kernel_plot_figure(fig_sz)
plot_kernel_result(axes, test_result_m4shift2, "m4shift2")
plot_kernel_result(axes, test_result_m4shift4, "m4shift4")
plot_kernel_result(axes, test_result_m4shift8, "m4shift8")
plot_kernel_result(axes, test_result_m4shift16, "m4shift16")
finalize_kernel_plot(axes)
plt.show()
