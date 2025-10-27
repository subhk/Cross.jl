import numpy as np

def operator_u(l, r2_D0_u, r3_D1_u, r4_D2_u):
    L = l * (l + 1)
    return L * (L * r2_D0_u - 2 * r3_D1_u - r4_D2_u)

def operator_coriolis_diagonal(l, m, r2_D0_u, r3_D1_u, r4_D2_u):
    L = l * (l + 1)
    return 2j * m * (-L * r2_D0_u + 2 * r3_D1_u + r4_D2_u)

def operator_viscous_diffusion(l, E, r0_D0_u, r2_D2_u, r3_D3_u, r4_D4_u):
    L = l * (l + 1)
    return E * L * (-L * (l + 2) * (l - 1) * r0_D0_u + 2 * L * r2_D2_u - 4 * r3_D3_u - r4_D4_u)

def operator_coriolis_offdiag(l, m, offset, r3_D0_u, r4_D1_u):
    if offset == -1:
        C = (l**2 - 1) * np.sqrt(l**2 - m**2) / (2 * l - 1)
        return 2 * C * ((l - 1) * r3_D0_u - r4_D1_u)
    elif offset == 1:
        C = l * (l + 2) * np.sqrt((l + m + 1) * (l - m + 1)) / (2 * l + 3)
        return 2 * C * (-(l + 2) * r3_D0_u - r4_D1_u)
    else:
        raise ValueError("offset must be ±1")

def operator_buoyancy(l, Ra, E, Pr, r4_D0_u):
    L = l * (l + 1)
    beyonce = -Ra * E**2 / Pr
    return beyonce * L * r4_D0_u

def operator_u_toroidal(l, r2_D0_v):
    L = l * (l + 1)
    return L * r2_D0_v

def operator_coriolis_toroidal(m, r2_D0_v):
    return -2j * m * r2_D0_v

def operator_viscous_toroidal(l, E, r0_D0_v, r1_D1_v, r2_D2_v):
    L = l * (l + 1)
    return E * L * (-L * r0_D0_v + 2 * r1_D1_v + r2_D2_v)

def operator_coriolis_v_to_u(l, m, offset, r1_D0_v, r2_D1_v):
    if offset == -1:
        C = (l**2 - 1) * np.sqrt(l**2 - m**2) / (2 * l - 1)
        return 2 * C * ((l - 1) * r1_D0_v - r2_D1_v)
    elif offset == 1:
        C = l * (l + 2) * np.sqrt((l + m + 1) * (l - m + 1)) / (2 * l + 3)
        return 2 * C * (-(l + 2) * r1_D0_v - r2_D1_v)
    else:
        raise ValueError("offset must be ±1")

def operator_theta(heating, r2_D0_h, r3_D0_h):
    if heating == "differential":
        return r3_D0_h
    else:
        return r2_D0_h

def operator_thermal_diffusion(l, Etherm, heating, r0_D0_h, r1_D0_h, r1_D1_h, r2_D1_h, r2_D2_h, r3_D2_h):
    L = l * (l + 1)
    if heating == "differential":
        return Etherm * (-L * r1_D0_h + 2 * r2_D1_h + r3_D2_h)
    else:
        return Etherm * (-L * r0_D0_h + 2 * r1_D1_h + r2_D2_h)

def operator_thermal_advection(l, heating, ricb, r0_D0_h, r2_D0_h):
    L = l * (l + 1)
    if heating == "differential":
        gap = 1.0 - ricb
        return L * (ricb / gap) * r0_D0_h
    else:
        return L * r2_D0_h


# -------------------------------------------------------------------------------------------------
# MHD helper formulas (axial background field, Boussinesq)
# -------------------------------------------------------------------------------------------------

def _get(mats, name):
    try:
        return mats[name]
    except KeyError:
        raise KeyError(f"Missing matrix '{name}' in mapping")


def lorentz_upol_bpol_axial(l, m, offset, mats):
    L = l * (l + 1)
    if offset == -2:
        denom = 3 - 8*l + 4*l**2
        C = (3 * (-2 - l + l**2) *
             np.sqrt((l - m)*(-1 + l + m)*(-1 + l - m)*(l + m))) / denom
        out1 = _get(mats, "r0_h0_D0_u") * (2*l + 3*l**2 + l**3)
        out1 += _get(mats, "r1_h0_D1_u") * ( -6 + 7*l - 3*l**2)
        out1 += _get(mats, "r1_h1_D0_u") * (2 + l - 6*l**2 + l**3)
        out1 += _get(mats, "r2_h0_D2_u") * (6 - l)
        out1 += _get(mats, "r2_h1_D1_u") * (-2) * (-2 + l)
        out1 += _get(mats, "r2_h2_D0_u") * (-2 + l)
        out1 += _get(mats, "r3_h2_D1_u") * (-1)
        out1 += _get(mats, "r3_h0_D3_u") * 3
        out1 += _get(mats, "r3_h1_D2_u") * ( - (l - 3))
        out1 += _get(mats, "r3_h3_D0_u") * ( -1 + l)
        return C * out1

    if offset == -1:
        denom = 2*l - 1
        C = np.sqrt(l**2 - m**2) * (l**2 - 1) / denom
        out1 = (l - 2) * _get(mats, "r0_h0_D0_u")
        out1 += l * _get(mats, "r1_h1_D0_u")
        out1 += -2 * _get(mats, "r1_h0_D1_u")
        out1 += -2 * (l - 2) * _get(mats, "r2_h1_D1_u")
        out1 += -(l - 4) * _get(mats, "r2_h0_D2_u")
        out1 += -(l - 2) * _get(mats, "r3_h1_D2_u")
        out1 += l * _get(mats, "r2_h2_D0_u")
        out1 += l * _get(mats, "r3_h3_D0_u")
        out1 += 2 * _get(mats, "r3_h0_D3_u")
        return C * out1

    if offset == 0:
        C = (3 * (l + l**2 - 3*m**2)) / (-3 + 4*l*(1 + l))
        out1 = 3 * L * (1 + l) * (-2 + l + l**2) * _get(mats, "r0_h0_D0_u")
        out1 += -3 * L**2 * _get(mats, "r1_h1_D0_u")
        out1 += 2 * (6 - 4*l - 5*l**2 - 2*l**3 - l**4) * _get(mats, "r1_h0_D1_u")
        out1 += 3 * L * _get(mats, "r2_h2_D0_u")
        out1 += (-12 + 5*l + 5*l**2) * _get(mats, "r2_h0_D2_u")
        out1 += 2 * (-6 + 5*l + 5*l**2) * _get(mats, "r2_h1_D1_u")
        out1 += 2 * L * _get(mats, "r3_h2_D1_u")
        out1 += L * _get(mats, "r3_h3_D0_u")
        out1 += 2 * (-3 + l + l**2) * _get(mats, "r3_h0_D3_u")
        out1 += 3 * (-2 + l + l**2) * _get(mats, "r3_h1_D2_u")
        return C * out1

    if offset == 1:
        denom = 2*l + 3
        C = np.sqrt((l + m + 1)*(l - m + 1)) * l * (l + 2) / denom
        out1 = -2 * (l**2 + 2*l + 3) * _get(mats, "r1_h0_D1_u")
        out1 += 2 * (l + 3) * _get(mats, "r2_h1_D1_u")
        out1 += (l + 5) * _get(mats, "r2_h0_D2_u")
        out1 += (l + 3) * _get(mats, "r3_h1_D2_u")
        out2 = -L * (l - 1) * _get(mats, "r0_h0_D0_u")
        out2 += -L * (l + 5) * _get(mats, "r1_h1_D0_u")
        out2 += -(l + 1) * _get(mats, "r2_h2_D0_u")
        out2 += -(l + 1) * _get(mats, "r3_h3_D0_u")
        out2 += 2 * _get(mats, "r3_h0_D3_u")
        return C * (out1 + out2)

    if offset == 2:
        denom = (3 + 2*l) * (5 + 2*l)
        C = (3 * l * (3 + l) *
             np.sqrt((2 + l - m)*(1 + l + m)) *
             np.sqrt((1 + l - m)*(2 + l + m))) / denom
        out1 = (l - l**3) * _get(mats, "r0_h0_D0_u")
        out1 += -(16 + 13*l + 3*l**2) * _get(mats, "r1_h0_D1_u")
        out1 += -(6 + 16*l + 9*l**2 + l**3) * _get(mats, "r1_h1_D0_u")
        out1 += 2 * (3 + l) * _get(mats, "r2_h1_D1_u")
        out1 += -(3 + l) * _get(mats, "r2_h2_D0_u")
        out1 += (7 + l) * _get(mats, "r2_h0_D2_u")
        out1 += -_get(mats, "r3_h2_D1_u")
        out1 += 3 * _get(mats, "r3_h0_D3_u")
        out1 += (4 + l) * _get(mats, "r3_h1_D2_u")
        out1 += -(2 + l) * _get(mats, "r3_h3_D0_u")
        return C * out1

    raise ValueError("offset must be between -2 and 2")


def lorentz_upol_btor_axial(l, m, offset, mats):
    if offset == -1:
        denom = 2*l - 1
        C = 6j * m * np.sqrt(l**2 - m**2) / denom
        out = -(3 - 3*l - 2*l**2) * _get(mats, "r1_h0_D0_u")
        out += -(l - 3) * _get(mats, "r2_h0_D1_u")
        out += (3 - 2*l - l**2) * _get(mats, "r2_h1_D0_u")
        out += 3 * _get(mats, "r3_h0_D2_u")
        out += -(l - 3) * _get(mats, "r3_h1_D1_u")
        out += -l * _get(mats, "r3_h2_D0_u")
        return C * out

    if offset == 0:
        return 2j * m * (
            -_get(mats, "r1_h0_D0_u")
            - (l**2 + l - 1) * _get(mats, "r2_h1_D0_u")
            + _get(mats, "r2_h0_D1_u")
            + _get(mats, "r3_h1_D1_u")
            + _get(mats, "r3_h0_D2_u")
        )

    if offset == 1:
        denom = 2*l + 3
        C = 6j * m * np.sqrt((l + 1)**2 - m**2) / denom
        out = (-4 + l + 2*l**2) * _get(mats, "r1_h0_D0_u")
        out += (4 + l) * _get(mats, "r2_h0_D1_u")
        out += (4 - l**2) * _get(mats, "r2_h1_D0_u")
        out += 3 * _get(mats, "r3_h0_D2_u")
        out += (1 + l) * _get(mats, "r3_h2_D0_u")
        out += (4 + l) * _get(mats, "r3_h1_D1_u")
        return C * out

    raise ValueError("offset must be ±1")


def lorentz_upol_diag_axial(l, m, mats):
    return lorentz_upol_bpol_axial(l, m, 0, mats)


def lorentz_utor_axial(l, m, mats):
    L = l * (l + 1)
    out = 4 * _get(mats, "r0_h0_D1_v")
    out += -L * (2 * _get(mats, "r0_h1_D0_v") + _get(mats, "r1_h2_D0_v"))
    out += 2 * _get(mats, "r1_h0_D2_v")
    return 1j * m * out


def lorentz_v_bpol_axial(l, m, offset, mats):
    if offset == -1:
        denom = 2*l - 1
        C = 3j * m * np.sqrt(l**2 - m**2) / denom
        out = 12 * _get(mats, "r0_h0_D1_v")
        out += -2 * (-1 + l) * l * _get(mats, "r0_h1_D0_v")
        out += 6 * _get(mats, "r1_h0_D2_v")
        out += -(-1 + l) * l * _get(mats, "r1_h2_D0_v")
        return C * out

    if offset == 0:
        out = 4 * _get(mats, "r0_h0_D1_v")
        out += -l*(l + 1) * (2 * _get(mats, "r0_h1_D0_v") + _get(mats, "r1_h2_D0_v"))
        out += 2 * _get(mats, "r1_h0_D2_v")
        return 1j * m * out

    if offset == 1:
        denom = 2*l + 3
        C = 3j * m * np.sqrt((1 + l - m)*(1 + l + m)) / denom
        out = 12 * _get(mats, "r0_h0_D1_v")
        out += -2 * (1 + l)*(2 + l) * _get(mats, "r0_h1_D0_v")
        out += 6 * _get(mats, "r1_h0_D2_v")
        out += -(1 + l)*(2 + l) * _get(mats, "r1_h2_D0_v")
        return C * out

    raise ValueError("offset must be -1, 0, or 1")


def lorentz_v_btor_axial(l, m, offset, mats):
    if offset == -2:
        denom = 3 - 8*l + 4*l**2
        C = (3 * (l - 2) * (l + 1) *
             np.sqrt((l - m)*(-1 + l + m)) *
             np.sqrt((-1 + l - m)*(l + m))) / denom
        out = (-4 + l) * _get(mats, "r0_h0_D0_v")
        out += -3 * _get(mats, "r1_h0_D1_v")
        out += (-1 + l) * _get(mats, "r1_h1_D0_v")
        return C * out

    if offset == -1:
        denom = 2*l - 1
        C = np.sqrt((l - m)*(l + m)) * (l**2 - 1) / denom
        out = (l - 2) * _get(mats, "r0_h0_D0_v")
        out += l * _get(mats, "r1_h1_D0_v")
        out += -2 * _get(mats, "r1_h0_D1_v")
        return C * out

    if offset == 0:
        C = (3 * (l + l**2 - 3*m**2)) / (-3 + 4*l*(1 + l))
        out = (6 - l - l**2) * _get(mats, "r0_h0_D0_v")
        out += l*(l + 1) * _get(mats, "r1_h1_D0_v")
        out += -2 * (-3 + l + l**2) * _get(mats, "r1_h0_D1_v")
        return C * out

    if offset == 1:
        denom = 2*l + 3
        C = -np.sqrt((l + m + 1)*(l + 1 - m)) * l * (l + 2) / denom
        out = (l + 3) * _get(mats, "r0_h0_D0_v")
        out += (l + 1) * _get(mats, "r1_h1_D0_v")
        out += 2 * _get(mats, "r1_h0_D1_v")
        return C * out

    if offset == 2:
        denom = (3 + 2*l)*(5 + 2*l)
        C = (3 * l * (3 + l) *
             np.sqrt((2 + l - m)*(1 + l + m)) *
             np.sqrt((1 + l - m)*(2 + l + m))) / denom
        out = -(5 + l) * _get(mats, "r0_h0_D0_v")
        out += -3 * _get(mats, "r1_h0_D1_v")
        out += -(2 + l) * _get(mats, "r1_h1_D0_v")
        return C * out

    raise ValueError("offset must be in -2..2")


def induction_f_upol_axial(l, m, offset, mats):
    if offset == -2:
        denom = 3 - 8*l + 4*l**2
        C = 3 * (l - 2) * (l + 1) * np.sqrt((l - m)*(-1 + l + m)*(-1 + l - m)*(l + m)) / denom
        out = (-4 + l) * _get(mats, "r0_h0_D0_f")
        out += -3 * _get(mats, "r1_h0_D1_f")
        out += (-1 + l) * _get(mats, "r1_h1_D0_f")
        return C * out

    if offset == -1:
        denom = 2*l - 1
        C = np.sqrt(l**2 - m**2) * (l**2 - 1) / denom
        out = (l - 2) * _get(mats, "r0_h0_D0_f")
        out += l * _get(mats, "r1_h1_D0_f")
        out += -2 * _get(mats, "r1_h0_D1_f")
        return C * out

    if offset == 0:
        C = 3 * (l + l**2 - 3*m**2) / (-3 + 4*l*(1 + l))
        out = (6 - l - l**2) * _get(mats, "r0_h0_D0_f")
        out += l*(1 + l) * _get(mats, "r1_h1_D0_f")
        out += -2 * (-3 + l + l**2) * _get(mats, "r1_h0_D1_f")
        return C * out

    if offset == 1:
        denom = 2*l + 3
        C = np.sqrt((l + 1)**2 - m**2) * l * (l + 2) / denom
        out = -(l + 3) * _get(mats, "r0_h0_D0_f")
        out += -(l + 1) * _get(mats, "r1_h1_D0_f")
        out += -2 * _get(mats, "r1_h0_D1_f")
        return C * out

    if offset == 2:
        denom = (3 + 2*l)*(5 + 2*l)
        sqrt1 = np.sqrt((2 + l - m)*(1 + l + m))
        sqrt2 = np.sqrt((1 + l - m)*(2 + l + m))
        C = 3 * l * (l + 3) * sqrt1 * sqrt2 / denom
        out = -(l + 5) * _get(mats, "r0_h0_D0_f")
        out += -3 * _get(mats, "r1_h0_D1_f")
        out += -(l + 2) * _get(mats, "r1_h1_D0_f")
        return C * out

    raise ValueError("offset must be in -2..2")


def induction_f_utor_axial(l, m, offset, mats):
    term = _get(mats, "r1_h0_D0_f")
    if offset == -1:
        denom = 1 - 2*l
        C = 18j * m * np.sqrt(l**2 - m**2) / denom
        return C * term
    if offset == 0:
        return -2j * m * term
    if offset == 1:
        denom = 3 + 2*l
        C = -18j * m * np.sqrt((l + 1)**2 - m**2) / denom
        return C * term
    return np.zeros_like(term)


def induction_g_upol_axial(l, m, offset, mats):
    if offset == -1:
        denom = 2*l - 1
        C = 3j * m * np.sqrt(l**2 - m**2) / denom
        out = -2 * (-3 + l) * _get(mats, "r0_h1_D0_u")
        out += -2 * (-3 + l) * _get(mats, "r0_h0_D1_u")
        out += -2 * (3 + l**2) * _get(mats, "r-1_h0_D0_u")
        out += 6 * _get(mats, "r1_h0_D2_u")
        out += -2 * (-3 + l) * _get(mats, "r1_h1_D1_u")
        out += (-1 + l) * l * _get(mats, "r1_h2_D0_u")
        return C * out

    if offset == 1:
        denom = 2*l + 3
        C = 3j * m * np.sqrt((l + 1)**2 - m**2) / denom
        out = 2 * (4 + l) * _get(mats, "r0_h1_D0_u")
        out += 2 * (4 + l) * _get(mats, "r0_h0_D1_u")
        out += -2 * (4 + 2*l + l**2) * _get(mats, "r-1_h0_D0_u")
        out += 6 * _get(mats, "r1_h0_D2_u")
        out += (2 + 3*l + l**2) * _get(mats, "r1_h2_D0_u")
        out += 2 * (4 + l) * _get(mats, "r1_h1_D1_u")
        return C * out

    return np.zeros_like(_get(mats, "r0_h0_D0_u"))


def induction_g_utor_axial(l, m, offset, mats):
    if offset == -2:
        denom = 3 - 8*l + 4*l**2
        C = 3 * (l - 2) * (l + 1) * np.sqrt((l - m)*(-1 + l + m)*(-1 + l - m)*(l + m)) / denom
        out = (-4 + l) * _get(mats, "r0_h0_D0_v")
        out += -3 * _get(mats, "r1_h0_D1_v")
        out += (-1 + l) * _get(mats, "r1_h1_D0_v")
        return C * out

    if offset == -1:
        denom = 2*l - 1
        C = np.sqrt((l - m)*(l + m)) * (l**2 - 1) / denom
        out = (l - 2) * _get(mats, "r0_h0_D0_v")
        out += l * _get(mats, "r1_h1_D0_v")
        out += -2 * _get(mats, "r1_h0_D1_v")
        return C * out

    if offset == 0:
        C = 3 * (l + l**2 - 3*m**2) / (-3 + 4*l*(1 + l))
        out = (6 - l - l**2) * _get(mats, "r0_h0_D0_v")
        out += (l + l**2) * _get(mats, "r1_h1_D0_v")
        out += -2 * (-3 + l + l**2) * _get(mats, "r1_h0_D1_v")
        return C * out

    if offset == 1:
        denom = 2*l + 3
        C = -np.sqrt((l + m + 1)*(l + 1 - m)) * l * (l + 2) / denom
        out = (l + 3) * _get(mats, "r0_h0_D0_v")
        out += (l + 1) * _get(mats, "r1_h1_D0_v")
        out += 2 * _get(mats, "r1_h0_D1_v")
        return C * out

    if offset == 2:
        denom = (3 + 2*l)*(5 + 2*l)
        C = 3 * l * (l + 3) * np.sqrt((2 + l - m)*(1 + l + m)) * np.sqrt((1 + l - m)*(2 + l + m)) / denom
        out = -(5 + l) * _get(mats, "r0_h0_D0_v")
        out += -3 * _get(mats, "r1_h0_D1_v")
        out += -(2 + l) * _get(mats, "r1_h1_D0_v")
        return C * out

    return np.zeros_like(_get(mats, "r0_h0_D0_v"))


def magnetic_diffusion_f_axial(l, Etherm, mats):
    L = l * (l + 1)
    return Etherm * (-L * _get(mats, "r0_D0_f") + 2 * _get(mats, "r1_D1_f") + _get(mats, "r2_D2_f"))


def magnetic_diffusion_g_axial(l, Em, mats):
    L = l * (l + 1)
    return Em * L * (-L * _get(mats, "r0_D0_g") + 2 * _get(mats, "r1_D1_g") + _get(mats, "r2_D2_g"))


def b_poloidal_axial(l, mats):
    L = l * (l + 1)
    return L * _get(mats, "r2_D0_f")


def b_toroidal_axial(l, mats):
    L = l * (l + 1)
    return L * _get(mats, "r2_D0_g")
