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
