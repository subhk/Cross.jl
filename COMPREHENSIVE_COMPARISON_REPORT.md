# COMPREHENSIVE COMPARISON: Cross.jl vs Kore Implementation

**Date**: October 26, 2025  
**Scope**: Detailed analysis of implementation differences between Cross.jl and Kore  
**Focus Areas**: Boundary conditions, mode selection, matrix construction, heating implementation, and numerical parameters  

---

## EXECUTIVE SUMMARY

A comprehensive comparison has identified **ONE CRITICAL BUG** in Cross.jl's mode selection logic for m=0 cases, along with several implementation details that match Kore perfectly.

**Critical Finding**: Cross.jl's `compute_l_modes()` function incorrectly excludes l=0 (when m=0 and symm=1) and fails to reach lmax+1 for m=0 cases. This can cause missing modes in the spectral expansion for axially symmetric flows.

---

## 1. MODE SELECTION AND SYMMETRY (CRITICAL ISSUE)

### Location
- **Kore**: `kore-main/bin/utils.py:174-183`, function `ell()`
- **Cross.jl**: `src/SparseOperator.jl:240-260`, function `compute_l_modes()`

### Issue Description

The mode selection logic uses different approaches that appear equivalent but have subtle differences for m=0 cases.

#### Kore Implementation (CORRECT)
```python
def ell(m, lmax, vsymm):
    lm1 = lmax - m + 1
    s = int(vsymm*0.5 + 0.5)  # s=0 if antisymm, s=1 if symm
    idp = np.arange((np.sign(m) + s) % 2, lm1, 2, dtype=int)
    idt = np.arange((np.sign(m) + s + 1) % 2, lm1, 2, dtype=int)
    ll = np.arange(m + 1 - np.sign(m), lmax + 2 - np.sign(m), dtype=int)
    return [ll[idp], ll[idt], ll]
```

#### Cross.jl Implementation (BUG FOR m=0)
```julia
function compute_l_modes(m::Int, lmax::Int, symm::Int)
    if symm == 1
        ll_top = collect(m:2:lmax)      # Range: [m, m+2, m+4, ...]
        ll_bot = collect((m+1):2:lmax)  # Range: [m+1, m+3, m+5, ...]
    # ...
end
```

### Detailed Analysis: m=0, lmax=10, symm=1

#### Kore Calculation
```
lm1 = 10 - 0 + 1 = 11
s = int(1*0.5 + 0.5) = 1 (symmetric)
sign(m) = sign(0) = 0

idp indices = arange((0+1)%2, 11, 2) = arange(1, 11, 2) = [1,3,5,7,9]
idt indices = arange((0+1+1)%2, 11, 2) = arange(0, 11, 2) = [0,2,4,6,8,10]

ll range = arange(0+1-0, 10+2-0) = arange(1, 12) = [1,2,3,4,5,6,7,8,9,10,11]

ll[idp] = [2,4,6,8,10]              <- POLOIDAL (symmetric)
ll[idt] = [1,3,5,7,9,11]            <- TOROIDAL (antisymmetric)
```

#### Cross.jl Calculation
```
ll_top = collect(0:2:10) = [0,2,4,6,8,10]    <- POLOIDAL
ll_bot = collect(1:2:10) = [1,3,5,7,9]       <- TOROIDAL (MISSING l=11!)

BUG: l=0 is invalid (l must be >= m and >= 1 for scalar spherical harmonics)
BUG: l=11 is missing from toroidal modes
```

### Test Cases and Results

| m | lmax | symm | Kore (top) | Cross.jl (top) | Match? | Kore (bot) | Cross.jl (bot) | Match? |
|---|------|------|-----------|----------------|--------|-----------|----------------|--------|
| 0 | 10 | 1 | [2,4,6,8,10] | [0,2,4,6,8,10] | ✗ BUG | [1,3,5,7,9,11] | [1,3,5,7,9] | ✗ BUG |
| 1 | 10 | 1 | [1,3,5,7,9] | [1,3,5,7,9] | ✓ | [2,4,6,8,10] | [2,4,6,8,10] | ✓ |
| 2 | 10 | 1 | [2,4,6,8,10] | [2,4,6,8,10] | ✓ | [3,5,7,9] | [3,5,7,9] | ✓ |
| 0 | 10 | -1 | [1,3,5,7,9,11] | [1,3,5,7,9] | ✗ BUG | [2,4,6,8,10] | [0,2,4,6,8,10] | ✗ BUG |

### Impact Assessment
- **Severity**: CRITICAL for m=0 problems
- **Affects**: All axially symmetric (m=0) convection problems
- **Consequence**: Missing l-modes (l=0 invalid, l=lmax+1 missing) reduces spectral accuracy

### Recommended Fix
Replace Cross.jl's `compute_l_modes()` with logic that matches Kore's approach:
```julia
function compute_l_modes(m::Int, lmax::Int, symm::Int)
    lm1 = lmax - m + 1
    s = div(symm + 1, 2)  # 0 if antisymm, 1 if symm
    
    # Generate full l-range accounting for sign(m) behavior
    if m == 0
        ll = collect(1:(lmax+1))
    else
        ll = collect(m:(lmax))
    end
    
    # Select modes based on parity
    idp_start = (sign(m) + s) % 2 + 1  # Convert to 1-based indexing
    idp = idp_start:2:length(ll)
    idt_start = (sign(m) + s + 1) % 2 + 1
    idt = idt_start:2:length(ll)
    
    return ll[idp], ll[idt]
end
```

---

## 2. HEATING MODE IMPLEMENTATION

### Location
- **Kore**: `kore-main/bin/operators.py:699-770`
- **Cross.jl**: `src/SparseOperator.jl:504-562`

### Implementation Comparison

#### Theta (Temperature) Time Derivative Operator

**Kore** (lines 712-715):
```python
if par.heating == 'differential':
    out = r3_D0_h  # eq. times r³
else:
    out = r2_D0_h  # eq. times r²
```

**Cross.jl** (lines 505-509):
```julia
if op.params.heating == :differential
    return op.r3_D0_h  # eq. times r³
else  # :internal
    return op.r2_D0_h  # eq. times r²
end
```

**Status**: ✓ IDENTICAL

#### Thermal Diffusion Operator

**Kore** (lines 758-761):
```python
if par.heating == 'differential':
    difus = -L*r1_D0_h + 2*r2_D1_h + r3_D2_h  # eq. times r**3
else:
    difus = -L*r0_D0_h + 2*r1_D1_h + r2_D2_h  # eq. times r**2
```

**Cross.jl** (lines 528-533):
```julia
if op.params.heating == :differential
    # eq. times r³
    return Etherm * (-L * op.r1_D0_h + 2 * op.r2_D1_h + op.r3_D2_h)
else  # :internal
    # eq. times r²
    return Etherm * (-L * op.r0_D0_h + 2 * op.r1_D1_h + op.r2_D2_h)
end
```

**Status**: ✓ IDENTICAL

#### Thermal Advection Operator

**Kore** (lines 733-736):
```python
if par.heating == 'internal':
    conv = r2_D0_h  # dT/dr = -beta*r. Heat equation is times r**2
elif par.heating == 'differential':
    conv = r0_D0_h * par.ricb/gap  # dT/dr = -beta * r**2. Heat equation is times r**3
```

**Cross.jl** (lines 553-561):
```julia
if op.params.heating == :differential
    # dT/dr = -beta * r⁻², eq. times r³
    ricb = op.params.ricb
    gap = one(T) - ricb
    return L * op.r0_D0_h * (ricb / gap)
else  # :internal
    # dT/dr = -beta * r, eq. times r²
    return L * op.r2_D0_h
end
```

**Status**: ✓ IDENTICAL (different order of conditionals but same logic)

### Summary
All heating-related operators are **CORRECTLY IMPLEMENTED** in Cross.jl and match Kore exactly.

---

## 3. BOUNDARY CONDITION IMPLEMENTATION

### Location
- **Kore**: `kore-main/bin/assemble.py:1188-1296` (poloidal/toroidal), lines 1300-1340 (thermal)
- **Cross.jl**: `src/SparseOperator.jl:816-921`

### Poloidal Velocity Boundary Conditions

#### Kore Implementation (lines 1231-1260)

**Stress-free CMB** (bco=0):
```python
out[0,:] = Tbu[:,0]  # P = 0
out[1,:] = ut.rcmb*Tbu[:,2] - lho1_b*Tbu[:,1]  # rcmb*P'' - log_rho'*P' = 0
```

**No-slip CMB** (bco=1):
```python
out[0,:] = Tbu[:,0]  # P = 0
out[1,:] = Tbu[:,1]  # P' = 0
```

**Stress-free ICB** (bci=0):
```python
out[2,:] = bv.Ta[:,0]  # P = 0
out[3,:] = par.ricb * bv.Ta[:,2] - lho1_a * bv.Ta[:,1]  # ricb*P'' - log_rho'*P' = 0
```

**No-slip ICB** (bci=1):
```python
out[2,:] = bv.Ta[:,0]  # P = 0
out[3,:] = bv.Ta[:,1]  # P' = 0
```

#### Cross.jl Implementation (lines 833-859)

**No-slip CMB** (bco=1):
```julia
bc_rows = [row_base + 1, row_base + 2]
apply_boundary_conditions!(A, B, bc_rows, :dirichlet, N, params.ricb, one(T))
```

**Stress-free CMB** (bco=0):
```julia
apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N, params.ricb, one(T))
apply_boundary_conditions!(A, B, [row_base + 2], :neumann2, N, params.ricb, one(T))
```

**Status**: ✓ LOGICALLY EQUIVALENT
- Cross.jl uses tau method with `apply_boundary_conditions!` helper
- Kore explicitly constructs BC rows with Chebyshev evaluations
- Different implementation methods but same physical BCs

### Toroidal Velocity Boundary Conditions

#### Kore Implementation (lines 1274-1288)

**Stress-free** (bco=0 or bci=0):
```python
out[0,:] = -ut.rcmb * Tbv[:,1] + (1+ut.rcmb*lho1_b)*Tbv[:,0]
# -r*v'_(r) + (1 + r*log_rho')*v = 0
```

**No-slip** (bco=1 or bci=1):
```python
out[0,:] = Tbv[:,0]  # v = 0
```

#### Cross.jl Implementation (lines 869-888)

**No-slip** (bco=1):
```julia
apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N, params.ricb, one(T))
```

**Stress-free** (bco=0):
```julia
apply_boundary_conditions!(A, B, [row_base + 1], :neumann, N, params.ricb, one(T))
```

**Status**: ✓ LOGICALLY EQUIVALENT
- Same physics, different implementation via tau method

### Thermal Boundary Conditions

#### Kore Implementation (lines 1318-1328)

**Fixed temperature** (bco_thermal=0):
```python
out[0,:] = Tbh[:,0]  # θ = 0
```

**Fixed flux** (bco_thermal=1):
```python
out[0,:] = Tbh[:,1]  # θ' = 0
```

#### Cross.jl Implementation (lines 898-917)

**Fixed temperature** (bco_thermal=0):
```julia
apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N, params.ricb, one(T))
```

**Fixed flux** (bco_thermal=1):
```julia
apply_boundary_conditions!(A, B, [row_base + 1], :neumann, N, params.ricb, one(T))
```

**Status**: ✓ LOGICALLY EQUIVALENT

### Summary
Boundary condition implementations are **EQUIVALENT IN PHYSICS**, using different technical approaches (tau method in Cross.jl vs explicit Chebyshev evaluation in Kore).

---

## 4. MATRIX CONSTRUCTION AND ASSEMBLY

### Location
- **Kore**: `kore-main/bin/assemble.py:439-523` (B matrix), lines 606-812 (A matrix)
- **Cross.jl**: `src/SparseOperator.jl:578-800`

### B Matrix Assembly (Time Derivatives)

#### Poloidal Velocity

**Kore** (line 445):
```python
mtx = -op.u(l,'u','upol',0)
```
From `operators.py` line 35:
```python
out = L*( L*r2_D0_u - 2*r3_D1_u - r4_D2_u )  # r4* r.2curl(u)
```

**Cross.jl** (line 636):
```julia
u_op = -operator_u(op, l)
```
From `SparseOperator.jl` line 300:
```julia
return L * (L * op.r2_D0_u - 2 * op.r3_D1_u - op.r4_D2_u)
```

**Status**: ✓ IDENTICAL (with negative sign for eigenvalue problem)

#### Toroidal Velocity

**Kore** (line 461):
```python
mtx = -op.u(l,'v','utor',0)
```
From `operators.py` line 42:
```python
out = L*r2_D0_v  # r2* r.1curl(u)
```

**Cross.jl** (line 692):
```julia
u_tor_op = -operator_u_toroidal(op, l)
```
From `SparseOperator.jl` line 446:
```julia
return L * op.r2_D0_v
```

**Status**: ✓ IDENTICAL

#### Temperature (Theta)

**Kore** (line 513):
```python
mtx = op.theta(l,'h','', 0)
```
Returns `r3_D0_h` for differential or `r2_D0_h` for internal heating.

**Cross.jl** (line 746):
```julia
theta_op = operator_theta(op, l)
```
Returns same operators.

**Status**: ✓ IDENTICAL

### A Matrix Assembly (Physics Operators)

#### Coriolis Force (Diagonal)

**Kore** (line 624):
```python
cori = op.coriolis(l,'u','upol',0)[0]
```
From `operators.py` line 63:
```python
out = 2j*par.m*( -L*r2_D0_u + 2*r3_D1_u + r4_D2_u )
```
Multiplied by `par.Gaspard = 1.0`

**Cross.jl** (line 640):
```julia
cori_op = operator_coriolis_diagonal(op, l, m)
```
From `SparseOperator.jl` line 314:
```julia
return 2im * m * (-L * op.r2_D0_u + 2 * op.r3_D1_u + op.r4_D2_u)
```

**Status**: ✓ IDENTICAL

#### Viscous Diffusion

**Kore** (line 625):
```python
visc = op.viscous_diffusion(l,'u','upol',0)
```
From `operators.py` line 174:
```python
out = L*( -L*(l+2)*(l-1)*r0_D0_u + 2*L*r2_D2_u - 4*r3_D3_u - r4_D4_u )
```
Multiplied by `par.ViscosD = E`

**Cross.jl** (line 644):
```julia
visc_op = -operator_viscous_diffusion(op, l, E)
```
From `SparseOperator.jl` line 361:
```julia
return E * L * (-L * (l + 2) * (l - 1) * op.r0_D0_u +
                2 * L * op.r2_D2_u - 4 * op.r3_D3_u - op.r4_D4_u)
```

**Status**: ✓ IDENTICAL (negative sign in assembly matches)

#### Buoyancy Force

**Kore** (line 695):
```python
mtx = op.buoyancy(l,'u','',0)
```
From `operators.py` line 403:
```python
out = L * buoy
```
where `buoy = r4_D0_u` (non-anelastic, non-dipole)
Multiplied by `par.Beyonce` which should be `-Ra * E² / Pr`

**Cross.jl** (line 666):
```julia
buoy_op = operator_buoyancy(op, l, Ra, Pr)
```
From `SparseOperator.jl` line 390:
```julia
beyonce = -Ra * E^2 / Pr
return beyonce * L * op.r4_D0_u
```

**Status**: ✓ IDENTICAL

#### Coriolis Coupling (Off-diagonal)

**Kore** (lines 642-650):
```python
for i in [-1,1]:
    if l+i in ll_flo[1]:
        mtx = op.coriolis(l,'u','utor',i)
        col = basecol + col0 + mtx[1] * ut.N1
        loc_list = ut.packit(loc_list, mtx[0], row, col)
```
From `operators.py` lines 67-87, constructs coupling with coefficient factors and `mtx[1]` indicating offset.

**Cross.jl** (lines 648-661):
```julia
for offset in [-1, 1]
    l_coupled = l + offset
    if l_coupled in op.ll_bot
        k_coupled = findfirst(==(l_coupled), op.ll_bot)
        col_coupled = (nb_top + k_coupled - 1) * n_per_mode
        cori_off, _ = operator_coriolis_offdiag(op, l, m, offset)
        add_block!(A_rows, A_cols, A_vals, cori_off, row_base, col_coupled)
    end
end
```

**Status**: ✓ EQUIVALENT
- Same physics (Coriolis coupling between adjacent l-modes)
- Different index calculation but equivalent

### Summary
Matrix assembly is **CORRECTLY IMPLEMENTED** in Cross.jl, matching Kore's physics exactly.

---

## 5. NUMERICAL PARAMETERS AND TOLERANCES

### Sparsity Calculation

**Kore** (`utils.py`): Uses various implicit tolerances in Chebyshev computations  
**Cross.jl** (`SparseOperator.jl:267-279`):
```julia
function estimate_sparsity(N::Int, nl_modes::Int)
    total_size = 3 * nl_modes * (N + 1)
    nnz_estimate = 5 * nl_modes * N^2
    sparsity = 100.0 * (1.0 - nnz_estimate / total_size^2)
    return round(sparsity, digits=2)
end
```

**Status**: ✓ CONSISTENT (estimates only, not used in actual computation)

### Interior DOF Selection

**Cross.jl** (line 787):
```julia
interior_dofs = findall(i -> abs(B_diag[i]) > 1e-14, 1:n)
```

This uses threshold 1e-14 to identify non-zero B diagonal entries after BCs.

**Kore**: No explicit interior DOF filtering shown in assemble.py  

**Status**: ✓ REASONABLE (numerical stability)

---

## 6. CORIOLIS AND VELOCITY COUPLING

### v → u Coupling (Important for completeness)

**Kore** (lines 89-113, section v, component upol):
Implements coupling from toroidal velocity to poloidal equation with explicit handling of symm=±1 cases.

**Cross.jl** (lines 703-724):
```julia
for offset in [-1, 1]
    l_coupled = l + offset
    if l_coupled in op.ll_top
        k_coupled = findfirst(==(l_coupled), op.ll_top)
        col_coupled = (k_coupled - 1) * n_per_mode
        cori_v_to_u = operator_coriolis_v_to_u(op, l, m, offset)
        add_block!(A_rows, A_cols, A_vals, cori_v_to_u, row_base, col_coupled)
    end
end
```

With operators defined in `SparseOperator.jl:410-425`:
```julia
function operator_coriolis_v_to_u(op::SparseStabilityOperator{T},
                                 l::Int, m::Int, offset::Int) where {T}
    if offset == -1
        C = (l^2 - 1) * sqrt(l^2 - m^2) / (2l - 1)
        return 2 * C * ((l - 1) * op.r1_D0_v - op.r2_D1_v)
    elseif offset == 1
        C = l * (l + 2) * sqrt((l + m + 1) * (l - m + 1)) / (2l + 3)
        return 2 * C * (-(l + 2) * op.r1_D0_v - op.r2_D1_v)
    end
end
```

**Status**: ✓ IDENTICAL (correct bidirectional Coriolis coupling)

---

## 7. SYMMETRY CONSTRAINT APPLICATION

### Kore Mode Selection Logic

The Kore `ell()` function's symmetry constraint is encoded in:
```python
if ut.symm1 == 1:   offd = -1  # symmetric, couple to l-1
elif ut.symm1 == -1: offd = 1  # antisymmetric, couple to l+1
```
This is applied in `operators.py` lines 75-87, 101-113.

### Cross.jl Mode Structure

Cross.jl determines l-mode distributions upfront in `compute_l_modes()` and then validates couplings exist:
```julia
if l_coupled in op.ll_bot
    # Only assemble if coupling target exists
end
```

**Status**: ✓ EQUIVALENT
- Different implementation strategy (upfront vs on-demand)
- Same physical constraints

---

## SUMMARY OF FINDINGS

### Bugs Found
1. **CRITICAL**: `compute_l_modes()` excludes l=0 for m=0 and misses l=lmax+1
   - Severity: HIGH for m=0 problems
   - Impact: Reduced spectral completeness for axially symmetric cases

### Correctly Implemented
- ✓ Heating mode operators (differential/internal)
- ✓ Boundary condition physics (tau method vs explicit)
- ✓ All matrix operators (Coriolis, viscous, buoyancy, advection)
- ✓ Bidirectional Coriolis coupling (u↔v)
- ✓ Symmetry constraints
- ✓ Numerical stability measures

### Recommendations

**URGENT**:
Fix `compute_l_modes()` to handle m=0 case correctly - this is essential for axially symmetric flows.

**OPTIONAL**:
Consider adding comments referencing Kore line numbers for cross-verification.

---

## APPENDIX: KEY CODE LOCATIONS

### Kore Reference Points
- Mode selection: `bin/utils.py:174-183`
- Heating: `bin/operators.py:699-770`
- Boundary conditions: `bin/assemble.py:1188-1340`
- Matrix assembly: `bin/assemble.py:429-812`
- Coriolis coupling: `bin/operators.py:48-124`

### Cross.jl Reference Points
- Mode selection: `src/SparseOperator.jl:240-260`
- Heating: `src/SparseOperator.jl:504-562`
- Boundary conditions: `src/SparseOperator.jl:816-921`
- Matrix assembly: `src/SparseOperator.jl:578-800`
- Coriolis coupling: `src/SparseOperator.jl:311-348`, 410-425

