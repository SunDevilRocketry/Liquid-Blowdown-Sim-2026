# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Sun Devil Rocketry

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import CoolProp.CoolProp as CP

'''
This module simulates pressure drop across a series of components

Define the component parameters below, for overview of input variables scroll to the class definitions'''

valve1 = valve(Kv=10, Cv=12, K=0.5, dia=0.1)
pipe1 = pipe(E_D=0.0001, L=5.0)

pres_path = [tank, valve1, pipe1, plenum]
deltaP = 300 # psi
P_tank = 800 # psi

class PresDrop:
    class tank
    class valve
        '''
         Params:
        Kv: flow coefficient
        Cv: flow coefficient
        K: pressure loss coefficient
        dia: diameter
        '''
        params = [Kv Cv K dia, length]]
    class pipe(P_in, P_out, flow_rate):
        '''
       Params:
       E_D: roughness
         L: length
        '''
        params = [E_d ]


"""
pressuredrop_framework.py

Simple pressure-drop solver for an incompressible liquid flowing through a path of components.
- Components: Tank (holds fluid), Valve (Kv or Cv or K loss), Pipe (Darcy-Weisbach)
- Fluid: simple dict with density (kg/m3) and viscosity (Pa·s)
- Solve for Q such that sum(dP_i(Q)) == DeltaP

Author: ChatGPT (code for you to play with)
"""

from dataclasses import dataclass
import math

# -----------------------
# Units helpers
# -----------------------
PA_PER_PSI = 6894.757293168361  # Pa per psi

def psi_to_Pa(psi):
    return psi * PA_PER_PSI

def Pa_to_psi(pa):
    return pa / PA_PER_PSI

# -----------------------
# Cv/Kv conversions & Kv flow function
# -----------------------
def Cv_to_Kv(Cv):
    # Kv ≈ 0.865 * Cv (engineering approx)
    return 0.865 * Cv

def Kv_to_Cv(Kv):
    return Kv / 0.865

def Kv_flow_Q_from_dP(dP_Pa, Kv):
    """
    Kv convention: Q_m3h = Kv * sqrt(dP_bar)
    dP_bar = dP_Pa / 1e5
    returns Q in m^3/s
    """
    if dP_Pa <= 0 or Kv <= 0:
        return 0.0
    dP_bar = dP_Pa / 1e5
    Q_m3h = Kv * math.sqrt(dP_bar)
    return Q_m3h / 3600.0

def Kv_dP_from_Q(Q_m3s, Kv):
    if Q_m3s <= 0 or Kv <= 0:
        return 0.0
    Q_m3h = Q_m3s * 3600.0
    dP_bar = (Q_m3h / Kv) ** 2
    return dP_bar * 1e5

# -----------------------
# Friction factor (Haaland)
# -----------------------
def friction_factor(Re, D, eps):
    """
    Haaland approximation for turbulent flow; laminar fallback.
    Re: Reynolds number
    D: diameter (m)
    eps: absolute roughness (m)
    returns f (Darcy friction factor)
    """
    if Re <= 0:
        return 0.0
    if Re < 2000:
        return 64.0 / Re  # laminar
    # Haaland:
    term = (eps / D) / 3.7
    # avoid math domain error
    inv_sqrt_f = -1.8 * math.log10(term**1.11 + 6.9 / Re)
    f = 1.0 / (inv_sqrt_f**2)
    return f

# -----------------------
# Component classes
# -----------------------
@dataclass
class Tank:
    name: str
    pressure_pa: float  # absolute upstream pressure in Pa
    fluid: dict         # {'rho': kg/m3, 'mu': Pa.s}
    # Tank does not produce dP but holds fluid/pressure

@dataclass
class Valve:
    name: str
    Kv: float = None   # m^3/h per sqrt(bar)
    Cv: float = None   # US gallons per minute per sqrt(psi)
    K_loss: float = None  # dimensionless loss coeff (optional)
    dia: float = None     # m (optional, for area)
    # Behavior:
    # - If Kv/Cv provided, we'll use Kv relation (preferred for incompressible liquids)
    # - If K_loss provided, we use dP = K * (rho * v^2 / 2) using throat area (dia required)

    def __post_init__(self):
        if self.Kv is None and self.Cv is None and self.K_loss is None:
            raise ValueError("Valve must have Kv or Cv or K_loss defined.")
        if self.Kv is None and self.Cv is not None:
            self.Kv = Cv_to_Kv(self.Cv)
        if self.Cv is None and self.Kv is not None:
            self.Cv = Kv_to_Cv(self.Kv)

    def dP_given_Q(self, Q_m3s, fluid):
        rho = fluid['rho']
        # If Kv known, use Kv formula (most common)
        if self.Kv is not None and self.Kv > 0:
            return Kv_dP_from_Q(Q_m3s, self.Kv)
        # else use K_loss and area
        if self.K_loss is not None and self.dia is not None:
            A = math.pi * (self.dia**2) / 4.0
            if A <= 0:
                return 0.0
            v = Q_m3s / A
            return self.K_loss * 0.5 * rho * v**2
        # fallback: no drop
        return 0.0

    def mdot_given_Q(self, Q_m3s, fluid):
        return Q_m3s * fluid['rho']

@dataclass
class Pipe:
    name: str
    D: float         # diameter (m)
    L: float         # length (m)
    eps: float = 1e-5  # roughness (m), default small
    # head loss: Darcy-Weisbach

    def dP_given_Q(self, Q_m3s, fluid):
        if Q_m3s <= 0:
            return 0.0
        rho = fluid['rho']
        mu = fluid['mu']
        A = math.pi * (self.D**2) / 4.0
        v = Q_m3s / A
        Re = rho * v * self.D / mu if mu > 0 else 1e9
        f = friction_factor(Re, self.D, self.eps)
        dp = f * (self.L / self.D) * 0.5 * rho * v**2
        return dp

    def mdot_given_Q(self, Q_m3s, fluid):
        return Q_m3s * fluid['rho']

@dataclass
class Plenum:
    name: str
    pressure_pa: float  # downstream absolute pressure

# -----------------------
# Solver: find Q such that sum(dP_i(Q)) = DeltaP
# -----------------------
def total_dP_for_Q(Q_m3s, path, fluid):
    """
    path: list of components in flow order, e.g.
          [Tank(...), Valve(...), Pipe(...), Valve(...), Plenum(...)]
    Returns total dP from upstream tank pressure to downstream plenum pressure
    by summing component contributions.
    """
    total = 0.0
    for comp in path:
        if isinstance(comp, Tank):
            continue
        if isinstance(comp, Plenum):
            continue
        # each component knows how to compute dP given Q and fluid
        total += comp.dP_given_Q(Q_m3s, fluid)
    return total

def solve_flow_for_deltaP(path, fluid, deltaP_pa, Q_min=1e-12, Q_max_guess=1.0):
    """
    Solve for Q (m^3/s) such that total_dP_for_Q(Q) ~= deltaP_pa.
    Uses bracketing + bisection. Q_max increases until bracket found.
    Returns (Q_m3s, mdot, per_component_dPs, cumulative_pressures)
    """
    # bracket search: ensure total_dP(Q_min) < deltaP and total_dP(Q_hi) > deltaP
    def total_minus_target(Q):
        return total_dP_for_Q(Q, path, fluid) - deltaP_pa

    # find Q_hi
    Q_lo = Q_min
    Q_hi = Q_max_guess
    # increase Q_hi exponentially until total > deltaP or reach huge
    for _ in range(60):
        if total_minus_target(Q_hi) >= 0:
            break
        Q_hi *= 2.0
    else:
        raise RuntimeError("Cannot bracket solution: deltaP too large for modeled path or Q_max too small.")

    # bisection
    for _ in range(60):
        Q_mid = 0.5 * (Q_lo + Q_hi)
        f_mid = total_minus_target(Q_mid)
        if abs(f_mid) < 1e-6:
            Q_lo = Q_mid
            break
        # choose side
        if f_mid > 0:
            Q_hi = Q_mid
        else:
            Q_lo = Q_mid
    Q = 0.5 * (Q_lo + Q_hi)
    mdot = Q * fluid['rho']

    # get per-component dPs and cumulative pressures
    per_dps = []
    p_up = None
    # upstream pressure is first Tank pressure
    for c in path:
        if isinstance(c, Tank):
            p_up = c.pressure_pa
            break
    if p_up is None:
        p_up = 0.0
    pressures = [p_up]
    p_current = p_up
    for comp in path:
        if isinstance(comp, Tank):
            continue
        if isinstance(comp, Plenum):
            # final downstream pressure set by plenum
            per = p_current - comp.pressure_pa
            per_dps.append(comp.pressure_pa - p_current)  # negative by convention
            p_current = comp.pressure_pa
            pressures.append(p_current)
            continue
        dp = comp.dP_given_Q(Q, fluid)
        per_dps.append(dp)
        p_current = p_current - dp
        pressures.append(p_current)

    return {
        'Q_m3s': Q,
        'mdot_kg_s': mdot,
        'per_component_dP_Pa': per_dps,
        'pressures_Pa': pressures
    }

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Define two sample fluids (simple properties)
    LOX = {'rho': 1141.0, 'mu': 1.2e-3}     # approximate LOX liquid density, viscosity
    RP1 = {'rho': 810.0, 'mu': 2.5e-3}      # kerosene-like density, viscosity

    # Build components
    tank_ox = Tank("Tank_OX", pressure_pa=psi_to_Pa(800), fluid=LOX)   # 800 psi tank
    v_vent = Valve("Ox_VENT", Cv=1.5)   # small vent
    pipe1 = Pipe("lineOx1", D=0.05, L=2.0, eps=1e-5)
    v_main = Valve("Ox_MAIN", Cv=12.0)
    chamber = Plenum("chamber", pressure_pa=psi_to_Pa(14.7))  # downstream ambient

    # Define path (upstream to downstream)
    path = [tank_ox, v_vent, pipe1, v_main, chamber]

    # desired deltaP: upstream - downstream
    deltaP_pa = tank_ox.pressure_pa - chamber.pressure_pa
    print(f"Available ΔP = {Pa_to_psi(deltaP_pa):.1f} psi ({deltaP_pa:.1f} Pa)")

    sol = solve_flow_for_deltaP(path, LOX, deltaP_pa)
    print("Solution:")
    print(f"  Q = {sol['Q_m3s']:.6e} m^3/s   mdot = {sol['mdot_kg_s']:.4f} kg/s")
    for c, dp in zip([c for c in path if not isinstance(c, Tank) and not isinstance(c, Plenum)], sol['per_component_dP_Pa']):
        print(f"  component {c.name:10s}  dP = {dp/1e3:.3f} kPa ({Pa_to_psi(dp):.3f} psi)")
    print(f"  downstream pressure (Pa): {sol['pressures_Pa'][-1]:.1f}")
