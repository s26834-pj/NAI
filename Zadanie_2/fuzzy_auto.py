#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projekt: Fuzzy AutoShift — rekomendacja biegu  + symulator czasu rzeczywistego

Autorzy: Wiktor Rapacz, Hanna Paczoska
Data: 2025-11-05

OPIS ROZWIĄZANIA (PL)
---------------------
To prosty symulator automatycznej skrzyni biegów sterowanej układem rozmytym (FIS), z trzema wejściami
sterowanymi z klawiatury: gaz (throttle), nachylenie drogi (slope) i ładunek (payload). Wyjściem FIS jest
„docelowy bieg” (gear), defuzyfikowany metodą centroid-over-singletons (wartość liczbowa 1..5). Na wyjściu
zastosowano histerezę, aby zapobiegać migotaniu przełączeń. Dodatkowo symulator zawiera prostą fizykę
jazdy (opór powietrza, wpływ nachylenia i obciążenia na przyspieszenie),

ZASADY DZIAŁANIA (PL)
---------------------
1) FIS:
   - Wejścia: throttle ∈ [0,100], slope ∈ [-0.10,+0.10], payload ∈ [0,100].
   - Zbiory rozmyte (triangles/trapezoids) oraz reguły Rule(throttle × slope × payload → gear-set).
   - Agregacja: AND = min; OR (między regułami do tego samego biegu) = max.
   - Defuzyfikacja: centroid-over-singletons (średnia ważona numerów biegów 1..5).
2) Stabilizacja: histereza liczona w klatkach (rekomendacja musi się utrzymać kilka klatek).
3) Heurystyki (czytelne i proste, bez dodatkowych wejść):
   - Smart launch: na 1. biegu nie wychodzimy, dopóki prędkość ≤ 7 km/h lub gaz ≤ 20% (z wyjątkiem zjazdu).
   - „Governor” przy przyspieszaniu: zapobiega przeskokom biegów (1→3, 3→5 itd.).
   - Ograniczenia najwyższego biegu na stromych podjazdach.
   - Ostateczny limit biegu od obciążenia (ciężkie auto nie pojedzie na 5).
   - Miękkie „zatrzymanie” na 2. biegu przy zwalnianiu ciężkiego auta (żeby 5→3→1 nie było zbyt gwałtowne).
4) Fizyka:
   - Prosty model: siła „silnika” ~ gaz × przełożenie biegu, pomniejszana o opór aero, nachylenie i masę.
   - Hamulec odcina gaz (przy większym wciśnięciu) i dokłada siłę hamującą skalowaną prędkością.

INSTRUKCJA ŚRODOWISKA (Python 3.14)
-----------------------------------
    pip install numpy pygame-ce fuzzylogic==1.2.0 matplotlib
    python fuzzy_auto.py

STEROWANIE
----------
    G          — throttle (gaz) 0..100% (puszczenie = powolny spadek gazu)
    S          — brake (hamulec) 0..100% (puszczenie = szybkie odpuszczenie hamulca)
    A / Z      — slope (nachylenie) –0.10..+0.10
    J / K      — payload (ładunek) 0..100%
    ESC        — wyjście

Uwaga nt. funkcji przynależności (biblioteka `fuzzylogic` :
- triangular(low, high)  — 2 parametry; maksimum w połowie [low, high].
- trapezoid(low, c_low, c_high, high) — low < c_low <= c_high < high (brak równości).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame

# --------- fuzzylogic  ---------
try:
    from fuzzylogic.classes import Domain, Set, Rule
    from fuzzylogic.functions import R, S, triangular, trapezoid
except Exception:
    print("Błąd importu 'fuzzylogic' — zainstaluj: pip install fuzzylogic==1.2.0 matplotlib", file=sys.stderr)
    raise

# ==============================
# Parametry symulatora i UI
# ==============================
WIDTH, HEIGHT = 920, 560
FPS = 30

MAX_SPEED = 180.0
DRAG = 0.010
BASE_ENGINE = 14.0
GEAR_POWER = {1: 1.40, 2: 1.15, 3: 0.95, 4: 0.78, 5: 0.70}  # 5. bieg 0.75 → 0.70

GEAR_STICK_FRAMES = 10
GEAR_SNAP_EPS = 0.35

MAX_PAYLOAD_KG = 500.0
BRAKE_COEFF = 5.0   # siła hamowania


@dataclass
class CarState:
    """Container for the simulated vehicle state.

    Attributes
    ----------
    speed_kmh : float
        Instantaneous vehicle speed in km/h (derived from the physics step).
    throttle_pct : float
        Throttle pedal position in percent [0..100].
    slope : float
        Road grade as a fraction [-0.10 .. +0.10], where +0.10 ≈ +10% uphill.
    payload_pct : float
        Additional payload as a percent [0..100], scaled to MAX_PAYLOAD_KG for HUD.
    engine_load_pct : float
        Derived engine load metric (for HUD only), clipped to [0..100].
    gear : int
        Stable gear after hysteresis (1..5).
    brake_pct : float
        Brake pedal position in percent [0..100].
    """
    speed_kmh: float = 0.0
    throttle_pct: float = 0.0
    slope: float = 0.00
    payload_pct: float = 0.0
    engine_load_pct: float = 0.0
    gear: int = 1
    brake_pct: float = 0.0     # hamulec 0..100% (S)


# ==============================
# Definicje MF i reguł
# ==============================
def define_mf(domain: Domain, name: str, func: Any) -> None:
    """Attach a membership function to a Domain under a given attribute name."""
    setattr(domain, name, func)


throttle = Domain("throttle", 0, 100)
slope    = Domain("slope", -0.10, 0.10)
payload  = Domain("payload", 0, 100)
gear     = Domain("gear", 1, 5)

# throttle
define_mf(throttle, 'low', trapezoid(-10.0, 0.0, 20.0, 40.0))
define_mf(throttle, 'mid',   triangular(25.0, 80.0))
define_mf(throttle, 'high',  trapezoid(60.0, 80.0, 100.0, 120.0))

# slope
define_mf(slope,  'downhill', trapezoid(-0.101, -0.095, -0.04, -0.01))
define_mf(slope,  'flat',     triangular(-0.02, 0.02))
define_mf(slope,  'uphill',   trapezoid(0.01, 0.04, 0.095, 0.101))

# payload
define_mf(payload, 'light',  trapezoid(-1.0, 0.0, 30.0, 60.0))   # 0% → μ=1
define_mf(payload, 'medium', triangular(25.0, 75.0))
define_mf(payload, 'heavy',  trapezoid(60.0, 75.0, 100.0, 120.0))

# gear sets
define_mf(gear, 'first',  Set(R(0, 10)))
define_mf(gear, 'second', Set(S(5, 30))  & Set(R(30, 45)))
define_mf(gear, 'third',  Set(S(25, 50)) & Set(R(50, 65)))
define_mf(gear, 'fourth', Set(S(45, 70)) & Set(R(65, 85)))
define_mf(gear, 'fifth',  Set(S(75, 90)))

# rules
rules = [
    Rule({(throttle.low,  slope.uphill,   payload.heavy):  gear.first}),
    Rule({(throttle.mid,  slope.uphill,   payload.heavy):  gear.second}),
    Rule({(throttle.high, slope.uphill,   payload.heavy):  gear.third}),

    Rule({(throttle.low,  slope.uphill,   payload.medium): gear.second}),
    Rule({(throttle.mid,  slope.uphill,   payload.medium): gear.third}),
    Rule({(throttle.high, slope.uphill,   payload.medium): gear.fourth}),

    Rule({(throttle.low,  slope.uphill,   payload.light):  gear.second}),
    Rule({(throttle.mid,  slope.uphill,   payload.light):  gear.third}),
    Rule({(throttle.high, slope.uphill,   payload.light):  gear.fourth}),

    Rule({(throttle.low,  slope.flat,     payload.heavy):  gear.second}),
    Rule({(throttle.mid,  slope.flat,     payload.heavy):  gear.third}),
    Rule({(throttle.high, slope.flat,     payload.heavy):  gear.third}),

    Rule({(throttle.low,  slope.flat,     payload.medium): gear.third}),
    Rule({(throttle.mid,  slope.flat,     payload.medium): gear.fourth}),
    Rule({(throttle.high, slope.flat,     payload.medium): gear.fourth}),

    Rule({(throttle.low,  slope.flat,     payload.light):  gear.second}),
    Rule({(throttle.mid,  slope.flat,     payload.light):  gear.third}),
    Rule({(throttle.high, slope.flat,     payload.light):  gear.fourth}),

    Rule({(throttle.low,  slope.downhill, payload.heavy):  gear.third}),
    Rule({(throttle.mid,  slope.downhill, payload.heavy):  gear.fourth}),
    Rule({(throttle.high, slope.downhill, payload.heavy):  gear.fourth}),

    Rule({(throttle.low,  slope.downhill, payload.medium): gear.fourth}),
    Rule({(throttle.mid,  slope.downhill, payload.medium): gear.fourth}),
    Rule({(throttle.high, slope.downhill, payload.medium): gear.fifth}),

    Rule({(throttle.low,  slope.downhill, payload.light):  gear.fourth}),
    Rule({(throttle.mid,  slope.downhill, payload.light):  gear.fifth}),
    Rule({(throttle.high, slope.downhill, payload.light):  gear.fifth}),
]

GEAR_SET_TO_INT = {gear.first: 1, gear.second: 2, gear.third: 3, gear.fourth: 4, gear.fifth: 5}


def calculate_gear(throttle_pct: float, slope_val: float, payload_pct: float) -> int:
    """Compute fuzzy gear recommendation (centroid-over-singletons)."""
    gear_memberships = {}
    for rule in rules:
        (conditions, target_set) = list(rule.conditions.items())[0]
        thr_cond, slp_cond, pay_cond = conditions
        mu_t = float(thr_cond(throttle_pct))
        mu_s = float(slp_cond(slope_val))
        mu_p = float(pay_cond(payload_pct))
        activation = min(mu_t, mu_s, mu_p)
        gear_memberships[target_set] = max(activation, gear_memberships.get(target_set, 0.0))

    num = den = 0.0
    for g_set, mu in gear_memberships.items():
        g_num = GEAR_SET_TO_INT.get(g_set)
        if g_num is None:
            for s, n in GEAR_SET_TO_INT.items():
                if str(s) == str(g_set):
                    g_num = n
                    break
        if g_num is None:
            continue
        num += mu * g_num
        den += mu

    return int(round(np.clip(num / den, 1, 5))) if den > 1e-12 else 1


class GearHysteresis:
    """Discrete recommendation stabilizer (hysteresis over frames)."""
    def __init__(self, stick_frames=GEAR_STICK_FRAMES, snap_eps=GEAR_SNAP_EPS):
        self.current = 1
        self._cnt = 0
        self._need = int(stick_frames)
        self._eps = float(snap_eps)

    def update(self, rec: int) -> int:
        """Update the latched gear given a new recommendation."""
        if abs(rec - self.current) > self._eps:
            self._cnt += 1
            if self._cnt >= self._need:
                self.current = int(np.clip(rec, 1, 5))
                self._cnt = 0
        else:
            self._cnt = 0
        return self.current


def payload_factor(payload_pct: float) -> float:
    """Map payload percent to a mass penalty factor for physics."""
    return 0.5 * (payload_pct / 100.0)


def step_physics(state: CarState, target_gear: int, dt: float) -> None:
    """Advance simple longitudinal dynamics by one frame."""
    if state.brake_pct > 5:
        effective_throttle = 0.0
    else:
        effective_throttle = state.throttle_pct

    power = BASE_ENGINE * GEAR_POWER.get(int(target_gear), 1.0) * (effective_throttle / 100.0)
    slope_drag = 20.0 * state.slope
    aero = DRAG * (state.speed_kmh ** 1.3)
    mass_penalty = 1.0 + payload_factor(state.payload_pct)
    speed_scale = 1.0 + 0.5 * (state.speed_kmh / MAX_SPEED)  # 1.0 .. 1.5
    brake_force = BRAKE_COEFF * (state.brake_pct / 100.0) * speed_scale

    accel = (power - aero - slope_drag - brake_force) / mass_penalty
    state.speed_kmh = float(np.clip(state.speed_kmh + accel * dt * 5.0, 0.0, MAX_SPEED))


def estimate_engine_load(state: CarState) -> float:
    """Estimate a synthetic engine load metric for HUD purposes."""
    hill = 50.0 * (state.slope * 10.0)
    speed_factor = 0.25 * (state.speed_kmh / MAX_SPEED) * 100.0
    gear_factor = (6 - state.gear) * 4.0
    payload_eff = 0.6 * state.payload_pct
    base = 0.5 * state.throttle_pct + 0.3 * (hill + speed_factor + gear_factor) + 0.2 * payload_eff
    return float(np.clip(base, 0.0, 100.0))


def draw_ui(screen: pygame.Surface, font: pygame.font.Font, state: CarState, rec_gear: int) -> None:
    """Render HUD with labels and side-by-side bars plus a gear dial."""
    screen.fill((12, 18, 24))

    lines = [
        f"Prędkość (stan):        {state.speed_kmh:6.1f} km/h",
        f"Gaz (wejście):          {state.throttle_pct:5.1f} %   (G)",
        f"Nachylenie (wejście):   {state.slope:+.3f}           (A/Z)",
        f"Ładunek (wejście):      {state.payload_pct:5.1f} %  (~{state.payload_pct/100*MAX_PAYLOAD_KG:4.0f} kg)  (J/K)",
        f"Hamulec (wejście):      {state.brake_pct:5.1f} %     (S)",
        f"Obciążenie (pochodne):  {state.engine_load_pct:5.1f} %",
        f"Bieg (stabilny): {state.gear:d}   |   Rekomendacja (centroid): {rec_gear:d}",
        "ESC = wyjście",
    ]
    for i, txt in enumerate(lines):
        surf = font.render(txt, True, (230, 235, 240))
        screen.blit(surf, (24, 24 + i * 28))

    # Paski + etykiety w jednej linii
    bar_w = 260
    bar_h = 18
    label_x = 24
    bar_x = 200
    y0 = 24 + 9 * 28

    rows = [
        ("Gaz (throttle)",     state.throttle_pct / 100.0),
        ("Nachylenie (slope)", (state.slope + 0.10) / 0.20),
        ("Ładunek (payload)",  state.payload_pct / 100.0),
        ("Prędkość",           state.speed_kmh / MAX_SPEED),
        ("Obciążenie silnika", state.engine_load_pct / 100.0),
        ("Hamulec",            state.brake_pct / 100.0),
    ]

    def bar(xb, yb, w, h, value):
        pygame.draw.rect(screen, (60, 70, 80), (xb, yb, w, h), border_radius=8)
        pygame.draw.rect(screen, (180, 200, 210),
                         (xb + 2, yb + 2, int((w - 4) * np.clip(value, 0, 1)), h - 4),
                         border_radius=6)

    for i, (label, frac) in enumerate(rows):
        y = y0 + i * 30
        lbl = font.render(label, True, (180, 190, 200))
        screen.blit(lbl, (label_x, y))
        bar(bar_x, y, bar_w, bar_h, frac)

    # Okrąg z biegami
    cx, cy = WIDTH - 240, HEIGHT // 2
    pygame.draw.circle(screen, (35, 42, 52), (cx, cy), 110)
    for g in range(1, 6):
        ang = math.radians(225 - (g - 1) * (180 / 4))
        gx = cx + int(80 * math.cos(ang))
        gy = cy - int(80 * math.sin(ang))
        color = (220, 230, 240) if g != state.gear else (120, 200, 120)
        txt = font.render(str(g), True, color)
        rect = txt.get_rect(center=(gx, gy))
        screen.blit(txt, rect)


def handle_input(state: CarState) -> None:
    """Read keyboard state and update inputs (throttle, brake, slope, payload).

    Notes
    -----
    - Throttle increases while holding **G** and decays slowly when released,
      mimicking a real pedal return.
    - Brake rises fast on **S** and releases quickly when not pressed.
    - Slope and payload adjust with **A/Z** and **J/K** respectively.
    """
    keys = pygame.key.get_pressed()

    # *** GAZ jak prawdziwy pedał — pod klawiszem G ***
    if keys[pygame.K_g]:
        # gaz rośnie umiarkowanie (łatwa kontrola)
        state.throttle_pct = float(np.clip(state.throttle_pct + 40.0 / FPS, 0, 100))
    else:
        # gaz opada BARDZO powoli (utrzymuje się podczas zmiany obciążenia)
        state.throttle_pct = float(np.clip(state.throttle_pct - 5.0 / FPS, 0, 100))

    # *** HAMULEC ***
    if keys[pygame.K_s]:
        state.brake_pct = float(np.clip(state.brake_pct + 180.0 / FPS, 0, 100))
    else:
        state.brake_pct = float(np.clip(state.brake_pct - 180.0 / FPS, 0, 100))

    # NACHYLENIE
    if keys[pygame.K_a]:
        state.slope = float(np.clip(state.slope + 0.25 / FPS, -0.10, 0.10))
    if keys[pygame.K_z]:
        state.slope = float(np.clip(state.slope - 0.25 / FPS, -0.10, 0.10))

    # ŁADUNEK
    if keys[pygame.K_j]:
        state.payload_pct = float(np.clip(state.payload_pct + 60.0 / FPS, 0, 100))
    if keys[pygame.K_k]:
        state.payload_pct = float(np.clip(state.payload_pct - 60.0 / FPS, 0, 100))


def main() -> None:
    """Run the realtime loop: input → fuzzy recommendation → heuristics → physics → render."""
    pygame.init()
    pygame.display.set_caption("Fuzzy AutoShift")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    state = CarState()
    hyster = GearHysteresis()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        handle_input(state)

        # 1) Rekomendacja z FIS
        rec_int = calculate_gear(state.throttle_pct, state.slope, state.payload_pct)

        # 2) Postój
        if state.speed_kmh < 5 and state.throttle_pct < 15:
            rec_int = 1

        # 3) Inteligentne ruszanie – tylko gdy NIE zjeżdżamy
        if hyster.current == 1 and state.slope >= -0.03:
            if not (state.speed_kmh > 7.0 and state.throttle_pct > 20.0):
                rec_int = 1

        # 3.5) Minimalny bieg przy dużej prędkości (tylko jeśli gaz jest dodawany)
        if state.speed_kmh >= 60.0 and state.throttle_pct > 20.0:
            rec_int = max(rec_int, 3)

        # 3.55) Governor przy przyspieszaniu: nie przeskakuj biegów
        if state.throttle_pct > 25.0:
            if state.speed_kmh < 20.0:
                rec_int = min(rec_int, 2)
            elif state.speed_kmh < 40.0:
                if state.payload_pct < 60:
                    rec_int = min(rec_int, 3)
                else:
                    rec_int = min(rec_int, 4)
            elif state.speed_kmh < 70.0:
                rec_int = min(rec_int, 4)
            # powyżej 70 km/h nie ograniczamy

        # 3.6) Miękkie przejście przez bieg 2 przy WYTRACANIU (nie na zjeździe)
        if state.throttle_pct < 15.0 and state.slope >= -0.02:
            if 8.0 < state.speed_kmh < 28.0:
                rec_int = 2
            elif state.speed_kmh <= 8.0:
                rec_int = 1

        # 3.62) Wymuszenie 2. biegu przy ciężkim aucie w zakresie 10–30 km/h
        if state.payload_pct > 60.0 and 10.0 < state.speed_kmh < 30.0:
            rec_int = min(max(rec_int, 2), 2)

        # 3.65) Limit najwyższego biegu na podjazdach
        if state.slope >= 0.08:
            rec_int = min(rec_int, 3)
        elif state.slope >= 0.06:
            rec_int = min(rec_int, 4)

        # 3.7) Pozwalamy na 5 TYLKO gdy auto jest lekkie i warunki są sprzyjające
        is_fast = state.speed_kmh > 95
        is_push = state.throttle_pct > 45
        is_flat = state.slope <= 0.03
        is_light = state.payload_pct < 45
        if is_fast and is_push and is_flat and is_light:
            rec_int = max(rec_int, 5)

        # 3.66) Ostateczny limit biegu według obciążenia (zawsze działa)
        if state.payload_pct > 75:
            rec_int = min(rec_int, 3)
        elif state.payload_pct > 45:
            rec_int = min(rec_int, 4)

        # 4) Histereza
        stable = hyster.update(rec_int)
        state.gear = stable

        # 5) Fizyka + obciążenie
        step_physics(state, stable, dt)
        state.engine_load_pct = estimate_engine_load(state)

        # 6) UI
        draw_ui(screen, font, state, rec_gear=rec_int)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Błąd:", e, file=sys.stderr)
        sys.exit(1)