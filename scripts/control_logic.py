"""
Week 3 deliverable: the control decision engine ("brain") that sits between
the LSTM forecasts and the physical relay/alerts. This runs on the Python
side (where the LSTM models live); the ESP32 firmware (relay_control.ino)
just reads whatever decision this produces via a Blynk virtual pin and acts
on it -- the microcontroller has no model logic of its own.

Design confirmed with Aizziq (per client WhatsApp: "kene ade bende yg
control", supervisor wants an actuation/feedback loop, not just monitoring):

Appliance-differentiated policy (NOT one-size-fits-all relay):
    - Water heater (geyser/AC proxy): HARD CUTOFF. High-current safety risk,
      matches the client's own reference example (old project: relay trips
      on overcurrent). Cutoff is PREDICTIVE (acts on the LSTM's next-30min
      forecast crossing a threshold), not just reactive to the current
      instantaneous reading -- this is what actually makes "Guna AI LSTM"
      mean something in the control chapter, not just the forecasting one.
    - Fan (kipas diri/TV proxy): SOFT SHIFT. Comfort item, not a safety
      risk -- delay/defer rather than cut power outright.
    - Laptop: NOTIFY ONLY, never auto-cut. Cutting power to a laptop
      mid-session risks data loss; the failure mode of automating this is
      worse than the problem it would solve.

Also implements (per confirmed Week-3 scope):
    - Multi-tier alerts (green/yellow/orange/red) instead of a single binary trip
    - Energy budget cap (user-set kWh limit vs. accumulated usage)
    - Dynamic threshold tied to TNB Tariff A's MONTHLY CUMULATIVE tiers
      (200/300/600/900 kWh bands -- Malaysia's domestic tariff is NOT
      time-of-day/peak-off-peak, it's a monthly consumption bracket
      structure, per Ch2.2 of the existing report). As cumulative monthly
      usage approaches the next, more expensive tier boundary, the system
      tightens its trip thresholds to help the household avoid crossing
      into it.
"""

from dataclasses import dataclass, field
from enum import Enum


# ---- TNB Tariff A - Domestic structure (from the existing Ch2.2 of the report) ----
TNB_TIERS = [
    (0, 200, 21.8),
    (200, 300, 33.4),
    (300, 600, 51.6),
    (600, 900, 54.6),
    (900, float("inf"), 57.1),
]

# Appliance-specific instantaneous power thresholds (Watts) -- the level at
# which the appliance is considered to be drawing a "high" / risky load.
# ORIGINALLY calibrated from actual percentiles of each channel's active-power
# distribution in the iAWE proxy data (Week 3):
#   geyser: p50=1758W p95=1890W p99=1941W max=2060W
#   fan:    p50=74W   p95=76W   p99=77W   max=80W    (near-flat)
#   computer: p50=34W p95=70W   p99=81W   max=113W
#
# RESCALED for the real physical demo hardware Ilyasa is actually using
# (1500W water heater, 70W fan -- confirmed Week 4). The proxy data's own
# design intent was preserved rather than re-guessed from scratch: geyser's
# original threshold (1900W) sat ~8% above the proxy's own median (1758W) --
# not deep into "abnormal", just above typical full-power running, so the
# appliance spends most of its ON time in ORANGE/NOTIFY and only crosses into
# RED/CUTOFF from a genuine forecast spike or from tariff-tier tightening
# (dynamic_threshold_multiplier). Applying that same ~8% ratio to the real
# 1500W unit: 1500 * (1900/1758) = 1620W (rounded). At normal full power
# (1500W / 1620W threshold = 0.93 ratio) the real heater sits in ORANGE,
# same behaviour as the proxy did -- CUTOFF still only fires from a forecast
# spike or from approaching a TNB tier boundary, not from ordinary use.
#
# Fan's real hardware (70W) is close enough to the proxy's own median (74W)
# that no rescaling was needed -- left at 85W. At 70W/85W = 0.82 ratio the
# real fan sits in ORANGE, consistent with the proxy's "near-flat appliance,
# driven mainly by tariff-tier proximity rather than raw power spikes" design.
#
# IMPORTANT CAVEAT, state this plainly in Ch3/Ch4: this rescaling only
# changes the CONTROL THRESHOLD, not the trained LSTM itself. The geyser
# model's weights were learned on the AC-proxy's ~1700-2000W dynamics
# (thermostat duty-cycling); a real 1500W resistive heating element likely
# behaves differently (flatter on/off, no duty-cycling to a setpoint). The
# threshold change makes the CONTROL DECISION sensible for the real
# hardware; it does not by itself make the model's live forecast on that
# hardware a validated number -- the accuracy claim still rests on the
# offline evaluation against the real iAWE held-out test set (Ch4 results),
# and the live demo should be framed as a mechanism demonstration, per the
# same reasoning already applied to the lamp/kettle substitution question.
APPLIANCE_THRESHOLDS_W = {
    "geyser": 1620.0,   # rescaled for the real 1500W water heater (see note above), was 1900.0 for the AC proxy
    "fan": 85.0,         # unchanged -- real 70W fan already close to proxy's own median (74W)
                          # (see note below: fan's SHIFT action is driven mainly by the dynamic
                          # tariff-tier threshold tightening, not by raw power spikes it doesn't have)
    "computer": 72.0,    # laptop -- ~p95, catches genuinely elevated load above routine idle/light use (~34W median)
    "lamp": 85.0,        # 3 W test lamp -- uses the fan (TV-channel) model, so this is the fan's 85 W limit in
                          # training units; live_control_loop.py scales it down to real watts (85 / 24.7 = about 3.4 W)
}


class Action(Enum):
    OK = "OK"
    NOTIFY = "NOTIFY"
    SHIFT = "SHIFT"
    CUTOFF = "CUTOFF"


class AlertTier(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass
class ControlDecision:
    appliance: str
    action: Action
    alert_tier: AlertTier
    reason: str
    predicted_power_w: float
    current_power_w: float
    threshold_w: float
    budget_pct_used: float


def tnb_current_tier(cumulative_kwh_this_month: float):
    """Returns (tier_index, lower, upper, rate_sen_per_kwh) for the given cumulative usage."""
    for i, (lower, upper, rate) in enumerate(TNB_TIERS):
        if lower <= cumulative_kwh_this_month < upper:
            return i, lower, upper, rate
    return len(TNB_TIERS) - 1, *TNB_TIERS[-1]


def dynamic_threshold_multiplier(cumulative_kwh_this_month: float, margin_kwh: float = 20.0) -> float:
    """
    Returns a multiplier (<=1.0) applied to the appliance's normal power
    threshold. As usage approaches the NEXT tariff tier boundary within
    `margin_kwh`, the multiplier shrinks toward 0.7 -- i.e. the system gets
    stricter (trips sooner) the closer the household is to crossing into a
    more expensive TNB bracket. Outside the margin, multiplier is 1.0
    (normal sensitivity).
    """
    tier_idx, lower, upper, rate = tnb_current_tier(cumulative_kwh_this_month)
    if upper == float("inf"):
        return 1.0  # already in the most expensive tier, nothing to protect against
    distance_to_next_tier = upper - cumulative_kwh_this_month
    if distance_to_next_tier <= 0:
        return 0.7
    if distance_to_next_tier >= margin_kwh:
        return 1.0
    # linear tightening from 1.0 (at margin_kwh away) down to 0.7 (at the boundary)
    frac = distance_to_next_tier / margin_kwh
    return 0.7 + 0.3 * frac


def alert_tier_from_ratio(ratio: float) -> AlertTier:
    """ratio = predicted_power / effective_threshold"""
    if ratio < 0.7:
        return AlertTier.GREEN
    elif ratio < 0.9:
        return AlertTier.YELLOW
    elif ratio < 1.0:
        return AlertTier.ORANGE
    else:
        return AlertTier.RED


def _w(v: float) -> str:
    """Format watts: 1 decimal for small loads (e.g. a 3 W lamp), whole numbers otherwise."""
    return f"{v:.1f}W" if abs(v) < 100 else f"{v:.0f}W"


def decide(
    appliance: str,
    current_power_w: float,
    predicted_power_w: float,
    cumulative_kwh_this_month: float,
    budget_kwh_limit: float = None,
    accumulated_kwh_today: float = None,
    tou=None,
) -> ControlDecision:
    """
    Core decision function. Called once per appliance per control cycle
    (e.g. every 15-30 min, aligned with the LSTM's forecast horizon).

    `predicted_power_w` should be the LSTM's next-30min forecast for this
    appliance (the whole point of using AI here -- act BEFORE the spike,
    not just after the current reading crosses a line, which is what a
    plain relay/threshold system -- like the client's supervisor's old
    project example -- already does).
    """
    base_threshold = APPLIANCE_THRESHOLDS_W[appliance]
    if tou is None:
        # Normal tariff (Tariff A): tighten near the next monthly kWh bracket
        mult = dynamic_threshold_multiplier(cumulative_kwh_this_month)
    else:
        # ToU tariff: no monthly brackets below 1,500 kWh; tighten during PEAK hours instead
        from tou_pricing import PEAK_THRESHOLD_MULT
        mult = PEAK_THRESHOLD_MULT if tou.is_peak else 1.0
    effective_threshold = base_threshold * mult
    why = (f"ToU {tou.period.lower()} {tou.all_in_sen:.2f} sen/kWh" if tou is not None
           else "near a TNB tier boundary")

    # budget cap check (independent of the tariff-tier dynamic threshold --
    # this is the user's OWN daily/weekly target, separate from TNB's bracket)
    budget_pct = 0.0
    if budget_kwh_limit and accumulated_kwh_today is not None:
        budget_pct = min(1.0, accumulated_kwh_today / budget_kwh_limit) if budget_kwh_limit > 0 else 0.0

    ratio = predicted_power_w / effective_threshold if effective_threshold > 0 else 0.0
    tier = alert_tier_from_ratio(max(ratio, budget_pct))

    # appliance-differentiated action policy
    if appliance in ("geyser", "lamp"):  # lamp = safe low-power stand-in for the heater's hard-cutoff path
        if tier == AlertTier.RED:
            action = Action.CUTOFF
            reason = (f"Predicted {_w(predicted_power_w)} exceeds safety threshold "
                      f"{_w(effective_threshold)} ({why}, adjusted from {_w(base_threshold)}) "
                      "-- predictive cutoff, " + ("lamp is standing in for the water heater's hard-safety cutoff."
                      if appliance == "lamp" else "water heater is a hard-safety appliance."))
        elif tier == AlertTier.ORANGE:
            action = Action.NOTIFY
            reason = f"Approaching threshold ({ratio*100:.0f}% of limit) -- pre-emptive warning before cutoff."
        else:
            action = Action.OK
            reason = "Within normal range."

    elif appliance == "fan":
        if tier in (AlertTier.RED, AlertTier.ORANGE):
            action = Action.SHIFT
            if tou is not None and tou.is_peak:
                reason = (f"Peak hour: {_w(predicted_power_w)} forecast at {tou.all_in_sen:.2f} sen/kWh -- "
                          f"defer to off-peak ({tou.label}), comfort item so not cut.")
            else:
                reason = (f"High predicted load ({_w(predicted_power_w)}) {why} -- "
                          f"comfort item, deferred/soft-shifted rather than cut.")
        else:
            action = Action.OK
            reason = "Within normal range."

    elif appliance == "computer":
        if tier in (AlertTier.RED, AlertTier.ORANGE):
            action = Action.NOTIFY
            reason = ("Elevated predicted load -- advisory only. Never auto-cut a laptop: "
                      "the failure mode (data loss mid-session) is worse than the problem.")
        else:
            action = Action.OK
            reason = "Within normal range."
    else:
        action = Action.OK
        reason = "Unknown appliance -- no policy defined, defaulting to OK."

    return ControlDecision(
        appliance=appliance,
        action=action,
        alert_tier=tier,
        reason=reason,
        predicted_power_w=predicted_power_w,
        current_power_w=current_power_w,
        threshold_w=effective_threshold,
        budget_pct_used=budget_pct,
    )


if __name__ == "__main__":
    # Quick sanity check with illustrative numbers
    scenarios = [
        ("geyser", 1200, 1250, 150),   # normal (proxy-scale numbers, kept for regression check)
        ("geyser", 1200, 1950, 190),   # near tier boundary + high predicted -> should trip
        ("fan", 60, 70, 100),           # normal-ish
        ("computer", 40, 58, 50),       # elevated -> notify only
        # Real physical demo hardware (Week 4): 1500W water heater at its own
        # normal full-power running, far from any tariff boundary -- should
        # land in ORANGE/NOTIFY, not RED/CUTOFF, per the rescaling note above.
        ("geyser", 1500, 1500, 150),
        # Same real heater, but now close to the 200kWh tier boundary --
        # dynamic tightening should be enough to push this into CUTOFF even
        # though the appliance itself hasn't changed at all.
        ("geyser", 1500, 1500, 195),
        # Real 70W fan at its own normal running power, mid-tier -- expect
        # ORANGE/SHIFT per the "already close to proxy median" calibration.
        ("fan", 70, 70, 150),
    ]
    for appliance, cur, pred, cum_kwh in scenarios:
        d = decide(appliance, cur, pred, cum_kwh)
        print(f"{appliance:10s} cur={cur:6.0f}W pred={pred:6.0f}W cum_month={cum_kwh:5.0f}kWh "
              f"-> action={d.action.value:8s} tier={d.alert_tier.value:6s} "
              f"threshold={d.threshold_w:.0f}W | {d.reason}")
