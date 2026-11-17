"""
TNB domestic Time of Use (ToU) pricing, used by the live loops and pushed to Blynk.

TNB domestic ToU (optional scheme, open since 1 July 2025), usage up to 1,500 kWh/month:
    PEAK      weekdays 2:00 pm - 10:00 pm           28.52 sen/kWh energy
    OFF-PEAK  weekdays 10:00 pm - 2:00 pm,           24.43 sen/kWh energy
              all of Saturday, Sunday and public holidays
  Above 1,500 kWh/month: 38.52 peak / 34.43 off-peak.
  Every kWh also pays capacity 4.55 sen + network 12.85 sen (same at any hour),
  plus RM10/month retail charge, AFA, and the Energy Efficiency Incentive.
  So the all-in per-kWh price used here is energy + capacity + network:
    peak 45.92 sen/kWh, off-peak 41.83 sen/kWh (about 10% more at peak).

IMPORTANT for the report: these prices only apply if the household has
enrolled in ToU with TNB. A household on the normal tariff pays the same
price at every hour, so moving usage to off-peak saves nothing there.
TARIFF_MODE (or the Blynk V19 switch) says which one to use.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

MYT = timezone(timedelta(hours=8))          # Malaysia time, whatever the laptop clock zone is

PEAK_START_H, PEAK_END_H = 14, 22           # 2 pm to 10 pm, weekdays only
ENERGY_SEN = {"PEAK": 28.52, "OFF-PEAK": 24.43}          # up to 1,500 kWh/month
ENERGY_SEN_ABOVE_1500 = {"PEAK": 38.52, "OFF-PEAK": 34.43}
CAPACITY_SEN, NETWORK_SEN = 4.55, 12.85
# Non-ToU (general) domestic tariff since 1 July 2025: 27.03 energy + 4.55 + 12.85, same at every hour
GENERAL_ALL_IN_SEN = 44.43

# Gazetted public holidays are off-peak all day. Fixed-date national ones are
# listed; add the moving ones (Raya, Deepavali, CNY, Wesak, state holidays)
# for the days you demo, format date(YYYY, M, D).
PUBLIC_HOLIDAYS = {
    date(2026, 5, 1),    # Labour Day
    date(2026, 8, 31),   # National Day
    date(2026, 9, 16),   # Malaysia Day
    date(2026, 12, 25),  # Christmas
}

# How much stricter the control thresholds get during peak (ToU mode only).
# 0.8 = limits drop to 80%: a fan that is "fine" off-peak gets a SHIFT at peak,
# and the water heater is cut at peak (classic water-heater load shifting).
PEAK_THRESHOLD_MULT = 0.8


@dataclass
class TouInfo:
    period: str            # "PEAK" or "OFF-PEAK"
    energy_sen: float      # energy part only
    all_in_sen: float      # energy + capacity + network
    next_change: datetime  # when the period flips next (Malaysia time)
    label: str             # short text for Blynk / LCD, e.g. "PEAK til 10pm"

    @property
    def is_peak(self) -> bool:
        return self.period == "PEAK"


def _is_peak(t: datetime) -> bool:
    return (t.weekday() < 5 and t.date() not in PUBLIC_HOLIDAYS
            and PEAK_START_H <= t.hour < PEAK_END_H)


def _fmt_hour(t: datetime, ref: datetime) -> str:
    h = t.hour % 12 or 12
    day = "" if t.date() == ref.date() else t.strftime(" %a")
    return f"{h}{'am' if t.hour < 12 else 'pm'}{day}"


def tou_info(now: datetime | None = None, monthly_kwh: float = 0.0) -> TouInfo:
    t = (now or datetime.now(MYT)).astimezone(MYT)
    period = "PEAK" if _is_peak(t) else "OFF-PEAK"
    table = ENERGY_SEN_ABOVE_1500 if monthly_kwh > 1500 else ENERGY_SEN
    energy = table[period]
    # next change: step hour by hour (at most a few days) until the period flips
    nxt = t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    for _ in range(24 * 5):
        if _is_peak(nxt) != (period == "PEAK"):
            break
        nxt += timedelta(hours=1)
    return TouInfo(period, energy, round(energy + CAPACITY_SEN + NETWORK_SEN, 2), nxt,
                   f"{period} til {_fmt_hour(nxt, t)}")


def cost_rm(power_w: float, hours: float, info: TouInfo) -> float:
    """Cost in RM of running power_w for `hours` at the current all-in rate."""
    return power_w / 1000.0 * hours * info.all_in_sen / 100.0


if __name__ == "__main__":
    print("now:", tou_info())
    for s in ["2026-10-07 13:59", "2026-10-07 14:00", "2026-10-07 21:59", "2026-10-07 22:00",
              "2026-10-09 15:00", "2026-10-10 15:00"]:
        i = tou_info(datetime.fromisoformat(s).replace(tzinfo=MYT))
        print(s, i.period, i.energy_sen, i.all_in_sen, i.label)
    i = tou_info(datetime(2026, 10, 7, 15, tzinfo=MYT))
    print("1500 W heater for 1 h at peak: RM", round(cost_rm(1500, 1, i), 3))
