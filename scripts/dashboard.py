"""
ML dashboard for the Ilyasa PSM2 energy system. Runs on the laptop, in the
browser, next to the live Python loop.

    pip install streamlit
    cd scripts
    streamlit run dashboard.py

Tabs:
  Today           every appliance: power and price today, actual vs historical vs LSTM expected
  Live            decisions from live_control_loop(_1h).py and live_control_fast.py,
                  power vs forecast vs threshold, auto-refresh, optional Blynk fetch
  Dataset         the iAWE training data: any day, daily energy, hour-of-day profile
  Model           LSTM (1h and 2h) vs ARIMA metrics, replay the model on test data
  Decision logic  how the tariff threshold tightens, and a what-if tester

Nothing here writes to Blynk. "Fetch from Blynk" only reads (4 messages per click).
"""

import json
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import control_logic as C
import live_control_loop as loop

BASE = Path(__file__).resolve().parent.parent
MODES = {
    "1-hour models (models_1h)": dict(models=BASE / "models_1h", processed=BASE / "data" / "processed_1h",
                                       results=BASE / "results_1h", live=BASE / "data" / "live_1h", steps=4),
    "2-hour models (models)": dict(models=BASE / "models", processed=BASE / "data" / "processed",
                                    results=BASE / "results", live=BASE / "data" / "live", steps=8),
}
APPS = {
    "fan": "Fan (kipas diri), trained on iAWE TV channel",
    "geyser": "Water heater, trained on iAWE air-con channel",
    "computer": "Laptop, iAWE laptop channel (exact match)",
    "lamp": "3 W test lamp, uses the fan model (iAWE TV channel), scaled",
}
FAST_LOG = BASE / "data" / "live_fast" / "fast_test_log.csv"
ACTION_COLORS = alt.Scale(domain=["OK", "NOTIFY", "SHIFT", "CUTOFF"],
                          range=["#2e8b57", "#e6a100", "#e67e22", "#c0392b"])

st.set_page_config(page_title="Ilyasa PSM2 ML dashboard", layout="wide")

# ---------------- sidebar ----------------
st.sidebar.title("Ilyasa PSM2")
mode_name = st.sidebar.radio("Model set", list(MODES),
                             help="Which trained LSTM to use. 1-hour = looks at the last 4 readings (1 h), warms up in "
                                  "about 1 h. 2-hour = last 8 readings (2 h). Both predict the next 30 min. "
                                  "Use 1-hour unless told otherwise.")
M = MODES[mode_name]
app = st.sidebar.selectbox("Appliance", list(APPS), format_func=lambda a: APPS[a],
                           index=list(APPS).index(loop.CURRENT_APPLIANCE) if loop.CURRENT_APPLIANCE in APPS else 0,
                           help="Which appliance the Live, Dataset, Model and Decision logic tabs show. "
                                "(The Today tab has its own picker.)")
refresh = st.sidebar.select_slider("Live tab auto-refresh", options=[0, 10, 30, 60, 300], value=30,
                                   format_func=lambda s: "off" if s == 0 else f"every {s}s",
                                   help="How often the Today and Live tabs re-read the log files. Off = refresh the "
                                        "browser page yourself.")
mapping_on = st.sidebar.toggle("Wattage mapping (what-if tab)", value=True,
                               help="The LIVE loop's mapping is switched from Blynk V15 or --no-scaling; "
                                    "this toggle only affects the what-if tester here.")
loop.set_scaling(mapping_on)
k = loop.scale_factor(app)
src = loop.MODEL_FOR.get(app, app)   # which trained model/data this appliance uses (lamp -> fan)
st.sidebar.caption(f"Wattage scale factor for {app}: x{k:.2f}"
                   + (f" (real {loop.REAL_TYPICAL_W[app]:.0f} W -> training {loop.TRAIN_TYPICAL_W[app]:.0f} W)"
                      if app in loop.REAL_TYPICAL_W else " (no scaling)"))
st.sidebar.caption(f"Threshold for {app}: {C.APPLIANCE_THRESHOLDS_W[app]:.1f} W (before tariff tightening)")


# ---------------- helpers ----------------
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_scaler(models_dir: str, processed_dir: str, appliance: str):
    import pickle
    from tensorflow.keras.models import load_model
    model = load_model(Path(models_dir) / f"lstm_{appliance}.keras")
    with open(Path(processed_dir) / appliance / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def forecast(window_w: np.ndarray) -> np.ndarray:
    """Same maths as live_control_loop.predict_next (incl. wattage scaling), returns both steps."""
    model, scaler = load_model_and_scaler(str(M["models"]), str(M["processed"]), src)
    x = scaler.transform((np.asarray(window_w, float) * k).reshape(-1, 1)).reshape(1, M["steps"], 1)
    y = model.predict(x, verbose=0)
    return scaler.inverse_transform(y.reshape(-1, 1)).ravel() / k


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def load_metrics(results_dir: Path, appliance: str, kind: str):
    p = results_dir / f"{appliance}_{kind}_results.json"
    return json.loads(p.read_text())["metrics"] if p.exists() else None


with st.expander("What is this dashboard? (read first)"):
    st.markdown("""
**The idea:** a smart socket measures an appliance's power. An LSTM (a neural network for time series) looks at the
last hour of readings and forecasts the **next 30 minutes**. A rule set then compares that forecast with a
**threshold** and decides what to do. The threshold gets stricter as the month's electricity use nears the next
TNB price bracket (200 / 300 / 600 / 900 kWh).

**Decisions**
| Decision | Meaning |
|---|---|
| OK (green) | Nothing to do. |
| NOTIFY (yellow) | Warn the user only. Used for the laptop, which is never cut off. |
| SHIFT (orange) | Suggest or defer use to a cheaper time. Used for the fan. |
| CUTOFF (red) | Buzzer warning, then the relay switches the appliance off. Used for the water heater and the test lamp. |

**Tabs**
- **Today: power & price** - how the day looks for each appliance: real readings vs the typical day vs the LSTM's forecast, and what it costs.
- **Live decisions** - every decision the system made, with the numbers behind it. This is the main demo page.
- **Dataset** - the iAWE data the LSTM was trained on. Not live data.
- **Model** - how accurate the LSTM is, compared with the ARIMA baseline.
- **Decision logic** - how the threshold tightens, plus a what-if box to try your own numbers.

**Sidebar:** Model set = which trained model; Appliance = what the tabs show; Auto-refresh = how often live tabs reload;
Wattage mapping = rescales a small real appliance (e.g. a 3 W lamp) to the size the model was trained on.
Everything here only reads. Nothing is sent to the appliance from this page.
""")

tab_ctrl, tab_today, tab_live, tab_data, tab_model, tab_logic = st.tabs(
    ["Control panel", "Today: power & price", "Live decisions", "Dataset", "Model", "Decision logic"])

# ---------------- CONTROL PANEL: run everything from here, no terminal ----------------
SCRIPTS = Path(__file__).resolve().parent
DEMO_SCENARIOS = ["fan_normal", "fan_near_tariff", "lamp_normal", "lamp_near_tariff", "heater_off", "heater_normal",
                  "heater_near_tariff"]
DEMO_TOU = ["fan_offpeak", "fan_peak", "lamp_offpeak", "lamp_peak", "heater_offpeak", "heater_peak"]


def run_script(args: list[str], timeout: int = 300) -> str:
    import subprocess
    import sys
    r = subprocess.run([sys.executable, *args], cwd=SCRIPTS, capture_output=True, text=True, timeout=timeout)
    out = r.stdout + (("\n" + r.stderr[-1500:]) if r.returncode else "")
    return "\n".join(l for l in out.splitlines() if "tensorflow" not in l.lower() and "WARNING" not in l)


with tab_ctrl:
    st.subheader("Control panel")
    st.caption("Pick what you want to show, press a button. Nothing needs a terminal. "
               "Results appear in the other tabs (Live decisions, Today).")
    mode = st.radio("What do you want to run?",
                    ["Demo without hardware", "Live with hardware (ESP32 + Blynk)"], horizontal=True)

    if mode == "Demo without hardware":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. Fill the dashboard with a simulated day**")
            st.caption("Runs the real LSTM over made-up readings for the fan, heater and lamp. Fills the Live and "
                       "Today tabs.")
            if st.button("Fill dashboard with simulated data"):
                with st.spinner("Running the LSTM..."):
                    st.code(run_script(["simulate_live.py"]) or "done")
            if st.button("Clear simulated / live logs"):
                for p in [BASE / "data" / "live_1h" / "decisions.csv", BASE / "data" / "live_fast" / "readings.csv",
                          FAST_LOG]:
                    p.unlink(missing_ok=True)
                st.success("Cleared. The Live tab is empty again.")
        with c2:
            st.markdown("**2. Instant what-happens-if scenario**")
            st.caption("Real LSTM + real decision rules on a preset situation. Prints the forecast and decision.")
            kind = st.radio("Scenario type", ["Tariff brackets (Tariff A)", "Time of day (ToU peak/off-peak)"],
                            horizontal=True)
            sc = st.selectbox("Scenario", DEMO_SCENARIOS if kind.startswith("Tariff") else DEMO_TOU)
            if st.button("Run scenario"):
                flag = "--scenario" if kind.startswith("Tariff") else "--tou-scenario"
                with st.spinner("Running..."):
                    st.code(run_script(["demo_decision_now.py", flag, sc]))
        st.info("These use scenarios, not sensor data. Say so when presenting.")

        st.markdown("**3. Send a demo to Blynk** (so the Blynk app shows it too)")
        st.caption("Runs the real LSTM + rules on 6 simulated hours and pushes power, LSTM/historical chart lines "
                   "(V21/V22), decision, reason and cost to Blynk, one 15-min step every 5 s (about 2 min). "
                   "Needs BLYNK_TOKEN in live_control_loop.py. Uses about 190 Blynk messages.")
        b1, b2, b3 = st.columns(3)
        bl_app = b1.selectbox("Appliance", ["geyser", "fan", "lamp"], key="bl_app")
        bl_tar = b2.selectbox("Tariff", ["a", "tou"], key="bl_tar",
                              format_func=lambda t: "Normal (A): NOTIFY -> CUTOFF as kWh rises" if t == "a"
                              else "TNB ToU (peak/off-peak now)")
        esp_off = b3.checkbox("The ESP32 is switched OFF",
                              help="If it is on, it overwrites V0 with the real reading and ACTS on the demo "
                                   "decisions (buzzer, relay). Unplug it first.")
        bproc = st.session_state.get("blynk_demo")
        b_running = bproc is not None and bproc.poll() is None
        if st.button("Send demo to Blynk", disabled=b_running or not esp_off):
            import subprocess
            import sys
            st.session_state["blynk_demo"] = subprocess.Popen(
                [sys.executable, "demo_to_blynk.py", "--yes", "--appliance", bl_app, "--tariff", bl_tar], cwd=SCRIPTS)
            st.rerun()
        st.write("Blynk demo: " + ("**sending** - open the Blynk dashboard now" if b_running else "not running"))

    else:
        import subprocess
        import sys
        st.caption("Needs the ESP32 powered and sending to Blynk, and BLYNK_TOKEN set in live_control_loop.py.")
        c1, c2, c3, c4 = st.columns(4)
        l_app = c1.selectbox("Appliance plugged in", ["lamp", "fan", "geyser"])
        l_step = c2.number_input("Seconds between readings", 5, 900, 30, help="30 = fast test, 900 = real 15-min loop")
        l_kwh = c3.number_input("Month's kWh so far", 0.0, 950.0, 150.0)
        l_tar = c4.selectbox("Tariff", ["tou", "a"], format_func=lambda t: "TNB ToU" if t == "tou" else "Normal (A)")
        proc = st.session_state.get("live_proc")
        running = proc is not None and proc.poll() is None
        b1, b2 = st.columns(2)
        if b1.button("Start live loop", disabled=running):
            st.session_state["live_proc"] = subprocess.Popen(
                [sys.executable, "live_control_fast.py", "--appliance", l_app, "--step", str(int(l_step)),
                 "--kwh", str(l_kwh), "--tariff", l_tar], cwd=SCRIPTS)
            st.rerun()
        if b2.button("Stop live loop", disabled=not running):
            proc.terminate()
            st.rerun()
        st.write("Status: " + ("**running** - first decision in about 2 min, then see the Live decisions tab"
                               if running else "stopped"))

# ---------------- TODAY: power + price, actual vs historical vs LSTM ----------------
with tab_today:
    import day_view as DV
    import tou_pricing as TP

    st.subheader("Today: power and price, actual vs historical vs LSTM expected")
    tc1, tc2 = st.columns([3, 2])
    today_apps = tc1.multiselect("Appliances", list(APPS), default=list(APPS),
                                 format_func=lambda a: a, key="today_apps")
    tariff_choice = tc2.radio("Tariff for the price graphs", ["TNB ToU", "Normal tariff"], horizontal=True,
                              index=0 if loop.TARIFF_MODE == "TOU" else 1, key="today_tariff")
    tariff_code = "TOU" if tariff_choice == "TNB ToU" else "A"
    st.caption("**actual** = today's live readings (15-min average; faint dots are the raw readings). "
               "**historical** = the average day in the iAWE training data for this appliance's model, "
               "in this appliance's real watts. **LSTM expected** = the model's forecast for every 15 min "
               "(before now: forecast from the previous window; after now: rolled forward from the latest "
               "readings). Price = power x TNB all-in rate at that time. Grey band = ToU peak (2pm-10pm weekdays).")

    SERIES = alt.Scale(domain=["actual", "historical", "LSTM expected"], range=["#2a78d6", "#9a9a9a", "#eb6834"])
    DASH = alt.Scale(domain=["actual", "historical", "LSTM expected"], range=[[1, 0], [4, 4], [6, 2]])

    def today_view():
        now = DV.today_myt()
        info = TP.tou_info()
        peak_band = pd.DataFrame()
        if tariff_code == "TOU" and now.weekday() < 5 and now.date() not in TP.PUBLIC_HOLIDAYS:
            d0 = pd.Timestamp(now.date())
            peak_band = pd.DataFrame({"start": [d0 + pd.Timedelta(hours=14)], "end": [d0 + pd.Timedelta(hours=22)]})
        rate_now = info.all_in_sen if tariff_code == "TOU" else TP.GENERAL_ALL_IN_SEN
        st.markdown(f"**Now {now:%a %d %b %H:%M}**: "
                    + (f"ToU **{info.period}**, {rate_now:.2f} sen/kWh ({info.label})" if tariff_code == "TOU"
                       else f"normal tariff, {rate_now:.2f} sen/kWh at every hour"))

        for a in today_apps:
            a_src = loop.MODEL_FOR.get(a, a)
            loop.set_scaling(mapping_on)
            try:
                mdl, scl = load_model_and_scaler(str(M["models"]), str(M["processed"]), a_src)
            except Exception as e:  # noqa: BLE001
                st.warning(f"{a}: model not found ({e})")
                continue
            f, raw = DV.build_day(mdl, scl, M["steps"], M["processed"], BASE, a, tariff_code, now)
            if f is None:
                st.warning(f"{a}: no training data found in {M['processed']}")
                continue
            tot = DV.day_totals(f)
            st.markdown(f"#### {a}: {APPS[a]}")
            last_w = raw["power_w"].iloc[-1] if raw is not None and not raw.empty else None
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Power now", f"{last_w:.1f} W" if last_w is not None else "no reading today")
            m2.metric("Cost now", f"RM{(last_w or 0) / 1000 * rate_now / 100:.4f}/h")
            m3.metric("Cost so far today", f"RM{tot['so_far_rm']:.3f}")
            m4.metric("LSTM expected today", f"RM{tot['expected_today_rm']:.3f}",
                      help="actual so far + LSTM forecast for the rest of the day")
            m5.metric("Historical average day", f"RM{tot['historical']['rm']:.3f}",
                      help=f"{tot['historical']['kwh']:.2f} kWh")

            def long(cols, unit):
                d = f.melt(id_vars=["time"], value_vars=cols, var_name="series", value_name=unit)
                d["series"] = d["series"].map(dict(zip(cols, ["actual", "historical", "LSTM expected"])))
                return d.dropna()

            now_rule = alt.Chart(pd.DataFrame({"t": [pd.Timestamp(now)]})).mark_rule(color="#c0392b").encode(x="t:T")
            band = (alt.Chart(peak_band).mark_rect(opacity=0.12, color="#555").encode(x="start:T", x2="end:T")
                    if not peak_band.empty else None)

            pw = long(["actual_w", "historical_w", "lstm_w"], "W")
            lines = alt.Chart(pw).mark_line(point=alt.OverlayMarkDef(size=12)).encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
                y=alt.Y("W:Q", title="power (W)"),
                color=alt.Color("series:N", scale=SERIES, title=None),
                strokeDash=alt.StrokeDash("series:N", scale=DASH, legend=None),
                tooltip=[alt.Tooltip("time:T", format="%H:%M"), "series:N", alt.Tooltip("W:Q", format=".1f")])
            layers = [x for x in (band, lines, now_rule) if x is not None]
            if raw is not None and not raw.empty:
                layers.insert(1, alt.Chart(raw).mark_circle(size=10, opacity=0.25, color="#2a78d6").encode(
                    x="time:T", y="power_w:Q"))
            cA, cB = st.columns(2)
            cA.altair_chart(alt.layer(*layers).properties(height=260, title="Power"), use_container_width=True)

            cs = long(["cost_actual_rm_h", "cost_historical_rm_h", "cost_lstm_rm_h"], "RM_per_h")
            cost_lines = alt.Chart(cs).mark_line(point=alt.OverlayMarkDef(size=12)).encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
                y=alt.Y("RM_per_h:Q", title="cost (RM/hour)"),
                color=alt.Color("series:N", scale=SERIES, title=None),
                strokeDash=alt.StrokeDash("series:N", scale=DASH, legend=None),
                tooltip=[alt.Tooltip("time:T", format="%H:%M"), "series:N",
                         alt.Tooltip("RM_per_h:Q", format=".4f")])
            rate_line = alt.Chart(f).mark_line(interpolate="step-after", color="#7b4fb5", strokeWidth=1.5,
                                               opacity=0.7).encode(
                x="time:T", y=alt.Y("price_sen:Q", title="price (sen/kWh, purple)",
                                    scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("time:T", format="%H:%M"), alt.Tooltip("price_sen:Q", format=".2f")])
            left = alt.layer(*[x for x in (band, cost_lines, now_rule) if x is not None])
            cB.altair_chart(alt.layer(left, rate_line).resolve_scale(y="independent")
                            .properties(height=260, title="Price"), use_container_width=True)
            if raw is None or raw.empty:
                st.caption(f"No live readings for {a} today yet: run `live_control_fast.py --appliance {a}` "
                           f"or the live loop. LSTM expected is rolled forward from the historical average.")
        if not today_apps:
            st.info("Pick at least one appliance.")
        with st.expander("How to read these charts"):
            st.markdown("""
- **Blue solid line = actual**: what the socket really measured today (15-min average). Faint blue dots = every raw reading.
- **Grey dashed line = historical**: the average day for this appliance in the training data. A "normal day" reference.
- **Orange dashed line = LSTM expected**: the model's forecast. Before the red line it is the forecast made from the previous hour (so you can compare it with actual); after the red line it is the forecast for the rest of the day.
- **Red vertical line = now.**
- **Grey shaded band = ToU peak** (2pm-10pm weekdays), when electricity is dearer under the TNB Time-of-Use tariff.
- **Purple stepped line (Price chart, right axis) = price in sen/kWh.** Cost (RM/hour) = power x price.
- **Metrics:** *Cost so far today* uses actual readings. *LSTM expected today* = actual so far + forecast for the rest of the day. *Historical average day* = the typical full-day cost.
- If actual is above the orange line, the appliance is being used more than the model expected.
""")

    if refresh:
        st.fragment(run_every=refresh)(today_view)()
    else:
        today_view()


# ---------------- LIVE ----------------
with tab_live:
    def live_view():
        st.subheader(f"Live decisions: {app}")
        dec = read_csv(M["live"] / "decisions.csv")
        fast = read_csv(FAST_LOG)
        if not dec.empty:
            dec = dec[dec["appliance"] == app].assign(source="live loop")
            dec["time"] = pd.to_datetime(dec["time"], utc=True).dt.tz_convert("Asia/Kuala_Lumpur").dt.tz_localize(None)
        if not fast.empty:
            fast = fast[fast["appliance"] == app].copy()
            fast["current_w"] = fast["readings_w"].str.split().str[-1].astype(float)
            t = fast["time"].astype(str)   # older logs have time only, newer ones date + time
            fast["time"] = pd.to_datetime(t.where(t.str.len() > 8, datetime.now().strftime("%Y-%m-%d ") + t),
                                          errors="coerce")
            fast["source"] = "fast test"
        both = pd.concat([d for d in (dec, fast) if not d.empty], ignore_index=True) if (
            not dec.empty or not fast.empty) else pd.DataFrame()

        if both.empty:
            script = "live_control_loop_1h.py" if M["steps"] == 4 else "live_control_loop.py"
            st.info(f"No decisions yet for **{app}**. Start `{script}` (logs to "
                    f"`{M['live'].relative_to(BASE)}/decisions.csv`) or `live_control_fast.py`. "
                    f"The first live decision comes about {M['steps'] * 15} min after the loop starts.")
        else:
            src = st.radio("Source", sorted(both["source"].unique()), horizontal=True)
            d = both[both["source"] == src].sort_values("time")
            last = d.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latest reading", f"{last['current_w']:.1f} W")
            c2.metric("LSTM forecast (next 30 min)", f"{last['forecast_w']:.1f} W")
            c3.metric("Threshold now", f"{last['threshold_w']:.0f} W")
            c4.metric("Decision", last["action"])
            st.caption(f"Last decision at {last['time']}. {last.get('reason', '')}")

            long = d.melt(id_vars=["time"], value_vars=["current_w", "forecast_w", "threshold_w"],
                          var_name="series", value_name="watts")
            long["series"] = long["series"].map({"current_w": "actual", "forecast_w": "LSTM forecast",
                                                 "threshold_w": "threshold"})
            lines = alt.Chart(long).mark_line(point=True).encode(
                x="time:T", y=alt.Y("watts:Q", title="W"),
                color=alt.Color("series:N", scale=alt.Scale(range=["#2a78d6", "#eb6834", "#888780"])),
                strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(range=[[1, 0], [5, 3], [2, 2]])))
            marks = alt.Chart(d).mark_tick(thickness=4, size=14).encode(
                x="time:T", color=alt.Color("action:N", scale=ACTION_COLORS, title="decision"),
                tooltip=["time:T", "action:N", "current_w:Q", "forecast_w:Q", "reason:N"])
            st.altair_chart(alt.vconcat(lines.properties(height=280), marks.properties(height=40))
                            .resolve_scale(color="independent"), use_container_width=True)

            with st.expander("How to read this page (legend)"):
                st.markdown("""
**Top boxes** (latest decision): *Latest reading* = power measured now. *LSTM forecast* = the highest power the model expects in the next 30 min. *Threshold now* = the limit for this appliance, already tightened for the month's usage. *Decision* = what the rules chose.

**Chart lines**
- **Blue solid = actual** power measured.
- **Orange dashed = LSTM forecast** for the next 30 min, made at that time.
- **Grey dotted = threshold.** It moves down as the month's kWh nears a TNB bracket. When the forecast goes above it, the rules escalate.

**Coloured ticks under the chart = the decision at that moment:** green OK, yellow NOTIFY, orange SHIFT, red CUTOFF.

**Tables:** *Decision counts* = how many of each decision. *Recent decisions* = newest first, with `cumulative_kwh` (month's use so far) and `reason` (why that decision).

**Source:** *live loop* = the 15-min loop; *fast test* = the 30-second test loop (its forecast is not a true 30-min forecast, it only proves the chain works).
""")

            if "temp_c" in d.columns and d["temp_c"].notna().any():
                st.write("Room temperature and humidity (DHT22, logged with each decision)")
                st.line_chart(d.set_index("time")[["temp_c", "humidity_pct"]], height=180)

            cc1, cc2 = st.columns([1, 2])
            cc1.write("Decision counts")
            cc1.dataframe(d["action"].value_counts().rename("count"))
            cc2.write("Recent decisions")
            cols = [c for c in ["time", "current_w", "forecast_w", "threshold_w", "cumulative_kwh",
                                "scaling", "temp_c", "action", "reason"] if c in d.columns]
            cc2.dataframe(d[cols].iloc[::-1].head(50), hide_index=True, use_container_width=True)

        buf = read_csv(M["live"] / f"{app}_buffer.csv")
        if not buf.empty:
            buf["timestamp"] = pd.to_datetime(buf["timestamp"], utc=True).dt.tz_convert(
                "Asia/Kuala_Lumpur").dt.tz_localize(None)
            n = len(buf)
            st.write(f"Raw readings collected by the live loop: {n} "
                     f"(needs {M['steps']} x 15-min buckets before the first decision)")
            st.line_chart(buf.set_index("timestamp")["power_w"], height=180)

    if refresh:
        st.fragment(run_every=refresh)(live_view)()
    else:
        live_view()

    with st.expander("Fetch current values from Blynk (reads only, 12 messages)"):
        if st.button("Fetch now"):
            if loop.BLYNK_TOKEN.startswith("YOUR_"):
                st.error("Set BLYNK_TOKEN in live_control_loop.py first.")
            else:
                vals = {}
                for pin, name in [("V0", "live power W"), ("V5", "15-min avg W"), ("V6", "temp C"),
                                  ("V7", "humidity %"), ("V10", "control"), ("V12", "relay on"),
                                  ("V13", "manual relay"), ("V14", "buzzer mode"), ("V15", "mapping"),
                                  ("V17", "ToU rate sen/kWh"), ("V18", "cost RM/hour"), ("V19", "tariff 1=ToU")]:
                    try:
                        vals[name] = loop.blynk_get(pin)
                    except Exception as e:  # noqa: BLE001
                        vals[name] = f"error: {e}"
                st.json(vals)

# ---------------- DATASET ----------------
with tab_data:
    st.subheader(f"Training data: {APPS[app]}")
    st.caption("iAWE dataset, Delhi 2013, resampled to 15-minute buckets. This is what the LSTM learned from, "
               "not live data. Gaps are periods the dataset's meter wasn't recording.")
    if src != app:
        st.info(f"There is no {app} data in iAWE. Showing the **{src}** training data this appliance's model uses.")
    ds = read_csv(M["processed"] / src / "resampled_15min.csv")
    if ds.empty:
        st.warning("No resampled data found. Run preprocess / train_lstm_1h.py first.")
    else:
        ds["timestamp"] = pd.to_datetime(ds["timestamp"])
        ds = ds.set_index("timestamp")
        ds["day"] = ds.index.date
        daily = ds.groupby("day")["power_w"].agg(
            kwh=lambda s: s.sum() * 0.25 / 1000, coverage=lambda s: s.notna().mean() * 100)
        daily = daily[daily["coverage"] > 0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Days with data", len(daily))
        c2.metric("Median power when on", f"{ds['power_w'][ds['power_w'] > 5].median():.0f} W")
        c3.metric("Date range", f"{daily.index.min()} to {daily.index.max()}")

        good_days = daily[daily["coverage"] >= 50].index
        day = st.selectbox("Pick a day (days with at least 50% coverage)", list(good_days)[::-1] or list(daily.index))
        one = ds[ds["day"] == day]["power_w"]
        st.line_chart(one, height=260)
        st.caption(f"{day}: {daily.loc[day, 'kwh']:.2f} kWh recorded, {daily.loc[day, 'coverage']:.0f}% of the day covered")

        with st.expander("How to read this page"):
            st.markdown("- **Top line chart** = the power of the chosen day, one point per 15 min. Breaks = the dataset's meter wasn't recording.\n"
                        "- **Energy per day** = kWh recorded each day (only the recorded part, so low bars may just be missing data).\n"
                        "- **Average power by hour** = the typical daily pattern the LSTM can learn from.")
        cA, cB = st.columns(2)
        cA.write("Energy per day (kWh, only the recorded part of each day)")
        cA.bar_chart(daily["kwh"], height=220)
        cB.write("Average power by hour of day (W)")
        prof = ds.groupby(ds.index.hour)["power_w"].mean()
        prof.index.name = "hour"
        cB.bar_chart(prof, height=220)

# ---------------- MODEL ----------------
with tab_model:
    st.subheader("Model performance")
    with st.expander("How to read this page"):
        st.markdown("""
- **RMSE (W)** = typical forecast error in watts. Lower is better. **MAE (W)** = average absolute error, also lower is better.
- **R2** = how much of the ups and downs the model explains. 1 = perfect, 0 = no better than guessing the average, negative = worse than that.
- **LSTM 1h / 2h** = the neural network with a 1-hour or 2-hour input window. **ARIMA** = the classic statistical baseline to beat.
- **Replay chart:** orange = what the LSTM predicted, blue = what really happened, for held-out windows it never trained on. The closer the lines, the better.
- Loss curve image: training error vs validation error per epoch. If the validation line stays near the training line, the model is not overfitting.
- Honest note: ARIMA wins on the laptop and the fan's RMSE; the LSTM wins on the water-heater proxy. Models are scored on different test samples, so compare with care.
""")
    rows = []
    for a in APPS:
        for label, rdir, kind in [("LSTM 1h", BASE / "results_1h", "lstm"), ("LSTM 2h", BASE / "results", "lstm"),
                                  ("ARIMA", BASE / "results", "arima")]:
            m = load_metrics(rdir, a, kind)
            if m:
                rows.append({"appliance": a, "model": label,
                             "RMSE (W)": m.get("rmse_watts", m.get("rmse")),
                             "MAE (W)": m.get("mae_watts", m.get("mae")),
                             "R2": m.get("r2")})
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
        st.caption("Each model is scored on its own held-out test set, so treat small differences as indicative. "
                   "ARIMA is the statistical baseline.")

    st.write(f"**Replay on test data** ({mode_name}, {app}): the model predicts each held-out window, "
             "compared with what actually happened.")
    xp, yp = M["processed"] / src / "X_test.npy", M["processed"] / src / "y_test.npy"
    if xp.exists():
        X, y = np.load(xp), np.load(yp)
        model, scaler = load_model_and_scaler(str(M["models"]), str(M["processed"]), src)
        yh = model.predict(X, verbose=0)
        inv = lambda a_: scaler.inverse_transform(a_.reshape(-1, 1)).reshape(a_.shape)  # noqa: E731
        y_w, yh_w = inv(y), inv(yh)
        rmse = float(np.sqrt(np.mean((y_w - yh_w) ** 2)))
        st.metric("RMSE on this test set (recomputed now)", f"{rmse:.1f} W")
        n = st.slider("Windows to show", 20, len(y_w), min(150, len(y_w)))
        rep = pd.DataFrame({"actual (next 15 min)": y_w[:n, 0], "LSTM predicted": yh_w[:n, 0]})
        st.line_chart(rep, height=280)
        st.caption("Test windows come from different recording sessions, so the x-axis is window number, not time.")
    for img in [f"{src}_prediction_vs_actual.png", f"{src}_loss_curve.png"]:
        p = M["results"] / img
        if p.exists():
            st.image(str(p), caption=img)

# ---------------- DECISION LOGIC ----------------
with tab_logic:
    st.subheader("How the decision is made")
    st.write("decision = LSTM forecast compared with a threshold that **tightens as the month's usage gets "
             "close to the next TNB bracket** (200 / 300 / 600 / 900 kWh). "
             f"Base threshold for {app}: **{C.APPLIANCE_THRESHOLDS_W[app]:.1f} W**.")
    kwh_axis = np.arange(0, 1000, 1.0)
    thr = pd.DataFrame({"cumulative kWh this month": kwh_axis,
                        "effective threshold (W)": [C.APPLIANCE_THRESHOLDS_W[app] *
                                                    C.dynamic_threshold_multiplier(v) for v in kwh_axis]})
    st.line_chart(thr, x="cumulative kWh this month", y="effective threshold (W)", height=240)
    st.caption("**Line** = the power limit for this appliance. It drops as the month's usage approaches 200, 300, 600 and "
               "900 kWh (the TNB price brackets), because crossing one makes every further kWh dearer, so the system "
               "gets stricter just before it. The dips are the brackets.")

    st.write("**What-if tester**: type the last hour of readings and the month's usage, see what it decides.")
    defaults = {"fan": loop.REAL_TYPICAL_W.get("fan", 70.0), "geyser": 1500.0, "computer": 35.0,
                "lamp": loop.REAL_TYPICAL_W.get("lamp", 3.0)}
    cols = st.columns(M["steps"])
    readings = [cols[i].number_input(f"reading {i + 1} (W)", 0.0, 5000.0, float(defaults[app]), 0.1,
                                     key=f"r{i}_{app}_{M['steps']}") for i in range(M["steps"])]
    kwh = st.slider("Cumulative kWh this month", 0.0, 950.0, 150.0, 1.0)
    fc = forecast(np.array(readings))
    d = C.decide(app, readings[-1], float(np.max(fc)), kwh)
    c1, c2, c3 = st.columns(3)
    c1.metric("LSTM forecast (next 30 min)", f"{np.max(fc):.1f} W", help=f"both steps: {fc[0]:.1f}, {fc[1]:.1f} W")
    c2.metric("Effective threshold", f"{d.threshold_w:.1f} W")
    c3.metric("Decision", f"{d.action.value} ({d.alert_tier.value})")
    st.caption(d.reason)
    st.caption("How to use: the boxes are the last readings (oldest first, 15 min apart). The LSTM forecasts from them; "
               "the decision compares that forecast with the threshold. Try setting the kWh slider near 195 with a "
               "heater at 1500 W to see it flip to CUTOFF.")
