# streamlit_app.py
# Airshow Trajectory Envelope (Wind = 0)
# - Axes naming: Display Line (x), Crowd (y), Height (z)
# - Plotly animation (XY, XZ, YZ) with Play/Pause/Stop/Replay
# - Session state to prevent reset when changing animation speed
# - Optional Auto-update on input changes; or explicit Run

import math
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import xlsxwriter

from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Airshow Trajectory (Wind=0)", layout="wide")

# === Session state defaults (persist results across UI interactions) ===
if "sim_df" not in st.session_state:
    st.session_state["sim_df"] = None
if "sim_summary" not in st.session_state:
    st.session_state["sim_summary"] = None
if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = None
if "auto_update" not in st.session_state:
    st.session_state["auto_update"] = True

# -----------------------------
# Physics & constants
# -----------------------------
def en_exp(vn, e0, einf, vc):
    """Velocity-dependent COR: e(vn) = einf + (e0 - einf)*exp(-vn/vc)."""
    vn = max(0.0, vn)
    return max(0.0, min(1.0, einf + (e0 - einf) * math.exp(-vn / max(1e-6, vc))))

# Surface parameter presets (impact friction, slide friction, restitution curve)
SURFSETS = {
    "concrete": dict(mu_imp=0.55, mu_slide=0.50, e0=0.20, einf=0.05, vc=15.0),
    "asphalt":  dict(mu_imp=0.45, mu_slide=0.40, e0=0.18, einf=0.05, vc=12.0),
    "grass":    dict(mu_imp=0.35, mu_slide=0.55, e0=0.12, einf=0.03, vc=8.0),  # short/dry turf
}

def simulate_3d(
    m, A, Cd, rho, g, dt,
    alt_ft, ktas, angle_deg, surface="grass",
    vz0=0.0, include_ground_drag=True,
    vz_bounce_min=0.5, max_steps=300000
):
    """
    3D point-mass with quadratic drag, wind = 0.
    Axes: x (Display Line), y (Crowd), z (Height above ground).
    Euler forward integration with event-based impact, bounce, and slide.
    """
    # Unit conversions & initial conditions
    alt_m = float(alt_ft) * 0.3048
    V     = float(ktas) * 0.514444444  # kt -> m/s
    theta = math.radians(angle_deg)
    vx0, vy0 = V * math.cos(theta), V * math.sin(theta)

    # Surface params
    s = SURFSETS[surface]
    mu_imp, mu_slide, e0, einf, vc = s["mu_imp"], s["mu_slide"], s["e0"], s["einf"], s["vc"]

    # Drag factor
    K = 0.5 * rho * Cd * A / m

    # State (z is height above ground)
    t = 0.0
    x = y = 0.0
    z = alt_m
    vx, vy, vz = vx0, vy0, vz0
    airborne = True

    def clamp_eps(u, eps=1e-12):
        return 0.0 if abs(u) < eps else u

    rows = []
    impact_recorded = False
    x_imp = y_imp = None
    impacts = 0

    for _ in range(max_steps):
        if airborne:
            # Relative velocity (wind = 0)
            vrelx, vrely, vrelz = vx, vy, vz
            vmag = math.sqrt(vrelx*vrelx + vrely*vrely + vrelz*vrelz)

            # Accelerations (vz is positive downward in this convention)
            ax = -K * vmag * vrelx
            ay = -K * vmag * vrely
            az =  g - K * vmag * vrelz

            vx_new = clamp_eps(vx + ax*dt)
            vy_new = clamp_eps(vy + ay*dt)
            vz_new = clamp_eps(vz + az*dt)

            # Update positions; z is height above ground -> decrease by downward vz
            x_new = x + vx_new * dt
            y_new = y + vy_new * dt
            z_new = max(0.0, z - vz_new * dt)  # height cannot go below ground

            # Impact event?
            if z > 0.0 and z_new <= 0.0:
                vn_pre = abs(vz_new)
                eN = en_exp(vn_pre, e0, einf, vc)  # restitution

                # Post-impact vertical speed (rebound upward => decrease downward vz)
                vz_post = -eN * vz_new

                # Impact friction impulse on tangential (horizontal) velocity
                vt_mag_pre = math.sqrt(vx_new*vx_new + vy_new*vy_new)
                dv_t = mu_imp * (1.0 + eN) * vn_pre
                if vt_mag_pre > 0.0:
                    scale = max(0.0, (vt_mag_pre - dv_t) / vt_mag_pre)
                    vx_post = vx_new * scale
                    vy_post = vy_new * scale
                else:
                    vx_post = vy_post = 0.0

                # Commit impact state
                x, y, z = x_new, y_new, 0.0
                vx, vy, vz = clamp_eps(vx_post), clamp_eps(vy_post), clamp_eps(vz_post)

                impacts += 1
                event_note = f"impact#{impacts}"

                if not impact_recorded:
                    x_imp, y_imp = x, y
                    impact_recorded = True

                rows.append(dict(t=t+dt, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, phase="air", event=event_note))

                # Transition to slide if bounce is negligible
                if abs(vz) < vz_bounce_min:
                    airborne = False

            else:
                # No impact this step
                x, y, z = x_new, y_new, z_new
                vx, vy, vz = vx_new, vy_new, vz_new
                rows.append(dict(t=t+dt, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, phase="air", event=None))

        else:
            # Ground slide (z = 0)
            vt_mag = math.sqrt(vx*vx + vy*vy)
            if vt_mag <= 1e-6:
                vx = vy = 0.0
                rows.append(dict(t=t+dt, x=x, y=y, z=0.0, vx=vx, vy=vy, vz=0.0, phase="slide", event=None))
                break

            # Kinetic friction
            ax_fric_x = -mu_slide * g * (vx / vt_mag)
            ax_fric_y = -mu_slide * g * (vy / vt_mag)

            # Optional aero drag during slide
            ax_drag = ay_drag = 0.0
            if include_ground_drag:
                vmag = vt_mag
                ax_drag = -K * vmag * vx
                ay_drag = -K * vmag * vy

            ax = ax_fric_x + ax_drag
            ay = ax_fric_y + ay_drag

            vx = clamp_eps(vx + ax * dt)
            vy = clamp_eps(vy + ay * dt)

            # Prevent numeric reversal
            if vx * (vx + ax*dt) < 0: vx = 0.0
            if vy * (vy + ay*dt) < 0: vy = 0.0

            x = x + vx * dt
            y = y + vy * dt
            z = 0.0

            rows.append(dict(t=t+dt, x=x, y=y, z=z, vx=vx, vy=vy, vz=0.0, phase="slide", event=None))

        t += dt
        if t > 3600.0:  # hard stop (1 hour)
            break

    df = pd.DataFrame(rows)

    # Distances in ground plane
    if impact_recorded:
        air_dist_xy = math.hypot(x_imp, y_imp)
        ground_dist_xy = math.hypot(x - x_imp, y - y_imp)
    else:
        air_dist_xy = math.hypot(x, y)
        ground_dist_xy = 0.0

    summary = dict(
        alt_ft=alt_ft,
        alt_m=alt_m,
        ktas=ktas,
        angle_deg=angle_deg,
        surface=surface,
        mass_kg=m,
        area_m2=A,
        Cd=Cd,
        air_dist_xy_m=air_dist_xy,
        ground_dist_xy_m=ground_dist_xy,
        total_dist_xy_m=air_dist_xy + ground_dist_xy,
        impacts=impacts
    )
    return summary, df

# -----------------------------
# Input fingerprint (to detect changes without recompute on UI-only tweaks)
# -----------------------------
def pack_inputs(mass_kg, area_m2, cd, alt_ft, ktas, angle, surface, rho, g, dt,
                include_ground_drag, vz_bounce_min):
    """Return a plain-Python dict capturing inputs; used to detect changes."""
    return dict(
        mass_kg=float(mass_kg),
        area_m2=float(area_m2),
        cd=float(cd),
        alt_ft=float(alt_ft),
        ktas=float(ktas),
        angle=int(angle),
        surface=str(surface),
        rho=float(rho),
        g=float(g),
        dt=float(dt),
        include_ground_drag=bool(include_ground_drag),
        vz_bounce_min=float(vz_bounce_min),
    )

# -----------------------------
# Plotly animation builder (XY, XZ, YZ)
# -----------------------------
def build_plotly_animation(df, title="Trajectory Animation", frame_ms=25, max_frames=800):
    """
    Build a 3‑pane Plotly animation (XY, XZ, YZ) from the simulation time series.
    Axes: Display Line (x), Crowd (y), Height (z).
    """
    phases = df["phase"].tolist()
    i_slide = phases.index("slide") if "slide" in phases else None

    # Subplots titled with required nomenclature
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Top View — Display Line (x) vs Crowd (y)",
            "Side View — Display Line (x) vs Height (z)",
            "Side View — Crowd (y) vs Height (z)"
        ],
        horizontal_spacing=0.06
    )

    # Background paths
    if i_slide is not None:
        fig.add_trace(go.Scatter(x=df["x"][:i_slide+1], y=df["y"][:i_slide+1],
                                 mode="lines", name="air"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["x"][i_slide:],   y=df["y"][i_slide:],
                                 mode="lines", name="ground"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df["x"], y=df["y"], mode="lines", name="trajectory"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["x"], y=df["z"], mode="lines", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["y"], y=df["z"], mode="lines", showlegend=False), row=1, col=3)

    # Axis labels (your convention)
    fig.update_xaxes(title_text="Display Line (x) [m]", row=1, col=1)
    fig.update_yaxes(title_text="Crowd (y) [m]",        row=1, col=1)
    fig.update_xaxes(title_text="Display Line (x) [m]", row=1, col=2)
    fig.update_yaxes(title_text="Height (z) [m]",       row=1, col=2)
    fig.update_xaxes(title_text="Crowd (y) [m]",        row=1, col=3)
    fig.update_yaxes(title_text="Height (z) [m]",       row=1, col=3)
    # Equal aspect for XY (plan view) so angles look right
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

    # Moving marker (red dot) on each subplot
    marker_style = dict(mode="markers", marker=dict(color="red", size=8))
    fig.add_trace(go.Scatter(x=[df["x"].iloc[0]], y=[df["y"].iloc[0]], showlegend=False, **marker_style), row=1, col=1)
    fig.add_trace(go.Scatter(x=[df["x"].iloc[0]], y=[df["z"].iloc[0]], showlegend=False, **marker_style), row=1, col=2)
    fig.add_trace(go.Scatter(x=[df["y"].iloc[0]], y=[df["z"].iloc[0]], showlegend=False, **marker_style), row=1, col=3)
    marker_traces = [len(fig.data)-3, len(fig.data)-2, len(fig.data)-1]

    # Downsample frames for web performance
    n = len(df)
    if n == 0:
        return fig
    step = max(1, n // max_frames)
    idxs = list(range(0, n, step))
    if idxs[-1] != n-1:
        idxs.append(n-1)

    frames = []
    for k, i in enumerate(idxs):
        r = df.iloc[i]
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[r["x"]], y=[r["y"]]),
                go.Scatter(x=[r["x"]], y=[r["z"]]),
                go.Scatter(x=[r["y"]], y=[r["z"]])
            ],
            traces=marker_traces,
            name=str(k)
        ))
    fig.frames = frames

    # Controls: Play/Pause/Stop/Replay + slider (animation speed = frame_ms)
    fig.update_layout(
        title=title,
        margin=dict(t=50, l=20, r=20, b=80),
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.5, "y": -0.12, "xanchor": "center",
            "showactive": False,
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": frame_ms, "redraw": True},
                                 "fromcurrent": True, "mode": "immediate"}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate"}]},
                {"label": "Stop", "method": "animate",
                 "args": [[frames[0].name], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate"}]},
                {"label": "Replay", "method": "animate",
                 "args": [None, {"frame": {"duration": frame_ms, "redraw": True},
                                 "mode": "immediate"}]},
            ]
        }],
        sliders=[{
            "currentvalue": {"prefix": "Frame: "},
            "pad": {"t": 50},
            "len": 0.9, "x": 0.05, "y": -0.05,
            "steps": [
                {"args": [[fr.name], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate"}],
                 "label": fr.name, "method": "animate"}
                for fr in frames
            ]
        }]
    )
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("Airshow Trajectory Envelope (Wind = 0)")
st.caption("Axes: **Display Line (x)**, **Crowd (y)**, **Height (z)**. Event-based impact, friction impulse, and ground slide. No wind advection.")

colL, colR = st.columns([1.1, 1.2])

with colL:
    st.subheader("Aircraft & Scenario")

    ac = st.selectbox("Aircraft", ["F-16 (preset)", "Hawk (preset)", "Custom"])

    # Presets
    if ac == "F-16 (preset)":
        mass_kg = st.number_input("Mass (kg)", value=9000.0, min_value=1000.0, step=500.0)
        area_m2 = st.number_input("Frontal area A (m²)", value=8.0, min_value=0.5, step=0.5)
        cd      = st.number_input("Drag coefficient Cd", value=1.1, min_value=0.2, step=0.1)
    elif ac == "Hawk (preset)":
        mass_kg = st.number_input("Mass (kg)", value=5000.0, min_value=1000.0, step=500.0)
        area_m2 = st.number_input("Frontal area A (m²)", value=5.0, min_value=0.5, step=0.5)
        cd      = st.number_input("Drag coefficient Cd", value=1.1, min_value=0.2, step=0.1)
    else:
        mass_kg = st.number_input("Mass (kg)", value=9000.0, min_value=100.0, step=100.0)
        area_m2 = st.number_input("Frontal area A (m²)", value=8.0, min_value=0.1, step=0.1)
        cd      = st.number_input("Drag coefficient Cd", value=1.1, min_value=0.2, step=0.1)

    alt_ft  = st.number_input("Altitude (ft AGL)", value=500.0, min_value=50.0, step=50.0)
    ktas    = st.number_input("KTAS (knots true airspeed)", value=350.0, min_value=50.0, step=10.0)
    angle   = st.select_slider("Angle to display line (deg)", options=[0,15,30,45,60,90], value=45)

    surface = st.selectbox("Surface", ["grass", "asphalt", "concrete"], index=0)

    st.markdown("**Environment / numerics**")
    rho = st.number_input("Air density ρ (kg/m³)", value=1.225, min_value=0.5, step=0.05)
    g   = st.number_input("Gravity g (m/s²)", value=9.81, min_value=9.7, max_value=9.9, step=0.01)
    dt  = st.number_input("Time step Δt (s)", value=0.01, min_value=0.002, step=0.002, format="%.3f")

    st.markdown("**Impact / slide physics**")
    vz_bounce_min = st.number_input("|vz| cutoff for bounce → slide (m/s)", value=0.5, min_value=0.1, max_value=2.0, step=0.1)
    include_ground_drag = st.checkbox("Include aerodynamic drag during slide", value=True)

with colR:
    # --- Simulation control ---
    st.markdown("### Simulation Control")
    st.checkbox("Auto‑update results when inputs change", key="auto_update",
                help="If on, any change on the left recomputes automatically.")
    run_btn = st.button("Run simulation", type="primary", use_container_width=True)

    # Fingerprint current inputs
    current_inputs = pack_inputs(
        mass_kg, area_m2, cd, alt_ft, ktas, angle, surface, rho, g, dt,
        include_ground_drag, vz_bounce_min
    )
    inputs_changed = (st.session_state["last_inputs"] != current_inputs)

    # Recompute if: pressed Run OR inputs changed with Auto‑update ON
    compute_now = run_btn or (st.session_state["auto_update"] and inputs_changed)
    if compute_now:
        summary, df = simulate_3d(
            m=mass_kg, A=area_m2, Cd=cd, rho=rho, g=g, dt=dt,
            alt_ft=alt_ft, ktas=ktas, angle_deg=angle, surface=surface,
            vz0=0.0, include_ground_drag=include_ground_drag,
            vz_bounce_min=vz_bounce_min
        )
        st.session_state["sim_summary"] = summary
        st.session_state["sim_df"] = df
        st.session_state["last_inputs"] = current_inputs
    else:
        if inputs_changed and not st.session_state["auto_update"]:
            st.info("Inputs changed. Click **Run simulation** to recompute.")

    # Render if we have results
    if st.session_state["sim_df"] is not None and st.session_state["sim_summary"] is not None:
        summary = st.session_state["sim_summary"]
        df = st.session_state["sim_df"]

        st.subheader("Results")
        mcols = st.columns(4)
        mcols[0].metric("Air distance to first impact (m)", f"{summary['air_dist_xy_m']:.1f}")
        mcols[1].metric("Ground distance to rest (m)", f"{summary['ground_dist_xy_m']:.1f}")
        mcols[2].metric("Total ground‑planar distance (m)", f"{summary['total_dist_xy_m']:.1f}")
        mcols[3].metric("Impacts (incl. first)", f"{summary['impacts']}")

        # Visualization
        st.markdown("---")
        st.subheader("Visualization")
        use_animation = st.checkbox("Interactive animation (Plotly)", value=True,
                                    help="Play/Pause/Stop/Replay in XY, XZ, YZ.")
        frame_ms = st.slider("Animation speed (ms per frame)", 10, 200, 25, step=5)

        if use_animation:
            fig_anim = build_plotly_animation(
                df,
                title="Trajectory — Display Line (x), Crowd (y), Height (z)",
                frame_ms=frame_ms
            )
            st.plotly_chart(fig_anim, use_container_width=True)
        else:
            # Static Matplotlib fallback (renamed axes)
            fig_xy = plt.figure(figsize=(5.5,4.5))
            if "slide" in df["phase"].values:
                i_slide = df["phase"].tolist().index("slide")
                plt.plot(df["x"].iloc[:i_slide+1], df["y"].iloc[:i_slide+1], label="air")
                plt.plot(df["x"].iloc[i_slide:],   df["y"].iloc[i_slide:],   label="ground")
            else:
                plt.plot(df["x"], df["y"], label="air")
            plt.xlabel("Display Line (x) [m]"); plt.ylabel("Crowd (y) [m]")
            plt.axis("equal"); plt.legend(); plt.tight_layout()
            st.pyplot(fig_xy, use_container_width=True)

            fig_xz = plt.figure(figsize=(5.5,3.5))
            plt.plot(df["x"], df["z"])
            plt.xlabel("Display Line (x) [m]"); plt.ylabel("Height (z) [m]")
            plt.tight_layout()
            st.pyplot(fig_xz, use_container_width=True)

            fig_yz = plt.figure(figsize=(5.5,3.5))
            plt.plot(df["y"], df["z"])
            plt.xlabel("Crowd (y) [m]"); plt.ylabel("Height (z) [m]")
            plt.tight_layout()
            st.pyplot(fig_yz, use_container_width=True)

        # Downloads
        st.subheader("Download outputs")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download time series (CSV)", data=csv_bytes,
                           file_name="timeseries.csv", mime="text/csv")

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
            df.to_excel(writer, sheet_name="TimeSeries", index=False)
        st.download_button("Download summary + series (XLSX)", data=out.getvalue(),
                           file_name="trajectory_summary.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Configure inputs on the left, then click **Run simulation** (or enable **Auto‑update**).")
