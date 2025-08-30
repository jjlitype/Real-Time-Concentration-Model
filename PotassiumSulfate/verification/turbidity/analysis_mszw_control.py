# Updated analysis_mszw_control.py to fix the TypeError in compute_sigma

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import seaborn as sns
from multiprocessing import Pool, cpu_count

# Set plotting styles and fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
c = sns.color_palette("deep")

def solubility_k2so4(T: float | np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    return 174.26 * (0.418 + 1.138e-2 * T + 1.688e-5 * T**2)

def compute_sigma(c: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
    """
    Calculates the supersaturation sigma.
    Corrected to handle both single floats and numpy arrays.
    """
    T = np.asarray(T)
    c = np.asarray(c)
    sol = solubility_k2so4(T)
    
    # Calculate the ratio
    val = c / sol
    
    # Corrected logic to handle element-wise assignment for arrays,
    # and direct assignment for scalars to avoid TypeError.
    val_corrected = np.where(val <= 1.0, 1.0, val)
    
    return np.log(val_corrected)


def detect_nucleation(time, turbidity, threshold=300, window=5):
    """Simple heuristic: first index where moving average exceeds threshold."""
    turb = np.asarray(turbidity)
    if len(turb) < window:
        return None
    ma = pd.Series(turb).rolling(window, min_periods=1).mean().to_numpy()
    idx = np.argmax(ma > threshold)
    if ma[idx] <= threshold:
        return None
    return idx

@dataclass
class BatchResult:
    Rc: float
    c0: float
    Ts: float
    Tn: float
    MSZW: float
    t_ind: float
    sigma_max: float

# -------------------- MSZW ANALYSIS -------------------- #

def analyze_batches(df: pd.DataFrame, cooling_rates: list[float]) -> list[BatchResult]:
    """Analyzes a series of batch crystallization data to compute MSZW metrics."""
    results = []
    for Rc in cooling_rates:
        dfr = df[df['Rc'] == Rc].copy()
        if dfr.empty:
            continue
        c0 = float(dfr['concentration'].iloc[0])
        dfr['m_sat'] = solubility_k2so4(dfr['temperature'])
        supersat_mask = dfr['concentration'] > dfr['m_sat']
        if supersat_mask.any():
            Ts_idx = supersat_mask.idxmax()
            Ts = float(dfr.loc[Ts_idx, 'temperature'])
            t_s = float(dfr.loc[Ts_idx, 'time'])
        else:
            Ts = float('nan')
            t_s = float('nan')
        idx_n = detect_nucleation(dfr['time'].values, dfr['turbidity'].values)
        if idx_n is not None:
            Tn = float(dfr['temperature'].values[idx_n])
            t_n = float(dfr['time'].values[idx_n])
        else:
            Tn = float('nan')
            t_n = float('nan')
        MSZW = Ts - Tn if (not math.isnan(Ts) and not math.isnan(Tn)) else float('nan')
        t_ind = t_n - t_s if (not math.isnan(t_n) and not math.isnan(t_s)) else float('nan')
        if not math.isnan(t_n):
            pre = dfr[dfr['time'] <= t_n]
            sigmas = compute_sigma(pre['concentration'], pre['temperature'])
            sigma_max = float(np.nanmax(sigmas))
        else:
            sigma_max = float('nan')
        results.append(BatchResult(Rc, c0, Ts, Tn, MSZW, t_ind, sigma_max))
    return results

# -------------------- CONTROL SIMULATION -------------------- #

def run_simulation(simulation_type, params):
    """A generic function to run a simulation with given parameters."""
    print(f"Starting {simulation_type} simulation...")
    if simulation_type == 'open_loop':
        result = simulate_open_loop(**params)
    elif simulation_type == 'closed_loop':
        result = simulate_closed_loop(**params)
    else:
        raise ValueError("Invalid simulation_type")
    print(f"Finished {simulation_type} simulation.")
    return result

def simulate_open_loop(
    T0=50.0,
    Tend=10.0,
    c0=120.0,
    Rc_const=0.5,
    dt=1.0,
    k_g=0.005,
    sigma_target=0.15,
    sigma_crit=0.25,
    noise_ec=0.8,
    noise_uv=0.6,
    seed=0,
    max_time=200,
):
    """Simulates an open-loop control process with a fixed cooling rate."""
    rng = np.random.default_rng(seed)
    dt_min = dt / 60.0
    T = T0
    c = c0
    nucleated = False
    t = 0.0
    rows = []
    
    while t < max_time:
        if T > Tend:
            T -= Rc_const * dt_min
            Rc = Rc_const
        else:
            T = Tend
            Rc = 0.0
        
        sigma_true = compute_sigma(c, T)
        
        if T == Tend and c <= solubility_k2so4(Tend):
            break

        if nucleated:
            c -= k_g * (c - solubility_k2so4(T)) * dt_min
        elif sigma_true > sigma_crit:
            nucleated = True
            print(f"Nucleation event detected at t={t:.2f} min in open-loop simulation with Rc={Rc_const}.")
        
        c_ec = c + rng.normal(0, noise_ec)
        c_uv = c + rng.normal(0, noise_uv)
        var_ec = noise_ec ** 2
        var_uv = noise_uv ** 2
        w_uv = (1 / var_ec) / (1 / var_ec + 1 / var_uv)
        c_fused = w_uv * c_uv + (1 - w_uv) * c_ec
        sigma_fused = compute_sigma(c_fused, T)

        rows.append({
            'time_min': t,
            'T_C': T,
            'c_gL': c,
            'sigma_true': float(sigma_true),
            'sigma_fused': float(sigma_fused),
            'Rc_C_per_min': Rc,
        })
        t += dt_min
    
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), {'overshoot': np.nan, 'IAE': np.nan, 'time_sigma_above_crit_min': np.nan}
    
    overshoot = max(0.0, df['sigma_true'].max() - sigma_target)
    iae = (df['sigma_true'] - sigma_target).abs().sum() * dt_min
    time_above = (df['sigma_true'] > sigma_crit).sum() * dt_min
    
    metrics = {
        'overshoot': overshoot,
        'IAE': iae,
        'time_sigma_above_crit_min': time_above,
    }
    return df, metrics

def simulate_closed_loop(
    T0=50.0,
    Tend=10.0,
    c0=120.0,
    Rc_bounds=(0.1, 1.0),
    dt=1.0,
    k_g=0.005,
    sigma_target=0.15,
    sigma_crit=0.25,
    Kp=0.8,
    Ki=0.05,
    noise_ec=0.8,
    noise_uv=0.6,
    seed=0,
    max_time=200,
):
    """Simulates a closed-loop control system using real-time feedback."""
    rng = np.random.default_rng(seed)
    dt_min = dt / 60.0

    T = T0
    c = c0
    integral = 0.0
    nucleated = False
    t = 0.0
    rows = []

    while t < max_time:
        if T == Tend and c <= solubility_k2so4(Tend):
            break

        c_ec = c + rng.normal(0, noise_ec)
        c_uv = c + rng.normal(0, noise_uv)
        
        var_ec = noise_ec ** 2
        var_uv = noise_uv ** 2
        w_uv = (1 / var_ec) / (1 / var_ec + 1 / var_uv)
        c_fused = w_uv * c_uv + (1 - w_uv) * c_ec
        sigma_fused = compute_sigma(c_fused, T)

        if T > Tend:
            error = sigma_target - sigma_fused
            integral += error * dt_min
            Rc = np.clip(Kp * error + Ki * integral, *Rc_bounds)
            T -= Rc * dt_min
        else:
            T = Tend
            Rc = 0.0
            
        sigma_true = compute_sigma(c, T)
        
        if nucleated:
            c -= k_g * (c - solubility_k2so4(T)) * dt_min
        elif sigma_true > sigma_crit:
            nucleated = True
            print(f"Nucleation event detected at t={t:.2f} min in closed-loop simulation.")
            
        rows.append({
            'time_min': t,
            'T_C': T,
            'c_gL': c,
            'sigma_true': float(sigma_true),
            'sigma_fused': float(sigma_fused),
            'Rc_C_per_min': Rc,
        })
        t += dt_min

    df_ctrl = pd.DataFrame(rows)
    if df_ctrl.empty:
        return pd.DataFrame(), {'overshoot': np.nan, 'IAE': np.nan, 'time_sigma_above_crit_min': np.nan}

    overshoot = max(0.0, df_ctrl['sigma_true'].max() - sigma_target)
    iae = (df_ctrl['sigma_true'] - sigma_target).abs().sum() * dt_min
    time_above = (df_ctrl['sigma_true'] > sigma_crit).sum() * dt_min

    metrics = {
        'overshoot': overshoot,
        'IAE': iae,
        'time_sigma_above_crit_min': time_above,
    }
    return df_ctrl, metrics


def plot_control(df_ctrl: pd.DataFrame, file_name='control_closedloop'):
    if df_ctrl.empty:
        print(f"Skipping plot for {file_name} due to empty data.")
        return
        
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    fig.suptitle('Closed-Loop Control Simulation', fontsize=16)
    
    axes[0].plot(df_ctrl['time_min'], df_ctrl['T_C'], label='T (°C)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('(a)', loc='left', fontsize='medium')

    axes[1].plot(df_ctrl['time_min'], df_ctrl['sigma_true'], label='σ true')
    axes[1].plot(df_ctrl['time_min'], df_ctrl['sigma_fused'], '--', label='σ fused')
    axes[1].axhline(y=compute_sigma(solubility_k2so4(df_ctrl['T_C'].min()), df_ctrl['T_C'].min()), color='k', linestyle=':', label='σ final')
    axes[1].set_ylabel('Supersaturation (σ)')
    axes[1].legend()
    axes[1].set_title('(b)', loc='left', fontsize='medium')

    axes[2].plot(df_ctrl['time_min'], df_ctrl['Rc_C_per_min'], label='Rc')
    axes[2].set_ylabel('Cooling Rate (°C/min)')
    axes[2].set_xlabel('Time (min)')
    axes[2].set_title('(c)', loc='left', fontsize='medium')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{file_name}.tif', dpi=300)
    plt.savefig(f'{file_name}.pdf')
    plt.close(fig)

def plot_all_comparison(df_open_05: pd.DataFrame, df_open_01: pd.DataFrame, df_ctrl: pd.DataFrame, sigma_target=0.15):
    """Plots all simulation results on a single figure for direct comparison."""
    if df_open_05.empty or df_open_01.empty or df_ctrl.empty:
        print("Skipping combined comparison plot due to empty dataframes.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    # fig.suptitle('Comparison of Simulation Control Strategies', fontsize=16)
    
    # Temperature
    axes[0].plot(df_open_05['time_min'], df_open_05['T_C'], label='Open-Loop Rc=0.5 $^\circ$C/min', color='tab:blue', linestyle='-')
    axes[0].plot(df_open_01['time_min'], df_open_01['T_C'], label='Open-Loop Rc=0.1 $^\circ$C/min', color='tab:blue', linestyle=':')
    axes[0].plot(df_ctrl['time_min'], df_ctrl['T_C'], label='Closed-Loop T', color='tab:red', linestyle='--')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend()
    axes[0].set_title('(a) Temperature over Time')
    
    # Supersaturation
    axes[1].plot(df_open_05['time_min'], df_open_05['sigma_true'], label='Open-Loop Rc=0.5 $^\circ$C/min', color='tab:blue', linestyle='-')
    axes[1].plot(df_open_01['time_min'], df_open_01['sigma_true'], label='Open-Loop Rc=0.1 $^\circ$C/min', color='tab:blue', linestyle=':')
    axes[1].plot(df_ctrl['time_min'], df_ctrl['sigma_true'], label='Closed-Loop', color='tab:red', linestyle='--')
    axes[1].axhline(y=sigma_target, color='gray', linestyle='-.', label='Target σ')
    axes[1].set_ylabel('Supersaturation (σ)')
    axes[1].legend()
    axes[1].set_title('(b) Supersaturation over Time')

    # Cooling rate
    axes[2].plot(df_open_05['time_min'], df_open_05['Rc_C_per_min'], label='Open-Loop Rc=0.5 $^\circ$C/min', color='tab:blue', linestyle='-')
    axes[2].plot(df_open_01['time_min'], df_open_01['Rc_C_per_min'], label='Open-Loop Rc=0.1 $^\circ$C/min', color='tab:blue', linestyle=':')
    axes[2].plot(df_ctrl['time_min'], df_ctrl['Rc_C_per_min'], label='Closed-Loop Rc (Variable)', color='tab:red', linestyle='--')
    axes[2].set_ylabel('Cooling Rate (°C/min)')
    axes[2].set_xlabel('Time (min)')
    axes[2].legend()
    axes[2].set_title('(c) Cooling Rate over Time')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('control_all_in_one.tif', dpi=300)
    plt.savefig('control_all_in_one.pdf')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Single merged CSV: must contain columns time,temperature,turbidity,concentration,Rc')
    parser.add_argument('--exp-files', type=str, nargs='*', help='Multiple raw experiment CSVs: must contain columns Temperature, Turbidity, Concentration, Time_elapsed (min or sec)')
    parser.add_argument('--cooling-rates', type=float, nargs='*', default=[0.1,0.3,0.5], help='For synthetic data or when unable to auto-infer')
    parser.add_argument('--sigma-target', type=float, default=0.15)
    parser.add_argument('--sigma-crit', type=float, default=0.25)
    parser.add_argument('--Kp', type=float, default=0.8)
    parser.add_argument('--Ki', type=float, default=0.05)
    parser.add_argument('--Rc-open', type=float, default=0.5, help='Constant cooling rate for open-loop simulation')
    parser.add_argument('--Rc-min', type=float, default=0.1)
    parser.add_argument('--Rc-max', type=float, default=1.0)
    parser.add_argument('--noise-ec', type=float, default=0.8)
    parser.add_argument('--noise-uv', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--turbidity-threshold', type=float, default=None, help='Manually set nucleation turbidity threshold (default: auto)')
    args = parser.parse_args()

    exp_data = {
        0.1: {'c0': 149.52, 'Ts': 41.86, 'Tn': 33, 't_ind': 88.6, 'sigma_max': 0.12},
        0.3: {'c0': 147.7, 'Ts': 41.03, 'Tn': 30, 't_ind': 36.8, 'sigma_max': 0.16},
        0.5: {'c0': 149.1, 'Ts': 41.32, 'Tn': 28, 't_ind': 26.6, 'sigma_max': 0.2},
    }

    if args.exp_files:
        frames = []
        for f in args.exp_files:
            p = Path(f)
            raw = pd.read_csv(p)
            if 'Time_elapsed' in raw.columns:
                time_raw = raw['Time_elapsed'].astype(float)
                time_max = time_raw.max()
                if time_max > 200:
                    time_col = time_raw / 60.0
                else:
                    time_col = time_raw
            else:
                time_col = pd.Series(np.arange(len(raw))) / 60.0
            T_col = raw.get('Temperature')
            turb_col = raw.get('Turbidity')
            c_col = raw.get('Concentration')
            m = re.search(r'_(\d+(?:\.\d+)?)\.csv$', p.name)
            if m:
                Rc_val = float(m.group(1))
            else:
                if T_col is not None:
                    try:
                        z = np.polyfit(time_col, T_col, 1)
                        Rc_val = -z[0]
                    except Exception:
                        Rc_val = np.nan
                else:
                    Rc_val = np.nan
            frames.append(pd.DataFrame({
                'time': time_col,
                'temperature': T_col,
                'turbidity': turb_col,
                'concentration': c_col,
                'Rc': Rc_val,
            }))
        df = pd.concat(frames, ignore_index=True)
        if args.turbidity_threshold is None and 'turbidity' in df.columns:
            early = df['time'] < df['time'].max()*0.1
            base = df.loc[early, 'turbidity'].dropna()
            if not base.empty:
                auto_th = base.mean() + 3*base.std()
            else:
                auto_th = 300.0
        else:
            auto_th = args.turbidity_threshold if args.turbidity_threshold is not None else 300.0
        def detect_nucleation_wrapper(time, turbidity, threshold=auto_th, window=5):
            return detect_nucleation(time, turbidity, threshold=threshold, window=window)
        global detect_nucleation
        detect_nucleation = detect_nucleation_wrapper
        valid_Rc = df['Rc'].dropna().unique()
        if len(valid_Rc):
            args.cooling_rates = sorted(valid_Rc)
    elif args.data:
        df = pd.read_csv(args.data)
    else:
        records = []
        for Rc in args.cooling_rates:
            if Rc in exp_data:
                Tn = exp_data[Rc]['Tn']
                T0 = 50.0
                total_time = (T0 - Tn) / Rc + 30
                times = np.arange(0, total_time, 1)
                T = T0 - Rc * times
                mask = T >= 10
                times = times[mask]
                T = T[mask]
                c0 = exp_data[Rc]['c0']
                
                turbidity = np.random.normal(50, 20, size=T.size)
                after = T < Tn
                turbidity[after] += 600 + 50*np.random.randn(after.sum())
                concentration = np.full_like(T, c0, dtype=float)
                c_sat_at_Tn = solubility_k2so4(Tn)
                concentration[after] = c_sat_at_Tn + 0.1*(c0 - c_sat_at_Tn)
                for t, TT, tu, cc in zip(times, T, turbidity, concentration):
                    records.append({'time': t, 'temperature': TT, 'turbidity': tu, 'concentration': cc, 'Rc': Rc})
        df = pd.DataFrame(records)
    
    results = analyze_batches(df, args.cooling_rates)
    if results:
        df_out = pd.DataFrame([r.__dict__ for r in results])
        df_out.to_csv('mszw_summary.csv', index=False)
        print('Saved mszw_summary.csv')

    num_cores = cpu_count()
    print(f"Running simulations on {num_cores} cores...")

    # --- 修改部分开始 ---
    # 运行所有模拟，并存储结果
    df_results = {}
    metrics_results = {}

    # 运行闭环控制模拟
    closed_loop_params = {
        'T0': 50.0, 'Tend': 10.0, 'c0': 150.0, 'Rc_bounds': (args.Rc_min, args.Rc_max),
        'dt': 1.0, 'k_g': 0.05, 'sigma_target': args.sigma_target,
        'sigma_crit': args.sigma_crit, 'Kp': args.Kp, 'Ki': args.Ki,
        'noise_ec': args.noise_ec, 'noise_uv': args.noise_uv, 'seed': args.seed
    }
    df_results['closed_loop'], metrics_results['closed_loop'] = run_simulation('closed_loop', closed_loop_params)
    
    # 运行两组开环控制模拟
    for rate in [0.5, 0.1]:
        open_loop_params = {
            'T0': 50.0, 'Tend': 10.0, 'c0': 150.0, 'Rc_const': rate,
            'dt': 1.0, 'k_g': 0.05, 'sigma_target': args.sigma_target,
            'sigma_crit': args.sigma_crit, 'noise_ec': args.noise_ec,
            'noise_uv': args.noise_uv, 'seed': args.seed
        }
        df_results[f'open_loop_Rc={rate}'], metrics_results[f'open_loop_Rc={rate}'] = run_simulation('open_loop', open_loop_params)
        
    # 生成所有模拟的综合图表
    plot_all_comparison(
        df_open_05=df_results['open_loop_Rc=0.5'],
        df_open_01=df_results['open_loop_Rc=0.1'],
        df_ctrl=df_results['closed_loop'],
        sigma_target=args.sigma_target
    )

    # 打印和保存所有模拟的指标
    all_metrics = []
    for key, metrics in metrics_results.items():
        metrics['type'] = key
        all_metrics.append(metrics)
    
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv('control_metrics_comparison_all.csv', index=False)
    print('\nAll simulation metrics:')
    print(df_metrics.set_index('type'))
    print('\nAll comparison plots and metrics saved.')

# --- 修改部分结束 ---

if __name__ == '__main__':
    main()