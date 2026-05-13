import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
app = FastAPI(title="HEMS Dispatch System - Strict Real-time API")

# =============================================================
# 1. 系统参数配置
# =============================================================
BATTERY_CAPACITY = 30.0
MAX_CHG_PWR = 6.0
MAX_DIS_PWR = 6.0
MIN_SOC, MAX_SOC = 0.10, 0.90
DT = 0.5  # 30分钟步长
DAILY_CYCLE_LIMIT = 1.5
MAX_CUMULATIVE_KWH = BATTERY_CAPACITY * DAILY_CYCLE_LIMIT
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================
# 2. 核心调度函数库
# =============================================================

def traditional_pcs_dispatch(df):
    """【传统模式】保底逻辑：简单的自发自用"""
    results = []
    soc = 0.3
    for i, row in df.iterrows():
        p_pv, p_load = row['光伏预测'], row['负载预测']
        net_p = p_pv - p_load
        if net_p > 0:
            p_bat = min(net_p, MAX_CHG_PWR, (MAX_SOC - soc) * BATTERY_CAPACITY / (DT * 0.95))
        else:
            p_bat = -min(abs(net_p), MAX_DIS_PWR, (soc - MIN_SOC) * BATTERY_CAPACITY * 0.95 / DT)

        delta_soc = (p_bat * DT * 0.95) if p_bat > 0 else (p_bat * DT / 0.95)
        soc = np.clip(soc + delta_soc / BATTERY_CAPACITY, 0, 1)
        p_grid = p_load - p_pv + p_bat

        results.append({
            'Step': i, 'Time': row['Time'], '电量SOC': soc * 100,
            '光伏': p_pv, '负载': p_load, '网购电': max(0, p_grid), '馈网电': abs(min(0, p_grid)),
            '电池充电': max(0, p_bat), '电池放电': abs(min(0, p_bat)),
            '买电价格': row.get('买电价格', 0.1), '策略模式': '传统PCS模式(保底)',
            'mode': 0, '当日累计吞吐': 0.0
        })
    return pd.DataFrame(results), 0.0


def simple_energy_dispatch(df):
    """【AI模式】优化策略：电价优化调度"""
    results = []
    soc = 0.3
    low_price_threshold = df['买电价格'].quantile(0.2)
    cumulative_throughput_kwh = 0.0
    for i, row in df.iterrows():
        p_pv, p_load = row['光伏预测'], row['负载预测']
        buy_p = row['买电价格']
        max_p_chg = min(MAX_CHG_PWR, (MAX_SOC - soc) * BATTERY_CAPACITY / (DT * 0.95))
        max_p_dis = min(MAX_DIS_PWR, (soc - MIN_SOC) * BATTERY_CAPACITY * 0.95 / DT)

        p_bat = 0.0
        mode_code = 0
        if cumulative_throughput_kwh < MAX_CUMULATIVE_KWH:
            if buy_p <= low_price_threshold and soc < 0.8:
                p_bat = max_p_chg
                mode_code = 3  # 强制充电模式
            elif p_pv >= p_load:
                p_bat = min(p_pv - p_load, max_p_chg)
            else:
                p_bat = -min(p_load - p_pv, max_p_dis)

        cumulative_throughput_kwh += abs(p_bat) * DT
        delta_soc = (p_bat * DT * 0.95) if p_bat > 0 else (p_bat * DT / 0.95)
        soc = np.clip(soc + delta_soc / BATTERY_CAPACITY, 0, 1)
        p_grid = p_load - p_pv + p_bat

        results.append({
            'Step': i, 'Time': row['Time'], '电量SOC': soc * 100,
            '光伏': p_pv, '负载': p_load, '网购电': max(0, p_grid), '馈网电': abs(min(0, p_grid)),
            '电池充电': max(0, p_bat), '电池放电': abs(min(0, p_bat)),
            '当日累计吞吐': cumulative_throughput_kwh, '买电价格': buy_p,
            '策略模式': 'AI优化模式', 'mode': mode_code
        })
    final_df = pd.DataFrame(results)
    total_cycles = cumulative_throughput_kwh / (2 * BATTERY_CAPACITY)
    return final_df, total_cycles


# =============================================================
# 3. 综合绘图代码库
# =============================================================
def plot_integrated_cn(df, cycles, target_date, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    time_labels = df['Time'].dt.strftime('%H:%M')
    steps = range(len(time_labels))

    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.bar(steps, df['光伏'], label='光伏 (PV)', color='#FFD700', alpha=0.7)
    ax1.bar(steps, df['电池放电'], bottom=df['光伏'], label='电池放电', color='#1E90FF', alpha=0.8)
    ax1.bar(steps, df['网购电'], bottom=df['光伏'] + df['电池放电'], label='网购电', color='#FF4500', alpha=0.6)
    ax1.bar(steps, -df['负载'], label='负载需求', color='#808080', alpha=0.7)
    ax1.bar(steps, -df['电池充电'], bottom=-df['负载'], label='电池充电', color='#32CD32', alpha=0.8)
    ax1.bar(steps, -df['馈网电'], bottom=-(df['负载'] + df['电池充电']), label='馈网卖电', color='#A020F0', alpha=0.5)
    
    ax1.set_xticks(steps[::4])  # 每2小时显示一个标签
    ax1.set_xticklabels(time_labels[::4], rotation=45, fontsize=8)
    ax1.axhline(0, color='black', lw=1.5)
    ax1.set_ylabel("功率 (kW)", fontsize=12)
    ax1.set_xlabel("时间", fontsize=12)
    ax1.set_title(f"HEMS 综合调度分析图 - {target_date} ({df['策略模式'].iloc[0]})", fontsize=14)

    ax_soc = ax1.twinx()
    ax_soc.plot(steps, df['电量SOC'], color='blue', lw=2.5, ls='--', label='电池 SOC (%)')
    ax_soc.set_ylabel("电池 SOC (%)", color='blue', fontsize=12)
    ax_soc.set_ylim(0, 105)

    ax_price = ax1.twinx()
    ax_price.spines['right'].set_position(('outward', 60))
    ax_price.step(steps, df['买电价格'], where='post', color='#FF4500', lw=2, label='买电价格 (EUR)')
    ax_price.set_ylabel("买电价格 (EUR/kWh)", color='#FF4500', fontsize=12)

    lines, labels = [], []
    for ax in [ax1, ax_soc, ax_price]:
        l, lb = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lb)
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.15, 1))

    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'hems_plot_{target_date}.png'))
    plt.close()


# =============================================================
# 4. 接口路由 (强制实时日期与文件名匹配)
# =============================================================
@app.get("/admin/hems/get-dispatch-data")
async def get_dispatch_data(startDate: str = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
    # 核心修改：无视外部传入的 startDate，强制使用当前系统日期
    now = datetime.now()  
    target_date = now.strftime("%Y-%m-%d")

    # 实时对齐：将调度起点设置为“当前时刻”的整点或半点
    current_aligned_start = now.replace(minute=30 if now.minute >= 30 else 0, second=0, microsecond=0)

    UNIT_CONVERSION = 0.001
    pv_file = os.path.join(DATA_DIR, f'pv_prediction_{target_date}.csv')
    # 文件名严格跟随当前实时日期
    load_file = os.path.join(DATA_DIR, f"load_prediction_{target_date}.csv")
    price_file = os.path.join(DATA_DIR, f"electricity_price_{target_date}.csv")

    try:
        # B. 数据读取与自动检测
        if os.path.exists(pv_file) and os.path.exists(load_file) and os.path.exists(price_file):
            df_pv = pd.read_csv(pv_file)
            df_load = pd.read_csv(load_file)
            df_price = pd.read_csv(price_file)

            pv_col = next((c for c in df_pv.columns if c in ['Predicted_PV(W)', 'PVLOAD', 'PV_kW']), df_pv.columns[1])
            load_col = next((c for c in df_load.columns if c in ['predicted_load', 'Load_kW']), df_load.columns[1])

            # 生成以“当前系统时刻”为起点、日期为当天的 48 个调度时段
            df_pred = pd.DataFrame({
                'Time': pd.date_range(current_aligned_start, periods=48, freq='30min'),
                '光伏预测': df_pv[pv_col].values[:48] * UNIT_CONVERSION,
                '负载预测': df_load[load_col].values[::6][:48] * UNIT_CONVERSION,
                '买电价格': np.repeat(df_price['含税电价'].values / 100.0, 2)[:48]
            })
            res_df, cycles = simple_energy_dispatch(df_pred)
        else:
            # 降级模式提示
            missing = []
            if not os.path.exists(pv_file): missing.append("PV预测")
            if not os.path.exists(load_file): missing.append(f"负载预测({target_date})")
            if not os.path.exists(price_file): missing.append(f"电价数据({target_date})")

            print(f"⚠️ 实时警告：未找到 {target_date} 所需的完整数据文件 (缺少: {', '.join(missing)})")

            df_fallback = pd.DataFrame({
                'Time': pd.date_range(current_aligned_start, periods=48, freq='30min'),
                '光伏预测': 0.0, '负载预测': 0.0, '买电价格': 0.1
            })
            res_df, cycles = traditional_pcs_dispatch(df_fallback)

        # C. 终端报告打印
        total_pv = res_df['光伏'].sum() * DT
        total_load = res_df['负载'].sum() * DT
        total_cost = (res_df['网购电'] * DT * res_df['买电价格']).sum()

        print(f"\n" + "=" * 70)
        print(f"📊 HEMS 调度实时报告 | 锁定日期文件: {target_date}")
        print(f"⏰ 执行时刻: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🚩 运行模式: {res_df['策略模式'].iloc[0]}")
        print(f"☀️ 发电: {total_pv:.2f} kWh | 🏠 负载: {total_load:.2f} kWh")
        print(f"🔋 循环: {cycles:.2f} | 💵 预计支出: {total_cost:.2f} EUR")
        print("=" * 70)

        # D. 本地产出
        res_df.to_csv(os.path.join(DATA_DIR, f'hems_data_{target_date}.csv'), index=False, encoding='utf_8_sig')
        plot_integrated_cn(res_df, cycles, target_date, DATA_DIR)

        # E. JSON 组装 (确保每个 Step 的日期都是实时动态计算的)
        obj_list = []
        for _, row in res_df.iterrows():
            obj_list.append({
                "timeSwitch": 1,
                "startTime": row['Time'].strftime("%H:%M"),
                "endTime": (row['Time'] + timedelta(minutes=29, seconds=59)).strftime("%H:%M"),
                "date": row['Time'].strftime("%Y-%m-%d"),  # 这里的日期随 Time 步进（可能跨天）
                "forcedPower": 0,
                "temporaryPower": 0,
                "mode": int(row['mode']),
                "electricPrice": str(round(row['买电价格'], 4)),
                "dischargingSOC": int(MIN_SOC * 100),
                "chargingSOC": int(MAX_SOC * 100),
                "pvPrediction": round(row['光伏'], 4),
                "loadPrediction": round(row['负载'], 4),
                "buyPower": round(row['网购电'], 4),
                "sellPower": round(row['馈网电'], 4),
                "batteryChargePower": round(row['电池充电'], 4),
                "batteryDischargePower": round(row['电池放电'], 4),
                "soc": round(row['电量SOC'], 2)
            })

        return {"result": 0, "msg": "Request successfully.", "obj": obj_list}

    except Exception as e:
        print(f"❌ 严重错误: {str(e)}")
        return {"result": 1, "msg": f"Internal Error: {str(e)}", "obj": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
