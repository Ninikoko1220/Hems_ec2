import os
os.chdir("/home/ec2-user/hems")

import numpy
import requests
import pandas as pd
import numpy as np
import numpy.core.multiarray
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime, timedelta
import warnings
# 解决 PyTorch 2.6 加载模型报错问题

torch.serialization.add_safe_globals([MinMaxScaler])
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
import json

# --- 载入统一配置 ---
def load_sys_config():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
        return json.load(f)

sys_cfg = load_sys_config()

# 1. API 配置 (自动同步 json)
BASE_URL = sys_cfg["BASE_URL"]
API_URL = BASE_URL + "/inner/ai/power/list"
TOKEN = sys_cfg["TOKEN"]

DEVICE_ID = sys_cfg["DEVICE_ID"]

# 2. 调度时间 (每天北京时间早上 06:10 自动运行)
SCHEDULE_HOUR = 6
SCHEDULE_MINUTE = 10

# 3. 文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, 'dataset_winter_strict.csv')
MODEL_FILE = os.path.join(BASE_DIR, 'load_forecast_model.pth')
TARGET_COL = 'loadElec'  # 对应接口中的负载功率字段

# 4. 【核心阈值】熔断标准
POINTS_PER_DAY = 288
ABNORMAL_THRESHOLD = 0.5  # 异常比例超过 50% (半天) -> 直接熔断，启动 B 计划
MAX_GAP_REPAIR = 12  # 连续缺失 < 1小时 -> 允许插值修复

# 5. 模型参数
SEQUENCE_LENGTH = 288
PREDICT_STEPS = 288


# ===========================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        output = self.linear(out)
        return output


# --- 模块 1: 数据质检官 (Judge) ---
def check_data_quality(df_day):
    """判断一天的数据是否可用"""
    nan_count = df_day[TARGET_COL].isna().sum()
    zero_count = (df_day[TARGET_COL] <= 0).sum()
    bad_mask = df_day[TARGET_COL].isna() | (df_day[TARGET_COL] <= 0)
    total_bad_points = bad_mask.sum()
    bad_ratio = total_bad_points / POINTS_PER_DAY

    print(f"   📊 质量检测: 缺失 {nan_count} 点, 0值 {zero_count} 点. 总异常率: {bad_ratio:.1%}")

    if bad_ratio > ABNORMAL_THRESHOLD:
        return False, f"异常比例 {bad_ratio:.1%} > 50% (超过半天不可用)"
    return True, "合格"


# --- 模块 2: 智能修复师 (Doctor) ---
def repair_data(df_day):
    """对轻微受损的数据进行修复"""
    df_day[TARGET_COL] = df_day[TARGET_COL].interpolate(method='linear', limit=MAX_GAP_REPAIR)
    df_day[TARGET_COL] = df_day[TARGET_COL].fillna(method='ffill').fillna(method='bfill')
    if df_day[TARGET_COL].isna().any():
        print("   🔧 发现大缺口，填充 0 处理...")
        df_day[TARGET_COL] = df_day[TARGET_COL].fillna(0)
    return df_day


# --- 模块 3: B 计划执行者 (Backup) ---
def load_historical_backup():
    """从本地 CSV 中寻找最近的一个“好日子”作为替补"""
    if not os.path.exists(CSV_FILE):
        return None
    try:
        df_hist = pd.read_csv(CSV_FILE)
        df_hist['datetime'] = pd.to_datetime(df_hist['datetime'])
        df_hist = df_hist.sort_values('datetime')
        if len(df_hist) >= POINTS_PER_DAY:
            backup_data = df_hist.tail(POINTS_PER_DAY).copy()
            backup_date = backup_data['datetime'].iloc[0].date()
            print(f"   🔄 [B计划] 已加载历史替补数据: {backup_date}")
            return backup_data[TARGET_COL].values.reshape(-1, 1)
    except Exception as e:
        print(f"   ❌ 读取历史数据失败: {e}")
    return None


# --- 模块 4: 每日主流程 (自适应版) ---
def perform_daily_task(target_date_obj=None):
    print(f"\n{'=' * 60}")
    # 自适应逻辑：如果未传入日期，则默认为“今天”
    today = target_date_obj if target_date_obj else datetime.now()
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    print(f"⏰ 任务执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 预测目标日期: {today_str}")

    input_values = None
    use_plan_b = False

    # 1. 获取昨日数据
    print(f"📅 1. 正在获取昨日历史功率 ({yesterday_str})...")
    headers = {"Authorization": TOKEN}
    # 获取昨日结束时间的时间戳
    day_end = datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59)
    params = {"deviceId": DEVICE_ID, "time": int(day_end.timestamp()), "date": yesterday_str}

    raw_data = []
    try:
        response = requests.get(API_URL, headers=headers, params=params, timeout=30)
        res_json = response.json()
        if res_json.get("code") == 0:
            raw_data = res_json.get("data", []) or []
        else:
            print(f"   ⚠️ 接口返回错误: {res_json.get('msg')}")
    except Exception as e:
        print(f"   ❌ 网络请求失败: {e}")

    # 2. 对齐与初步整理
    standard_idx = pd.date_range(start=f"{yesterday_str} 00:00", end=f"{yesterday_str} 23:55", freq='5min')
    df_raw = pd.DataFrame(raw_data)
    df_day = pd.DataFrame({'datetime': standard_idx})
    df_day[TARGET_COL] = np.nan

    if not df_raw.empty and 'time' in df_raw.columns:
        # 接口返回 time="00:05", 需要结合昨天的日期字符串拼接成 datetime
        df_raw['datetime'] = pd.to_datetime(yesterday_str + ' ' + df_raw['time'])
        df_raw = df_raw.drop_duplicates('datetime').set_index('datetime')
        df_day.set_index('datetime', inplace=True)
        df_day.update(df_raw)
        df_day.reset_index(inplace=True)
        # 强制转换为数值
        df_day[TARGET_COL] = pd.to_numeric(df_day[TARGET_COL], errors='coerce')

    # 3. 质检与修复
    is_usable, reason = check_data_quality(df_day)
    if is_usable:
        print("   ✅ 数据合格，正在执行本地存储与修复...")
        df_final = repair_data(df_day)
        input_values = df_final[TARGET_COL].values.reshape(-1, 1)

        # 增量更新本地数据库
        try:
            if os.path.exists(CSV_FILE):
                old_df = pd.read_csv(CSV_FILE)
                old_df['datetime'] = pd.to_datetime(old_df['datetime'])
                full_df = pd.concat([old_df, df_final]).drop_duplicates(subset=['datetime'], keep='last').sort_values(
                    'datetime')
                full_df.to_csv(CSV_FILE, index=False)
            else:
                df_final.to_csv(CSV_FILE, index=False)
        except Exception as e:
            print(f"   ⚠️ 本地存储失败: {e}")
    else:
        print(f"   ⛔ 触发熔断: {reason}. 启动 [B计划]")
        input_values = load_historical_backup()
        use_plan_b = True

    if input_values is None:
        print("   ❌ [崩溃] 无任何可用数据源，预测任务终止。")
        return

    # 4. 模型预测
    print(f"🔮 2. 正在基于 LSTM 生成 AI 负荷预测 ({today_str})...")
    if not os.path.exists(MODEL_FILE):
        print("   ❌ 错误: 模型文件丢失，请检查 load_forecast_model.pth")
        return

    device = torch.device('cpu')
    try:
        # 加载模型和归一化参数
        # torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])  # Nian
        checkpoint = torch.load(MODEL_FILE, map_location=device,weights_only=False)

        model = LSTMModel().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        scaler = checkpoint['scaler']

        # 归一化输入
        input_scaled = scaler.transform(input_values)
        input_seq = torch.FloatTensor(input_scaled).view(1, SEQUENCE_LENGTH, 1).to(device)

        # 执行滚动预测 (288 个点)
        future_preds_scaled = []
        with torch.no_grad():
            curr_seq = input_seq
            for _ in range(PREDICT_STEPS):
                pred = model(curr_seq)
                future_preds_scaled.append(pred.item())
                curr_seq = torch.cat((curr_seq[:, 1:, :], pred.view(1, 1, 1)), dim=1)

        # 逆归一化
        future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))

        # 5. 保存结果
        save_file_name = os.path.join(BASE_DIR, "data", f"load_prediction_{today_str}.csv")
        result_df = pd.DataFrame({
            'datetime': pd.date_range(start=f"{today_str} 00:00", end=f"{today_str} 23:55", freq='5min'),
            'predicted_load': future_preds.flatten()
        })
        result_df.to_csv(save_file_name, index=False)
        print(f"   🎉 预测完成! 调度文件已更新: {save_file_name}")

        # 6. 画图展示
        plt.figure(figsize=(10, 5))
        title_text = f"load_prediction {today_str} ({'Plan B' if use_plan_b else 'Normal'})"
        plt.plot(result_df['datetime'], result_df['predicted_load'], color='red', label='AI load_prediction')
        plt.title(title_text, fontweight='bold', color='orange' if use_plan_b else 'green')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR,"data", f"load_prediction_{today_str}.png"))
        plt.close()

    except Exception as e:
        print(f"   ❌ 预测计算过程中出错: {e}")


# --- 守护进程循环 ---
def main_loop():
    print(f"🚀 家庭负荷预测系统 Pro 版已启动 (2026-03-25 接口对齐)")
    print(f"🛡️ 熔断阈值: 异常数据占比 > {ABNORMAL_THRESHOLD * 100}%")
    print("-" * 50)

    while True:
        now = datetime.now()
        # 设定下一次运行时间
        target_time = now.replace(hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE, second=0, microsecond=0)
        if now >= target_time:
            target_time += timedelta(days=1)

        wait_seconds = (target_time - now).total_seconds()
        hours = int(wait_seconds // 3600)
        minutes = int((wait_seconds % 3600) // 60)

        print(f"💤 待机中... 下次自动唤醒: {target_time} (约 {hours}小时{minutes}分后)")
        time.sleep(wait_seconds)

        try:
            perform_daily_task()
        except Exception as e:
            print(f"⚠️ 任务执行异常: {e}")

        time.sleep(60)


# =============================================================
# 5. 主入口：自适应启动
# =============================================================
if __name__ == "__main__":
    # 启动后立刻执行一次预测，确保调度系统随时有数据可用
    try:
        print("🛠️ [启动初次同步] 正在生成今日预测...")
        perform_daily_task(target_date_obj=datetime.now())
    except Exception as e:
        print(f"⚠️ 初次同步异常: {e}")

    # 然后进入正常的每日定时循环
    main_loop()
