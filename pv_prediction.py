import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import schedule

#  Nini
# 在 import schedule 之后添加
# print("Schedule module path:", schedule.__file__)
# print("Schedule module attributes:", dir(schedule))

# ==============================================================================
# ================= 0. 全局统一配置区域 (GLOBAL CONFIG) ==========================
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {    # 🕒 数据时间范围 (起始日期固定，作为首次全量下载的起点)
    "start_date": "2025-01-06",

    # 接口与设备配置
    "device_id": "1762775382909366274",
    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6IjQ3MjZlNWIyLWI1MGEtNGM4NC1iNjNmLTI0NDg2ZDJkOTNlMCIsIm5hbWUiOiJhZG1pbiJ9.CcD5cBBHyHL_DR-SHYw17m1Z9gePjhFCXMKh4ojJL_g",
    "latitude": 51.1769066,
    "longitude": 4.3435924,
    "timezone": "Asia/Shanghai",

    # 💾 文件路径配置
    "merged_file": "aligned_pv_irradiance_merged_shift_minus8h.csv",
    "test_split_file": "test_split_auto_save.csv",
    "model_path": "best_model_fixed.pt",
    "scaler_x_path": "scaler_x.pkl",
    "scaler_y_path": "scaler_y.pkl",
    "future_pred_file": "data/pv_prediction_" + datetime.now().strftime("%Y-%m-%d") + ".csv",
    "last_train_record": "last_train_date.txt",  # 🌟 新增：记录上次训练时间的记忆文件

    # 模型与训练配置
    "seq_length": 144,  # 历史窗口长度
    "output_size": 48,  # 未来预测长度
    "batch_size": 64,
    "learning_rate": 0.01,
    "epochs": 200,  # 训练轮数
    "train_interval_days": 90,  # 每隔 90 天训练一次模型
    "seed": 1234567,  # 固定随机种子，用于提升训练可复现性
    "daily_run_time": "06:10",  # 🌟 已修改：自启动时间改为早上 6:10

    # 强化特征列 (对应 input_size = 9)
    "x_cols": ['PV', 'irradianceWm2', 'directRadiationWm2', 'directRadiationWm2', 'directRadiationWm2',
               'hour_sin', 'hour_cos', 'month_sin', 'month_cos'],
    "y_col": ['PV']
}

HEADERS = {"Authorization": CONFIG['token']}


# ==============================================================================
# ================= 1. 核心模型与工具类定义 (MODELS & UTILS) =======================
# ==============================================================================
class DualWeightedLoss(nn.Module):
    """
    双权重损失函数
    结合了类别权重(class_weight)和样本权重(sample_weight)

    常见应用：
    1. 处理类别不平衡 + 难易样本不平衡
    2. 时间序列中不同时间点重要性不同
    3. 多任务学习中不同样本重要性不同
    """
    def __init__(self, peak_weight=4.0, zero_penalty=5.0): # 针对峰值预测和零值处理的加权损失函数
        super().__init__()
        self.peak_weight = peak_weight
        self.zero_penalty = zero_penalty

    def forward(self, pred, target):
        """
               前向传播

               参数:
                   pred: 模型预测，shape任意
                   target: 真实标签，shape同pred
               返回:
                   加权损失值
               """
        loss = (pred - target) ** 2
        weights = torch.ones_like(target)
        weights += target * self.peak_weight
        zero_mask = (target < 1e-4).float()
        weights += zero_mask * self.zero_penalty
        return (loss * weights).mean()


class FixedCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, cnn_filters=64):
        super(FixedCNNLSTM, self).__init__()
        self.cnn = nn.Conv1d(input_size, cnn_filters, 3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(cnn_filters, hidden_size, 2, batch_first=True)
        self.attn_fc = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(nn.Linear(hidden_size, output_size), nn.ReLU())

    def forward(self, x):
        x = x.permute(0, 2, 1) # 卷积网络中的维度调整
        x = self.cnn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


class MyDataset(Dataset):
    def __init__(self, x_data, y_data, seq_len, pred_len):
        self.x, self.y, self.seq_len, self.pred_len = x_data, y_data, seq_len, pred_len

    def __len__(self):
        return len(self.x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.x[idx: idx + self.seq_len]
        y = self.y[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).squeeze()


def build_time_features(df):
    """构建时间循环特征"""
    dt = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['hour_sin'] = np.sin(2 * np.pi * (dt.dt.hour + dt.dt.minute / 60) / 24)
    df['hour_cos'] = np.cos(2 * np.pi * (dt.dt.hour + dt.dt.minute / 60) / 24)
    df['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    return df, dt


def make_naive_dt(df, date_col="date", time_col="time"):
    s = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
    return pd.to_datetime(s, errors="coerce")


def get_config_tz():
    tz_name = CONFIG.get("timezone", "Asia/Shanghai")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        print(f"⚠️ 无法识别时区 {tz_name}，回退到固定 UTC+8")
        return timezone(timedelta(hours=6))


def now_local():
    return datetime.now(get_config_tz())


def day_start_timestamp(date_str, fmt="%Y-%m-%d"):
    dt_local = datetime.strptime(date_str, fmt).replace(tzinfo=get_config_tz())
    return int(dt_local.timestamp())


# ==============================================================================
# ================= 2. 管道阶段一：获取并对齐数据 (DATA PIPELINE) ==================
# ==============================================================================

def set_global_seed(seed: int):
    """固定随机种子，尽量保证同环境下训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def run_data_pipeline(end_date_str):
    print("\n" + "=" * 50)

    merged_file = CONFIG["merged_file"]
    fmt = "%Y-%m-%d"
    end_date_dt = datetime.strptime(end_date_str, fmt)
    existing_df = None

    if os.path.exists(merged_file):
        try:
            existing_df = pd.read_csv(merged_file)
            if not existing_df.empty:
                last_dt = pd.to_datetime(existing_df['date']).max()
                fetch_start_dt = last_dt + timedelta(days=1)

                if fetch_start_dt > end_date_dt:
                    print(f"本地数据已是最新(至 {last_dt.strftime(fmt)})，无需网络抓取，直接跳过。")
                    return

                fetch_start_str = fetch_start_dt.strftime(fmt)
                print(
                    f"检测到本地历史数据 (至 {last_dt.strftime(fmt)})。将执行增量更新: {fetch_start_str} -> {end_date_str}")
            else:
                fetch_start_str = CONFIG["start_date"]
                fetch_start_dt = datetime.strptime(fetch_start_str, fmt)
                print(f"📡 本地文件为空，将执行全量下载: {fetch_start_str} -> {end_date_str}...")
        except Exception as e:
            print(f"⚠️ 读取本地数据出错: {e}。回退到全量下载模式。")
            fetch_start_str = CONFIG["start_date"]
            fetch_start_dt = datetime.strptime(fetch_start_str, fmt)
    else:
        fetch_start_str = CONFIG["start_date"]
        fetch_start_dt = datetime.strptime(fetch_start_str, fmt)
        print(f"📡 未检测到本地数据，将执行全量下载: {fetch_start_str} -> {end_date_str}...")

    base_url_pv = "http://esybackend.esysunhome.com:7074/inner/ai/power/list"
    curr_date = fetch_start_dt
    pv_points = []

    total_days = (end_date_dt - curr_date).days + 1
    day_count = 1

    print("   👉 开始下载光伏数据...")
    while curr_date <= end_date_dt:
        day_str = curr_date.strftime(fmt)
        timestamp = day_start_timestamp(day_str, fmt)
        params = {"deviceId": CONFIG["device_id"], "time": str(timestamp), "date": day_str}

        print(f"   ⏳ 进度 [{day_count}/{total_days}] 正在下载 {day_str} ...")

        try:
            res = requests.get(base_url_pv, headers=HEADERS, params=params, timeout=20)
            if res.status_code == 200 and str(res.json().get("code")) == "0":
                day_data = res.json().get("data", [])
                # 将“请求日期”写入每条记录，兼容接口未返回日期字段的情况
                for item in day_data:
                    if isinstance(item, dict):
                        row = item.copy()
                        row["_query_date"] = day_str
                        pv_points.append(row)
            else:
                err_msg = res.json().get('msg', '未知错误') if res.status_code == 200 else f"HTTP {res.status_code}"
                print(f"      ⚠️ {day_str} 无有效数据: {err_msg}")
        except Exception as e:
            print(f"      ❌ {day_str} 网络请求失败: {e}")

        curr_date += timedelta(days=1)
        day_count += 1
        time.sleep(0.2)

    new_merged = pd.DataFrame()

    if not pv_points:
        print("   ⚠️ 本次请求区间内未获取到光伏数据，不更新历史数据。")
    else:
        print("   👉 正在清洗和重采样光伏数据...")
        df_pv = pd.DataFrame(pv_points)

        # 日期优先使用接口原始列；若不存在，则使用请求日期列 _query_date
        date_col = next((c for c in ["日期", "date", "_query_date"] if c in df_pv.columns), None)
        time_col = next((c for c in ["时间", "time"] if c in df_pv.columns), None)
        power_col = next((c for c in ["pvElec", "PV"] if c in df_pv.columns), None)

        if date_col is None or time_col is None or power_col is None:
            raise KeyError(f"光伏数据缺少必要列。当前列: {list(df_pv.columns)}")

        df_pv[power_col] = pd.to_numeric(df_pv[power_col], errors='coerce')
        if power_col != "pvElec":
            df_pv.rename(columns={power_col: "pvElec"}, inplace=True)

        min_daily_points = 252
        target_daily_points = 288
        freq_5min = "5min"

        df_pv["datetime"] = pd.to_datetime(
            df_pv[date_col].astype(str).str.strip() + " " + df_pv[time_col].astype(str).str.strip(),
            errors="coerce"
        )
        df_pv = df_pv.dropna(subset=["datetime"]).copy()

        if df_pv.empty:
            print("   ⚠️ 光伏数据时间列解析失败，跳过合并。")
        else:
            df_pv["date_only"] = df_pv["datetime"].dt.date
            raw_counts = df_pv.groupby("date_only")["datetime"].nunique().sort_index()
            valid_days = raw_counts[raw_counts >= min_daily_points].index.tolist()
            df_pv_30min = pd.DataFrame()

            if not valid_days:
                print(f"   ⚠️ 本次增量区间数据不完整（单日不足{min_daily_points}条），跳过合并。")
            else:
                invalid_days = raw_counts[raw_counts < min_daily_points]
                if not invalid_days.empty:
                    bad_msg = ", ".join([f"{d}({int(c)}点)" for d, c in invalid_days.items()])
                    print(f"   ⚠️ 以下日期原始点数不足{min_daily_points}，已跳过: {bad_msg}")

                daily_frames = []
                for d in valid_days:
                    day_df = df_pv[df_pv["date_only"] == d][["datetime", "pvElec"]].copy()
                    day_df = day_df.groupby("datetime", as_index=False).mean(numeric_only=True)
                    day_df = day_df.set_index("datetime").sort_index()

                    day_start = pd.Timestamp(d)
                    full_idx = pd.date_range(
                        day_start,
                        day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=5),
                        freq=freq_5min
                    )
                    day_full = day_df.reindex(full_idx)
                    missing_cnt = int(day_full["pvElec"].isna().sum())

                    if missing_cnt > 0:
                        day_full["pvElec"] = day_full["pvElec"].interpolate(method="time", limit_direction="both")
                        day_full["pvElec"] = day_full["pvElec"].fillna(0.0)
                        print(f"   🔧 {d} 原始{int(raw_counts.loc[d])}点，已补齐{missing_cnt}点 -> {target_daily_points}点")
                    else:
                        print(f"   ✅ {d} 原始{int(raw_counts.loc[d])}点，无需补齐")

                    day_full = day_full.reset_index().rename(columns={"index": "datetime"})
                    daily_frames.append(day_full)

                df_pv = pd.concat(daily_frames, ignore_index=True)
                df_pv.set_index("datetime", inplace=True)
                df_pv_30min = df_pv.resample('30min').mean(numeric_only=True).dropna(how='all').reset_index()
            if df_pv_30min.empty:
                print("   ⚠️ 本次无满足入库条件的日数据，跳过合并。")
            else:
                df_pv_30min.rename(columns={'pvElec': 'PV'}, inplace=True)
                df_pv_30min['date'] = df_pv_30min['datetime'].apply(lambda x: f"{x.year}/{x.month}/{x.day}")
                df_pv_30min['time'] = df_pv_30min['datetime'].apply(lambda x: f"{x.hour}:{x.minute:02d}")

                print("   👉 正在获取气象数据并进行合并...")
                url_meteo = "http://esybackend.esysunhome.com:7074/inner/ai/meteo/get-meteo-data"
                params_meteo = {
                    'latitude': CONFIG['latitude'], 'longitude': CONFIG['longitude'],
                    'startDate': fetch_start_str, 'endDate': end_date_str,
                    'tilt': 45, 'azimuth': 0
                }

                try:
                    headers = {"Authorization": CONFIG['token']}
                    res = requests.get(url_meteo,headers=headers, params=params_meteo, timeout=30)
                    irr_list = res.json().get('data', {}).get('data', []) if res.status_code == 200 else []
                    if not irr_list: print("      ⚠️ 气象接口未返回有效数据！")
                except Exception as e:
                    print(f"      ❌ 气象数据请求失败: {e}")
                    irr_list = []

                if irr_list:
                    df_irr = pd.DataFrame(irr_list)
                    df_irr['dateTime'] = pd.to_datetime(df_irr['dateTime'])
                    df_irr_30min = df_irr.set_index('dateTime').resample('30min').interpolate('linear').reset_index()
                    df_irr_30min['date'] = df_irr_30min['dateTime'].apply(lambda x: f"{x.year}/{x.month}/{x.day}")
                    df_irr_30min['time'] = df_irr_30min['dateTime'].apply(lambda x: f"{x.hour}:{x.minute:02d}")
                    df_pv_30min["dt_key"] = make_naive_dt(df_pv_30min)
                    df_irr_30min["dt_key"] = make_naive_dt(df_irr_30min) - pd.Timedelta(hours=6)
                    irr_keep = df_irr_30min[["dt_key", "irradianceWm2", "directRadiationWm2"]].groupby("dt_key",
                                                                                                as_index=False).mean(
                        numeric_only=True)
                    new_merged = df_pv_30min.merge(irr_keep, on="dt_key", how="left").fillna(0.0)
                else:
                    new_merged = df_pv_30min.copy()
                    new_merged["dt_key"] = make_naive_dt(new_merged)
                    new_merged["irradianceWm2"] = 0.0
                    new_merged["directRadiationWm2"] = 0.0

                new_merged["date"] = new_merged["dt_key"].apply(lambda x: f"{x.year}/{x.month}/{x.day}")
                new_merged["time"] = new_merged["dt_key"].apply(lambda x: f"{x.hour}:{x.minute:02d}")
                new_merged = new_merged[["date", "time", "PV", "irradianceWm2", "directRadiationWm2"]]

    if existing_df is not None:
        if not new_merged.empty:
            final_df = pd.concat([existing_df, new_merged], ignore_index=True)
            final_df['dt_key'] = make_naive_dt(final_df)
            final_df = final_df.drop_duplicates(subset=['dt_key'], keep='last').sort_values('dt_key')
            final_df = final_df.drop(columns=['dt_key'])
            print(f"   ✅ 增量拼接完成！总行数: {len(final_df)}")
        else:
            final_df = existing_df
    else:
        final_df = new_merged

    if not final_df.empty:
        final_df.to_csv(merged_file, index=False, encoding="utf-8-sig")
        print("✅ 第一阶段执行完毕！数据文件已更新。")
    else:
        raise Exception("❌ 最终数据集为空，无法继续！")


# ==============================================================================
# ================= 3. 管道阶段二：模型训练 (TRAINING PIPELINE) ====================
# ==============================================================================
def run_training_pipeline():
    print("\n" + "=" * 50)
    print("🧠 [第二阶段] 正在构建特征并启动全量重训练...")
    set_global_seed(CONFIG["seed"])
    print(f"🎯 固定随机种子: {CONFIG['seed']}")
    df = pd.read_csv(CONFIG["merged_file"])
    df, dt = build_time_features(df)

    df['YearMonth'] = dt.dt.to_period('M')
    df['Day'] = dt.dt.day
    df['dt'] = dt
    train_list, val_list, test_list = [], [], []
    for ym in df['YearMonth'].unique():
        month_data = df[df['YearMonth'] == ym].sort_values('dt')
        unique_days = sorted(month_data['Day'].unique())
        if len(unique_days) < 3:
            train_list.append(month_data);
            continue
        t_end, v_end = int(len(unique_days) * 0.7), int(len(unique_days) * 0.85)
        train_list.append(month_data[month_data['Day'].isin(unique_days[:t_end])])
        val_list.append(month_data[month_data['Day'].isin(unique_days[t_end:v_end])])
        test_list.append(month_data[month_data['Day'].isin(unique_days[v_end:])])

    train_df = pd.concat(train_list).sort_values('dt')
    val_df = pd.concat(val_list).sort_values('dt')
    test_df = pd.concat(test_list).sort_values('dt')
    test_df.drop(columns=['Day', 'YearMonth', 'dt']).to_csv(CONFIG["test_split_file"], index=False)

    scaler_x, scaler_y = MinMaxScaler((0, 1)), MinMaxScaler((0, 1))
    train_x = scaler_x.fit_transform(train_df[CONFIG["x_cols"]])
    train_y = scaler_y.fit_transform(train_df[CONFIG["y_col"]])
    val_x = scaler_x.transform(val_df[CONFIG["x_cols"]])
    val_y = scaler_y.transform(val_df[CONFIG["y_col"]])

    joblib.dump(scaler_x, CONFIG["scaler_x_path"])
    joblib.dump(scaler_y, CONFIG["scaler_y_path"])

    train_loader = DataLoader(MyDataset(train_x, train_y, CONFIG["seq_length"], CONFIG["output_size"]),
                              batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(MyDataset(val_x, val_y, CONFIG["seq_length"], CONFIG["output_size"]),
                            batch_size=CONFIG["batch_size"], shuffle=False)

    model = FixedCNNLSTM(input_size=len(CONFIG["x_cols"]), hidden_size=128, output_size=CONFIG["output_size"]).to(
        device)
    criterion = DualWeightedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    best_val_loss = float('inf')
    epochs = CONFIG["epochs"]

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += criterion(model(x.to(device)), y.to(device)).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"   🔄 Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG["model_path"])

    print(f"✅ 模型训练完毕！最优权重已保存 (Val Loss: {best_val_loss:.5f})。")


# ==============================================================================
# ================= 4. 管道阶段三：测试集评估 (TESTING PIPELINE) ===================
# ==============================================================================
def run_testing_pipeline():
    print("\n" + "=" * 50)
    print("📲 [第三阶段] 正在评估新训练模型的准确率...")
    df_test = pd.read_csv(CONFIG["test_split_file"])
    df_test, dt = build_time_features(df_test)

    scaler_x, scaler_y = joblib.load(CONFIG["scaler_x_path"]), joblib.load(CONFIG["scaler_y_path"])
    test_x = scaler_x.transform(df_test[CONFIG["x_cols"]])
    test_y = scaler_y.transform(df_test[CONFIG["y_col"]])
    test_loader = DataLoader(MyDataset(test_x, test_y, CONFIG["seq_length"], CONFIG["output_size"]), batch_size=32,
                             shuffle=False)

    model = FixedCNNLSTM(len(CONFIG["x_cols"]), 128, CONFIG["output_size"]).to(device)
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device))[:, 0].cpu().numpy())
            trues.append(y[:, 0].numpy())

    pred_real = scaler_y.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
    true_real = scaler_y.inverse_transform(np.concatenate(trues).reshape(-1, 1)).flatten()
    pred_real[pred_real < 1e-3] = 0

    print(f"   📳 测试集 R2 Score: {r2_score(true_real, pred_real):.4f}")

    full_time = dt.iloc[CONFIG["seq_length"]: CONFIG["seq_length"] + len(pred_real)]
    plt.figure(figsize=(15, 6))
    plt.plot(full_time, true_real, label='True Power', color='black', alpha=0.5)
    plt.plot(full_time, pred_real, label='Predicted Power', color='#1f77b4', alpha=0.8)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.grid(True)
    plt.title("Evaluation: PV Prediction - Test Set Overview")
    plt.savefig("Eval_Figure_Overview.png", dpi=300);
    plt.close('all')
    print("✅ 测试图表已生成 (Eval_Figure_Overview.png)。")


# ==============================================================================
# ================= 5. 管道阶段四：预测未来 (PREDICTION PIPELINE) ==================
# ==============================================================================
def run_prediction_pipeline():
    print("\n" + "=" * 50)
    print("🔭 [第四阶段] 正在基于最新数据预测未来 24 小时光伏功率...")
    df = pd.read_csv(CONFIG["merged_file"])
    df, dt = build_time_features(df)

    if len(df) < CONFIG["seq_length"]:
        print("❌ 历史数据量不足以支持预测 (需要过去三天的完整数据)。")
        return

    df_input = df.iloc[-CONFIG["seq_length"]:][CONFIG["x_cols"]].copy()
    history_time = dt.iloc[-CONFIG["seq_length"]:].values
    history_pv = df_input['PV'].values
    latest_hist = pd.Timestamp(history_time[-1])
    now_cn = now_local()
    print(f"   🕒 历史数据最新时间: {latest_hist.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   🕒 当前北京时间: {now_cn.strftime('%Y-%m-%d %H:%M:%S')}")

    scaler_x, scaler_y = joblib.load(CONFIG["scaler_x_path"]), joblib.load(CONFIG["scaler_y_path"])
    x_scaled = scaler_x.transform(df_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    model = FixedCNNLSTM(len(CONFIG["x_cols"]), 128, CONFIG["output_size"]).to(device)
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.eval()

    with torch.no_grad():
        pred_scaled = model(x_tensor)

    pred_real = scaler_y.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
    pred_real[pred_real < 1e-3] = 0

    time_step = pd.Timedelta(history_time[-1] - history_time[-2])
    future_time = [history_time[-1] + time_step * (i + 1) for i in range(CONFIG["output_size"])]

    # 去除白天时段的孤立零点毛刺（前后都高、中间被截断为0）
    pred_smooth = pred_real.copy()
    for i in range(1, len(pred_smooth) - 1):
        hour = pd.Timestamp(future_time[i]).hour
        if 8 <= hour <= 17:
            left_v, cur_v, right_v = pred_smooth[i - 1], pred_smooth[i], pred_smooth[i + 1]
            if cur_v == 0 and left_v > 300 and right_v > 300:
                pred_smooth[i] = (left_v + right_v) / 2.0
    pred_real = pred_smooth

    plt.figure(figsize=(12, 5))
    plt.plot(history_time, history_pv, label='Historical PV (Last 144 pts)', color='black', linewidth=2)
    plt.plot(future_time, pred_real, label='Predicted PV (Next 48 pts)', color='red', linestyle='--', marker='o',
             markersize=4)
    plt.axvline(x=history_time[-1], color='gray', linestyle=':')
    plt.gcf().autofmt_xdate();
    plt.legend();
    plt.grid(True)
    plt.title("Future PV Forecasting (Next 24 Hours)")
    current_date = datetime.now().strftime("%Y-%m-%d")
    plt.savefig(f"data/pv_prediction_{current_date}.png", dpi=300);
    plt.close('all')

    output_df = pd.DataFrame(
        {'time': pd.to_datetime(future_time).strftime('%Y-%m-%d %H:%M:%S'), 'Predicted_PV(kW)': np.round(pred_real, 2)})
    primary_pred_file = CONFIG["future_pred_file"]
    try:
        output_df.to_csv(primary_pred_file, index=False, encoding='utf-8-sig')
        print(f"✅ 第四阶段执行完毕！预测数据已保存至 {primary_pred_file}")
    except Exception as e:
        ts = now_local().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(primary_pred_file)
        if not ext:
            ext = ".csv"
        backup_pred_file = f"{base}_backup_{ts}{ext}"
        try:
            output_df.to_csv(backup_pred_file, index=False, encoding='utf-8-sig')
            print(f"⚠️ 主预测文件写入失败: {e}")
            print(f"✅ 已写入备份预测文件: {backup_pred_file}")
        except Exception as e2:
            raise Exception(
                f"主预测文件与备份文件均写入失败。主文件: {primary_pred_file} ({e}); "
                f"备份文件: {backup_pred_file} ({e2})"
            )


# ==============================================================================
# ================= 6. 自动化调度器与入口 (SCHEDULER & MAIN) ======================
# ==============================================================================
def daily_auto_job():
    now_cn = now_local()
    current_time = now_cn.strftime('%Y-%m-%d %H:%M:%S')
    today_date_str = now_cn.strftime("%Y-%m-%d")

    print(f"\n\n{'*' * 60}")
    print(f"🏁 [{current_time}] 触发自动化流水线！")
    print(f"{'*' * 60}")

    # 判断今天是否需要训练模型
    need_training = False

    # 情况 1: 本地没有模型文件，强制训练一次
    if not os.path.exists(CONFIG["model_path"]):
        print("⚠️ 未检测到预训练模型文件，本次触发首次全量训练")
        need_training = True
    # 情况 2: 找不到上次训练时间记录，也强制训练一次
    elif not os.path.exists(CONFIG["last_train_record"]):
        need_training = True
    else:
        # 情况 3: 读取上次训练时间，判断距离今天是否达到阈值 (由 CONFIG 控制)
        with open(CONFIG["last_train_record"], "r") as f:
            last_train_str = f.read().strip()
        try:
            last_train_dt = datetime.strptime(last_train_str, "%Y-%m-%d")
            days_since_train = (now_cn.date() - last_train_dt.date()).days
            if days_since_train >= CONFIG["train_interval_days"]:
                print(f"📕 距离上次训练已过 {days_since_train} 天，达到触发阈值，今天将进行【模型重训练】。")
                need_training = True
            else:
                print(
                    f"⏩ 距离上次训练仅过 {days_since_train} 天 (需 {CONFIG['train_interval_days']} 天)，今天将【跳过】耗时的模型训练环节！")
        except:
            need_training = True

    try:
        # 第一阶段：每天雷打不动，获取今天的增量数据并拼接
        run_data_pipeline(today_date_str)

        # 根据上面的判断结果，决定是否执行训练阶段
        if need_training:
            run_training_pipeline()
            run_testing_pipeline()

            with open(CONFIG["last_train_record"], "w") as f:
                f.write(today_date_str)

        # 最后阶段：每天都执行，用现有模型预测未来功率
        run_prediction_pipeline()

        finish_time = now_local().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n🎀 [{finish_time}] 自动化流水线已全部完成！程序将挂机等待下次执行时间 ({CONFIG['daily_run_time']})...")
        print("*" * 60)
    except Exception as e:
        print(f"\n❌ 流水线执行期间发生致命错误: {e}")
        print("💡 提示：如果是因为 Token 导致的错误，请更新 CONFIG 里的 token。")


if __name__ == "__main__":

    print("⏰ 全自动 AI 光伏预测流水线已启动")
    print(f"🌍 当前时区配置：{CONFIG['timezone']}")
    print(f"⏰ 设定时间：每天 {CONFIG['daily_run_time']} 获取当日数据并生成预测。")
    print(f"🧠 模型进化：每 {CONFIG['train_interval_days']} 天自动进行一次重训练。")
    print("⚠️ 提示：请保持该终端窗口/进程处于后台运行状态。")

    # 刚启动时立即执行一遍
    daily_auto_job()

    # 设定每天固定时刻执行
    schedule.every().day.at(CONFIG["daily_run_time"]).do(daily_auto_job)

    # 守护进程
    while True:
        schedule.run_pending()
        time.sleep(60)
