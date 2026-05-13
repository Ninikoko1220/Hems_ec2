import json
import os

with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
    config = json.load(f)

import os
os.chdir("/home/ec2-user/hems")

import requests
import json
import csv
import os
from datetime import datetime


def get_and_save_tax_price():
    # ================= 配置区域 (自适应读取) =================
    # 1. 读取本地统一配置文件 config.json
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            sys_cfg = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：未找到配置文件 {config_path}，请先创建它。")
        return
    except Exception as e:
        print(f"❌ 错误：读取配置文件失败: {e}")
        return

    # 2. 从配置中提取参数
    host = config.get("BASE_URL", "http://esybackend.esysunhome.com:7074")
    token = config.get("TOKEN")
    device_id = config.get("DEVICE_ID")
    sn = config.get("SN")

    # 接口路径与固定参数
    api_path = "/inner/ai/ai-model/ai-tax-included"
    url = host + api_path
    price_company = "Belgium"  # 比利时电价
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Body 请求参数 (raw-json)
    payload = {
        "priceCompany": price_company,
        "dataTime": current_date,
        "sn": sn
    }
    # ========================================================

    headers = {
        "Content-Type": "application/json",
        "Authorization": token,
        "User-Agent": "Python-Client"
    }

    # 继续保持作为 Query 参数的 deviceId
    params = {"deviceId": device_id}

    try:
        print(f"🚀 正在通过最新接口请求电价数据 (日期: {current_date})...")

        # 发送 POST 请求
        response = requests.post(url, headers=headers, params=params,json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()

        # 校验响应编码 (0 为成功)
        if result.get("code") == 0:
            data_list = result.get("data")
            if not data_list:
                print("⚠️ 接口调用成功但返回数据为空，请确认该 SN 权限或日期是否有数据。")
                return

            print(f"✅ 获取成功！共拿到 {len(data_list)} 条电价数据。")

            # 文件名：electricity_price_日期.csv
            filename = os.path.join('data', f"electricity_price_{current_date}.csv")
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

            # 写入 CSV
            with open(save_path, mode='w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)

                # 写入表头
                writer.writerow(["开始时间", "结束时间", "基础电价", "含税电价", "峰谷状态"])

                # 写入数据行
                for item in data_list:
                    # peakValley: 峰谷电价差值
                    # taxIncludedPrice: 含税价
                    writer.writerow([
                        item.get("startTime"),
                        item.get("endTime"),
                        item.get("price"),
                        item.get("taxIncludedPrice"),
                        "高峰" if item.get("peakValley") == 1 else "平/谷"
                    ])

            print(f"💾 数据已成功保存到: {save_path}")

        else:
            # 失败处理逻辑
            print(f"❌ 获取失败，错误代码: {result.get('code')}")
            print(f"❌ 服务器消息: {result.get('msg')}")

    except Exception as e:
        print(f"❌ 发生网络或代码执行错误: {e}")


if __name__ == "__main__":
    get_and_save_tax_price()
