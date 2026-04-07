import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import random
import os
import warnings
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings("ignore")

# ===================== 全局参数配置 =====================
CONFIG = {
    # 必须参数
    "input_file": "sishuiliuyu(1).xlsx",      # 输入文件路径
    "output_file": "sihui_predictions.xlsx",  # 输出文件路径

    # 数据参数
    "window_size": 24,                    # 时间序列窗口大小
    "test_size": 0.2,                     # 验证集比例
    "random_seed": 42,                    # 基础随机种子

    # 模型结构
    "transformer": {
        "nhead": 8,                       # 注意力头数
        "num_layers": 3,                  # Transformer层数
        "dim_feedforward": 512,           # 前馈网络维度
        "dropout": 0.1                    # 丢弃率
    },
    "standard_conv": {                    # 修改后的普通卷积配置
        "channels": [64, 128],            # 卷积通道数
        "kernel_size": 3                  # 卷积核大小
    },
    "regressor": {
        "hidden_sizes": [256, 128],       # 回归头隐藏层结构
        "activation": "gelu",             # 激活函数
        "dropout": 0.3                    # 回归头丢弃率
    },

    # 训练参数
    "training": {
        "batch_size": 512,                # 批次大小
        "max_epochs": 500,                # 单次最大epoch
        "patience": 30,                   # 早停耐心值
        "learning_rate": 6e-5,            # 初始学习率
        "optimizer": "adamw",             # 优化器
        "weight_decay": 1e-4,             # 权重衰减
        "grad_clip": 1.0                  # 梯度裁剪
    },

    # 种子搜索参数
    "seed_search": {
        "enable": True,                   # 启用种子搜索
        "r2_threshold": 0.6,              # R²阈值
        "rmse_threshold": 0.2,           # RMSE阈值
        "rrmse_threshold": 0.2,          # 新增RRMSE阈值（小数形式，0.15即15%）
        "max_trials": 1,                  # 最大尝试次数
        "seed_range": [0, 1000000]        # 种子生成范围
    },

    # 系统参数
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_file": "sihui_training_log.csv"  # 训练日志文件
}

# ===================== 模型定义 =====================
class EnhancedTimeSeriesModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # 普通卷积模块
        self.conv_layers = nn.ModuleList()
        in_channels = input_size
        for out_channels in CONFIG["standard_conv"]["channels"]:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=CONFIG["standard_conv"]["kernel_size"],
                        padding=1  # 固定padding保持尺寸
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(CONFIG["transformer"]["dropout"])
                )
            )
            in_channels = out_channels

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=CONFIG["transformer"]["nhead"],
            dim_feedforward=CONFIG["transformer"]["dim_feedforward"],
            dropout=CONFIG["transformer"]["dropout"],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=CONFIG["transformer"]["num_layers"]
        )

        # 回归头
        self.regressor = self._build_regressor(in_channels)

    def _build_regressor(self, input_dim):
        layers = []
        current_dim = input_dim
        activation = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU()
        }[CONFIG["regressor"]["activation"]]

        for hidden_dim in CONFIG["regressor"]["hidden_sizes"]:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                activation,
                nn.Dropout(CONFIG["regressor"]["dropout"])
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 转换为通道优先 (batch, features, time)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.permute(0, 2, 1)  # 恢复时序优先 (batch, time, features)
        x = self.transformer(x)
        return self.regressor(x[:, -1, :])  # 取最后一个时间步

# ===================== 核心函数 =====================
def calculate_rmse(y_true, y_pred):
    """计算均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_rrmse(y_true, y_pred):
    """计算相对均方根误差（小数形式）"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mean_y = np.mean(np.abs(y_true))
    return (rmse / mean_y)

def set_global_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data():
    """数据预处理流程（读取第三个sheet，删除最后一列；过滤none列）"""
    # 读取Excel（第三个sheet，索引为2）
    try:
        df = pd.read_excel(CONFIG["input_file"], header=None, sheet_name=2, engine='openpyxl')
    except Exception as e1:
        print(f"使用openpyxl引擎失败: {e1}")
        try:
            df = pd.read_excel(CONFIG["input_file"], header=None, sheet_name=2, engine='xlrd')
        except Exception as e2:
            print(f"使用xlrd引擎失败: {e2}")
            import gc
            gc.collect()
            df = pd.read_excel(CONFIG["input_file"], header=None, sheet_name=2)

    print(f"读取第三个sheet，原始数据形状: {df.shape}")

    # 删除最后一列（保持你原逻辑）
    df = df.iloc[:, :-1]
    print(f"删除最后一列后，数据形状: {df.shape}")

    # 1) 把“none”类文本统一视为缺失
    none_tokens = {"none", "None", "NONE", "null", "NULL", "nan", "NaN", ""}
    df = df.replace(list(none_tokens), np.nan)

    # 2) 数值化：非数值 -> NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # 3) 删除“差特征列”：缺失占比过高 / 近似常数（第0列是目标，不参与删列）
    miss_ratio = df.isna().mean()
    nunique = df.nunique(dropna=True)

    bad_cols = []
    for c in df.columns[1:]:
        if miss_ratio[c] > 0.30:
            bad_cols.append(c)
        elif nunique[c] <= 1:
            bad_cols.append(c)

    if bad_cols:
        df = df.drop(columns=bad_cols)
        print(f"[Info] dropped feature cols: {bad_cols}")

    # 4) 目标缺失：删行；特征缺失：中位数填充
    df = df.dropna(subset=[0])
    feat_cols = df.columns[1:]
    if len(feat_cols) == 0:
        raise ValueError("所有特征列都被判定为无效（缺失过高/常数）。请检查Excel特征列或放宽阈值。")
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())

    print(f"清理/填充后，数据剩余 {len(df)} 行，特征列数={len(feat_cols)}")

    # 标准化处理（第一列为目标值，后面列为特征）
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    features = x_scaler.fit_transform(df.iloc[:, 1:].values)
    targets = y_scaler.fit_transform(df.iloc[:, 0].values.reshape(-1, 1))

    # 构造时间序列窗口
    window_size = CONFIG["window_size"]
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i-window_size:i])
        y.append(targets[i, 0])

    print(f"使用时间序列窗口（窗口大小={window_size}），样本数: {len(X)}")

    return (
        torch.from_numpy(np.array(X, dtype=np.float32)),
        torch.from_numpy(np.array(y, dtype=np.float32)).reshape(-1, 1),
        x_scaler,
        y_scaler,
        df.iloc[window_size:, 0].values
    )

def calculate_rmse_percent(y_true, y_pred):
    """计算RMSE的百分比形式（相对于平均值）"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mean_y = np.mean(np.abs(y_true))
    return (rmse / mean_y)

def train_single_seed(seed, X_train, y_train, X_val, y_val, y_scaler):
    """单种子训练流程"""
    set_global_seed(seed)
    model = EnhancedTimeSeriesModel(X_train.shape[2]).to(CONFIG["device"])

    optimizer = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop
    }[CONFIG["training"]["optimizer"]](
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"]
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )
    criterion = nn.HuberLoss()
    scaler = GradScaler()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True,
        pin_memory=True
    )

    best_r2 = -np.inf
    best_rmse = np.inf
    best_rrmse = np.inf
    patience_counter = 0

    for epoch in range(CONFIG["training"]["max_epochs"]):
        model.train()

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            X_batch = X_batch.to(CONFIG["device"], non_blocking=True)
            y_batch = y_batch.to(CONFIG["device"], non_blocking=True)

            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

        # 验证评估
        model.eval()
        with torch.no_grad():
            preds = model(X_val.to(CONFIG["device"])).cpu().numpy()

        y_val_orig = y_scaler.inverse_transform(y_val.numpy())
        preds_orig = y_scaler.inverse_transform(preds)

        current_r2 = r2_score(y_val_orig, preds_orig)
        current_rmse = calculate_rmse(y_val_orig, preds_orig)
        current_rrmse = calculate_rrmse(y_val_orig, preds_orig)
        current_rmse_pct = calculate_rmse_percent(y_val_orig, preds_orig)
        scheduler.step(current_r2)

        improvement = False
        if (current_r2 > best_r2 and current_rmse < best_rmse and current_rrmse < best_rrmse):
            improvement = True
        elif (current_r2 == best_r2 and current_rmse < best_rmse and current_rrmse < best_rrmse):
            improvement = True

        if improvement:
            best_r2 = current_r2
            best_rmse = current_rmse
            best_rrmse = current_rrmse
            patience_counter = 0
            torch.save(model.state_dict(), "sihui_temp_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["training"]["patience"]:
                break

    return best_r2, best_rmse, best_rrmse, current_rmse_pct

def main():
    if "cuda" in CONFIG["device"] and not torch.cuda.is_available():
        CONFIG["device"] = "cpu"
        print("检测到CUDA不可用，已自动切换至CPU模式")

    X, y, x_scaler, y_scaler, original_y = prepare_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_seed"]
    )

    log_data = []
    best_r2 = -np.inf
    best_rmse = np.inf
    best_rrmse = np.inf
    trial = 0

    while trial < CONFIG["seed_search"]["max_trials"]:
        current_seed = random.randint(*CONFIG["seed_search"]["seed_range"])
        current_r2, current_rmse, current_rrmse, current_rmse_pct = train_single_seed(
            current_seed, X_train, y_train, X_val, y_val, y_scaler
        )

        log_entry = {
            "trial": trial+1,
            "seed": current_seed,
            "r2_score": current_r2,
            "rmse_score": current_rmse,
            "rrmse_score": current_rrmse,
            "status": "达标" if (
                current_r2 >= CONFIG["seed_search"]["r2_threshold"] and
                current_rmse <= CONFIG["seed_search"]["rmse_threshold"] and
                current_rrmse <= CONFIG["seed_search"]["rrmse_threshold"]
            ) else "继续"
        }
        log_data.append(log_entry)

        if (current_r2 > best_r2 and current_rmse < best_rmse and current_rrmse < best_rrmse):
            best_r2 = current_r2
            best_rmse = current_rmse
            best_rrmse = current_rrmse
            os.replace("sihui_temp_model.pt", "sihui_best_model.pt")
            print(f"试验 {trial+1}: 种子 {current_seed} => R² {current_r2*100:.2f}% | RMSE {current_rmse_pct*100:.2f}% | RRMSE {current_rrmse*100:.2f}% (新最佳)")
        else:
            print(f"试验 {trial+1}: 种子 {current_seed} => R² {current_r2*100:.2f}% | RMSE {current_rmse_pct*100:.2f}% | RRMSE {current_rrmse*100:.2f}%")

        pd.DataFrame(log_data).to_csv(CONFIG["log_file"], index=False)

        if (CONFIG["seed_search"]["enable"] and
            best_r2 >= CONFIG["seed_search"]["r2_threshold"] and
            best_rmse <= CONFIG["seed_search"]["rmse_threshold"] and
            best_rrmse <= CONFIG["seed_search"]["rrmse_threshold"]):
            print(f"\n达到三指标阈值 R²>={CONFIG['seed_search']['r2_threshold']*100:.0f}% | RMSE<={CONFIG['seed_search']['rmse_threshold']} | RRMSE<={CONFIG['seed_search']['rrmse_threshold']*100:.0f}%")
            break

        trial += 1

    print("\n开始使用最佳模型进行预测...")
    model = EnhancedTimeSeriesModel(X.shape[2]).to(CONFIG["device"])
    model.load_state_dict(torch.load("sihui_best_model.pt"))
    model.eval()

    with torch.no_grad():
        raw_pred = model(X.to(CONFIG["device"])).cpu().numpy()

    final_pred = y_scaler.inverse_transform(raw_pred).flatten()

    final_r2 = r2_score(original_y, final_pred)
    final_rmse = calculate_rmse(original_y, final_pred)
    final_rrmse = calculate_rrmse(original_y, final_pred)
    final_rmse_pct = calculate_rmse_percent(original_y, final_pred)

    result_df = pd.DataFrame({
        "实际第一列值": original_y,
        "预测第一列值": final_pred
    })
    result_df.to_excel(CONFIG["output_file"], index=False)
    print(f"已保存 {len(result_df)} 行预测结果")

    print("\n" + "="*60)
    print(f"【第一列预测结果验证】")
    print("-"*60)
    print(f"预测目标: 第一列（列索引0）")
    print(f"样本数量: {len(final_pred)}")
    print(f"R²分数: {final_r2*100:.2f}%  (拟合优度)")
    print(f"RMSE: {final_rmse_pct*100:.2f}%  (相对误差)")
    print(f"RRMSE: {final_rrmse*100:.2f}%  (相对均方根误差)")
    print(f"RMSE绝对值: {final_rmse:.4f}  (原始单位)")
    print("-"*60)
    print(f"预测结果已保存: {CONFIG['output_file']}")
    print("="*60)

if __name__ == "__main__":
    main()
