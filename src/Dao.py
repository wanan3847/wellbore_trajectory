import pandas as pd
from sqlalchemy import create_engine
import os

# ================== 1. 自动处理路径 (核心优化) ==================
# 获取当前脚本所在目录 (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (pythonProject1)
project_root = os.path.dirname(current_dir)


def get_data_path(file_name):
    """根据文件名自动生成在 data 文件夹下的绝对路径"""
    return os.path.join(project_root, "data", file_name)


# ================== 2. 数据库连接 ==================
# 建议将连接字符串提取出来，方便修改
DB_URL = "mysql+pymysql://root:12345678@127.0.0.1:3306/wellbore_trajectory"
engine = create_engine(DB_URL)

# 列名映射字典 (共用，减少重复代码)
column_mapping = {
    "转换后JH": "well_id",
    "XJS": "depth",
    "JX": "inclination",
    "FW": "azimuth",
    "LJCZJS": "tvd",
    "JX_design": "design_incl",
    "FW_design": "design_azim",
    "LJCZJS_design": "design_tvd",
    "关键点": "label"
}

# ================== 3. 训练集处理 ==================
train_path = get_data_path("train.csv")

if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    train_df = train_df.rename(columns=column_mapping)

    train_df.to_sql(
        name="well_trajectory_train",
        con=engine,
        if_exists="replace",
        index=False
    )
    print(f"✅ 成功从 {train_path} 导入 well_trajectory_train")
else:
    print(f"❌ 错误: 未找到训练集文件 {train_path}")

# ================== 4. 验证集处理 ==================
val_path = get_data_path("validation_without_label.csv")

if os.path.exists(val_path):
    val_df = pd.read_csv(val_path)
    val_df = val_df.rename(columns=column_mapping)

    val_df.to_sql(
        name="well_trajectory_val",
        con=engine,
        if_exists="replace",
        index=False
    )
    print(f"✅ 成功从 {val_path} 导入 well_trajectory_val")
else:
    print(f"❌ 错误: 未找到验证集文件 {val_path}")