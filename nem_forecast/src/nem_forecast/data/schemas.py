"""
CSV / DataFrame 列约定（可与 AEMO 导出列映射到此规范）。

必需列（加载后应统一为小写别名）：
- TIMESTAMP_COL: 时间戳（tz-aware 或 naive，建议 UTC）
- REGION_COL: 区域代码 NSW/VIC/QLD/SA
- RRP_COL: 区域参考电价 AUD/MWh（5min）
- DEMAND_COL: 区域负荷需求（MW 或其它一致单位）

可选：
- 供需结构、可再生能源出力等，预留前缀 SUPPLY_
"""

TIMESTAMP_COL = "timestamp"
REGION_COL = "region"
RRP_COL = "rrp"
DEMAND_COL = "demand"

# 占位：气象列前缀（merge_weather_placeholder 期望）
WEATHER_PREFIX = "wx_"

REQUIRED_COLUMNS = {TIMESTAMP_COL, REGION_COL, RRP_COL, DEMAND_COL}


def normalize_column_names(mapping: dict[str, str]) -> dict[str, str]:
    """将原始 CSV 列名映射到规范名；调用方在 loaders 中传入。"""
    return {str(k).strip(): str(v).strip() for k, v in mapping.items()}
