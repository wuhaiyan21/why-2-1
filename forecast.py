import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

SEASONAL_MONTHS = [11, 12]
SEASONAL_WEIGHT = 1.3
MIN_HISTORY_DAYS = 30
MIN_DATA_COVERAGE = 0.6
FORECAST_DAYS = 30


def load_sales_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    col_map = {}
    for col in df.columns:
        if '日期' in col or 'date' in col.lower():
            if 'date' not in col_map.values():
                col_map[col] = 'date'
        elif '商品' in col or 'product' in col.lower() or 'sku' in col.lower() or 'code' in col.lower():
            if 'product_code' not in col_map.values():
                col_map[col] = 'product_code'
        elif '销量' in col or 'sale' in col.lower() or 'qty' in col.lower():
            if 'sales' not in col_map.values():
                col_map[col] = 'sales'

    if len(col_map) < 3:
        raise ValueError(f"CSV列名不正确。需要包含'日期'、'商品编码'、'销量'三列。当前列：{list(df.columns)}")

    df = df.rename(columns=col_map)
    df = df[['date', 'product_code', 'sales']]

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0)
    df['sales'] = df['sales'].clip(lower=0)
    df['product_code'] = df['product_code'].astype(str)

    df = df.groupby(['product_code', 'date'])['sales'].sum().reset_index()
    df = df.sort_values(['product_code', 'date']).reset_index(drop=True)

    return df


def load_product_master(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    col_map = {}
    for col in df.columns:
        if '商品' in col or 'product' in col.lower() or 'sku' in col.lower() or 'code' in col.lower():
            if 'product_code' not in col_map.values():
                col_map[col] = 'product_code'
        elif '品类' in col or 'category' in col.lower() or 'cat' in col.lower():
            if 'category' not in col_map.values():
                col_map[col] = 'category'

    if len(col_map) < 2:
        raise ValueError(f"商品主数据CSV列名不正确。需要包含'商品编码'和'品类名称'两列。当前列：{list(df.columns)}")

    df = df.rename(columns=col_map)
    df = df[['product_code', 'category']]
    df['product_code'] = df['product_code'].astype(str)
    df['category'] = df['category'].astype(str)
    df['category'] = df['category'].fillna('未分类')

    return df


def add_date_features(df, date_col='date'):
    df = df.copy()
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['time_index'] = (df[date_col] - df[date_col].min()).dt.days
    return df


def train_single_product(product_df):
    product_df = product_df.sort_values('date').reset_index(drop=True)

    date_span_days = (product_df['date'].max() - product_df['date'].min()).days + 1
    record_count = len(product_df)
    data_coverage = record_count / date_span_days if date_span_days > 0 else 0

    if (date_span_days < MIN_HISTORY_DAYS
            or record_count < MIN_HISTORY_DAYS
            or data_coverage < MIN_DATA_COVERAGE):
        return None

    product_df = add_date_features(product_df)

    feature_cols = ['time_index', 'day_of_week', 'day_of_month', 
                    'month', 'is_weekend']

    X = product_df[feature_cols].values
    y = product_df['sales'].values

    model = Ridge(alpha=10.0)
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred
    residual_std = np.std(residuals) if np.std(residuals) > 0 else (np.mean(y) * 0.1 if np.mean(y) > 0 else 1.0)

    avg_sales = np.mean(y)

    return {
        'model': model,
        'feature_cols': feature_cols,
        'residual_std': residual_std,
        'avg_sales': avg_sales,
        'history_days': date_span_days,
        'record_count': record_count,
        'data_coverage': data_coverage,
        'min_date': product_df['date'].min(),
        'max_date': product_df['date'].max(),
        'min_date_global': product_df['date'].min()
    }


def forecast_single_product(model_info, forecast_start_date, seasonal_months=None, seasonal_weight=None):
    if seasonal_months is None:
        seasonal_months = SEASONAL_MONTHS
    if seasonal_weight is None:
        seasonal_weight = SEASONAL_WEIGHT

    model = model_info['model']
    feature_cols = model_info['feature_cols']
    residual_std = model_info['residual_std']
    min_date = model_info['min_date_global']

    forecast_dates = pd.date_range(start=forecast_start_date, periods=FORECAST_DAYS, freq='D')

    forecast_df = pd.DataFrame({'date': forecast_dates})
    forecast_df = add_date_features(forecast_df)
    forecast_df['time_index'] = (forecast_df['date'] - min_date).dt.days

    X_forecast = forecast_df[feature_cols].values
    y_pred = model.predict(X_forecast)

    seasonal_mask = forecast_df['month'].isin(seasonal_months)
    y_pred[seasonal_mask] = y_pred[seasonal_mask] * seasonal_weight

    y_pred = np.maximum(y_pred, 0)

    ci_lower = y_pred - 1.96 * residual_std
    ci_upper = y_pred + 1.96 * residual_std

    ci_lower = np.maximum(ci_lower, 0)

    forecast_df['forecast'] = y_pred
    forecast_df['ci_lower'] = ci_lower
    forecast_df['ci_upper'] = ci_upper
    forecast_df['ci_width'] = ci_upper - ci_lower
    forecast_df['is_low_confidence'] = forecast_df['ci_width'] > (forecast_df['forecast'] * 0.5)
    forecast_df['model_type'] = '独立建模'

    return forecast_df


def compute_category_avg(sales_df, product_master, valid_products=None):
    merged = sales_df.merge(product_master, on='product_code', how='left')
    merged['category'] = merged['category'].fillna('未分类')

    if valid_products is not None and len(valid_products) > 0:
        merged = merged[merged['product_code'].isin(valid_products)]

    category_stats = merged.groupby('category').agg(
        total_sales=('sales', 'sum'),
        product_count=('product_code', 'nunique'),
        days_count=('date', 'nunique')
    ).reset_index()

    category_stats['avg_daily_per_product'] = np.where(
        (category_stats['product_count'] > 0) & (category_stats['days_count'] > 0),
        category_stats['total_sales'] / category_stats['days_count'] / category_stats['product_count'],
        0
    )

    daily_sales = merged.groupby(['category', 'date'])['sales'].sum().reset_index()
    daily_std = daily_sales.groupby('category')['sales'].std().reset_index()
    daily_std.columns = ['category', 'daily_std']

    category_stats = category_stats.merge(daily_std, on='category', how='left')
    category_stats['std_per_product'] = np.where(
        category_stats['product_count'] > 0,
        category_stats['daily_std'] / category_stats['product_count'],
        category_stats['avg_daily_per_product'] * 0.3
    )
    category_stats['std_per_product'] = category_stats['std_per_product'].fillna(
        category_stats['avg_daily_per_product'] * 0.3
    )

    zero_mask = category_stats['std_per_product'] == 0
    category_stats.loc[zero_mask, 'std_per_product'] = category_stats.loc[zero_mask, 'avg_daily_per_product'] * 0.3

    return category_stats


def forecast_by_category(category_stats, category_name, forecast_start_date, ci_multiplier=1.0, seasonal_months=None, seasonal_weight=None):
    if seasonal_months is None:
        seasonal_months = SEASONAL_MONTHS
    if seasonal_weight is None:
        seasonal_weight = SEASONAL_WEIGHT

    cat_row = category_stats[category_stats['category'] == category_name]

    if len(cat_row) == 0:
        avg_val = 0.0
        std_val = 1.0
    else:
        avg_val = cat_row['avg_daily_per_product'].values[0]
        std_val = cat_row['std_per_product'].values[0]

    if std_val <= 0:
        std_val = avg_val * 0.3 if avg_val > 0 else 1.0

    std_val = std_val * ci_multiplier

    forecast_dates = pd.date_range(start=forecast_start_date, periods=FORECAST_DAYS, freq='D')
    forecast_df = pd.DataFrame({'date': forecast_dates})
    forecast_df['month'] = forecast_df['date'].dt.month

    seasonal_mask = forecast_df['month'].isin(seasonal_months)
    forecast = np.full(FORECAST_DAYS, avg_val, dtype=float)
    forecast[seasonal_mask] = forecast[seasonal_mask] * seasonal_weight

    half_ci = 1.96 * std_val
    ci_lower = forecast - half_ci
    ci_upper = forecast + half_ci

    ci_lower = np.maximum(ci_lower, 0)

    forecast_df['forecast'] = forecast
    forecast_df['ci_lower'] = ci_lower
    forecast_df['ci_upper'] = ci_upper
    forecast_df['ci_width'] = ci_upper - ci_lower
    forecast_df['is_low_confidence'] = forecast_df['ci_width'] > (forecast_df['forecast'] * 0.5)
    forecast_df['model_type'] = '品类平均'

    forecast_df = forecast_df.drop(columns=['month'])

    return forecast_df


def run_forecast(sales_df, product_master_df, has_master_data=False, seasonal_months=None, seasonal_weight=None):
    if seasonal_months is None:
        seasonal_months = SEASONAL_MONTHS
    if seasonal_weight is None:
        seasonal_weight = SEASONAL_WEIGHT

    sales_df = sales_df.copy()
    sales_df['product_code'] = sales_df['product_code'].astype(str)

    if product_master_df is None or len(product_master_df) == 0:
        all_products = sales_df['product_code'].unique()
        product_master_df = pd.DataFrame({
            'product_code': all_products,
            'category': ['默认品类'] * len(all_products)
        })
    else:
        product_master_df = product_master_df.copy()
        product_master_df['product_code'] = product_master_df['product_code'].astype(str)
        product_master_df['category'] = product_master_df['category'].fillna('未分类')

        sales_products = set(sales_df['product_code'].unique())
        master_products = set(product_master_df['product_code'].unique())
        missing = sales_products - master_products
        if missing:
            missing_df = pd.DataFrame({
                'product_code': list(missing),
                'category': ['未分类'] * len(missing)
            })
            product_master_df = pd.concat([product_master_df, missing_df], ignore_index=True)

    products = sales_df['product_code'].unique()
    max_date = sales_df['date'].max()
    forecast_start = max_date + timedelta(days=1)

    product_model_map = {}
    individual_products = []
    category_products = []

    for product in products:
        product_data = sales_df[sales_df['product_code'] == product]

        date_span_days = (product_data['date'].max() - product_data['date'].min()).days + 1
        record_count = len(product_data)
        data_coverage = record_count / date_span_days if date_span_days > 0 else 0

        is_individual = (
            date_span_days >= MIN_HISTORY_DAYS
            and record_count >= MIN_HISTORY_DAYS
            and data_coverage >= MIN_DATA_COVERAGE
        )

        if is_individual:
            model_info = train_single_product(product_data)
            if model_info is not None:
                product_model_map[product] = {
                    'type': 'individual',
                    'model_info': model_info,
                    'history_days': date_span_days,
                    'record_count': record_count,
                    'data_coverage': data_coverage
                }
                individual_products.append(product)
            else:
                product_model_map[product] = {
                    'type': 'category',
                    'history_days': date_span_days,
                    'record_count': record_count,
                    'data_coverage': data_coverage
                }
                category_products.append(product)
        else:
            product_model_map[product] = {
                'type': 'category',
                'history_days': date_span_days,
                'record_count': record_count,
                'data_coverage': data_coverage
            }
            category_products.append(product)

    category_stats = compute_category_avg(sales_df, product_master_df, valid_products=individual_products)

    all_forecasts = []
    warnings = []

    for product in products:
        info = product_model_map[product]

        if info['type'] == 'individual':
            forecast_df = forecast_single_product(
                info['model_info'], forecast_start,
                seasonal_months=seasonal_months,
                seasonal_weight=seasonal_weight
            )
        else:
            cat_row = product_master_df[product_master_df['product_code'] == product]
            category = cat_row['category'].values[0] if len(cat_row) > 0 else '未分类'

            record_count = info['record_count']
            if record_count < MIN_HISTORY_DAYS:
                ci_multiplier = max(2.0, MIN_HISTORY_DAYS / max(record_count, 1))
            else:
                ci_multiplier = max(1.5, MIN_DATA_COVERAGE / max(info['data_coverage'], 0.01))

            if not has_master_data:
                warnings.append(
                    f"商品 {product} 历史数据不足（记录{record_count}条/覆盖度{info['data_coverage']:.0%}），"
                    f"且未上传商品主数据，无法使用真实品类均值回退。"
                    f"建议上传商品主数据CSV以获得更准确的预测。"
                )

            forecast_df = forecast_by_category(
                category_stats, category, forecast_start,
                ci_multiplier=ci_multiplier,
                seasonal_months=seasonal_months,
                seasonal_weight=seasonal_weight
            )

            if not has_master_data:
                forecast_df['model_type'] = '默认回退'
            else:
                forecast_df['model_type'] = '品类平均'

        forecast_df['product_code'] = product

        cat_row = product_master_df[product_master_df['product_code'] == product]
        category = cat_row['category'].values[0] if len(cat_row) > 0 else '未分类'
        forecast_df['category'] = category

        all_forecasts.append(forecast_df)

    result_df = pd.concat(all_forecasts, ignore_index=True)

    result_df = result_df[[
        'product_code', 'category', 'date', 'forecast',
        'ci_lower', 'ci_upper', 'ci_width',
        'is_low_confidence', 'model_type'
    ]]

    result_df = result_df.sort_values(['product_code', 'date']).reset_index(drop=True)

    summary = generate_summary(result_df, product_model_map, category_stats, sales_df=sales_df)
    summary['warnings'] = list(set(warnings))
    summary['has_master_data'] = has_master_data
    summary['individual_products'] = individual_products
    summary['product_model_map'] = product_model_map

    return result_df, summary


def generate_summary(forecast_df, product_model_map, category_stats, sales_df=None):
    total_products = forecast_df['product_code'].nunique()

    product_day_low_conf = forecast_df[forecast_df['is_low_confidence']]
    low_conf_products = product_day_low_conf['product_code'].nunique()

    category_summary = forecast_df.groupby('category').agg(
        product_count=('product_code', 'nunique'),
        total_forecast=('forecast', 'sum'),
        low_confidence_days=('is_low_confidence', 'sum')
    ).reset_index()

    category_summary['forecast_daily_avg'] = category_summary['total_forecast'] / FORECAST_DAYS

    if sales_df is not None:
        individual_products = [p for p, info in product_model_map.items() if info['type'] == 'individual']

        if len(individual_products) > 0:
            sales_individual = sales_df[sales_df['product_code'].isin(individual_products)]
        else:
            sales_individual = sales_df.iloc[0:0]

        sales_with_cat = sales_individual.merge(
            forecast_df[['product_code', 'category']].drop_duplicates(),
            on='product_code',
            how='left'
        )
        historical_stats = sales_with_cat.groupby('category').agg(
            historical_total=('sales', 'sum'),
            historical_days=('date', 'nunique'),
            historical_product_count=('product_code', 'nunique')
        ).reset_index()
        historical_stats['historical_daily_avg'] = np.where(
            (historical_stats['historical_days'] > 0) & (historical_stats['historical_product_count'] > 0),
            historical_stats['historical_total'] / historical_stats['historical_days'] / historical_stats['historical_product_count'],
            0
        )
        category_summary = category_summary.merge(
            historical_stats[['category', 'historical_daily_avg', 'historical_total', 'historical_days', 'historical_product_count']],
            on='category',
            how='left'
        )
        category_summary['historical_daily_avg'] = category_summary['historical_daily_avg'].fillna(0)
        category_summary['historical_total'] = category_summary['historical_total'].fillna(0)
        category_summary['historical_days'] = category_summary['historical_days'].fillna(0)
        category_summary['historical_product_count'] = category_summary['historical_product_count'].fillna(0)
    else:
        category_summary['historical_daily_avg'] = 0
        category_summary['historical_total'] = 0
        category_summary['historical_days'] = 0
        category_summary['historical_product_count'] = 0

    individual_count = sum(1 for v in product_model_map.values() if v['type'] == 'individual')
    category_count = sum(1 for v in product_model_map.values() if v['type'] == 'category')

    product_model_info = forecast_df.groupby('product_code').agg({
        'model_type': 'first',
        'category': 'first',
        'is_low_confidence': 'sum'
    }).reset_index()
    product_model_info.columns = ['product_code', 'model_type', 'category', 'low_conf_days']

    product_details = []
    for product, info in product_model_map.items():
        product_details.append({
            'product_code': product,
            'history_days': info.get('history_days', 0),
            'record_count': info.get('record_count', 0),
            'data_coverage': info.get('data_coverage', 0)
        })
    product_details_df = pd.DataFrame(product_details)
    product_model_info = product_model_info.merge(product_details_df, on='product_code', how='left')

    low_conf_details = product_model_info[product_model_info['low_conf_days'] > 0].copy()

    return {
        'total_products': total_products,
        'low_confidence_products': low_conf_products,
        'category_summary': category_summary,
        'individual_model_count': individual_count,
        'category_model_count': category_count,
        'product_model_info': product_model_info,
        'low_confidence_details': low_conf_details,
        'category_stats': category_stats
    }


def run_backtest(sales_df, product_code, backtest_start, backtest_end, seasonal_months=None, seasonal_weight=None):
    if seasonal_months is None:
        seasonal_months = SEASONAL_MONTHS
    if seasonal_weight is None:
        seasonal_weight = SEASONAL_WEIGHT

    product_data = sales_df[sales_df['product_code'] == product_code].copy()
    product_data = product_data.sort_values('date').reset_index(drop=True)

    if len(product_data) == 0:
        return None, "无销售数据"

    backtest_start = pd.to_datetime(backtest_start)
    backtest_end = pd.to_datetime(backtest_end)

    train_data = product_data[product_data['date'] < backtest_start]

    date_span_days = (train_data['date'].max() - train_data['date'].min()).days + 1 if len(train_data) > 0 else 0
    record_count = len(train_data)
    data_coverage = record_count / date_span_days if date_span_days > 0 else 0

    if (date_span_days < MIN_HISTORY_DAYS
            or record_count < MIN_HISTORY_DAYS
            or data_coverage < MIN_DATA_COVERAGE):
        return None, f"训练数据不足（{record_count}条记录，覆盖度{data_coverage:.0%}），至少需要{MIN_HISTORY_DAYS}天有效数据"

    model_info = train_single_product(train_data)
    if model_info is None:
        return None, "模型训练失败"

    backtest_dates = pd.date_range(start=backtest_start, end=backtest_end, freq='D')
    backtest_df = pd.DataFrame({'date': backtest_dates})

    actual_data = product_data[product_data['date'].between(backtest_start, backtest_end)][['date', 'sales']]
    backtest_df = backtest_df.merge(actual_data, on='date', how='left')
    backtest_df['sales'] = backtest_df['sales'].fillna(0)

    backtest_df_features = backtest_df.copy()
    backtest_df_features = add_date_features(backtest_df_features)
    backtest_df_features['time_index'] = (backtest_df_features['date'] - model_info['min_date_global']).dt.days

    X_backtest = backtest_df_features[model_info['feature_cols']].values
    y_pred = model_info['model'].predict(X_backtest)

    seasonal_mask = backtest_df_features['month'].isin(seasonal_months)
    y_pred[seasonal_mask] = y_pred[seasonal_mask] * seasonal_weight
    y_pred = np.maximum(y_pred, 0)

    backtest_df['forecast'] = y_pred
    backtest_df['abs_error'] = np.abs(backtest_df['forecast'] - backtest_df['sales'])

    backtest_df['error_rate'] = np.where(
        backtest_df['sales'] > 0,
        backtest_df['abs_error'] / backtest_df['sales'],
        np.nan
    )
    backtest_df['is_actual_zero'] = backtest_df['sales'] == 0

    valid_error_rates = backtest_df[~backtest_df['is_actual_zero']]['error_rate']
    mae = backtest_df['abs_error'].mean()
    max_deviation = backtest_df['abs_error'].max()
    avg_error_rate = valid_error_rates.mean() if len(valid_error_rates) > 0 else np.nan

    backtest_df['product_code'] = product_code

    stats = {
        'backtest_days': len(backtest_df),
        'non_zero_days': len(valid_error_rates),
        'zero_days': backtest_df['is_actual_zero'].sum(),
        'mae': mae,
        'max_deviation': max_deviation,
        'avg_error_rate': avg_error_rate,
        'total_actual': backtest_df['sales'].sum(),
        'total_forecast': backtest_df['forecast'].sum()
    }

    return backtest_df, stats
