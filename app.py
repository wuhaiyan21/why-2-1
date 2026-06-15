import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

from forecast import (
    load_sales_data,
    load_product_master,
    run_forecast,
    FORECAST_DAYS,
    MIN_HISTORY_DAYS,
    SEASONAL_MONTHS,
    SEASONAL_WEIGHT
)

st.set_page_config(
    page_title="销售预测工具",
    page_icon="📊",
    layout="wide"
)

st.title("📊 销售预测小工具")
st.markdown("上传历史销售数据和商品主数据，自动生成未来30天的销售预测")

with st.sidebar:
    st.header("📁 数据上传")

    st.subheader("历史销售数据")
    sales_file = st.file_uploader(
        "上传销售CSV（包含：日期、商品编码、销量）",
        type=["csv"],
        key="sales_upload"
    )

    st.subheader("商品主数据")
    master_file = st.file_uploader(
        "上传商品主数据CSV（包含：商品编码、品类名称）",
        type=["csv"],
        key="master_upload"
    )

    st.markdown("---")

    st.subheader("⚙️ 预测参数")
    st.info(f"预测天数：{FORECAST_DAYS} 天\n最少历史天数：{MIN_HISTORY_DAYS} 天\n\n季节性月份：{', '.join(str(m) + '月' for m in SEASONAL_MONTHS)}\n季节加权：{SEASONAL_WEIGHT}x")

    st.markdown("---")

    if st.button("生成示例数据", type="secondary"):
        import numpy as np
        from datetime import datetime, timedelta

        np.random.seed(42)
        products = [f'P{i:03d}' for i in range(1, 11)]
        categories = ['食品', '饮料', '日用品', '电器']

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        sales_records = []
        for i, prod in enumerate(products):
            base = np.random.randint(20, 200)
            cat = categories[i % len(categories)]
            for d in dates:
                if len(products) - i <= 2:
                    if d < datetime(2024, 12, 1):
                        continue
                trend = 0.05 * (d - start_date).days / 365
                seasonality = 1.3 if d.month in [11, 12] else 1.0
                noise = np.random.normal(0, base * 0.15)
                sales = max(0, int(base * (1 + trend) * seasonality + noise))
                sales_records.append({
                    '日期': d.strftime('%Y-%m-%d'),
                    '商品编码': prod,
                    '销量': sales
                })

        sample_sales = pd.DataFrame(sales_records)
        sample_sales_csv = sample_sales.to_csv(index=False)

        master_records = []
        for i, prod in enumerate(products):
            cat = categories[i % len(categories)]
            master_records.append({
                '商品编码': prod,
                '品类名称': cat
            })
        sample_master = pd.DataFrame(master_records)
        sample_master_csv = sample_master.to_csv(index=False)

        st.download_button(
            "📥 下载示例销售数据",
            data=sample_sales_csv,
            file_name="sample_sales.csv",
            mime="text/csv"
        )
        st.download_button(
            "📥 下载示例主数据",
            data=sample_master_csv,
            file_name="sample_master.csv",
            mime="text/csv"
        )

sales_df = None
master_df = None

if sales_file is not None:
    try:
        sales_df = load_sales_data(sales_file)
        st.sidebar.success(f"✓ 销售数据加载成功\n共 {len(sales_df):,} 条记录\n{sales_df['product_code'].nunique()} 个商品")
    except Exception as e:
        st.sidebar.error(f"销售数据加载失败：{str(e)}")

if master_file is not None:
    try:
        master_df = load_product_master(master_file)
        st.sidebar.success(f"✓ 主数据加载成功\n共 {len(master_df):,} 条记录\n{master_df['category'].nunique()} 个品类")
    except Exception as e:
        st.sidebar.error(f"主数据加载失败：{str(e)}")

if sales_file is not None and sales_df is not None:
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 预测结果", "📊 品类汇总", "⚠️ 低可信预警", "📋 数据预览"])

    with st.spinner("正在进行预测分析..."):
        has_master = master_df is not None and len(master_df) > 0
        forecast_df, summary = run_forecast(sales_df, master_df, has_master_data=has_master)

    if 'warnings' in summary and len(summary['warnings']) > 0:
        for warning in list(set(summary['warnings'])):
            st.warning(warning)

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总商品数", summary['total_products'])
        col2.metric("独立建模商品", summary['individual_model_count'])
        col3.metric("品类平均商品", summary['category_model_count'])
        col4.metric("低可信商品", summary['low_confidence_products'], delta_color="inverse")

        st.markdown("### 🔍 商品筛选")

        product_options = sorted(forecast_df['product_code'].unique())
        selected_product = st.selectbox(
            "选择商品查看详细预测",
            product_options,
            index=0
        )

        product_forecast = forecast_df[forecast_df['product_code'] == selected_product].copy()

        model_type = product_forecast['model_type'].iloc[0]
        category = product_forecast['category'].iloc[0]

        prod_info = summary['product_model_info'][summary['product_model_info']['product_code'] == selected_product]
        history_days = prod_info['history_days'].values[0] if len(prod_info) > 0 else 0
        record_count = prod_info['record_count'].values[0] if len(prod_info) > 0 else 0

        col_info1, col_info2 = st.columns(2)
        col_info1.info(f"**商品编码**：{selected_product}")
        col_info2.info(f"**品类**：{category}")
        col_info3, col_info4 = st.columns(2)
        col_info3.success(f"**建模方式**：{model_type}")
        low_conf_days = product_forecast['is_low_confidence'].sum()
        if low_conf_days > 0:
            col_info4.warning(f"**低可信天数**：{low_conf_days} / {FORECAST_DAYS} 天")
        else:
            col_info4.success(f"**低可信天数**：{low_conf_days} / {FORECAST_DAYS} 天")

        col_info5, col_info6 = st.columns(2)
        col_info5.info(f"**历史日历跨度**：{history_days} 天")
        col_info6.info(f"**历史记录数**：{record_count} 条")

        st.markdown("### 📉 预测趋势图")

        plot_df = product_forecast.copy()
        plot_df['日期'] = plot_df['date']
        plot_df['预测销量'] = plot_df['forecast']
        plot_df['置信下限'] = plot_df['ci_lower']
        plot_df['置信上限'] = plot_df['ci_upper']

        fig = px.line(
            plot_df,
            x='日期',
            y='预测销量',
            title=f'商品 {selected_product} - 未来{FORECAST_DAYS}天销量预测',
            labels={'预测销量': '销量', '日期': '日期'}
        )

        fig.add_scatter(
            x=plot_df['日期'],
            y=plot_df['置信上限'],
            fill=None,
            mode='lines',
            line=dict(color='rgba(0,100,255,0.3)'),
            name='置信上限'
        )
        fig.add_scatter(
            x=plot_df['日期'],
            y=plot_df['置信下限'],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(0,100,255,0.3)'),
            name='置信下限'
        )

        fig.update_layout(
            hovermode='x unified',
            showlegend=True,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 预测明细")

        display_df = product_forecast.copy()
        display_df['日期'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['预测销量'] = display_df['forecast'].round(2)
        display_df['置信区间'] = display_df.apply(
            lambda x: f"[{x['ci_lower']:.2f}, {x['ci_upper']:.2f}]",
            axis=1
        )
        display_df['可信度'] = display_df['is_low_confidence'].map({True: '⚠️ 低可信', False: '✓ 正常'})
        display_df['建模方式'] = display_df['model_type']

        st.dataframe(
            display_df[['日期', '预测销量', '置信区间', '可信度', '建模方式']],
            use_container_width=True,
            hide_index=True
        )

        csv_buffer = io.StringIO()
        forecast_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

        st.download_button(
            label="📥 导出全部预测结果 (CSV)",
            data=csv_buffer.getvalue(),
            file_name="sales_forecast_result.csv",
            mime="text/csv",
            type="primary"
        )

    with tab2:
        st.markdown("### 📊 品类汇总对比")

        cat_summary = summary['category_summary'].copy()

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_bar = px.bar(
                cat_summary,
                x='category',
                y='total_forecast',
                title='各品类预测总销量',
                color='category',
                text_auto='.0f',
                labels={'category': '品类', 'total_forecast': '预测总销量'}
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_chart2:
            fig_count = px.bar(
                cat_summary,
                x='category',
                y='product_count',
                title='各品类商品数量',
                color='category',
                text_auto='.0f',
                labels={'category': '品类', 'product_count': '商品数量'}
            )
            fig_count.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_count, use_container_width=True)

        st.markdown("### � 历史日均 vs 预测日均对比")

        compare_df = cat_summary.copy()
        compare_df = compare_df.rename(columns={
            'category': '品类',
            'historical_daily_avg': '历史日均销量',
            'forecast_daily_avg': '预测日均销量'
        })

        compare_melted = compare_df.melt(
            id_vars=['品类'],
            value_vars=['历史日均销量', '预测日均销量'],
            var_name='类型',
            value_name='日均销量'
        )

        fig_compare = px.bar(
            compare_melted,
            x='品类',
            y='日均销量',
            color='类型',
            barmode='group',
            title='各品类历史日均销量 vs 预测日均销量对比',
            text_auto='.1f'
        )
        fig_compare.update_layout(
            height=500,
            xaxis_title='品类',
            yaxis_title='日均销量',
            legend_title='数据类型'
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("### 📋 品类明细")
        display_cols = ['category', 'product_count', 'historical_daily_avg', 'forecast_daily_avg', 'total_forecast', 'low_confidence_days']
        display_cat = cat_summary[display_cols].copy()
        display_cat.columns = ['品类', '商品数量', '历史日均销量', '预测日均销量', '预测总销量', '低可信天数']
        display_cat['历史日均销量'] = display_cat['历史日均销量'].round(2)
        display_cat['预测日均销量'] = display_cat['预测日均销量'].round(2)
        display_cat['预测总销量'] = display_cat['预测总销量'].round(2)

        display_cat['预测涨幅'] = np.where(
            display_cat['历史日均销量'] > 0,
            ((display_cat['预测日均销量'] - display_cat['历史日均销量']) / display_cat['历史日均销量'] * 100).round(1).astype(str) + '%',
            'N/A'
        )

        st.dataframe(
            display_cat,
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.markdown("### ⚠️ 低可信预测汇总")
        st.info("置信区间宽度超过预测值50%的日期标记为低可信。")

        low_conf_details = summary['low_confidence_details']

        if len(low_conf_details) > 0:
            st.warning(f"共有 {len(low_conf_details)} 个商品存在低可信预测")

            display_low = low_conf_details.copy()
            display_low = display_low[['product_code', 'category', 'model_type', 'low_conf_days']]
            display_low.columns = ['商品编码', '品类', '建模方式', '低可信天数']
            display_low = display_low.sort_values('低可信天数', ascending=False)

            st.dataframe(
                display_low,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 📉 低可信商品预测详情")

            for _, row in low_conf_details.head(10).iterrows():
                with st.expander(f"商品 {row['product_code']} - {row['category']} - 低可信天数: {row['low_conf_days']}天"):
                    prod_df = forecast_df[forecast_df['product_code'] == row['product_code']].copy()
                    prod_df['日期'] = prod_df['date'].dt.strftime('%Y-%m-%d')
                    prod_df['预测销量'] = prod_df['forecast'].round(2)
                    prod_df['置信区间'] = prod_df.apply(
                        lambda x: f"[{x['ci_lower']:.2f}, {x['ci_upper']:.2f}]",
                        axis=1
                    )
                    prod_df['可信度'] = prod_df['is_low_confidence'].map({True: '⚠️ 低可信', False: '✓ 正常'})

                    st.dataframe(
                        prod_df[['日期', '预测销量', '置信区间', '可信度']],
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.success("🎉 所有商品的预测置信度均符合要求！")

    with tab4:
        st.markdown("### 📋 销售数据预览")

        st.info(f"数据时间范围：{sales_df['date'].min().strftime('%Y-%m-%d')} ~ {sales_df['date'].max().strftime('%Y-%m-%d')}")

        preview_df = sales_df.copy()
        preview_df.columns = ['日期', '商品编码', '销量']
        preview_df['日期'] = preview_df['日期'].dt.strftime('%Y-%m-%d')

        st.dataframe(
            preview_df.head(100),
            use_container_width=True,
            hide_index=True
        )

        if master_df is not None:
            st.markdown("### 📋 商品主数据预览")
            master_preview = master_df.copy()
            master_preview.columns = ['商品编码', '品类名称']
            st.dataframe(
                master_preview.head(100),
                use_container_width=True,
                hide_index=True
            )

else:
    st.info("👈 请在左侧上传历史销售数据CSV文件开始使用")

    st.markdown("### 🔧 功能说明")

    col_feat1, col_feat2, col_feat3 = st.columns(3)

    with col_feat1:
        st.info("**📈 智能预测**\n\n基于历史数据，使用多项式回归模型预测未来30天销量，支持季节性调整。")

    with col_feat2:
        st.info("**📊 品类汇总**\n\n按品类汇总预测结果，柱状图直观对比各品类销售情况。")

    with col_feat3:
        st.info("**⚠️ 置信区间**\n\n每条预测附带置信区间，自动标记低可信预测，辅助决策。")

    st.markdown("### 📝 CSV格式要求")

    col_form1, col_form2 = st.columns(2)

    with col_form1:
        st.markdown("**销售数据 CSV**")
        st.code("日期,商品编码,销量\n2024-01-01,P001,100\n2024-01-02,P001,120\n2024-01-01,P002,80")

    with col_form2:
        st.markdown("**商品主数据 CSV**")
        st.code("商品编码,品类名称\nP001,食品\nP002,饮料\nP003,日用品")
