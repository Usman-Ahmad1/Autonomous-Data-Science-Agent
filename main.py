import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from graph import graph
import base64
import time

# Try to import visualization
try:
    from src.visualization import VisualizationAgent
    VISUALIZATION_AVAILABLE = True
except ImportError:
    try:
        from visualization import VisualizationAgent
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
        class VisualizationAgent:
            CHART_TYPES = {
                'bar': 'Bar Chart',
                'pie': 'Pie Chart',
                'histogram': 'Histogram',
                'box': 'Box Plot',
                'scatter': 'Scatter Plot',
                'line': 'Line Chart',
                'heatmap': 'Correlation Heatmap',
                'violin': 'Violin Plot',
                'count': 'Count Plot',
                'pairplot': 'Pair Plot'
            }

load_dotenv()

st.set_page_config(page_title="Autonomous Data Science Agent", layout="wide")

# ============ CUSTOM CSS FOR HEADER ============
st.markdown("""
    <style>
        .main-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        .main-header h1 {
            margin: 0;
            font-size: 2rem;
        }
        .main-header p {
            margin: 0;
            opacity: 0.9;
        }
        .reset-btn {
            background-color: #ff6b6b;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        .reset-btn:hover {
            background-color: #ee5a24;
            transform: scale(1.05);
        }
        .stButton button {
            border-radius: 8px !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============ HEADER WITH RESET BUTTON ============
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🤖 Autonomous Data Science Agent")
    st.markdown("### End-to-End ML Workflow for Tabular Classification and Regression")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Reset Everything", type="primary", use_container_width=True, key="reset_top"):
        # Clear ALL session state
        for key in list(st.session_state.keys()):
            if key != "file_uploader_key":
                del st.session_state[key]
        st.session_state.file_uploader_key = st.session_state.get("file_uploader_key", 0) + 1
        st.rerun()

# ============ SESSION STATE INITIALIZATION ============
if "df" not in st.session_state:
    st.session_state.df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "columns_dropped" not in st.session_state:
    st.session_state.columns_dropped = []
if "result" not in st.session_state:
    st.session_state.result = None
if "cleaning_report" not in st.session_state:
    st.session_state.cleaning_report = None
if "feature_report" not in st.session_state:
    st.session_state.feature_report = None
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "model_results_df" not in st.session_state:
    st.session_state.model_results_df = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "model_problem_type" not in st.session_state:
    st.session_state.model_problem_type = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "model_selection_done" not in st.session_state:
    st.session_state.model_selection_done = False
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

# ============ RESET BUTTON (Small) ============
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Quick Actions")
if st.sidebar.button("🗑️ Clear All Data", use_container_width=True, key="clear_data"):
    for key in list(st.session_state.keys()):
        if key not in ["file_uploader_key", "df", "cleaned_df"]:
            del st.session_state[key]
    st.session_state.df = None
    st.session_state.cleaned_df = None
    st.session_state.columns_dropped = []
    st.session_state.result = None
    st.session_state.cleaning_report = None
    st.session_state.feature_report = None
    st.session_state.model_results = None
    st.session_state.model_results_df = None
    st.session_state.best_model_name = None
    st.session_state.best_model = None
    st.session_state.model_problem_type = None
    st.session_state.target_column = None
    st.session_state.model_selection_done = False
    st.rerun()

if st.sidebar.button("📤 Upload New File", use_container_width=True, key="upload_new"):
    st.session_state.file_uploader_key += 1
    st.session_state.df = None
    st.session_state.cleaned_df = None
    st.session_state.columns_dropped = []
    st.session_state.result = None
    st.session_state.cleaning_report = None
    st.session_state.feature_report = None
    st.session_state.model_results = None
    st.session_state.model_results_df = None
    st.session_state.best_model_name = None
    st.session_state.best_model = None
    st.session_state.model_problem_type = None
    st.session_state.target_column = None
    st.session_state.model_selection_done = False
    st.rerun()

# ============ FILE UPLOADER ============
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)", 
    type=["csv", "xlsx", "xls"],
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        if st.session_state.df is None:
            st.session_state.df = df.copy()
            st.session_state.cleaned_df = df.copy()
            st.session_state.columns_dropped = []
            st.session_state.cleaning_report = None
            st.session_state.feature_report = None
            st.session_state.result = None
            st.session_state.model_selection_done = False
        
        st.success(f"✅ Dataset loaded! Shape: {df.shape}")
        st.dataframe(df.head(10))
        
        if len(df) > 50000:
            st.error("❌ Dataset too large (>50,000 rows)")
            st.stop()
            
    except Exception as e:
        st.error(f"Error: {e}")

# ============ MAIN APPLICATION ============
if st.session_state.df is not None:
    # ============ STEP 1: COLUMN SELECTION & CLEANING ============
    st.subheader("🧹 Column Selection & Cleaning")
    
    st.info(f"📊 **Current dataset in memory:** {st.session_state.cleaned_df.shape if st.session_state.cleaned_df is not None else 'None'}")
    
    current_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", current_df.shape[0])
    with col2:
        st.metric("Total Columns", current_df.shape[1])
    with col3:
        if st.session_state.columns_dropped:
            st.metric("Dropped", len(st.session_state.columns_dropped))
        else:
            st.metric("Status", "Original")
    
    if st.session_state.columns_dropped:
        st.warning(f"🗑️ Dropped: {', '.join(st.session_state.columns_dropped)}")
    
    missing_percent = (current_df.isnull().sum() / len(current_df) * 100).round(2)
    high_missing_cols = missing_percent[missing_percent > 40].index.tolist()
    
    if high_missing_cols:
        st.warning(f"⚠️ High missing (>40%): {', '.join(high_missing_cols)}")
    else:
        st.success("✅ No high missing columns")
    
    drop_cols = st.multiselect(
        "Select columns to DROP:",
        options=list(current_df.columns),
        default=[]
    )
    
    if st.button("🗑️ APPLY DROPPING", type="primary", key="apply_dropping"):
        if drop_cols:
            new_df = current_df.drop(columns=drop_cols)
            st.session_state.cleaned_df = new_df
            st.session_state.columns_dropped = drop_cols
            st.session_state.result = None
            st.session_state.cleaning_report = None
            st.session_state.feature_report = None
            st.session_state.model_selection_done = False
            
            st.success(f"✅ DROPPED {len(drop_cols)} columns: {', '.join(drop_cols)}")
            st.info(f"📊 NEW SHAPE: {new_df.shape[0]} rows × {new_df.shape[1]} columns")
            st.dataframe(new_df.head())
            st.rerun()
        else:
            st.warning("⚠️ No columns selected to drop")
    
    # ============ STEP 2: ADVANCED CLEANING ============
    st.divider()
    st.subheader("🔧 Advanced Data Cleaning")
    st.caption("Fill missing values, handle outliers, and remove duplicates")
    
    with st.expander("⚙️ Advanced Cleaning Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_strategy = st.selectbox(
                "Missing Value Strategy",
                ["median", "mean", "mode", "knn", "forward_fill", "backward_fill", "drop_rows", "drop_columns"],
                index=0,
                key="missing_strategy"
            )
        
        with col2:
            outlier_method = st.selectbox(
                "Outlier Detection",
                ["iqr", "zscore", "isolation_forest"],
                index=0,
                key="outlier_method"
            )
        
        with col3:
            outlier_handling = st.selectbox(
                "Outlier Handling",
                ["cap", "remove", "transform"],
                index=0,
                key="outlier_handling"
            )
        
        col4, col5 = st.columns(2)
        with col4:
            remove_duplicates = st.checkbox("Remove Duplicate Rows", value=True, key="remove_duplicates")
        with col5:
            remove_duplicate_cols = st.checkbox("Remove Duplicate Columns", value=True, key="remove_duplicate_cols")
    
    if st.button("🧹 Apply Advanced Cleaning", type="secondary", key="apply_cleaning"):
        try:
            try:
                from cleaning import CleaningPipeline
            except ImportError:
                try:
                    from src.cleaning import CleaningPipeline
                except ImportError:
                    st.error("❌ Cleaning module not found.")
                    st.stop()
            
            with st.spinner("Applying advanced cleaning..."):
                pipeline = CleaningPipeline()
                cleaned_df = pipeline.clean(
                    current_df,
                    missing_strategy=missing_strategy,
                    outlier_method=outlier_method,
                    outlier_handling=outlier_handling,
                    remove_duplicates=remove_duplicates,
                    remove_duplicate_cols=remove_duplicate_cols
                )
                
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_report = pipeline.get_report()
                st.session_state.feature_report = None
                st.session_state.result = None
                st.session_state.model_selection_done = False
                
                st.success(f"✅ Advanced cleaning applied successfully!")
                st.info(f"📊 New shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
                
                with st.expander("📋 Advanced Cleaning Report", expanded=True):
                    report = st.session_state.cleaning_report
                    st.write(f"**Original Shape:** {report['original_shape']}")
                    st.write(f"**Cleaned Shape:** {report['cleaned_shape']}")
                    st.write("**Details:**")
                    for key, value in report['details'].items():
                        if isinstance(value, dict):
                            st.write(f"- **{key}:**")
                            for k, v in value.items():
                                st.write(f"  - {k}: {v}")
                        else:
                            st.write(f"- **{key}:** {value}")
                
                st.dataframe(cleaned_df.head(10))
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Advanced cleaning failed: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # ============ STEP 3: CURRENT DATASET STATE (After Cleaning) ============
    st.divider()
    st.subheader("📊 Current Dataset State")
    
    if st.session_state.cleaned_df is not None:
        analyze_df = st.session_state.cleaned_df
        st.success(f"✅ **Cleaned dataset ready:** {analyze_df.shape[0]} rows × {analyze_df.shape[1]} columns")
        if st.session_state.columns_dropped:
            st.info(f"🗑️ Dropped: {', '.join(st.session_state.columns_dropped)}")
        if st.session_state.cleaning_report:
            st.info(f"🧹 Advanced cleaning applied")
    else:
        analyze_df = st.session_state.df
        st.info(f"ℹ️ **Original dataset:** {analyze_df.shape[0]} rows × {analyze_df.shape[1]} columns")
    
    with st.expander("📋 View Current Columns"):
        st.write(list(analyze_df.columns))
    
    # ============ STEP 4: START ANALYSIS ============
    if st.button("🚀 START ANALYSIS", type="primary", key="start_analysis"):
        dataset_to_use = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
        
        st.info(f"📊 **Analyzing:** {dataset_to_use.shape[0]} rows × {dataset_to_use.shape[1]} columns")
        
        with st.spinner("Running Analysis... This may take a moment..."):
            try:
                initial_state = {
                    "messages": [{"role": "user", "content": f"Analyzing dataset with {dataset_to_use.shape[0]} rows and {dataset_to_use.shape[1]} columns."}],
                    "dataset": dataset_to_use,
                    "cleaned_dataset": dataset_to_use,
                    "next_agent": "EDA_Agent",
                    "cleaning_report": {
                        "columns_dropped": st.session_state.columns_dropped,
                        "final_shape": dataset_to_use.shape,
                        "advanced_cleaning": st.session_state.cleaning_report
                    },
                    "feature_report": st.session_state.feature_report,
                    "eda_complete": False 
                }
                
                print(f"🚀 Starting analysis with dataset shape: {dataset_to_use.shape}")
                result = graph.invoke(initial_state)
                print(f"✅ Analysis complete! Result keys: {result.keys()}")
                
                st.session_state.result = result
                st.success("✅ Analysis Complete!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # ============ STEP 5: DISPLAY RESULTS ============
    if st.session_state.result:
        st.divider()
        st.subheader("📊 Analysis Results")
        
        result = st.session_state.result
        
        if st.session_state.cleaned_df is not None:
            st.success(f"✅ Analyzed dataset: {st.session_state.cleaned_df.shape}")
            if st.session_state.columns_dropped:
                st.info(f"🗑️ Dropped: {', '.join(st.session_state.columns_dropped)}")
            if st.session_state.cleaning_report:
                st.info(f"🧹 Advanced cleaning was applied")
        
        if st.session_state.cleaning_report:
            with st.expander("🧹 Advanced Cleaning Report"):
                report = st.session_state.cleaning_report
                st.write(f"**Original Shape:** {report['original_shape']}")
                st.write(f"**Cleaned Shape:** {report['cleaned_shape']}")
                st.write("**Details:**")
                for key, value in report['details'].items():
                    if isinstance(value, dict):
                        st.write(f"- **{key}:**")
                        for k, v in value.items():
                            st.write(f"  - {k}: {v}")
                    else:
                        st.write(f"- **{key}:** {value}")
        
        # ============ EDA REPORT ============
        st.subheader("📊 EDA Report")
        if "eda_report" in result and result["eda_report"] is not None:
            with st.container():
                st.markdown(result["eda_report"])
        else:
            st.warning("⚠️ No EDA report generated. Please check the data.")
            with st.expander("🔍 Debug Info"):
                st.write("Result keys:", list(result.keys()))
        
        # ============ AGENT LOGS ============
        if "messages" in result:
            with st.expander("💬 Agent Logs"):
                seen = set()
                for msg in result["messages"]:
                    content = ""
                    if hasattr(msg, 'content'):
                        content = msg.content
                    elif isinstance(msg, dict) and "content" in msg:
                        content = msg["content"]
                    if content and content not in seen:
                        seen.add(content)
                        st.text(content)
        
                       # ============ STEP 6: INTERACTIVE VISUALIZATION ============
        st.divider()
        st.subheader("📊 Interactive Visualization")
        st.caption("Create custom visualizations using your dataset")
        
        with st.expander("🎨 Create Visualization", expanded=True):
            viz_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
            
            if viz_df is not None and not viz_df.empty:
                st.info(f"📊 Using dataset: {viz_df.shape[0]} rows × {viz_df.shape[1]} columns")
                
                # Initialize visualization agent
                viz_agent = VisualizationAgent()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    available_cols = viz_df.columns.tolist()
                    
                    # Show column types hint
                    col_types = []
                    for col in available_cols:
                        if pd.api.types.is_numeric_dtype(viz_df[col]):
                            col_types.append(f"{col} (📊 numeric)")
                        else:
                            col_types.append(f"{col} (📝 categorical)")
                    
                    viz_mode = st.radio(
                        "Visualization Mode:",
                        ["Single Column Analysis", "Compare Two Columns", "Multiple Columns Analysis"],
                        horizontal=True,
                        key="viz_mode"
                    )
                    
                    if viz_mode == "Single Column Analysis":
                        selected_cols = st.multiselect(
                            "Select column to visualize:",
                            options=available_cols,
                            max_selections=1,
                            key="viz_cols_single",
                            help="Select one column to visualize its distribution"
                        )
                        # Show column type info
                        if selected_cols:
                            col_type = "numeric" if pd.api.types.is_numeric_dtype(viz_df[selected_cols[0]]) else "categorical"
                            st.info(f"📌 Selected column type: **{col_type.upper()}**")
                    
                    elif viz_mode == "Compare Two Columns":
                        selected_cols = st.multiselect(
                            "Select two columns to compare:",
                            options=available_cols,
                            max_selections=2,
                            key="viz_cols_compare",
                            help="Select two columns for comparison"
                        )
                        # Show column types info
                        if len(selected_cols) == 2:
                            col1_type = "numeric" if pd.api.types.is_numeric_dtype(viz_df[selected_cols[0]]) else "categorical"
                            col2_type = "numeric" if pd.api.types.is_numeric_dtype(viz_df[selected_cols[1]]) else "categorical"
                            st.info(f"📌 Column types: **{selected_cols[0]} ({col1_type.upper()})** vs **{selected_cols[1]} ({col2_type.upper()})**")
                    
                    else:  # Multiple Columns Analysis
                        selected_cols = st.multiselect(
                            "Select multiple columns for analysis:",
                            options=available_cols,
                            max_selections=10,
                            key="viz_cols_multi",
                            help="Select 3+ columns for correlation or comparison"
                        )
                        # Show column types summary
                        if len(selected_cols) >= 2:
                            num_count = sum(1 for col in selected_cols if pd.api.types.is_numeric_dtype(viz_df[col]))
                            cat_count = len(selected_cols) - num_count
                            st.info(f"📌 Selected: {num_count} numeric columns, {cat_count} categorical columns")
                
                with col2:
                    # Get suggested charts based on selected columns
                    if selected_cols:
                        suggested_charts = viz_agent.get_suggested_charts(viz_df, selected_cols)
                    else:
                        suggested_charts = list(VisualizationAgent.CHART_TYPES.keys())
                    
                    # Show suggested charts with indicator
                    chart_options = list(VisualizationAgent.CHART_TYPES.keys())
                    
                    # Create display labels with suggestions
                    chart_display_names = []
                    for chart in chart_options:
                        chart_info = VisualizationAgent.CHART_TYPES[chart]
                        if chart in suggested_charts:
                            chart_display_names.append(f"⭐ {chart_info['name']} (suggested)")
                        else:
                            chart_display_names.append(chart_info['name'])
                    
                    # Find default index (first suggested chart if available)
                    default_index = 0
                    if suggested_charts and suggested_charts[0] in chart_options:
                        default_index = chart_options.index(suggested_charts[0])
                    
                    chart_type = st.selectbox(
                        "Select Chart Type:",
                        options=chart_options,
                        format_func=lambda x: VisualizationAgent.CHART_TYPES[x]['name'],
                        index=default_index,
                        key="chart_type"
                    )
                    
                    # Show chart description and compatibility
                    chart_info = VisualizationAgent.CHART_TYPES.get(chart_type, {})
                    st.caption(f"💡 {chart_info.get('description', '')}")
                    st.caption(f"🎯 Best for: {chart_info.get('best_for', '')}")
                    
                    # Show compatibility check
                    if selected_cols:
                        compat = viz_agent.get_chart_compatibility(viz_df, selected_cols, chart_type)
                        if compat['compatible']:
                            st.success("✅ Compatible with selected columns")
                        else:
                            st.error(compat['reason'])
                    
                    chart_title = st.text_input("Chart Title (optional):", placeholder="Enter custom title...", key="chart_title")
                
                if st.button("📊 Generate Visualization", type="primary", key="generate_viz"):
                    if not selected_cols:
                        st.warning("⚠️ Please select at least one column")
                    elif not VISUALIZATION_AVAILABLE:
                        st.error("❌ Visualization module not available. Install: pip install plotly")
                    else:
                        try:
                            from src.visualization import VisualizationAgent
                            with st.spinner("Generating visualization..."):
                                # Re-initialize agent for safety
                                viz_agent = VisualizationAgent()
                                
                                # Check compatibility before generating
                                compat = viz_agent.get_chart_compatibility(viz_df, selected_cols, chart_type)
                                if not compat['compatible']:
                                    st.error(compat['reason'])
                                else:
                                    result_viz = viz_agent.create_visualization(
                                        viz_df, chart_type=chart_type,
                                        columns=selected_cols,
                                        title=chart_title if chart_title else None
                                    )
                                    
                                    if result_viz['type'] == 'plotly':
                                        st.plotly_chart(result_viz['figure'], use_container_width=True)
                                        st.success("✅ Visualization generated successfully!")
                                    elif result_viz['type'] == 'matplotlib':
                                        img_data = base64.b64decode(result_viz['image'])
                                        st.image(img_data, use_container_width=True)
                                        st.success("✅ Visualization generated successfully!")
                                    else:
                                        st.error(f"❌ {result_viz.get('message', 'Error')}")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                            st.code(traceback.format_exc())
            else:
                st.info("ℹ️ No dataset available for visualization.")
        
        # ============ STEP 7: FEATURE ENGINEERING ============
        st.divider()
        st.subheader("🔧 Feature Engineering")
        st.caption("Create new features, transform existing ones, and select best features")
        
        fe_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
        
        st.info(f"📊 **Current dataset shape:** {fe_df.shape[0]} rows × {fe_df.shape[1]} columns")
        
        with st.expander("⚙️ Feature Engineering Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                create_interactions = st.checkbox("Create Interaction Features", value=True, key="create_interactions")
                create_ratios = st.checkbox("Create Ratio Features", value=False, key="create_ratios")
                create_polynomial = st.checkbox("Create Polynomial Features", value=False, key="create_polynomial")
                
            with col2:
                scale_features = st.checkbox("Scale Features", value=True, key="scale_features")
                scale_method = st.selectbox("Scaling Method", ["standard", "minmax", "robust"],
                                           index=0, disabled=not scale_features, key="scale_method")
                encode_categorical = st.checkbox("Encode Categorical Features", value=True, key="encode_categorical")
        
        if st.button("🔧 Apply Feature Engineering", type="secondary", key="apply_fe"):
            try:
                try:
                    from src.feature_engineering import FeatureEngineeringAgent
                except ImportError:
                    st.error("❌ Feature Engineering module not found.")
                    st.stop()
                
                with st.spinner("Applying feature engineering..."):
                    feature_eng = FeatureEngineeringAgent()
                    engineered_df = feature_eng.engineer_features(
                        fe_df,
                        create_interactions=create_interactions,
                        create_ratios=create_ratios,
                        create_polynomial=create_polynomial,
                        create_bins=False,
                        scale_features=scale_features,
                        scale_method=scale_method,
                        encode_categorical=encode_categorical,
                        encoding_method='label',
                        log_transform=False,
                        select_features=False
                    )
                    
                    st.session_state.cleaned_df = engineered_df
                    st.session_state.feature_report = feature_eng.get_report()
                    st.session_state.model_selection_done = False
                    
                    st.success(f"✅ Feature engineering applied successfully!")
                    st.info(f"📊 New shape: {engineered_df.shape[0]} rows × {engineered_df.shape[1]} columns")
                    
                    with st.expander("📋 Feature Engineering Report", expanded=True):
                        report = st.session_state.feature_report
                        st.write(f"**Original Shape:** {report['original_shape']}")
                        st.write(f"**New Shape:** {report['new_shape']}")
                        st.write(f"**Features Created:** {report['features_created']}")
                        st.write("**Details:**")
                        for key, value in report['details'].items():
                            if isinstance(value, dict) and value:
                                st.write(f"- **{key}:** {len(value.get('features', [])) if value.get('features') else 0} features created")
                                if value.get('features'):
                                    st.write(f"  - {', '.join(value['features'][:5])}")
                                    if len(value['features']) > 5:
                                        st.write(f"  - ... and {len(value['features']) - 5} more")
                    
                    st.dataframe(engineered_df.head(10))
                    st.info("💡 **Note:** Analysis results are still available above. The final dataset below has been updated with new features.")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Feature engineering failed: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # ============ STEP 8: MODEL SELECTION ============
        st.divider()
        st.subheader("🧪 Model Selection")
        st.caption("Automatically select the best machine learning model for your data")
        
        model_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
        
        # ============ DISPLAY SAVED MODEL RESULTS IF AVAILABLE ============
        if st.session_state.model_selection_done and st.session_state.best_model_name is not None:
            st.success(f"✅ Model Selection Complete!")
            st.info(f"🏆 Best Model: **{st.session_state.best_model_name}**")
            st.info(f"📊 Problem Type: **{st.session_state.model_problem_type.upper()}**")
            st.info(f"🎯 Target Column: **{st.session_state.target_column}**")
            
            with st.expander("📊 Model Performance Comparison", expanded=True):
                if st.session_state.model_results_df is not None:
                    st.dataframe(st.session_state.model_results_df.style.highlight_max(subset=['Test Score']))
            
            with st.expander("🏆 Best Model Details", expanded=True):
                st.write(f"**Model Name:** {st.session_state.best_model_name}")
                st.write(f"**Problem Type:** {st.session_state.model_problem_type.upper()}")
                
                if st.session_state.model_results and st.session_state.best_model_name in st.session_state.model_results:
                    best_result = st.session_state.model_results[st.session_state.best_model_name]
                    if st.session_state.model_problem_type == 'classification':
                        st.write(f"**Test Accuracy:** {best_result['test_score']:.4f}")
                        st.write(f"**CV Mean Accuracy:** {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})")
                    else:
                        st.write(f"**Test R2 Score:** {best_result['test_score']:.4f}")
                        st.write(f"**CV Mean R2:** {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})")
                    st.write(f"**Training Time:** {best_result['training_time']:.2f} seconds")
                
                if st.session_state.best_model is not None:
                    st.write("**Model Parameters:**")
                    st.code(str(st.session_state.best_model.get_params()))
            
            if st.button("🔄 Run New Model Selection", type="secondary", key="run_new_model"):
                st.session_state.model_selection_done = False
                st.session_state.best_model_name = None
                st.session_state.best_model = None
                st.session_state.model_results = None
                st.session_state.model_results_df = None
                st.rerun()
            
            st.divider()
        
        # ============ MODEL SELECTION SETTINGS ============
        if not st.session_state.model_selection_done:
            if model_df is not None and not model_df.empty:
                with st.expander("⚙️ Model Selection Settings", expanded=True):
                    target_column = st.selectbox(
                        "Select Target Column (What you want to predict):",
                        options=model_df.columns.tolist(),
                        help="Choose the column you want to predict",
                        key="target_column_select"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        problem_type = st.selectbox(
                            "Problem Type:",
                            ["auto", "classification", "regression"],
                            index=0,
                            help="Auto-detect or manually specify",
                            key="problem_type_select"
                        )
                    
                    with col2:
                        test_size = st.slider(
                            "Test Size:",
                            min_value=0.1,
                            max_value=0.4,
                            value=0.2,
                            step=0.05,
                            help="Portion of data to use for testing",
                            key="test_size_slider"
                        )
                
                if st.button("🧪 Select Best Model", type="primary", key="select_model"):
                    if target_column is None:
                        st.warning("⚠️ Please select a target column")
                    else:
                        try:
                            import sys
                            import os
                            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                            
                            try:
                                from src.model_selection import ModelSelector, ModelComparator
                            except ImportError:
                                from model_selection import ModelSelector, ModelComparator
                            
                            with st.spinner("Testing models... This may take a moment..."):
                                clean_model_df = model_df.dropna(subset=[target_column])
                                
                                if len(clean_model_df) < 50:
                                    st.warning(f"⚠️ Only {len(clean_model_df)} rows available. Need at least 50 rows for reliable model selection.")
                                else:
                                    X = clean_model_df.drop(columns=[target_column])
                                    y = clean_model_df[target_column]
                                    
                                    selector = ModelSelector(problem_type=problem_type)
                                    results = selector.select_models(
                                        X, y, 
                                        test_size=test_size, 
                                        cv_folds=min(5, max(2, len(X)//20))
                                    )
                                    
                                    results_df = selector.get_model_results()
                                    best_model_name = selector.get_best_model_name()
                                    best_model = selector.get_best_model()
                                    
                                    st.session_state.model_results = results
                                    st.session_state.model_results_df = results_df
                                    st.session_state.best_model_name = best_model_name
                                    st.session_state.best_model = best_model
                                    st.session_state.model_problem_type = selector.problem_type
                                    st.session_state.target_column = target_column
                                    st.session_state.model_selection_done = True
                                    
                                    st.success(f"✅ Model Selection Complete!")
                                    st.info(f"🏆 Best Model: **{best_model_name}**")
                                    st.info(f"📊 Problem Type: **{selector.problem_type.upper()}**")
                                    st.rerun()
                                    
                        except ImportError as e:
                            st.error(f"❌ Model Selection module not found.")
                            st.info("💡 Make sure the following files exist:")
                            st.code("""
src/model_selection/
├── __init__.py
├── model_selector.py
├── model_evaluator.py
└── model_comparator.py
                            """)
                        except Exception as e:
                            st.error(f"❌ Model selection failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.info("ℹ️ No dataset available for model selection.")
        
        # ============ STEP 9: FINAL DATASET ============
        st.divider()
        st.subheader("📋 Final Dataset After All Steps")
        
        applied_steps = []
        if st.session_state.columns_dropped:
            applied_steps.append("Column Dropping")
        if st.session_state.cleaning_report:
            applied_steps.append("Advanced Cleaning")
        if st.session_state.feature_report:
            applied_steps.append("Feature Engineering")
        if st.session_state.model_selection_done:
            applied_steps.append("Model Selection")
        
        if applied_steps:
            st.caption(f"✅ Applied: {' → '.join(applied_steps)}")
        else:
            st.caption("This is the dataset after column dropping and cleaning")
        
        final_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", final_df.shape[0])
        with col2:
            st.metric("Total Columns", final_df.shape[1])
        with col3:
            original_cols = st.session_state.df.shape[1] if st.session_state.df is not None else 0
            changes = original_cols - final_df.shape[1]
            if st.session_state.feature_report:
                st.metric("Total Features", final_df.shape[1])
            else:
                st.metric("Columns Changed", f"{changes} removed" if changes > 0 else "No changes")
        
        with st.expander("📊 View Final Dataset", expanded=True):
            st.dataframe(final_df.head(20))
            st.write("**Columns in final dataset:**")
            st.write(", ".join(final_df.columns.tolist()))
            
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Final Dataset (CSV)",
                data=csv,
                file_name="final_cleaned_dataset.csv",
                mime="text/csv",
                key="download_final"
            )
        
        # ============ STEP 10: SUMMARY OF CHANGES ============
        with st.expander("📋 Summary of All Changes Applied"):
            changes_list = []
            
            if st.session_state.columns_dropped:
                changes_list.append(f"🗑️ Dropped {len(st.session_state.columns_dropped)} columns: {', '.join(st.session_state.columns_dropped)}")
            
            if st.session_state.cleaning_report:
                report = st.session_state.cleaning_report
                changes_list.append(f"🧹 Advanced cleaning applied: {report['original_shape']} → {report['cleaned_shape']}")
                for key, value in report['details'].items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if k == 'outliers_handled' and v > 0:
                                changes_list.append(f"  - Handled {v} outliers")
                            elif k == 'removed_rows' and v > 0:
                                changes_list.append(f"  - Removed {v} duplicate rows")
            
            if st.session_state.feature_report:
                report = st.session_state.feature_report
                changes_list.append(f"🔧 Feature engineering applied")
                changes_list.append(f"  - {report['original_shape']} → {report['new_shape']}")
                changes_list.append(f"  - Created {report['features_created']} new features")
                for key, value in report['details'].items():
                    if isinstance(value, dict) and value and 'features' in value:
                        if value.get('features'):
                            changes_list.append(f"  - {key}: {len(value['features'])} features created")
            
            if st.session_state.model_selection_done:
                changes_list.append(f"🧪 Model Selection completed")
                changes_list.append(f"  - Best Model: {st.session_state.best_model_name}")
                changes_list.append(f"  - Problem Type: {st.session_state.model_problem_type.upper()}")
                changes_list.append(f"  - Target Column: {st.session_state.target_column}")
                if st.session_state.model_results and st.session_state.best_model_name in st.session_state.model_results:
                    best_result = st.session_state.model_results[st.session_state.best_model_name]
                    if st.session_state.model_problem_type == 'classification':
                        changes_list.append(f"  - Accuracy: {best_result['test_score']:.4f}")
                    else:
                        changes_list.append(f"  - R2 Score: {best_result['test_score']:.4f}")
            
            if not changes_list:
                changes_list.append("No changes were applied to the dataset.")
            
            for change in changes_list:
                st.write(change)
        
        # ============ RESET BUTTON AT BOTTOM ============
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Reset Everything", type="primary", use_container_width=True, key="reset_bottom"):
                for key in list(st.session_state.keys()):
                    if key not in ["file_uploader_key"]:
                        del st.session_state[key]
                st.session_state.file_uploader_key = st.session_state.get("file_uploader_key", 0) + 1
                st.rerun()