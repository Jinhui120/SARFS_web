import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 修复NumPy bool弃用问题
if not hasattr(np, 'bool'):
    np.bool = bool

# 设置页面标题和布局
st.set_page_config(
    page_title="Obesity Level Prediction Model Based on SARFS",
    page_icon="⚖️",
    layout="wide"
)

# 定义全局变量
global feature_names, feature_dict, variable_descriptions

# 特征名称和描述 - 肥胖预测模型（根据实际数据集）
feature_names = [
    'gender', 'age', 'height', 'weight', 'family_history_with_overweight',
    'favc', 'fcvc', 'ncp', 'caec', 'ch2o', 'faf', 'tue', 'calc', 'mtrans', 'bmi'
]

feature_names_en = [
    'Gender', 'Age', 'Height', 'Weight', 'Family History with Overweight',
    'Frequent Consumption of High Caloric Food', 'Frequency of Consumption of Vegetables',
    'Number of Main Meals', 'Consumption of Food Between Meals', 'Daily Water Consumption',
    'Physical Activity Frequency', 'Time Using Technology Devices', 'Alcohol Consumption',
    'Transportation Used', 'BMI'
]

feature_dict = dict(zip(feature_names, feature_names_en))

# Variable descriptions - according to your mapping
variable_descriptions = {
    'gender': 'Gender (0=Male, 1=Female)',
    'age': 'Age in years',
    'height': 'Height in meters',
    'weight': 'Weight in kilograms',
    'family_history_with_overweight': 'Family history with overweight (0=yes, 1=no)',
    'favc': 'Frequent consumption of high caloric food (0=yes, 1=no)',
    'fcvc': 'Frequency of consumption of vegetables (1=Never, 2=Sometimes, 3=Always)',
    'ncp': 'Number of main meals (1-4 meals)',
    'caec': 'Consumption of food between meals (0=Always, 1=Frequently, 2=no, 3=Sometimes)',
    'ch2o': 'Daily water consumption (1=Less than 1L, 2=1-2L, 3=More than 2L)',
    'faf': 'Physical activity frequency (0=No exercise, 1=1-2 days, 2=2-4 days, 3=4-5 days)',
    'tue': 'Time using technology devices (0=0-2 hours, 1=3-5 hours, 2=More than 5 hours)',
    'calc': 'Alcohol consumption (0=no, 1=Sometimes, 2=Frequently, 3=Always)',
    'mtrans': 'Transportation used (0=Automobile, 1=Bike, 2=Motorbike, 3=Public_Transportation, 4=Walking)',
    'bmi': 'BMI index (kg/m²)'
}

# Obesity level descriptions - according to your mapping
obesity_levels = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Obesity Type I",
    3: "Obesity Type II",
    4: "Obesity Type III",
    5: "Overweight Level I",
    6: "Overweight Level II"
}

# 加载RF模型
@st.cache_resource
def load_model():
    # 加载随机森林模型
    with open('rf_model_optimized.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# 主应用
def main():
    global feature_names, feature_dict, variable_descriptions

    # 侧边栏标题
    st.sidebar.title("Obesity Level Prediction Model Based on SARFS")

    # 添加系统说明到侧边栏
    st.sidebar.markdown("""
    # System Description

    ## About This System
    This is an Obesity Level prediction system based on Random Forest algorithm, which predicts obesity levels by analyzing lifestyle and dietary habits.

    ## Prediction Results
    The system predicts obesity levels:
    - 0: Insufficient Weight
    - 1: Normal Weight
    - 2: Obesity Type I
    - 3: Obesity Type II
    - 4: Obesity Type III
    - 5: Overweight Level I
    - 6: Overweight Level II

    ## How to Use
    1. Fill in personal information and lifestyle habits
    2. Click the prediction button to generate prediction results
    3. View prediction results and feature importance analysis

    ## Important Notes
    - Please ensure accurate information input
    - All fields need to be filled
    - Numeric fields require number input
    - Selection fields require choosing from options
    """)

    # 添加变量说明到侧边栏
    with st.sidebar.expander("Variable Descriptions"):
        for feature in feature_names:
            st.markdown(f"**{feature_dict[feature]}**: {variable_descriptions[feature]}")

    # 主页面标题
    st.title("Obesity Level Prediction Model Based on SARFS")
    st.markdown("### Lifestyle and Dietary Habits Assessment")
    
    # 加载模型
    try:
        model = load_model()
        st.sidebar.success("SARFS Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {e}")
        return

    # 创建输入表单
    st.sidebar.header("Personal Information Input")

    # 创建三列布局用于输入
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Basic Information")
        gender = st.selectbox(f"{feature_dict['gender']}", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        age = st.number_input(f"{feature_dict['age']} (years)", min_value=14, max_value=61, value=25)
        height = st.number_input(f"{feature_dict['height']} (m)", min_value=1.45, max_value=1.98, value=1.70, step=0.01)
        weight = st.number_input(f"{feature_dict['weight']} (kg)", min_value=39.0, max_value=173.0, value=70.0, step=0.1)
        family_history = st.selectbox(f"{feature_dict['family_history_with_overweight']}", options=[0, 1], format_func=lambda x: "yes" if x == 0 else "no")

    with col2:
        st.subheader("Dietary Habits")
        favc = st.selectbox(f"{feature_dict['favc']}", options=[0, 1], format_func=lambda x: "yes" if x == 0 else "no")
        fcvc = st.selectbox(f"{feature_dict['fcvc']}", options=[1, 2, 3], format_func=lambda x: "Never" if x == 1 else "Sometimes" if x == 2 else "Always")
        ncp = st.selectbox(f"{feature_dict['ncp']}", options=[1, 2, 3, 4])
        caec = st.selectbox(f"{feature_dict['caec']}", options=[0, 1, 2, 3], format_func=lambda x: "Always" if x == 0 else "Frequently" if x == 1 else "no" if x == 2 else "Sometimes")
        ch2o = st.selectbox(f"{feature_dict['ch2o']}", options=[1, 2, 3], format_func=lambda x: "Less than 1L" if x == 1 else "1-2L" if x == 2 else "More than 2L")
        calc = st.selectbox(f"{feature_dict['calc']}", options=[0, 1, 2, 3], format_func=lambda x: "no" if x == 0 else "Sometimes" if x == 1 else "Frequently" if x == 2 else "Always")

    with col3:
        st.subheader("Lifestyle")
        faf = st.selectbox(f"{feature_dict['faf']}", options=[0, 1, 2, 3], format_func=lambda x: "No exercise" if x == 0 else "1-2 days" if x == 1 else "2-4 days" if x == 2 else "4-5 days")
        tue = st.selectbox(f"{feature_dict['tue']}", options=[0, 1, 2], format_func=lambda x: "0-2 hours" if x == 0 else "3-5 hours" if x == 1 else "More than 5 hours")
        mtrans = st.selectbox(f"{feature_dict['mtrans']}", options=[0, 1, 2, 3, 4], format_func=lambda x: "Automobile" if x == 0 else "Bike" if x == 1 else "Motorbike" if x == 2 else "Public_Transportation" if x == 3 else "Walking")

        # 计算BMI
        bmi = weight / (height ** 2)
        st.metric("Calculated BMI", f"{bmi:.2f}")

    # 创建预测按钮
    predict_button = st.button("Predict Obesity Level")

    if predict_button:
        # 收集所有输入特征 - 按照训练数据的顺序
        features = [gender, age, height, weight, family_history, favc, fcvc, ncp, caec, ch2o, faf, tue, calc, mtrans, bmi]

        # 转换为DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)

        # 进行预测
        prediction_proba = model.predict_proba(input_df)[0]
        prediction_class = model.predict(input_df)[0]

        # 显示预测结果
        st.header("Obesity Level Prediction Results")

        # 显示预测的肥胖等级
        predicted_level = obesity_levels[prediction_class]
        st.markdown(f"## Predicted Obesity Level: **{prediction_class} - {predicted_level}**")

        # 显示所有等级的概率
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            'Obesity Level': [f"{i} - {obesity_levels[i]}" for i in range(7)],
            'Probability': prediction_proba
        })

        # 创建概率条形图
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(prob_df['Obesity Level'], prob_df['Probability'],
                     color=['red' if i == prediction_class else 'lightblue' for i in range(7)])
        ax.set_xlabel('Obesity Level')
        ax.set_ylabel('Probability')
        ax.set_title('Obesity Level Prediction Probability Distribution')
        plt.xticks(rotation=45, ha='right')

        # 在最高的柱子上添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0.01:  # 只显示概率大于1%的标签
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # 显示概率表格
        st.table(prob_df)
        
        # 健康建议
        st.header("Health Recommendations")
        st.write("Based on the obesity level prediction results, the following health recommendations are provided:")

        if prediction_class in [2, 3, 4]:  # 肥胖类型 (Obesity Type I, II, III)
            obesity_type = {2: "I", 3: "II", 4: "III"}[prediction_class]
            st.error(f"""
            **Obesity Type {obesity_type}** - Immediate action required:
            - Consult healthcare professionals for weight management plan
            - Consider medical intervention if necessary
            - Implement strict dietary control under professional guidance
            - Regular physical activity under medical supervision
            - Monitor for obesity-related health complications
            - Consider bariatric surgery consultation if BMI > 40
            """)
        elif prediction_class in [5, 6]:  # 超重 (Overweight Level I, II)
            overweight_level = {5: "I", 6: "II"}[prediction_class]
            st.warning(f"""
            **Overweight Level {overweight_level}** - Lifestyle modifications recommended:
            - Reduce caloric intake by 500-750 calories per day
            - Increase physical activity to 150-300 minutes per week
            - Focus on whole foods, vegetables, and lean proteins
            - Limit processed foods and sugary beverages
            - Regular monitoring of weight and BMI
            - Consider consulting a nutritionist
            """)
        elif prediction_class == 1:  # 正常体重
            st.success("""
            **Normal Weight** - Maintain current healthy lifestyle:
            - Continue balanced diet and regular exercise
            - Monitor weight regularly to prevent weight gain
            - Maintain current physical activity levels
            - Focus on overall health and wellness
            """)
        else:  # 体重不足 (prediction_class == 0)
            st.info("""
            **Insufficient Weight** - Weight gain may be beneficial:
            - Increase caloric intake with nutrient-dense foods
            - Focus on healthy weight gain through proper nutrition
            - Consider consulting healthcare professionals
            - Regular monitoring of nutritional status
            - Strength training to build muscle mass
            """)
        
        # 添加模型解释
        st.write("---")
        st.subheader("Model Interpretation")

        try:
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            # 处理SHAP值格式 - 多分类情况
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                # 3D数组: (samples, features, classes)
                shap_value = shap_values[0, :, prediction_class]  # 第一个样本，所有特征，预测类别
                expected_value = explainer.expected_value[prediction_class]
            elif isinstance(shap_values, list):
                # 列表格式
                shap_value = np.array(shap_values[prediction_class][0])
                expected_value = explainer.expected_value[prediction_class] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                # 2D数组或其他格式
                shap_value = np.array(shap_values[0])
                expected_value = explainer.expected_value

            # 特征贡献分析表格
            st.subheader("Feature Contribution Analysis")

            # 创建贡献表格
            feature_values = []
            feature_impacts = []

            # 获取SHAP值
            for i, feature in enumerate(feature_names):
                feature_values.append(float(input_df[feature].iloc[0]))
                # SHAP值现在应该是一维数组
                impact_value = float(shap_value[i])
                feature_impacts.append(impact_value)

            shap_df = pd.DataFrame({
                'Feature': [feature_dict.get(f, f) for f in feature_names],
                'Value': feature_values,
                'Impact': feature_impacts
            })

            # 按绝对影响排序
            shap_df['Absolute Impact'] = shap_df['Impact'].abs()
            shap_df = shap_df.sort_values('Absolute Impact', ascending=False)

            # 显示表格
            st.table(shap_df[['Feature', 'Value', 'Impact']])
            
            # SHAP瀑布图
            st.subheader("SHAP Waterfall Plot")

            try:
                # 创建SHAP瀑布图
                fig_waterfall = plt.figure(figsize=(12, 8))

                # 使用新版本的waterfall plot
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_value,
                        base_values=expected_value,
                        data=input_df.iloc[0].values,
                        feature_names=[feature_dict.get(f, f) for f in feature_names]
                    ),
                    max_display=15,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)
            except Exception as e:
                st.error(f"Unable to generate waterfall plot: {str(e)}")
                # 使用条形图作为替代
                fig_bar = plt.figure(figsize=(12, 8))
                sorted_idx = np.argsort(np.abs(shap_value))[-15:]
                plt.barh(range(len(sorted_idx)), shap_value[sorted_idx])
                plt.yticks(range(len(sorted_idx)), [feature_dict.get(feature_names[i], feature_names[i]) for i in sorted_idx])
                plt.xlabel('SHAP Value')
                plt.title(f'Feature Impact on Obesity Level {prediction_class} Prediction')
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            # SHAP Force Plot - Official Implementation
            st.subheader("SHAP Force Plot")

            # SHAP Force Plot for multi-class models - Official Implementation
            try:

                # Generate force plot using official SHAP method
                force_plot = shap.force_plot(
                    expected_value,
                    shap_value,
                    input_df.iloc[0],
                    feature_names=[feature_dict.get(f, f) for f in feature_names],
                    matplotlib=True,
                    show=False,
                    figsize=(20, 3)
                )

                # Display the plot
                st.pyplot(force_plot)

            except Exception as e:
                st.error(f"SHAP Force Plot error: {str(e)}")

                # Alternative: Create a custom force plot visualization
                try:
                    st.info("Creating alternative force plot visualization...")

                    # Create custom force plot
                    fig, ax = plt.subplots(figsize=(16, 4))

                    # Calculate cumulative effects
                    cumulative = expected_value

                    # Sort features by absolute SHAP value
                    feature_importance = list(zip(feature_names, shap_value, input_df.iloc[0].values))
                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

                    # Plot base value
                    ax.barh(0, expected_value, left=0, height=0.5, color='gray', alpha=0.3, label=f'Base value: {expected_value:.3f}')

                    # Plot each feature contribution
                    for i, (feature, shap_val, feature_val) in enumerate(feature_importance[:10]):  # Top 10 features
                        color = 'red' if shap_val > 0 else 'blue'
                        ax.barh(0, shap_val, left=cumulative, height=0.5, color=color, alpha=0.7)

                        # Add feature labels
                        label_x = cumulative + shap_val/2
                        feature_display = feature_dict.get(feature, feature)
                        ax.text(label_x, 0, f'{feature_display}\\n{feature_val}\\n{shap_val:.3f}',
                               ha='center', va='center', fontsize=8, rotation=0)

                        cumulative += shap_val

                    # Final prediction line
                    final_prediction = cumulative
                    ax.axvline(x=final_prediction, color='black', linestyle='--', alpha=0.8, label=f'Prediction: {final_prediction:.3f}')

                    ax.set_ylabel('SHAP Force Plot')
                    ax.set_xlabel('Model Output')
                    ax.set_title(f'SHAP Force Plot for Obesity Level {prediction_class} Prediction')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as e2:
                    st.error(f"Alternative visualization also failed: {str(e2)}")
                    st.info("SHAP force plot visualization is not available for this configuration.")


        except Exception as e:
            st.error(f"Unable to generate SHAP explanation: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.info("Using model feature importance as alternative")

        # 显示模型特征重要性
        st.write("---")
        st.subheader("Feature Importance")

        # 从随机森林模型获取特征重要性
        try:
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': [feature_dict.get(f, f) for f in feature_names],
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)

            fig, _ = plt.subplots(figsize=(12, 8))
            plt.barh(range(len(importance_df)), importance_df['Importance'], color='skyblue')
            plt.yticks(range(len(importance_df)), importance_df['Feature'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance for Obesity Level Prediction')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # 显示重要性表格
            st.table(importance_df)
        except Exception as e2:
            st.error(f"Unable to display feature importance: {str(e2)}")

if __name__ == "__main__":
    main()
