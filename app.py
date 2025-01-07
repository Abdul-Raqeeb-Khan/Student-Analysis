import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency,f_oneway
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
from scipy.stats import kruskal
from sklearn.ensemble import GradientBoostingClassifier
from streamlit_option_menu import option_menu


# App Configuration
st.set_page_config(page_title="Teaching Methods Effectiveness", page_icon="📘", layout="wide")


# Colors for the App
primary_color =  "#B0E0E6" #"#56A8CB"#"#004466"  # Dark blue
secondary_color = "#66b2b2"  # Light teal
accent_color = "#ffcc66"  # Warm yellow
background_color = "#e6f2f2"  # Light aqua
navbar_color = "#F0FFFF" # Very dark blue for the sidebar
navbar_text_color = "#ffffff"  # White for text links in the navigation bar



# Define the navigation menu
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "Data Overview", "Data Visualization", "Model", "Conclusion"],
        icons=["house", "table", "bar-chart", "robot", "check-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": navbar_color},  # Sidebar background color
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": primary_color,  # Hover color matches the primary color
            },
            "nav-link-selected": {
                "background-color": secondary_color,  # Light teal for selected item
                "color": "white",  # Text color for selected item
            },
        },
    )

# Add custom CSS to change the right-side background color
st.markdown(
    f"""
    <style>
    /* Change the background color of the main content area */
    div[data-testid="stAppViewContainer"] {{
        background-color: {background_color} !important;  /* Light aqua for main content background */
    }}

    /* Optional: Remove padding/margin if needed */
    div[data-testid="stAppViewContainer"] > div {{
        padding: 10px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)




# Initialize session state for data persistence
if "data" not in st.session_state:
    st.session_state["data"] = None



# App Pages
if selected == "Home":
    st.title("📘 Evaluating the Effectiveness of Teaching Methods")
    st.write("""
        Education is a cornerstone of societal progress, and the effectiveness of teaching methods 
        significantly impacts students' learning outcomes. This app provides a comprehensive analysis 
        of various teaching methods using data-driven approaches, visualization, and machine learning.
        
        Whether you are an educator, policymaker, or researcher, this app will guide you in exploring 
        insights and models to make informed decisions for enhancing educational strategies.
    """)

elif selected == "Data Overview":
    st.title("📊 Data Overview")
    st.write("Upload your dataset to analyze the effectiveness of different teaching methods.")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        st.session_state["data"] = pd.read_csv(uploaded_file)
        st.write("### Dataset Overview")
        st.dataframe(st.session_state["data"].head())
        st.write("### Summary Statistics")
        st.dataframe(st.session_state["data"].describe())
    elif st.session_state["data"] is not None:
        st.write("### Dataset Overview")
        st.dataframe(st.session_state["data"].head())
        st.write("### Summary Statistics")
        st.dataframe(st.session_state["data"].describe())
    else:
        st.warning("Please upload a dataset.")

elif selected == "Data Visualization":
    st.title("📈 Data Visualization")
    if st.session_state["data"] is None:
        st.warning("Please upload a dataset in the 'Data Overview' section first.")
    else:
        data = st.session_state["data"]
        analysis_type = st.radio("Choose Analysis Type", ["Univariate Analysis", "Bivariate Analysis","Pairwise Analysis", "Correlations"])

        if analysis_type == "Univariate Analysis":
            st.write("### Univariate Analysis")
            feature = st.selectbox("Select a feature for Univariate Analysis", data.columns)

            if pd.api.types.is_numeric_dtype(data[feature]):
                st.write(f"### Distribution of {feature}")
                plt.figure(figsize=(10, 6))
                sns.histplot(data[feature], kde=True, color='skyblue')
                plt.title(f"Distribution of {feature}", fontsize=16)
                plt.xlabel(feature, fontsize=14)
                plt.ylabel("Frequency", fontsize=14)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(plt)
            else:
                st.write(f"### Distribution of {feature}")
                target_distribution = data[feature].value_counts()
                plt.figure(figsize=(10, 6))
                target_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
                plt.title(f'Distribution of {feature}', fontsize=16)
                plt.xlabel(feature, fontsize=14)
                plt.ylabel('Frequency', fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout(pad=1.5)
                st.pyplot(plt)

        elif analysis_type == "Bivariate Analysis":
            st.write("### Bivariate Analysis")
            x_feature = st.selectbox("Select X-axis feature", data.columns, key="x_feature")
            y_feature = st.selectbox("Select Y-axis feature", data.columns, key="y_feature")

            plot_type = st.radio("Select Plot Type", ["Countplot", "Crosstab"], key="plot_type")

            # Convert object columns to categorical
            if pd.api.types.is_object_dtype(data[x_feature]):
                data[x_feature] = pd.Categorical(data[x_feature])
            if pd.api.types.is_object_dtype(data[y_feature]):
                data[y_feature] = pd.Categorical(data[y_feature])

            # Handle missing values
            if data[x_feature].isnull().any() or data[y_feature].isnull().any():
                st.warning("The selected features contain missing values. Please clean the data.")
            else:
                # Generate plots
                if plot_type == "Countplot":
                    if pd.api.types.is_categorical_dtype(data[x_feature]) and pd.api.types.is_categorical_dtype(data[y_feature]):
                        plt.figure(figsize=(10, 6))
                        sns.countplot(x=data[x_feature], hue=data[y_feature], palette="coolwarm")
                        plt.title(f"{x_feature} vs {y_feature} (Countplot)", fontsize=16)
                        plt.xlabel(x_feature, fontsize=14)
                        plt.ylabel("Count", fontsize=14)
                        plt.xticks(rotation=45, ha='right', fontsize=12)
                        plt.grid(axis="y", linestyle="--", alpha=0.7)
                        st.pyplot(plt)
                    else:
                        st.warning("Both selected features should be categorical for a countplot.")

                elif plot_type == "Crosstab":
                        st.write(f"### Crosstab: {x_feature} vs {y_feature}")
                        if pd.api.types.is_categorical_dtype(data[x_feature]) and pd.api.types.is_categorical_dtype(data[y_feature]):
                            # Create Crosstab
                            crosstab_result = pd.crosstab(data[x_feature], data[y_feature], normalize='index') * 100

                            st.write("Crosstab Table (Normalized by Index)")
                            st.dataframe(crosstab_result)

                            # Chart selection option
                            chart_type = st.radio("Select Chart Type", ["Stacked Bar Chart", "Line Chart"])

                            if chart_type == "Stacked Bar Chart":
                                st.write("### Stacked Bar Chart")
                                plt.figure(figsize=(25, 13))
                                crosstab_result.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Paired', edgecolor='black')
                                plt.title(f'{y_feature} Distribution by {x_feature}', fontsize=16)
                                plt.xlabel(x_feature, fontsize=14)
                                plt.ylabel('Percentage (%)', fontsize=14)
                                plt.xticks(rotation=45, fontsize=12)
                                plt.legend(title=y_feature, fontsize=12)
                                plt.tight_layout()
                                st.pyplot(plt)

                            elif chart_type == "Line Chart":
                                st.write("### Line Chart")
                                plt.figure(figsize=(25, 13))
                                crosstab_result.plot(kind='line', marker='o', figsize=(10, 6), colormap='coolwarm')
                                plt.title(f'{y_feature} Distribution by {x_feature}', fontsize=16)
                                plt.xlabel(x_feature, fontsize=14)
                                plt.ylabel('Percentage (%)', fontsize=14)
                                plt.xticks(rotation=45, fontsize=12)
                                plt.grid(axis="y", linestyle="--", alpha=0.7)
                                plt.legend(title=y_feature, fontsize=12)
                                plt.tight_layout()
                                st.pyplot(plt)
                        else:
                            st.warning("Both selected features should be categorical for a crosstab.")
       
        elif analysis_type == "Pairwise Analysis":
            st.write("### Kruskal-Wallis Test")

            # Allow the user to select columns for Kruskal-Wallis Test (either ordinal or categorical encoded columns)
            encoded_columns = [col for col in data.columns if '_numeric' in col]  # Select encoded columns (e.g., 'col_numeric')

            # Allow the user to select columns for Kruskal-Wallis Test
            selected_columns = st.multiselect("Select Columns for Kruskal-Wallis Test", encoded_columns, default=[])

            if len(selected_columns) > 0:
                # Initialize a dictionary to store the Kruskal-Wallis test results
                results = {}

                # Perform the Kruskal-Wallis test for each selected column across EngagingMethod groups
                for col in selected_columns:
                    # Group data by EngagingMethod
                    groups = [group[col].dropna() for _, group in data.groupby('EngagingMethod')]

                    # Apply Kruskal-Wallis test
                    stat, p = kruskal(*groups)
                    results[col] = {'Statistic': stat, 'p-value': p}

                # Display the results
                for col, result in results.items():
                    # Remove the '_numeric' suffix for display
                    clean_col_name = col.replace('_numeric', '')
                    st.write(f"\n### Kruskal-Wallis Test for {clean_col_name} across EngagingMethod:")
                    st.write(f"Test Statistic: {result['Statistic']:.4f}")
                    st.write(f"p-value: {result['p-value']:.4f}")
                    if result['p-value'] <= 0.05:
                        st.write("Conclusion: There is a significant difference across EngagingMethod groups.")
                    else:
                        st.write("Conclusion: There is no significant difference across EngagingMethod groups.")

                # Visualize distributions using box plots for each selected column
                for col in selected_columns:
                    # Remove the '_numeric' suffix for display
                    clean_col_name = col.replace('_numeric', '')
                    st.write(f"### Distribution of {clean_col_name} Across Engaging Methods")
                    plt.figure(figsize=(8, 4))
                    sns.boxplot(data=data, x='EngagingMethod', y=col, palette='coolwarm')
                    plt.title(f"Distribution of {clean_col_name} Across Engaging Methods", fontsize=16)
                    plt.xlabel("Engaging Method", fontsize=14)
                    plt.ylabel(clean_col_name, fontsize=14)  # Display clean column name here
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(plt)

            else:
                st.warning("Please select at least one column for Kruskal-Wallis Test.")

            # Perform Chi-Square Test for categorical variables
            st.write("### Chi-Square Test")

            # Identify categorical columns (either as object types or encoded categorical columns)
            categorical_columns = [col for col in data.columns if data[col].dtype == 'object' or '_numeric' in col]
            selected_categorical_columns = st.multiselect("Select Categorical Columns for Chi-Square Test", categorical_columns, default=[])

            if len(selected_categorical_columns) > 0:
                # Initialize list to store significant results
                significant_columns = []

                for col in selected_categorical_columns:
                    target = 'EngagingMethod'  # Target variable for Chi-Square Test

                    # Perform Chi-Square test on the selected categorical column and target
                    contingency_table = pd.crosstab(data[col], data[target])
                    chi2, p, dof, expected = chi2_contingency(contingency_table)

                    # Display Chi-Square results
                    null_hypothesis = f"Null Hypothesis (H0): {col} and {target} are independent."
                    alt_hypothesis = f"Alternative Hypothesis (H1): {col} and {target} are not independent (they are associated)."

                    st.write(f"\n### Chi-Square Test for {col} and {target}:")
                    st.write(null_hypothesis)
                    st.write(alt_hypothesis)
                    st.write(f"Chi-Square Statistic: {chi2:.4f}")
                    st.write(f"p-value: {p:.4f}")
                    st.write(f"Degrees of Freedom: {dof}")
                    if p <= 0.05:
                        st.write(f"Decision: Reject the null hypothesis. {col} and {target} are associated.")
                        significant_columns.append(col)
                    else:
                        st.write(f"Decision: Fail to reject the null hypothesis. {col} and {target} are independent.")

                # Visualize Chi-Square results with bar plots for significant variables
                if selected_categorical_columns:
                    for col in selected_categorical_columns:
                        st.write(f"### Relationship Between {col} and EngagingMethod")
                        plt.figure(figsize=(10, 6))
                        sns.countplot(data=data, x='EngagingMethod', hue=col, palette='coolwarm')
                        plt.title(f'Relationship Between {col} and EngagingMethod', fontsize=16)
                        plt.xlabel('Engaging Method', fontsize=14)
                        plt.ylabel('Count', fontsize=14)
                        plt.xticks(rotation=45, fontsize=12, ha='right')
                        plt.legend(title=col, fontsize=12)
                        plt.tight_layout()
                        st.pyplot(plt)

            else:
                st.warning("Please select at least one categorical column for Chi-Square Test.")
        elif analysis_type == "Correlations":
            st.write("### Correlation Analysis")

            # Columns to analyze
            columns_to_check = [
                'EngagingMethod_numeric', 'MotivationDuringTL_numeric', 'HandsOnActivityHelp_numeric',
                'TLPerformance_numeric', 'GroupDiscussion_numeric', 'MultimediaEffectiveness_numeric',
                'TeacherInteraction_numeric', 'IMEncourageness_numeric', 'PeerLearning_numeric',
                'Retention_numeric', 'ChallengesTL_numeric'
            ]

            # Calculate Spearman's rank correlation matrix using pandas
            corr_df = data[columns_to_check].corr(method="spearman")

            # Display the correlation matrix
            st.write("#### Spearman's Rank Correlation Matrix")
            st.dataframe(corr_df)

            # Reverse the DataFrame if necessary
            #corr_df_reversed = corr_df.iloc[::-1, ::-1]

            # Plot the heatmap
            st.write("#### Heatmap of Spearman's Rank Correlation Matrix")
            plt.figure(figsize=(10, 8))

            sns.heatmap(
                corr_df,  # Use reversed matrix if necessary
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                square=True,
            )

            # Add title and tick adjustments
            plt.title("Spearman's Rank Correlation Matrix", fontsize=16)
            plt.xticks(rotation=45, ha="right", fontsize=12)
            plt.yticks(rotation=0, fontsize=12)

            # Render the plot in Streamlit
            st.pyplot(plt)

            
elif selected == "Model":
    st.title("🤖 Machine Learning Model")

    # Check if data is available in the session
    if st.session_state["data"] is None:
        st.warning("Please upload a dataset in the 'Data Overview' section first.")
    else:
        data = st.session_state["data"]

        # Define mappings for Gender, AgeGroup, and Engaging Method
        gender_mapping = {'Male': 0, 'Female': 1}
        age_group_mapping = {'18–24': 1, '25–34': 2, 'Above 34': 3}
        TLPerformance_mapping = {0: 'No', 1: 'Not sure', 2: 'Yes'}
        ChallengesTL_mapping = {
            0: 'Lack of engagement', 1: 'No hands-on learning opportunities',
            2: 'One-way communication', 3: 'Overwhelming content'
        }
        GroupDiscussion_mapping = {
            3: 'Very effective', 2: 'Neutral', 0: 'Effective',
            1: 'Ineffective', 4: 'Very ineffective'
        }
        ordinal_mappings = {
            'MotivationDuringTL': {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5},
            'HandsOnActivityHelp': {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5},
            'PeerLearning': {'No': 1, 'Not sure': 2, 'Sometimes': 3, 'Neutral': 4, 'Yes': 5, 'Strongly agree': 6},
            'Retention': {'Traditional lectures': 1, 'Experiential methods': 2, 'Interactive methods': 3}
        }
        engaging_method_mapping = {
            'Experiential methods': 1, 'Interactive methods': 2, 'Traditional lectures': 3
        }

        selected_features = [
            'TLPerformance_numeric', 'GroupDiscussion_numeric', 'ChallengesTL_numeric',
            'MotivationDuringTL_numeric', 'HandsOnActivityHelp_numeric', 'PeerLearning_numeric',
            'Retention_numeric', 'AgeGroup_numeric', 'Gender_numeric'
        ]

        # Prepare data
        X = data[selected_features]
        y = data['EngagingMethod_numeric']  # Target variable (already encoded)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train the Gradient Boosting Classifier
        gbc_classifier = GradientBoostingClassifier(random_state=42)
        gbc_classifier.fit(X_train, y_train)

        # Input fields for the user to make predictions
        st.write("### Enter values for the following features:")

        user_input = {}
        user_input['Gender'] = st.selectbox("Select Gender", options=['Male', 'Female'])
        user_input['AgeGroup'] = st.selectbox("Select Age Group", options=['18–24', '25–34', 'Above 34'])
        user_input['Gender_numeric'] = gender_mapping[user_input['Gender']]
        user_input['AgeGroup_numeric'] = age_group_mapping[user_input['AgeGroup']]

        # Other ordinal feature inputs
        for feature, mapping in ordinal_mappings.items():
            user_input[feature] = st.selectbox(f"Select {feature}", options=list(mapping.keys()))
            user_input[f"{feature}_numeric"] = mapping[user_input[feature]]

        # Categorical encoded features
        user_input['TLPerformance_numeric'] = st.selectbox("Select TLPerformance", options=TLPerformance_mapping.values())
        user_input['GroupDiscussion_numeric'] = st.selectbox("Select GroupDiscussion", options=GroupDiscussion_mapping.values())
        user_input['ChallengesTL_numeric'] = st.selectbox("Select ChallengesTL", options=ChallengesTL_mapping.values())

        # Map categorical inputs
        user_input['TLPerformance_numeric'] = [
            k for k, v in TLPerformance_mapping.items() if v == user_input['TLPerformance_numeric']
        ][0]
        user_input['GroupDiscussion_numeric'] = [
            k for k, v in GroupDiscussion_mapping.items() if v == user_input['GroupDiscussion_numeric']
        ][0]
        user_input['ChallengesTL_numeric'] = [
            k for k, v in ChallengesTL_mapping.items() if v == user_input['ChallengesTL_numeric']
        ][0]

        # Prepare user input for prediction
        input_data = pd.DataFrame([user_input])

        # Prediction logic
        make_prediction_button = st.button("Make Prediction")
        if make_prediction_button:
            if any(value is None or value == "" for value in user_input.values()):
                st.warning("Please fill all the input fields before making a prediction.")
            else:
                # Prepare input data for prediction
                input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
                prediction_encoded = gbc_classifier.predict(input_data)

                # Decode prediction
                prediction_label = {v: k for k, v in engaging_method_mapping.items()}
                predicted_value = prediction_encoded[0]
                predicted_label = prediction_label.get(predicted_value, "Unknown")

                st.success(f"### Predicted Engaging Method (Encoded): {predicted_value}")
                st.success(f"### Predicted Engaging Method (Label): {predicted_label}")

elif selected == "Conclusion":
    st.title("📝 Conclusion")
    st.write("Summarize the insights gained from the analysis and model evaluation.")
    st.write("""
    ### Key Insights:
    - Data visualization highlights the trends and variations across teaching methods.
    - Machine learning models, like Gradient Boosting Classifier , provide robust predictions of outcomes based on teaching strategies.
    - Combining data-driven insights with pedagogical expertise enhances the decision-making process.
    """)

# Footer with links to profiles
st.markdown(
    """
    <hr style='border: 1px solid #004466; margin-top: 50px; margin-bottom: 10px;'>
    <div style='text-align: center; font-size: 14px;'>
        © Developed By <strong>Abdul Raqeeb Khan</strong><br>
        <a href="https://www.kaggle.com/abdulraqeebkhan" target="_blank" style='text-decoration: none; color: #004466; font-weight: bold;'>Kaggle</a> | 
        <a href="https://github.com/Abdul-Raqeeb-Khan" target="_blank" style='text-decoration: none; color: #004466; font-weight: bold;'>GitHub</a> | 
        <a href="https://www.linkedin.com/in/abdul-raqeeb-khan-766ab9269/" target="_blank" style='text-decoration: none; color: #004466; font-weight: bold;'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)






