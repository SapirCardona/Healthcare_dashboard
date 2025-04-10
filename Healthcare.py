import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Mental Health in Tech - Interactive Dashboard", layout="wide")

st.markdown("""
    <style>
    html {
    background-color: #e3f2fd !important;
    color-scheme: light !important;
    color: #102027 !important;
    }
    .stSidebar, .css-1avcm0n, .css-1d391kg {
        background-color: #739dbf !important;
    }
    h1, h2, h3, .stSubheader {
        color: #1565c0 !important;
    }
    .stMetric label {
        font-weight: bold;
        font-size: 16px;
        color: #102027 !important;
    }
    .stMetric span {
        font-weight: bold;
        font-size: 26px;
        color: #1e88e5 !important;
    }
    .stMarkdown, .markdown-text-container, .stExpanderHeader {
        color: #102027 !important;
        font-size: 16px;
    }
    .stTabs [role="tab"] {
        background-color: #90caf9;
        color: #1565c0;
        padding: 10px;
        border-radius: 5px;
        margin-right: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #64b5f6;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Mental Health in Tech - Interactive Dashboard")
px.defaults.template = "seaborn"

# Load the CSV file from the backend
def load_data():
    return pd.read_csv("survey_cleaned.csv")

df = load_data()
with st.expander("Click to view a sample of the dataset"):
    st.dataframe(df.sample(10))

# Sidebar filters
country_options = ["All"] + sorted(df['Country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", country_options)

gender_options = ["All"] + sorted(df['Gender'].dropna().unique().tolist())
selected_gender = st.sidebar.selectbox("Select Gender", gender_options)

min_age = int(df['Age'].min())
max_age = int(df['Age'].max())
selected_age = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

# Filter the data
filtered_df = df[
    ((df['Country'] == selected_country) | (selected_country == "All")) &
    ((df['Gender'] == selected_gender) | (selected_gender == "All")) &
    (df['Age'] >= selected_age[0]) &
    (df['Age'] <= selected_age[1])
]

# Download button in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("Download Data")

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(filtered_df)

    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_mental_health_data.csv",
        mime="text/csv"
    )
    st.markdown("Export your filtered data for further analysis")

# Show warning if no data matches filters
if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your filters and try again.")
else:
    # Tabs
    tab1, tab2 = st.tabs(["Dashboard", "Prediction Lab"])

    with tab1:
        # KPIs
        st.subheader("Key Metrics (KPIs)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Respondents", filtered_df.shape[0])
        with col2:
            treated = filtered_df[filtered_df['treatment'] == 'Yes'].shape[0]
            st.metric("Sought Treatment", f"{treated} ({treated / filtered_df.shape[0] * 100:.1f}%)")
        with col3:
            st.metric("Avg. Age", f"{filtered_df['Age'].mean():.1f}")

        color_discrete_map = {"Yes": "#8fcbbc", "No": "#d98880"}

        def show_plotly_chart(title, fig):
            st.subheader(title)
            st.plotly_chart(fig, use_container_width=False, height=250)

        fig1 = px.histogram(filtered_df, x="Gender", color="treatment", barmode="group",
                            title="Treatment by Gender", color_discrete_map=color_discrete_map)
        show_plotly_chart("Treatment by Gender", fig1)

        fig2 = px.histogram(filtered_df, x="Age", nbins=15, title="Age Distribution",
                            color_discrete_sequence=["#d2b48c"])
        show_plotly_chart("Age Distribution", fig2)

        self_employed_counts = filtered_df["self_employed"].value_counts().reset_index()
        self_employed_counts.columns = ["self_employed", "count"]

        fig3 = px.pie(
            self_employed_counts,
            names="self_employed",
            values="count",
            title="Self-Employment Status",
            hole=0.4,
            color_discrete_map=color_discrete_map
        )
        fig3.update_traces(textinfo="percent+label")
        show_plotly_chart("Self-Employment Status", fig3)

        fig4 = px.histogram(
            filtered_df,
            x="work_interfere",
            category_orders={"work_interfere": ["Never", "Rarely", "Sometimes", "Often"]},
            title="Work Interference",
            color_discrete_sequence=["#b2a88d"]
        )
        show_plotly_chart("Work Interference", fig4)

        heat_df = pd.crosstab(filtered_df['family_history'], filtered_df['treatment'])
        fig5 = px.imshow(
            heat_df,
            text_auto=True,
            color_continuous_scale='Earth',
            title="Relationship between Family History and Seeking Treatment"
        )
        show_plotly_chart("Family History vs Treatment", fig5)

        fig6 = px.histogram(filtered_df, x="benefits", title="Mental Health Benefits at Work",
                            color_discrete_sequence=["#c5e2d5"])
        show_plotly_chart("Mental Health Benefits at Work", fig6)

        company_order = ["5-25", "26-100", "100-500", "500-1000", "More than 1000"]
        fig7 = px.histogram(
            filtered_df,
            x="no_employees",
            title="Company Size Distribution",
            category_orders={"no_employees": company_order},
            color_discrete_sequence=["#f2cfa0"]
        )
        show_plotly_chart("Company Size Distribution", fig7)

    with tab2:
        st.subheader("Predict Likelihood to Seek Mental Health Treatment")

        age = st.slider("Age", 18, 65, 30)
        gender = st.selectbox("Gender", df["Gender"].dropna().unique())
        family_history = st.selectbox("Family History", ["Yes", "No"])
        work_interfere = st.selectbox("Mental Health Interference at Work", df["work_interfere"].dropna().unique())
        benefits = st.selectbox("Mental Health Benefits Provided?", ["Yes", "No", "Don't know"])

        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "family_history": [family_history],
            "work_interfere": [work_interfere],
            "benefits": [benefits]
        })

        features = ["Age", "Gender", "family_history", "work_interfere", "benefits"]
        X = pd.get_dummies(df[features])
        y = df["treatment"].map({"Yes": 1, "No": 0})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

        probability = model.predict_proba(input_encoded)[0][1]
        st.metric("Predicted Probability of Seeking Treatment", f"{probability * 100:.1f}%")