import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly.express as px
# from IPython.core.display import HTML

file_path = r'C:\Users\Sarvesh\Desktop\STAT 4355 Assignments\STAT 4355 Final Project\admission_data.csv'
grad_df = pd.read_csv(file_path)

# Assign column names
grad_df.columns = ['GRE', 'TOEFL', 'UnivRtg', 'SOP', 'LOR', 'CGPA', 'Research', 'AdmitChance']
# Remove duplicates
grad_df = grad_df.drop_duplicates()
# Remove entries with missing values
grad_df = grad_df.dropna()
# Check variable data types
# print(grad_df.info())

# Change scaling for AdmitChance
grad_df['AdmitChance'] = grad_df['AdmitChance'] * 100

x_grad_df_ur = grad_df[['GRE', 'TOEFL', 'SOP', 'LOR', 'CGPA', 'Research', 'UnivRtg']]
y_grad_df_ur = grad_df['AdmitChance']
x_grad_df_ur['UnivRtg'] = pd.to_numeric(x_grad_df_ur['UnivRtg'], errors='coerce')
total_y = len(y_grad_df_ur)

def build_display_lm(x_df, y_df):
    # Build and Display Model
    x_df = sm.add_constant(x_df)
    model_df = sm.OLS(y_df, x_df).fit()
    print(model_df.summary())

    total_y = len(y_df)

    # Calculate Residuals for Analysis
    residuals = model_df.resid
    R_stud_resid = model_df.get_influence().resid_studentized_external
    stud_resid = model_df.get_influence().resid_studentized_internal
    std_resid = residuals / np.std(residuals)
    fitted = model_df.fittedvalues

    # resid = {'org_resid': residuals, 'RStud_resid' : R_stud_resid, 
              #'Stud_resid' : stud_resid, 'Std_resid' : std_resid}
    
    # Build Residual Plot
    plt.figure(figsize = (8, 6))
    sns.residplot(x = fitted, y = residuals, line_kws = {'color' : 'red', 'lw' : 2})
    plt.axhline(0, color = 'black', linestyle = '--', linewidth = 1)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.show()

    # Build R-Student Plot
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(R_stud_resid) + 1), R_stud_resid)
    plt.title('R-Student Residuals')
    plt.xlabel('Index')
    plt.ylabel('R-Student Residual')
    plt.ylim(-4, 4)
    
    plt.axhline(y=3, color='red', linewidth=2, linestyle='--')
    plt.axhline(y=-3, color='red', linewidth=2, linestyle='--')
    plt.show()
    
    R_stud_out = np.where((R_stud_resid >= 3) | (R_stud_resid <= -3))[0]
    print(R_stud_out)

    # Build Studentized Residuals Plot
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(stud_resid) + 1), stud_resid)
    plt.title('Studentized Residuals')
    plt.xlabel('Index')
    plt.ylabel('Studentized Residual')
    plt.ylim(-4, 4)
    
    plt.axhline(y=3, color='red', linewidth=2, linestyle='--')
    plt.axhline(y=-3, color='red', linewidth=2, linestyle='--')
    plt.show()
    
    stud_out = np.where((stud_resid >= 3.0) | (stud_resid <= -3.0))[0]
    print(stud_out)

    # Build QQ-Plot
    sm.qqplot(residuals, line = '45', fit = True)
    plt.title('Q-Q Plot of Residuals')
    plt.show()

    # Build Histogram of Residuals
    plt.figure(figsize = (8, 6))
    sns.histplot(residuals, kde = True, bins = math.ceil(math.log2(total_y) + 1), color = 'blue')
    plt.axvline(0, color = 'black', linestyle = '--', linewidth = 1)
    plt.xlabel('Residuals')
    plt.ylabel('Frequeny')
    plt.title('Histogram of Residuals')
    plt.show()

    return residuals

ur_resid = build_display_lm(x_grad_df_ur, y_grad_df_ur)

filtered_data_ur = x_grad_df_ur[(ur_resid >= -4.5) & (ur_resid <= 14.5)]
filtered_target_ur = y_grad_df_ur[(ur_resid >= -4.5) & (ur_resid <= 14.5)]

# Build the model that will be used for the app
app_model = sm.OLS(filtered_target_ur, sm.add_constant(filtered_data_ur)).fit()
print(app_model.summary())

# Save the model and its coefficients 
with open('admission_model.pkl', 'wb') as model_file:
   pickle.dump(app_model, model_file)

coefficients = app_model.params

with open('coefficients.pkl', 'wb') as coef_file:
    pickle.dump({'coefficients' : coefficients[1:], 'intercept' : coefficients['const']}, coef_file)

# Load the saved model
with open('admission_model.pkl', 'rb') as model_file:
    app_model = pickle.load(model_file)

# Load the model's coefficients
with open('coefficients.pkl', 'rb') as coef_file:
    app_model_data = pickle.load(coef_file)

st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="centered",  # Or 'wide' for a wider layout
    initial_sidebar_state="expanded",  # Set 'collapsed' or 'expanded' for sidebar
)

# Custom CSS for dark mode
custom_css = """
<style>
/* General background and text styling */
[data-testid="stAppViewContainer"] {
    background-color: #121212; /* Black background */
    color: #FFA500; /* Orange text */
}

[data-testid="stSidebar"] {
    background-color: #121212; /* Black sidebar background */
    color: #FFA500; /* Orange text */
}

[data-testid="stSidebar"]::before {
    background-color: #121212;
}

/* Fix top bar white space */
[data-testid="stHeader"] {
    background-color: #121212; /* Match background */
    color: #FFA500; /* Orange text */
}

/* Remove gradient around slider numbers */
div[data-testid="stMarkdownContainer"] div {
    background: none !important; /* Remove gradient */
    border: none !important; /* Remove border */
    color: #FFA500 !important; /* Orange text */
}

/* Style slider */
.stSlider > div > div > div {
    background: linear-gradient(to right, #800080, #FFA500); /* Purple-to-orange gradient */
}

.stSlider > div > div > div > div {
    background-color: #FFA500; /* Orange handle */
}

/* Labels for sliders */
.stSlider label {
    color: #8B0000; /* Purple text */
}

/* Input fields and select box styling */
input[type="number"], input[type="text"], [data-baseweb="select"] {
    background-color: #1f1f1f; /* Dark input field background */
    color: #FFA500; /* Orange text */
    border: 1px solid #800080; /* Purple border */
}

/* Ensure CGPA input box has the same styling */
input[type="text"]#cgpa {
    background-color: #1f1f1f; /* Dark input field background */
    color: #FFA500; /* Orange text */
    border: 1px solid #800080; /* Purple border */
}

/* Handle focus effect for input fields */
input[type="text"]:focus, input[type="number"]:focus {
    border: 1px solid #FFA500 !important; /* Orange border on focus */
    outline: none !important; /* Remove default focus outline */
}

/* Dropdown menu styling */
[data-baseweb="menu"] {
    background-color: #1f1f1f !important; /* Dark dropdown background */
    color: #FFA500 !important; /* Orange text */
    border: 1px solid #800080; /* Purple border */
}

[data-baseweb="menu"] div:hover {
    background-color: #800080 !important; /* Purple hover */
    color: #121212 !important; /* Black text on hover */
}

/* Style sidebar hover color */
[data-testid="stSidebar"] a:hover {
    background-color: #800080; /* Purple background */
    color: #FFA500; /* Orange text */
}

/* Radio button styling */
div[data-testid="stRadio"] label {
    background-color: #1f1f1f !important; /* Dark radio background */
    color: #FFA500 !important; /* Orange text */
}

div[data-testid="stRadio"] label:hover {
    background-color: #800080 !important; /* Purple background on hover */
    color: #121212 !important; /* Black text */
}

/* Buttons */
button {
    background-color: #2e2e2e; /* Dark button background */
    color: #FFA500; /* Orange text */
    border: 1px solid #FFA500; /* Orange border */
}

button:hover {
    background-color: #FFA500; /* Orange hover */
    color: #121212; /* Black text */
}

/* Style text associated with input fields */
div[data-testid="stText"], span, p {
    color: #FFA500 !important; /* Orange text */
}

/* Style for select box */
div[data-baseweb="select"] > div {
    background-color: #1f1f1f !important; /* Dark background for the select box */
    color: #FFA500 !important; /* Orange text */
}

/* Prevent gradient from affecting the select label and text */
div[data-baseweb="select"] > div > div {
    background: none !important; /* Remove gradient background */
}

/* Ensure the select option text color stays orange */
div[data-baseweb="select"] > div > div > div {
    color: #FFA500 !important; /* Orange text */
}
</style>
"""

# Inject custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

coefficients = app_model_data['coefficients']
intercept = app_model_data['intercept']

# Calculate the Percentiles for the Recommendations
percentiles = {
    'GRE' : np.percentile(filtered_data_ur['GRE'], 50),
    'TOEFL' : np.percentile(filtered_data_ur['TOEFL'], 50), 
    'SOP' : np.percentile(filtered_data_ur['SOP'], 50), 
    'LOR' : np.percentile(filtered_data_ur['LOR'], 50), 
    'CGPA' : np.percentile(filtered_data_ur['CGPA'], 50), 
    'UnivRtg' : np.percentile(filtered_data_ur['UnivRtg'], 50)
}

# Title of the App
st.title("Graduate Admission Predictor")
st.write("""Welcome to the Graduate Admission Predictor. Enter your
         academic profile to predict your graduate admission chance
         and to receive personalized academic recommendations.""")


# User Input
st.header("Enter Your Academic Profile")
gre = st.slider("GRE Score (260 - 340)", min_value = 260, max_value = 340, step = 1)
toefl = st.slider("TOEFL Score (0 - 120)", min_value = 0, max_value = 120, step = 1)
sop = st.selectbox("SOP Strength (0-5)", options = [0, 0.5, 1, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5])
lor = st.selectbox("LOR Strength (0-5)", options = [0, 0.5, 1, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5])
cgpa = st.number_input("CGPA (0-10, ex. 8.57)", min_value = 0.0, max_value = 10.0, step = 0.01)
research = st.radio("Research Experience", options = ["No", "Yes"])
univ_rtg = st.selectbox("University Rating (0-5)", options = [0, 1, 2, 3, 4, 5])

st.sidebar.write("**Your Inputs:**")
st.sidebar.write(f"GRE: {gre}")
st.sidebar.write(f"TOEFL: {toefl}")
st.sidebar.write(f"SOP: {sop}")
st.sidebar.write(f"LOR: {lor}")
st.sidebar.write(f"CGPA: {cgpa}")
st.sidebar.write(f"Research: {research}")
st.sidebar.write(f"University Rating: {univ_rtg}")

research_binary = 1 if research == "Yes" else 0

user_data = {
    "GRE" : gre, 
    "TOEFL" : toefl, 
    "SOP" : sop, 
    "LOR" : lor, 
    "CGPA" : cgpa, 
    "Research" : research_binary, 
    "UnivRtg" : univ_rtg
}

user_df = pd.DataFrame([user_data])

# Predict admission chance
def predict_admission(features, coefficients, intercept):
    prediction = intercept

    for feature in coefficients.index:
        feature_value = features[feature].values[0]
        feature_coefficient = coefficients[feature]
        cont = feature_value * feature_coefficient
        prediction += cont

    if (prediction <= 0):
        return 0.0
    elif (prediction >= 100):
        return 100.0
    else: 
        return prediction

st.markdown("### Prediction Result")

prediction = predict_admission(user_df, coefficients, intercept)
st.write(f"Predicted Chance of Admission: **{prediction:.2f}%**")

st.markdown("### Model Diagnostics")

# Interactive Scatter Plot (Actual vs Predicted Admission Chances)
import plotly.express as px
import pandas as pd

# Assuming actual_y and pred_y are already defined
# Replace this with your actual code for actual_y and pred_y
actual_y = filtered_target_ur
pred_y = app_model.predict(sm.add_constant(filtered_data_ur))

# Create DataFrame for the scatter plot
df = pd.DataFrame({'Actual': actual_y, 'Predicted': pred_y})

# Create the scatter plot
fig = px.scatter(df, 
                 x='Predicted', 
                 y='Actual', 
                 title='Actual vs Predicted Admission Chances',
                 labels={'Predicted': 'Predicted Admission Chances', 'Actual': 'Actual Admission Chances'})

# Update trace to set marker color to orange and customize the markers
fig.update_traces(marker=dict(size=10, opacity=0.5, line=dict(width=1, color='black'), color='#FFA500'))

# Update layout to set background colors to purple and text colors to orange
fig.update_layout(
    plot_bgcolor='#121212',  # Purple background for the plot area
    paper_bgcolor='#121212',  # Purple background for the entire figure
    title='Actual vs Predicted Admission Chances',
    title_x = 0,  # Center title
    title_font=dict(color='#FFA500'),  # Orange title color
    xaxis=dict(title='Predicted Admission Chances', title_font=dict(color='#FFA500')),
    yaxis=dict(title='Actual Admission Chances', title_font=dict(color='#FFA500'))
)

st.plotly_chart(fig)


# Interactive Feature Plot
fig = px.bar(x=coefficients.index, 
             y=coefficients.values, 
             labels={'x': 'Feature', 'y': 'Coefficient'}, 
             title='Feature Impact on Prediction')

# Update layout to set background color to purple and text color to orange
fig.update_layout(
    plot_bgcolor='#121212',  # Purple background for the plot area
    paper_bgcolor='#121212',  # Purple background for the entire figure
    title='Feature Impact on Prediction',
    title_x= 0,  # Center title
    title_font=dict(color='#FFA500'),  # Orange title color
    xaxis=dict(title='Feature', title_font=dict(color='#FFA500'), tickfont=dict(color='#FFA500')),
    yaxis=dict(title='Coefficient', title_font=dict(color='#FFA500'), tickfont=dict(color='#FFA500'))
)

# Update traces to set bar color to orange and make the edges black
fig.update_traces(marker=dict(color='#FFA500', line=dict(width=1, color='black')))

# Show the plot in Streamlit
st.plotly_chart(fig)
