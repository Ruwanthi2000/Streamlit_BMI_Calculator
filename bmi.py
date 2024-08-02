import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate BMI
def calculate_bmi(weight, height):
    bmi = weight / (height / 100) ** 2
    return bmi

# Function to classify BMI
def classify_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

# Function to create BMI category chart from CSV
def bmi_category_chart():
    df = pd.read_csv("bmi.csv")
    df['BMI'] = df.apply(lambda row: calculate_bmi(row['Weight'], row['Height']), axis=1)
    df['Category'] = df['BMI'].apply(classify_bmi)
    
    category_counts = df['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']

    fig = px.bar(category_counts, x="Category", y="Count", color="Category",
                 labels={"Count": "Number of People", "Category": "BMI Category"},
                 title="Distribution of BMI Categories")
    return fig

# Function to create height vs weight scatter plot
def height_weight_scatter_plot():
    df = pd.read_csv("bmi.csv")
    fig = px.scatter(df, x='Height', y='Weight', color='Gender',
                     labels={'Height': 'Height (cm)', 'Weight': 'Weight (kg)'},
                     title="Height vs Weight Scatter Plot")
    return fig

# Function to create BMI distribution plot
def bmi_distribution_plot():
    df = pd.read_csv("bmi.csv")
    df['BMI'] = df.apply(lambda row: calculate_bmi(row['Weight'], row['Height']), axis=1)
    fig = px.histogram(df, x='BMI', color='Gender', marginal='rug',
                       labels={'BMI': 'BMI'},
                       title="BMI Distribution")
    return fig

# Home Page
def home():
    
    st.title("BMI Calculator App")

    st.image("image1.jpg")
    st.write("""
    **Body Mass Index (BMI)** is a simple index of weight-for-height that is commonly used to classify underweight, normal weight, overweight, and obesity in adults.
    
    The BMI categories are:
    - **Underweight**: BMI < 18.5
    - **Normal weight**: BMI 18.5 - 24.9
    - **Overweight**: BMI 25 - 29.9
    - **Obese**: BMI ≥ 30
    
    The BMI is defined as the body mass divided by the square of the body height, and is universally expressed in units of kg/m².
    """)
    st.image("image2.jpg")

    st.write("""
    ### Significance of BMI
    - **Underweight**: Increased risk of malnutrition, osteoporosis, and anemia.
    - **Normal weight**: Lower risk of heart disease, diabetes, and other health issues.
    - **Overweight**: Higher risk of cardiovascular diseases, high blood pressure, and type 2 diabetes.
    - **Obese**: Significantly higher risk of serious health conditions, including heart disease, diabetes, and certain cancers.
    
    Maintaining a healthy BMI is important for overall health and well-being. It can help reduce the risk of developing chronic diseases and improve quality of life.
    """)

    st.write("### BMI Distribution")
    fig = bmi_distribution_plot()
    st.plotly_chart(fig)

    st.write("### Height vs Weight Scatter Plot")
    fig2 = height_weight_scatter_plot()
    st.plotly_chart(fig2)

    st.write("### BMI Category Distribution")
    fig3 = bmi_category_chart()
    st.plotly_chart(fig3)


    st.write("""
    BMI is a useful tool for identifying weight problems, but it should not be the only measure. It does not account for muscle mass, bone density, overall body composition, and racial and sex differences.
    
    For a more accurate assessment of an individual's health, it is important to consider other factors such as diet, physical activity, and family history of diseases.
    
    ### Tips for Maintaining a Healthy BMI
    - **Eat a balanced diet**: Include plenty of fruits, vegetables, whole grains, and lean proteins.
    - **Exercise regularly**: Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week.
    - **Stay hydrated**: Drink plenty of water throughout the day.
    - **Get enough sleep**: Aim for 7-9 hours of sleep each night.
    - **Avoid sugary drinks and excessive alcohol consumption**.
    """)
    st.audio("fit.mp3")
    st.video("meditate.mp4")

# Prediction Page with Model Integration
def prediction():
    st.title("BMI Prediction")
    
    st.write("""
    Use this tool to calculate your BMI and understand what it means for your health.
    
    ### Instructions
    1. **Enter your weight**: Provide your weight in kilograms (kg).
    2. **Enter your height**: Provide your height in centimeters (cm).
    3. **Click 'Calculate BMI'**: The app will compute your BMI and display the result.
    """)

    weight = st.number_input("Enter your weight (in kg):", min_value=1.0, max_value=200.0, step=0.1)
    height = st.number_input("Enter your height (in cm):", min_value=50.0, max_value=250.0, step=0.1)

    # Load and prepare the data for training the model
    df = pd.read_csv("bmi.csv")
    df['BMI'] = df.apply(lambda row: calculate_bmi(row['Weight'], row['Height']), axis=1)
    df['Category'] = df['BMI'].apply(classify_bmi)
    
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)  # Encode Gender as 0/1
    
    X = df[['Gender', 'Height', 'Weight']]
    y = df['Category'].apply(lambda x: ['Underweight', 'Normal weight', 'Overweight', 'Obese'].index(x))  # Encode categories

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'bmi_model.pkl')

    # Load the model
    model = joblib.load('bmi_model.pkl')

    # Predict BMI category based on user input
    gender = 1 if st.selectbox("Select your gender", ["Male", "Female"]) == "Male" else 0
    height_meters = height / 100  # convert height to meters
    input_data = pd.DataFrame({'Gender': [gender], 'Height': [height_meters], 'Weight': [weight]})
    
    if st.button("Calculate BMI and Predict Category"):
        bmi = calculate_bmi(weight, height)  # height in cm for BMI calculation
        st.write(f"Your BMI is: {bmi:.2f}")

        category = classify_bmi(bmi)
        st.write(f"Based on your BMI, you are classified as: **{category}**")

        # Make prediction using the model
        prediction = model.predict(input_data)[0]
        categories = ['Underweight', 'Normal weight', 'Overweight', 'Obese']
        predicted_category = categories[prediction]

        st.write(f"Model Prediction: **{predicted_category}**")

        if predicted_category == "Underweight":
            st.write("""
            <div style='color: blue;'>
                ### Underweight:
                - It's important to eat a nutritious diet and consult a healthcare provider if needed.
            </div>
            """, unsafe_allow_html=True)
        elif predicted_category == "Normal weight":
            st.write("""
            <div style='color: green;'>
                ### Normal weight:
                - Maintain your healthy habits to keep your BMI in this range.
            </div>
            """, unsafe_allow_html=True)
        elif predicted_category == "Overweight":
            st.write("""
            <div style='color: orange;'>
                ### Overweight:
                - Consider lifestyle changes such as a healthier diet and more physical activity.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("""
            <div style='color: red;'>
                ### Obese:
                - It's advisable to consult a healthcare provider for a comprehensive health assessment and personalized advice.
            </div>
            """, unsafe_allow_html=True)

    # Show model accuracy and classification report
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: **{accuracy:.2f}**")

    # Convert classification report to DataFrame
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Plot classification report as a highlight graph
    st.write("### Classification Report")
    fig, ax = plt.subplots()
    sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, cmap='Blues', fmt='.2f', ax=ax)
    plt.title("Classification Report")
    st.pyplot(fig)

# Main function
def main():
    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        home()
    elif choice == "Prediction":
        prediction()

if __name__ == '__main__':
    main()
