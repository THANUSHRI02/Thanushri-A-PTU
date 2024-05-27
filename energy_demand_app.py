import streamlit as st
import joblib
import pandas as pd

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "How It Works", "About Us"])


def home_page():
    """
    This function creates a Streamlit application for electricity load forecasting
    using a pre-trained XGBoost model.
    """

    # Load the XGBoost model
    model = joblib.load("Energy_demand_forecast_model.pkl") 

    # Streamlit app title and description
    st.title("Energy demand Forecasting")
    st.write("Use this app to predict energy demand based on historical data.")

    def get_numeric_input(label, min_value=None, max_value=None, data_type=float):
        while True:
            value = st.number_input(label, min_value=min_value, max_value=max_value)
            try:
                # Ensure data type is correct (e.g., float or int)
                converted_value = data_type(value)
                return converted_value
            except ValueError:
                st.error(f"Invalid input for '{label}'. Please enter a number.")

    # Define features dictionary with input fields
    features_dict = {
        'Day': get_numeric_input('Day',min_value=1, max_value=31, data_type=int),
        'Month': get_numeric_input('Month',min_value=1, max_value=12, data_type=int),
        'Year': get_numeric_input('Year', min_value=2000, data_type=int),
        'Time_of_Day_encoded': get_numeric_input('Time of Day: 0(Afternoon), 1(Evening), 2(Morning), 3(Night)',min_value=0, max_value=3,data_type=int),
        'Electricity_Load': st.number_input('Electricity Load (kW)'),
        'Temperature': st.number_input('Temperature (°C)'),
        'Humidity': st.number_input('Humidity (%)'),
        'Holiday_Indicator': get_numeric_input('Holiday Indicator: 0 (Not a holiday) , 1 (Holiday)', min_value=0, max_value=1,data_type=int), 
        'Previous_Load': st.number_input('Previous Load'), 
        'Transportation_Data': st.number_input('Transportation Data'), 
        'Operational_Metrics': st.number_input('Operation Metrics'),
        'System_Load': st.number_input('System Load'),
        'External_Factors_encoded': get_numeric_input('External Factor 0(Economic), 1(Other), 2(Regulatory)', min_value=0, max_value=1,data_type=int)
    }

    # Button to trigger prediction
    if st.button("Predict Energy demand"):

        # Convert user input to DataFrame
        user_data = pd.DataFrame(features_dict, index=[0])

        # Make prediction using the model
        prediction = model.predict(user_data)[0]

        # Display prediction to the user
        st.write("Predicted Energy demand:", prediction)

def how_it_works_page():

    st.title("How It Works: Forcast Your Energy Demand with Ease")

    # Explain the process visually with an infographic (optional)
    st.image("Our Model.png", width=800)  # Replace with your infographic image path

    st.subheader("A Step-by-Step Look at Our Prediction Process:")

    # Break down the steps with clear descriptions
    with st.expander("1. Data Collection: The Foundation of Accuracy"):
        st.write(
            """
            Imagine a chef crafting a delicious meal. Just as high-quality ingredients are essential, accurate energy demand prediction relies on comprehensive data. 
            We gather data from a wide range of sources to create a comprehensive picture of energy-influencing factors. 
            This includes weather forecasts, historical energy usage patterns, and operational metrics from your facility. 
            The more data we have, the more accurate your predictions will be!
            """
        )

    with st.expander("2. Data Cleaning & Preprocessing: Preparing the Ingredients"):
        st.write(
            """
            Just as a chef cleans and prepares ingredients before cooking, we meticulously clean and preprocess the acquired data. This critical step ensures:
                - Handling Missing Values: We address missing data points using appropriate techniques to maintain data integrity.
                - Format Standardization: Data from various sources may have inconsistencies. We convert everything into a uniform format for streamlined analysis.
                - Outlier Detection: We identify and address outliers that could skew the model's predictions.
            """
        )

    with st.expander("3. Feature Engineering: Crafting the Perfect Recipe"):
        st.write(
            """
            Think of a chef creating a unique flavor profile. Feature engineering is akin to this, where we transform the data to create new features that are:
                - More Informative: We derive additional features that enhance the model's ability to learn complex relationships between variables.
                - Dimensionality Reduction: Sometimes, having too many features can be detrimental. Feature engineering helps us select the most impactful ones.
            """
        )

    with st.expander("4. Model Selection: Choosing the Right Tool for the Job"):
        st.write(
            """
            Just as a chef uses specialized tools for different dishes, we carefully choose an appropriate machine learning model. In your case, we've selected a powerful XGBoost model. This model excels at learning complex patterns in data, making it ideal for predicting energy demand based on various influencing factors.
            """
        )

    with st.expander("5. Model Training: Unleashing the Predictive Power"):
        st.write(
            """
            Now comes the exciting part! We train the XGBoost model on the preprocessed data. Imagine the model learning from past data patterns, just like a chef perfecting a recipe through experience. This training process involves:
                - Feeding Data into the Model: The model analyzes the data, identifying relationships and patterns.
                - Learning from the Data: The model adjusts its internal parameters to make increasingly accurate predictions.
                - Fine-Tuning the Process: We refine the training process as needed to optimize the model's performance.
            """
        )

    with st.expander("6. Model Evaluation: Ensuring Recipe Perfection"):
        st.write(
            """
            Before serving the final dish, a chef rigorously tests it. In the same way, we rigorously test our model using a separate dataset. This helps us assess:
                - Accuracy: How well does the model predict actual energy demand?
                - Generalizability: Can the model perform well on unseen data?
                - Overfitting: Has the model memorized the training data without learning general patterns?

            Through rigorous evaluation, we ensure our model delivers reliable energy demand predictions you can trust.
            """
        )


def about_us_page():
    st.title("About Us")
   
    st.write(
        """
        This energy demand prediction tool was developed by us from the Information Technology department at Puducherry Technological University, working in collaboration with Coapps. We are passionate about applying machine learning to solve real-world problems and contribute to sustainable energy management.
        """
    )

    # Team member introductions
    st.subheader("Team Members:")

    # Member 1
    col1, col2 = st.columns(2)
    with col1:
        st.image("Thanu.jpg", width=200)  
    with col2:
        st.write(
            """
            **Name:** Thanushri A  
            **Email:** thanushri.a@pec.edu  
            **Phone:** +91 8508670449     
            I'm a data science enthusiast with a passion for turning data into actionable insights. I leverage my knowledge of Python, data analysis libraries like Pandas, and machine learning algorithms to build solutions that bridge the gap between technical concepts and real-world applications. Beyond code, I'm proficient in data visualization tools like Power BI and Excel, allowing me to effectively communicate insights. Additionally, my experience with JotForm and other software platforms enables me to gather and analyze data efficiently. My background in IT support further strengthens my problem-solving skills and understanding of user needs. 
            """
        )

    # Member 2
    col1, col2 = st.columns(2)
    with col1:
        st.image("Jeevithaa.jpg", width=200) 
    with col2:
        st.write(
            """
            **Name:** Jeevithaa S    
            **Email:** jeevithaa.s@pec.edu    
            **Phone:** +91 9944117462    
            I'm a passionate developer with expertise in both data science and web development. I leverage my knowledge of machine learning and data analysis techniques to extract insights from data.  Skilled in Python libraries like Pandas and scikit-learn, I can build and train models to solve real-world problems.  On the web development side, I excel at creating user-friendly interfaces using React and handling server-side logic with PHP. My foundation in Java provides a strong programming base, and I'm constantly learning new technologies to stay at the forefront of data-driven development. 
            """
        )




if page == "Home":
    home_page()
elif page == "How It Works":
    how_it_works_page()
elif page == "About Us":
    about_us_page()

