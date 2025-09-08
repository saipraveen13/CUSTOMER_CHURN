# Customer Churn Prediction (End-to-End Project)

This is a **basic end-to-end Data Science project** where we build, train, and deploy a **Customer Churn Prediction Model**.  
The model predicts whether a customer is likely to **churn (leave the company)** or **stay**, based on customer attributes like tenure, services, charges, and payment method.

The project is deployed using **Flask**, with a **frontend built in HTML + CSS**, and the prediction is shown dynamically on the same page with animations.


## Project Workflow

1. **Data Preprocessing & Training**
   - Dataset: `Telco-Customer-Churn.csv`
   - Encoded categorical features using `LabelEncoder`
   - Trained a **Logistic Regression** model
   - Saved model (`churn_model.pkl`), encoders (`encoders.pkl`), and feature order (`feature_order.pkl`)

2. **Flask Backend (`app.py`)**
   - Loads trained model and encoders
   - Accepts user inputs from form
   - Makes predictions and returns probability of churn

3. **Frontend (HTML + CSS)**
   - User-friendly input form (`index.html`)
   - Modern styling with `style.css`
   - Loader animation while predicting
   - Prediction result with fade-in effect

## 📂 File Structure
customer-churn/
│── app.py # Flask app for deployment
│── train.py # Model training script
│── churn_model.pkl # Trained Logistic Regression model
│── encoders.pkl # Encoders for categorical variables
│── feature_order.pkl # Feature order for consistent prediction
│── Telco-Customer-Churn.csv # Dataset
│── templates/
│ └── index.html # Frontend form
│── static/
│ └── style.css # Styling + animations
│── README.md # Project documentation

