import os
import sys
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add the project root directory to the path
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(app_dir)
sys.path.append(project_root)

# Create Flask app
app = Flask(__name__)

# Configuration for the app
categorical_features = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']


@app.route('/')
def index():
    """Render the home page with the form."""
    return render_template('index.html',
                           categorical_features=categorical_features,
                           numerical_features=numerical_features)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle the form submission and make a prediction."""
    try:
        # Get form data
        form_data = request.form.to_dict()

        # For demonstration purposes, generate a sample prediction
        # In a real app, this would use a trained model
        churn_probability = 35.5
        prediction = 0

        # Determine risk level
        if churn_probability < 30:
            risk_level = "Low"
            risk_class = "success"
        elif churn_probability < 60:
            risk_level = "Medium"
            risk_class = "warning"
        else:
            risk_level = "High"
            risk_class = "danger"

        # Generate sample insights based on form data
        insights = []

        # Contract type insights
        if 'Contract' in form_data:
            contract = form_data['Contract']
            if contract == 'Month-to-month':
                insights.append(
                    "Month-to-month contracts have a higher churn risk. Consider offering incentives for longer contract terms.")
            elif contract in ['One year', 'Two year']:
                insights.append(
                    "Longer contracts typically have lower churn rates. This customer has a term contract which is favorable.")

        # Tenure insights
        if 'tenure' in form_data:
            tenure = float(form_data['tenure'])
            if tenure < 12:
                insights.append(
                    "New customers (less than 12 months) have a higher churn risk. Consider special retention offers.")
            elif tenure >= 24:
                insights.append("Loyal customers (2+ years) typically have lower churn rates.")

        # Add a default insight if we have none
        if not insights:
            insights.append("Customer profile suggests moderate churn risk. Regular engagement recommended.")

        # Return result
        return render_template('result.html',
                               prediction=prediction,
                               churn_probability=churn_probability,
                               risk_level=risk_level,
                               risk_class=risk_class,
                               insights=insights,
                               customer_data=form_data)

    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500


# Run the application
if __name__ == '__main__':
    # Create required directories
    os.makedirs(os.path.join(project_root, 'config'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'data', 'raw'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'data', 'processed'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'models', 'trained'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'models', 'evaluation'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

    print(f"Starting Flask application on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)