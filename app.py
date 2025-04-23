from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoders
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('feature_columns.pkl', 'rb') as file:
    feature_columns = pickle.load(file)

numerical_features = [
    'Age', 'Weight_kg', 'Height_cm', 'BMI',
    'Daily_Caloric_Intake', 'Cholesterol_mg/dL',
    'Blood_Pressure_mmHg', 'Glucose_mg/dL',
    'Weekly_Exercise_Hours', 'Adherence_to_Diet_Plan',
    'Dietary_Nutrient_Imbalance_Score'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data['Blood_Pressure_mmHg'] = sum(map(int, data['Blood_Pressure_mmHg'].split('/'))) / 2

    df = pd.DataFrame([data])

    categorical_features = [
        'Gender', 'Disease_Type', 'Severity',
        'Physical_Activity_Level', 'Dietary_Restrictions',
        'Allergies', 'Preferred_Cuisine'
    ]
    df = pd.get_dummies(df, columns=categorical_features)

    missing_cols = set(feature_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    df = df[feature_columns]
    df[numerical_features] = scaler.transform(df[numerical_features])

    prediction_encoded = model.predict(df)
    prediction = label_encoder.inverse_transform(prediction_encoded)[0]

    cuisine = data['Preferred_Cuisine']

    # Diet plans dictionary with meal options
    diet_plans = {
        'Balanced': {
            'Italian': "Breakfast: Oatmeal with fruits<br>Lunch: Grilled chicken salad<br>Dinner: Pasta with vegetables",
            'Mexican': "Breakfast: Avocado toast<br>Lunch: Chicken tacos<br>Dinner: Quinoa and black bean bowl",
            'Indian': "Breakfast: Poha<br>Lunch: Dal and rice<br>Dinner: Grilled paneer with vegetables",
            'Chinese': "Breakfast: Congee<br>Lunch: Stir-fried tofu and vegetables<br>Dinner: Chicken and broccoli"
        },
        'Low Carb': {
            'Italian': "Breakfast: Scrambled eggs<br>Lunch: Zucchini noodles<br>Dinner: Grilled fish with salad",
            'Mexican': "Breakfast: Omelette<br>Lunch: Lettuce wrap tacos<br>Dinner: Grilled steak with vegetables",
            'Indian': "Breakfast: Egg bhurji<br>Lunch: Chicken curry with cauliflower rice<br>Dinner: Grilled fish",
            'Chinese': "Breakfast: Egg drop soup<br>Lunch: Chicken lettuce wraps<br>Dinner: Beef and broccoli"
        },
        'Low Sugar Diet': {
            'Italian': "Breakfast: Scrambled eggs<br>Lunch: Grilled chicken salad (no dressing)<br>Dinner: Grilled fish with salad",
            'Mexican': "Breakfast: Omelette<br>Lunch: Chicken tacos (no sauce)<br>Dinner: Grilled chicken with vegetables",
            'Indian': "Breakfast: Scrambled eggs with vegetables<br>Lunch: Dal and rice (less rice)<br>Dinner: Grilled paneer with vegetables (no sugar)",
            'Chinese': "Breakfast: Egg drop soup (no sugar)<br>Lunch: Stir-fried tofu (no sugar)<br>Dinner: Grilled chicken and broccoli"
        },
        'High Protein Diet': {
            'Italian': "Breakfast: Scrambled eggs<br>Lunch: Grilled chicken with quinoa<br>Dinner: Grilled fish with vegetables",
            'Mexican': "Breakfast: Omelette with cheese<br>Lunch: Chicken tacos with extra chicken<br>Dinner: Grilled steak with vegetables",
            'Indian': "Breakfast: Egg bhurji<br>Lunch: Chicken curry with cauliflower rice<br>Dinner: Grilled fish",
            'Chinese': "Breakfast: Egg drop soup<br>Lunch: Chicken lettuce wraps<br>Dinner: Beef and broccoli"
        },
        'Low Sodium Diet': {
            'Italian': "Breakfast: Oatmeal with fruits (no added salt)<br>Lunch: Grilled chicken (no added salt)<br>Dinner: Grilled fish with vegetables",
            'Mexican': "Breakfast: Avocado toast (no added salt)<br>Lunch: Chicken tacos (low sodium seasoning)<br>Dinner: Grilled fish with vegetables",
            'Indian': "Breakfast: Poha (low sodium)<br>Lunch: Dal (no salt) with rice<br>Dinner: Grilled paneer with vegetables",
            'Chinese': "Breakfast: Congee (no added salt)<br>Lunch: Stir-fried tofu with vegetables (low sodium sauce)<br>Dinner: Grilled chicken with broccoli"
        }
    }

    # Get the meal plan for the predicted diet type and preferred cuisine
    meal_plan = diet_plans.get(prediction, {}).get(cuisine, "Diet plan not available.")

    return render_template("result.html", diet_type=prediction, cuisine=cuisine, meal_plan=meal_plan)

if __name__ == '__main__':
    app.run(debug=True)
