import gradio as gr
import dill
import pandas as pd
import numpy as np

# ✅ Load model and preprocessor
model = dill.load(open('artifacts/model.pkl', 'rb'))
preprocessor = dill.load(open('artifacts/preprocessor.pkl', 'rb'))

# ---------- Prediction Function ----------
def predict_marks(gender, race_ethnicity, parental_level_of_education, lunch,
                   test_preparation_course, math_score, reading_score, writing_score):
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([{
            'gender': gender,
            'race_ethnicity': race_ethnicity,
            'parental_level_of_education': parental_level_of_education,
            'lunch': lunch,
            'test_preparation_course': test_preparation_course,
            'writing_score': writing_score,
            'reading_score': reading_score
        }])

        # Apply preprocessor transformation (same as training)
        transformed_input = preprocessor.transform(input_df)

        # Predict using trained model
        prediction = model.predict(transformed_input)[0]

        return f"🎯 Predicted Math Score: {round(prediction, 2)}"

    except Exception as e:
        return f"❌ Error: {e}"


# ---------- UI Components ----------
gender_options = ["male", "female"]
race_options = ["group A", "group B", "group C", "group D", "group E"]
education_options = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
]
lunch_options = ["standard", "free/reduced"]
test_prep_options = ["none", "completed"]

inputs = [
    gr.Radio(gender_options, label="Gender"),
    gr.Dropdown(race_options, label="Race/Ethnicity"),
    gr.Dropdown(education_options, label="Parental Level of Education"),
    gr.Radio(lunch_options, label="Lunch Type"),
    gr.Radio(test_prep_options, label="Test Preparation Course"),
    gr.Slider(0, 100, value=70, label="Writing Score"),
    gr.Slider(0, 100, value=70, label="Reading Score"),
    gr.Slider(0, 100, value=70, label="(Optional) Math Score", interactive=False)
]

# ---------- Output ----------
outputs = gr.Markdown(label="Predicted Score")

# ---------- Interface ----------
app = gr.Interface(
    fn=predict_marks,
    inputs=inputs,
    outputs=outputs,
    title="🎓 Student Marks Prediction Dashboard",
    description="""
### Predict a student's expected Math score  
Based on reading, writing, and demographic factors, estimate their performance using your trained ML model.
""",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan"),
    allow_flagging="never"
)

# ---------- Run ----------
if __name__ == "__main__":
    app.launch(share=True)
