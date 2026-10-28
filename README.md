# Heart Disease Predictor

## Flow
1. Fill in the heart-disease form.
2. Choose Existing Model or Train a New Model.
3. If training a new model, enter Testing Size (%) and Cross-Validation Folds, then choose Stratified or Non-Stratified split.
4. View the final prediction result.

## Training configuration rules
- Testing size accepts whole numbers from 1% through 99%; 100% is rejected because no data would remain for training.
- CV folds are user-entered (2-20), subject to dataset-size and class-count validation.
- Stratified split preserves target-class proportions approximately; non-stratified split does not enforce this.
- The result page only shows the final heart disease prediction, method and navigation buttons. It does not show the probability panel, evaluation scores, CV score, recommended next step, or confusion matrix.

## Project structure
```text
Heart_Disease_Predictor/
├── app.py
├── Heart_Disease_Predictor.ipynb
├── requirements.txt
├── README.md
├── data/heart.csv
├── models/heart_disease_pipeline.pkl
├── templates/
│   ├── index.html
│   ├── choose_method.html
│   ├── training_configuration.html
│   └── result.html
└── static/heart.css
```

## Run
```powershell
python -m pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000.

Educational demonstration only; not a medical diagnostic tool.
