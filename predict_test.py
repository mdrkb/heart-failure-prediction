import requests


url = "http://localhost:8080/predict"
# url = "https://mdrkb.pythonanywhere.com/predict"

record_true = {
    "age": 40.0,
    "anaemia": True,
    "creatinine_phosphokinase": 582,
    "diabetes": False,
    "ejection_fraction": 35,
    "high_blood_pressure": True,
    "platelets": 222000.0,
    "serum_creatinine": 1.0,
    "serum_sodium": 132,
    "sex": "male",
    "smoking": True,
    "time": 244,
}

record_false = {
    "age": 82.0,
    "anaemia": False,
    "creatinine_phosphokinase": 379,
    "diabetes": True,
    "ejection_fraction": 50,
    "high_blood_pressure": True,
    "platelets": 47000.0,
    "serum_creatinine": 1.3,
    "serum_sodium": 136,
    "sex": "male",
    "smoking": True,
    "time": 13,
}

# Always send input as json array to handle batch prediction
input_json = [record_true, record_false]


response = requests.post(url, json=input_json)
if response.status_code == 200:
    print(response.json())
else:
    print(response.raise_for_status())
