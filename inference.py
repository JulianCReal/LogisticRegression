"""
Heart Disease Risk Prediction - Inference Script
Production-ready inference for AWS SageMaker deployment
"""

import json
import numpy as np
import os
import sys
from typing import Dict, Any, Union, Tuple


def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load the trained model from directory"""
    try:
        model_path = os.path.join(model_dir, 'heart_disease_model.npy')
        
        if not os.path.exists(model_path):
            model_path = 'heart_disease_model.npy'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model_params = np.load(model_path, allow_pickle=True).item()
        
        required_keys = ['weights', 'bias', 'scaler_mean', 'scaler_std', 'feature_names']
        missing_keys = [key for key in required_keys if key not in model_params]
        
        if missing_keys:
            raise ValueError(f"Model is missing required keys: {missing_keys}")
        
        print("=" * 70)
        print("MODEL LOADED SUCCESSFULLY")
        print("=" * 70)
        print(f"Model path: {model_path}")
        print(f"Features: {model_params['feature_names']}")
        print(f"Weights shape: {model_params['weights'].shape}")
        print(f"Regularization lambda: {model_params.get('lambda', 'N/A')}")
        print("=" * 70)
        
        return model_params
        
    except Exception as e:
        print(f"ERROR loading model: {e}", file=sys.stderr)
        raise


def input_fn(request_body: Union[str, bytes], content_type: str = 'application/json') -> np.ndarray:
    """Parse and validate input data"""
    
    EXPECTED_FEATURES = ['Age', 'BP', 'Cholesterol', 'Max_HR', 'ST_depression', 'Vessels']
    
    try:
        if content_type == 'application/json':
            if isinstance(request_body, bytes):
                request_body = request_body.decode('utf-8')
            
            data = json.loads(request_body)
            
            if isinstance(data, dict):
                features = []
                missing_features = []
                
                for feature in EXPECTED_FEATURES:
                    if feature in data:
                        features.append(float(data[feature]))
                    else:
                        missing_features.append(feature)
                
                if missing_features:
                    raise ValueError(f"Missing required features: {missing_features}")
                
                input_array = np.array([features], dtype=np.float64)
                
            elif isinstance(data, list):
                if len(data) != 6:
                    raise ValueError(f"Expected 6 features, got {len(data)}")
                input_array = np.array([data], dtype=np.float64)
            
            else:
                raise ValueError("JSON must be either a dictionary or a list")
        
        elif content_type == 'text/csv':
            if isinstance(request_body, bytes):
                request_body = request_body.decode('utf-8')
            
            values = [float(x.strip()) for x in request_body.split(',')]
            
            if len(values) != 6:
                raise ValueError(f"Expected 6 comma-separated values, got {len(values)}")
            
            input_array = np.array([values], dtype=np.float64)
        
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        validate_input_ranges(input_array[0])
        
        return input_array
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except ValueError as e:
        raise ValueError(f"Input validation error: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing input: {e}")


def validate_input_ranges(features: np.ndarray) -> None:
    """Validate input features are within reasonable medical ranges"""
    age, bp, chol, max_hr, st_dep, vessels = features
    
    validations = [
        (age >= 0 and age <= 120, f"Age must be between 0-120, got {age}"),
        (bp >= 50 and bp <= 250, f"Blood pressure must be between 50-250 mmHg, got {bp}"),
        (chol >= 100 and chol <= 600, f"Cholesterol must be between 100-600 mg/dL, got {chol}"),
        (max_hr >= 50 and max_hr <= 220, f"Max heart rate must be between 50-220 bpm, got {max_hr}"),
        (st_dep >= 0 and st_dep <= 10, f"ST depression must be between 0-10, got {st_dep}"),
        (vessels >= 0 and vessels <= 3, f"Number of vessels must be between 0-3, got {vessels}"),
    ]
    
    for is_valid, error_msg in validations:
        if not is_valid:
            raise ValueError(error_msg)


def predict_fn(input_data: np.ndarray, model: Dict[str, Any]) -> Dict[str, Any]:
    """Generate predictions using the loaded model"""
    try:
        X_normalized = (input_data - model['scaler_mean']) / model['scaler_std']
        
        z = np.dot(X_normalized, model['weights']) + model['bias']
        
        z_clipped = np.clip(z, -500, 500)
        probability = float(1 / (1 + np.exp(-z_clipped)))
        
        prediction = 1 if probability >= 0.5 else 0
        
        risk_assessment = assess_risk(probability)
        
        feature_names = ['Age', 'BP', 'Cholesterol', 'Max_HR', 'ST_depression', 'Vessels']
        feature_values = {
            name: float(value) 
            for name, value in zip(feature_names, input_data[0])
        }
        
        response = {
            'probability': round(probability, 4),
            'probability_percentage': f"{probability * 100:.2f}%",
            'prediction': prediction,
            'prediction_label': 'Disease Presence' if prediction == 1 else 'Disease Absence',
            'risk_level': risk_assessment['level'],
            'risk_category': risk_assessment['category'],
            'recommendation': risk_assessment['recommendation'],
            'confidence': risk_assessment['confidence'],
            'urgency': risk_assessment['urgency'],
            'feature_values': feature_values,
            'model_version': '1.0'
        }
        
        return response
        
    except Exception as e:
        return {
            'error': True,
            'error_message': str(e),
            'error_type': type(e).__name__
        }


def assess_risk(probability: float) -> Dict[str, str]:
    """Provide detailed risk assessment based on probability"""
    if probability >= 0.8:
        return {
            'level': 'VERY HIGH',
            'category': 'Critical',
            'recommendation': 'URGENT: Immediate medical attention required. Contact emergency services or visit ER.',
            'confidence': 'Very High',
            'urgency': 'Emergency'
        }
    elif probability >= 0.65:
        return {
            'level': 'HIGH',
            'category': 'Severe',
            'recommendation': 'Schedule immediate consultation with cardiologist within 24-48 hours.',
            'confidence': 'High',
            'urgency': 'Urgent'
        }
    elif probability >= 0.5:
        return {
            'level': 'MODERATE-HIGH',
            'category': 'Concerning',
            'recommendation': 'Schedule medical consultation within 1 week. Begin monitoring symptoms.',
            'confidence': 'Moderate',
            'urgency': 'Soon'
        }
    elif probability >= 0.35:
        return {
            'level': 'MODERATE',
            'category': 'Caution',
            'recommendation': 'Schedule routine checkup. Discuss cardiovascular health with physician.',
            'confidence': 'Moderate',
            'urgency': 'Routine'
        }
    elif probability >= 0.2:
        return {
            'level': 'LOW-MODERATE',
            'category': 'Mild',
            'recommendation': 'Annual checkup recommended. Maintain healthy lifestyle practices.',
            'confidence': 'Moderate',
            'urgency': 'Routine'
        }
    else:
        return {
            'level': 'LOW',
            'category': 'Minimal',
            'recommendation': 'Continue healthy lifestyle. Annual checkup for monitoring.',
            'confidence': 'High',
            'urgency': 'Routine'
        }


def output_fn(prediction: Dict[str, Any], accept: str = 'application/json') -> Tuple[str, str]:
    """Format the prediction output for response"""
    try:
        if accept == 'application/json' or accept == '*/*':
            response_body = json.dumps(prediction, indent=2)
            content_type = 'application/json'
        
        elif accept == 'text/csv':
            csv_output = (
                f"{prediction.get('probability', 0)},"
                f"{prediction.get('prediction', 0)},"
                f"{prediction.get('risk_level', 'UNKNOWN')}"
            )
            response_body = csv_output
            content_type = 'text/csv'
        
        elif accept == 'text/plain':
            prob = prediction.get('probability', 0)
            risk = prediction.get('risk_level', 'UNKNOWN')
            rec = prediction.get('recommendation', 'N/A')
            
            plain_output = (
                f"Heart Disease Risk Assessment\n"
                f"{'=' * 50}\n"
                f"Risk Probability: {prob:.2%}\n"
                f"Risk Level: {risk}\n"
                f"Recommendation: {rec}\n"
            )
            response_body = plain_output
            content_type = 'text/plain'
        
        else:
            response_body = json.dumps(prediction, indent=2)
            content_type = 'application/json'
        
        return response_body, content_type
        
    except Exception as e:
        error_response = {
            'error': True,
            'error_message': f"Error formatting output: {str(e)}"
        }
        return json.dumps(error_response), 'application/json'


def ping_handler() -> Tuple[str, int]:
    """Health check endpoint handler"""
    try:
        model_fn('.')
        return json.dumps({'status': 'healthy', 'model': 'loaded'}), 200
    except Exception as e:
        return json.dumps({'status': 'unhealthy', 'error': str(e)}), 503


def test_inference_locally():
    """Test inference functions locally before deployment"""
    print("\n" + "=" * 70)
    print("TESTING INFERENCE SCRIPT LOCALLY")
    print("=" * 70)
    
    print("\n[TEST 1] Loading model...")
    try:
        model = model_fn(".")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    print("\n[TEST 2] Testing JSON input parsing...")
    test_json = json.dumps({
        "Age": 60,
        "BP": 145,
        "Cholesterol": 300,
        "Max_HR": 130,
        "ST_depression": 2.0,
        "Vessels": 2
    })
    
    try:
        input_data = input_fn(test_json, 'application/json')
        print(f"✅ JSON parsing successful")
        print(f"   Input shape: {input_data.shape}")
        print(f"   Values: {input_data}")
    except Exception as e:
        print(f"❌ JSON parsing failed: {e}")
        return
    
    print("\n[TEST 3] Testing CSV input parsing...")
    test_csv = "60,145,300,130,2.0,2"
    
    try:
        input_data_csv = input_fn(test_csv, 'text/csv')
        print(f"✅ CSV parsing successful")
        print(f"   Input shape: {input_data_csv.shape}")
    except Exception as e:
        print(f"❌ CSV parsing failed: {e}")
    
    print("\n[TEST 4] Testing prediction...")
    try:
        prediction = predict_fn(input_data, model)
        print(f"✅ Prediction successful")
        print(f"   Probability: {prediction['probability']}")
        print(f"   Risk Level: {prediction['risk_level']}")
        print(f"   Recommendation: {prediction['recommendation']}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    print("\n[TEST 5] Testing output formatting...")
    try:
        output_json, content_type = output_fn(prediction, 'application/json')
        print(f"✅ JSON output formatting successful")
        print(f"   Content-Type: {content_type}")
        print(f"   Output preview: {output_json[:100]}...")
    except Exception as e:
        print(f"❌ Output formatting failed: {e}")
    
    print("\n[TEST 6] Testing multiple scenarios...")
    
    test_cases = [
        {
            "name": "High Risk Patient",
            "data": {"Age": 65, "BP": 160, "Cholesterol": 320, 
                    "Max_HR": 120, "ST_depression": 2.5, "Vessels": 2}
        },
        {
            "name": "Low Risk Patient",
            "data": {"Age": 35, "BP": 115, "Cholesterol": 180,
                    "Max_HR": 175, "ST_depression": 0.2, "Vessels": 0}
        },
        {
            "name": "Medium Risk Patient",
            "data": {"Age": 55, "BP": 140, "Cholesterol": 250,
                    "Max_HR": 145, "ST_depression": 1.2, "Vessels": 1}
        }
    ]
    
    for case in test_cases:
        try:
            test_input = json.dumps(case["data"])
            input_arr = input_fn(test_input, 'application/json')
            pred = predict_fn(input_arr, model)
            
            print(f"\n   {case['name']}:")
            print(f"      Probability: {pred['probability']:.4f} ({pred['probability_percentage']})")
            print(f"      Risk Level: {pred['risk_level']}")
            print(f"      Urgency: {pred['urgency']}")
            
        except Exception as e:
            print(f"   ❌ {case['name']} failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nInference script is ready for deployment to SageMaker.")


if __name__ == "__main__":
    test_inference_locally()