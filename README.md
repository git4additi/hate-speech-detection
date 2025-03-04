# Multi-Model Hate Speech Detection System

A comprehensive hate speech detection system implementing multiple machine learning and deep learning approaches to classify text into three categories: Hate Speech, Offensive Language, and Neither. The system compares performance across different models including SVM, Decision Trees, LSTM, Bi-LSTM, and Bi-GRU.

## Use pre-trained models

```python
from tensorflow.keras.models import load_model
import pickle
import os

def load_pretrained_models(models_dir='trained-models'):
    traditional_models = {}
    dl_models = {}
    
    try:
        print("Loading traditional models...")
        for model_type in [ModelType.SVM, ModelType.DECISION_TREE]:
            model_name = model_type.value.lower()
            with open(os.path.join(models_dir, f'{model_name}.pkl'), 'rb') as f:
                traditional_models[model_type] = pickle.load(f)
        
        print("Loading deep learning models...")
        for model_type in [ModelType.LSTM, ModelType.BI_LSTM, ModelType.BI_GRU]:
            model_name = model_type.value.lower()
            model_path = os.path.join(models_dir, f'{model_name}.h5')
            dl_models[model_type] = load_model(model_path)
        
        # Load preprocessor
        print("Loading preprocessor...")
        with open(os.path.join(models_dir, 'preprocessor.pkl'), 'rb') as f:
            preprocessor = pickle.load(f)
        
        return traditional_models, dl_models, preprocessor
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None

# Example usage
traditional_models, dl_models, preprocessor = load_pretrained_models()

# Quick Tweet analysis
analyzer = TweetAnalyzer(preprocessor, traditional_models, dl_models)
result = analyzer.analyze_tweet('your offensive tweet')
```

<br />

> [!NOTE]  
> bi-gru and bi-lstm models are not available in the pre-trained directory because of GitHub file size limitations.