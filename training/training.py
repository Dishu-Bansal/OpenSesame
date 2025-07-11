import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score
import joblib # For parsing history sequence
from pathlib import Path

def load_training_data(file_path="./training/data.csv"):
    try:
        if not Path(file_path).exists():
            print(f"Training data file {file_path} not found")
            raise FileNotFoundError(f"Training data file {file_path} not found")
        
        df = pd.read_csv(file_path)
        required_columns = [
            'history_sequence', 'user_prompt', 'user_intent', 'candidate_endpoint',
            'candidate_http_method', 'candidate_summary', 'next_call_user_made',
            'llm_reason_for_candidate', 'is_actual_next_call'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns in training data: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df['is_actual_next_call'] = df['is_actual_next_call'].fillna(0)
        df = df.drop(columns=["candidate_has_path_parameters", "candidate_has_query_parameters", "candidate_requires_request_body", "candidate_path_depth"])
        print(f"Loaded training data with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Failed to load training data: {str(e)}")
        raise

def get_embedding(text, model):
    if pd.isna(text) or text == "":
        return np.zeros(model.get_sentence_embedding_dimension())
    return model.encode(str(text))

def parse_history_sequence(history_str):
    if pd.isna(history_str) or history_str == "":
        return {
            'last_call_endpoint': None,
            'last_call_http_method': None,
            'last_call_has_parameters': False,
            'last_call_path_depth': 0,
            'history_length': 0,
            'unique_endpoints_in_history': 0,
            'history_get_ratio': 0.0,
            'history_post_ratio': 0.0,
            'history_put_ratio': 0.0,
            'history_delete_ratio': 0.0
        }
    
    calls = history_str.split('; ')
    history_length = len(calls)
    
    endpoints = []
    methods = []
    has_params = []
    
    for call in calls:
        parts = call.strip().split(' ')
        if len(parts) == 2:
            method, endpoint = parts
            endpoints.append(endpoint)
            methods.append(method)
            has_params.append('{' in endpoint or endpoint.endswith('}')) # Simple check for path params
        else: # Handle cases like just an endpoint or malformed
            endpoints.append(call)
            methods.append('GET') # Default for malformed
            has_params.append(False)

    last_call_endpoint = endpoints[-1] if endpoints else None
    last_call_http_method = methods[-1] if methods else None
    last_call_has_parameters = has_params[-1] if has_params else False
    last_call_path_depth = len(last_call_endpoint.split('/')) - 1 if last_call_endpoint else 0

    unique_endpoints_in_history = len(set(endpoints))

    method_counts = {
        'GET': methods.count('GET'),
        'POST': methods.count('POST'),
        'PUT': methods.count('PUT'),
        'DELETE': methods.count('DELETE'),
        'PATCH': methods.count('PATCH'), # Added PATCH for completeness
        'HEAD': methods.count('HEAD'),
        'OPTIONS': methods.count('OPTIONS'),
    }
    total_methods = sum(method_counts.values())
    
    history_get_ratio = method_counts['GET'] / total_methods if total_methods > 0 else 0.0
    history_post_ratio = method_counts['POST'] / total_methods if total_methods > 0 else 0.0
    history_put_ratio = method_counts['PUT'] / total_methods if total_methods > 0 else 0.0
    history_delete_ratio = method_counts['DELETE'] / total_methods if total_methods > 0 else 0.0

    return {
        'last_call_endpoint': last_call_endpoint,
        'last_call_http_method': last_call_http_method,
        'last_call_has_parameters': last_call_has_parameters,
        'last_call_path_depth': last_call_path_depth,
        'history_length': history_length,
        'unique_endpoints_in_history': unique_endpoints_in_history,
        'history_get_ratio': history_get_ratio,
        'history_post_ratio': history_post_ratio,
        'history_put_ratio': history_put_ratio,
        'history_delete_ratio': history_delete_ratio
    }

def process_dataframe(df, embedding_model, is_training=True, fitted_encoder=None):
    try:    
        print("Starting feature engineering...")

        # Handle empty prompts (optional strings)
        df['user_prompt'] = df['user_prompt'].fillna('')
        # df['candidate_endpoint'] = df['predicted_next_call'].apply(lambda x: x.strip().split(" ")[1])
        # df['candidate_http_method'] = df['predicted_next_call'].apply(lambda x: x.strip().split(" ")[0])

        # --- Embeddings ---
        print("Generating embeddings...")
        df['prompt_embedding'] = df['user_prompt'].apply(lambda x: get_embedding(x, embedding_model))
        df['intent_embedding'] = df['user_intent'].apply(lambda x: get_embedding(x, embedding_model))
        df['candidate_endpoint_embedding'] = df['candidate_endpoint'].apply(lambda x: get_embedding(x, embedding_model))
        df['candidate_summary_embedding'] = df['candidate_summary'].apply(lambda x: get_embedding(x, embedding_model))

        # --- Parse History Sequence ---
        print("Parsing history sequences...")
        history_features_df = df['history_sequence'].apply(parse_history_sequence).apply(pd.Series)
        df = pd.concat([df, history_features_df], axis=1)

        # --- Embeddings for last_call_endpoint (after parsing) ---
        df['last_call_endpoint_embedding'] = df['last_call_endpoint'].apply(lambda x: get_embedding(x, embedding_model))

        # --- Cosine Similarities ---
        print("Calculating cosine similarities...")
        # Reshape embeddings for cosine_similarity (needs 2D arrays)
        def calculate_similarity(emb1_series, emb2_series):
            # Ensure they are aligned and convert to matrix
            emb1_matrix = np.array(list(emb1_series))
            emb2_matrix = np.array(list(emb2_series))
            
            # Calculate row-wise cosine similarity
            similarities = np.array([cosine_similarity(emb1_matrix[i].reshape(1, -1), emb2_matrix[i].reshape(1, -1))[0][0] for i in range(len(emb1_matrix))])
            return similarities

        df['sim_prompt_candidate_endpoint'] = calculate_similarity(df['prompt_embedding'], df['candidate_endpoint_embedding'])
        df['sim_prompt_candidate_summary'] = calculate_similarity(df['prompt_embedding'], df['candidate_summary_embedding'])
        df['sim_intent_candidate_endpoint'] = calculate_similarity(df['intent_embedding'], df['candidate_endpoint_embedding'])
        df['sim_intent_candidate_summary'] = calculate_similarity(df['intent_embedding'], df['candidate_summary_embedding'])
        df['sim_last_call_candidate_endpoint'] = calculate_similarity(df['last_call_endpoint_embedding'], df['candidate_endpoint_embedding'])

        # --- Categorical Encoding (One-Hot) ---
        print("One-hot encoding categorical features...")
        categorical_cols = [
            'candidate_http_method', 'last_call_http_method'
        ]
        # Ensure all possible categories are present for robust encoding
        # In a real scenario, you'd fit the encoder on the full dataset, not just this batch
        all_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'] 
        for col in categorical_cols:
            df[col] = pd.Categorical(df[col], categories=all_methods)
            # Fill NA for categorical columns before encoding if any, or OHE might treat NA as a category
            # df[col] = df[col].fillna('UNKNOWN') # Or a more appropriate default categor
        
        # df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dummy_na=False)
        # Apply OneHotEncoder
        if is_training:
            # Fit a new encoder during training
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = encoder.fit_transform(df[categorical_cols])
            # Store this encoder for later inference
            global global_one_hot_encoder
            global_one_hot_encoder = encoder
        else:
            # Use the pre-fitted encoder during inference
            if fitted_encoder is None:
                raise ValueError("Fitted OneHotEncoder must be provided for inference.")
            encoder = fitted_encoder
            encoded_data = encoder.transform(df[categorical_cols])
        
        # Create DataFrame for encoded columns
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

        # --- Numerical Features (ensure correct types) ---
        numerical_cols = [
            'last_call_has_parameters', 'last_call_path_depth',
            'history_length', 'unique_endpoints_in_history',
            'history_get_ratio', 'history_post_ratio', 'history_put_ratio', 'history_delete_ratio',
            'sim_prompt_candidate_endpoint', 'sim_prompt_candidate_summary',
            'sim_intent_candidate_endpoint', 'sim_intent_candidate_summary',
            'sim_last_call_candidate_endpoint'
        ]
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Fill NA from coerces with 0

        # Expand embeddings into multiple columns
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        for prefix, col in [
            ('prompt_emb', 'prompt_embedding'), 
            ('intent_emb', 'intent_embedding'),
            ('cand_ep_emb', 'candidate_endpoint_embedding'), 
            ('cand_sum_emb', 'candidate_summary_embedding'),
            ('last_ep_emb', 'last_call_endpoint_embedding')
        ]:
            if col in df.columns:
                # Ensure embeddings are lists of correct length, fill with zeros if empty/None
                df[col] = df[col].apply(lambda x: x if isinstance(x, np.ndarray) and x.shape[0] == embedding_dim else np.zeros(embedding_dim))
                df = pd.concat([df.drop(columns=[col]), pd.DataFrame(df[col].tolist(), index=df.index, columns=[f'{prefix}_{i}' for i in range(embedding_dim)])], axis=1)
                # df = df.drop(columns=[col])
        
        print("Feature engineering complete.")
        return df
    except Exception as e:
        print(f"Feature Engineering failed: {str(e)}")
        raise

def predict_next_api_call(user_history_str, user_prompt_str, user_intent_str, candidates_from_llm, trained_model, embedding_model, fitted_encoder, fitted_rescaler, feature_columns_at_train_time):
    try:
        if not candidates_from_llm:
            print("No candidates provided for prediction")
            raise ValueError("No candidates provided for prediction")
        
        if not feature_columns_at_train_time:
            print("Feature columns not provided")
            raise ValueError("Feature columns not provided")
        
        prediction_data = []
        for cand in candidates_from_llm:
            row = {
                'history_sequence': user_history_str,
                'user_prompt': user_prompt_str,
                'user_intent': user_intent_str,
                'candidate_endpoint': cand['endpoint'],
                'candidate_http_method': cand['method'],
                'candidate_summary': cand['summary']
            }
            prediction_data.append(row)
        
        predict_df = pd.DataFrame(prediction_data)

        # 2. Apply the SAME feature engineering pipeline
        # NOTE: It's critical that categorical encoders (like one-hot)
        # and column order are consistent with training.
        # We pass a copy of the original dataframe to `process_dataframe` to ensure
        # the categorical columns get correct categories for one-hot encoding.
        # In a real app, you'd store and load the fitted encoders.
        
        # For a real system, you'd fit the OneHotEncoder ONCE on ALL possible methods/categories
        # observed in your full training data, and then use that fitted encoder here.
        # For this example, `process_dataframe` assumes all_methods list for categories.
        
        processed_predict_df = process_dataframe(predict_df, embedding_model, is_training=False, fitted_encoder=fitted_encoder)

        # Ensure columns match training data's columns, fill missing with 0
        # This is vital for consistent prediction
        X_predict = processed_predict_df.reindex(columns=feature_columns_at_train_time, fill_value=0)

        # 3. Make predictions
        # Get raw probabilities from the LightGBM model
        raw_probabilities = trained_model.predict_proba(X_predict)[:, 1]

        # rescaled = calibrator.predict_proba(raw_probabilities.reshape(-1, 1))[:, 1]
        # Apply the custom rescaler to get desired scores
        rescaled_probabilities = fitted_rescaler.transform(raw_probabilities)
        sum  = rescaled_probabilities.sum()

        # 4. Rank candidates
        ranked_predictions = []
        for i, cand in enumerate(candidates_from_llm):
            ranked_predictions.append({
                'endpoint': cand["method"] + " " + cand['endpoint'],
                'score': rescaled_probabilities[i]/sum,
                'reason':cand['reason'],
                'summary':cand['summary']
            })

        # Sort by score in descending order
        ranked_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked_predictions
    except Exception as e:
        print(f"Prediction failed with error: {str(e)}")
        raise

def save_model_files(score_rescaler, model, encoder, feature_columns, version):
    try:
        version_file = Path("./training/version.txt")
        version_file.write_text(str(version))
        joblib.dump(score_rescaler, f'./training/score_rescaler_{version}.pkl')
        joblib.dump(model, f'./training/lgbm_model_{version}.pkl')
        joblib.dump(encoder, f'./training/one_hot_encoder_{version}.pkl')
        joblib.dump(feature_columns, f'./training/calls_list_{version}.joblib')
        print(f"Model files saved for version {version}")
    except Exception as e:
        print(f"Failed to save model files for version {version}: {str(e)}")
        raise

class ScoreRescaler:
    def __init__(self, desired_min=0.5, desired_max=0.95, power=0.3):
        self.desired_min = desired_min
        self.desired_max = desired_max
        self.power = power
        self.raw_min = None
        self.raw_max = None

    def fit(self, probabilities):
        """Learns the min/max from the input probabilities."""
        self.raw_min = probabilities.min()
        self.raw_max = probabilities.max()
        print(f"ScoreRescaler fitted: Raw min={self.raw_min:.4f}, Raw max={self.raw_max:.4f}")

    def transform(self, probabilities):
        """Applies a power transformation and then scales to the desired range."""
        if self.raw_min is None or self.raw_max is None:
            raise RuntimeError("ScoreRescaler must be fitted before transforming.")

        # Apply power transformation first
        # Add a tiny epsilon to avoid issues with 0^power
        transformed_probs = np.power(probabilities + 1e-9, self.power)

        # Rescale transformed values to the desired range
        transformed_min = np.power(self.raw_min + 1e-9, self.power)
        transformed_max = np.power(self.raw_max + 1e-9, self.power)
        
        if transformed_max - transformed_min == 0:
            return np.full_like(probabilities, self.desired_max) # All same, set to max

        scaled_probs = self.desired_min + \
                       (transformed_probs - transformed_min) * \
                       (self.desired_max - self.desired_min) / \
                       (transformed_max - transformed_min)
        
        # Clip to ensure within desired range (due to float precision or edge cases)
        scaled_probs = np.clip(scaled_probs, self.desired_min, self.desired_max)
        return scaled_probs

if __name__ == "__main__":
    try:
        from sklearn.model_selection import train_test_split
        # Load the Sentence Transformer model
        print("Loading Sentence Transformer model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")

        global_one_hot_encoder = None
        global_feature_columns = None # Store the final feature columns after encoding

        df = load_training_data()
        processed_df = process_dataframe(df.copy(), embedding_model)

        # Define features (X) and target (y)
        # Drop original columns used for feature engineering, and the labels/reasons
        features_to_drop = [
            'history_sequence', 'user_prompt', 'user_intent', 
            'candidate_endpoint', 'candidate_summary', 
            'last_call_endpoint', # Original string, now embedded
            'next_call_user_made', 'llm_reason_for_candidate'
        ]
        X = processed_df.drop(columns=features_to_drop + ['is_actual_next_call'])
        y = processed_df['is_actual_next_call']

        # Store these column names globally after training data processing
        global_feature_columns = X.columns.tolist()

        print("\nFeatures (X) shape:", X.shape)
        print("Target (y) shape:", y.shape)
        # print("\nSample of X (first 5 rows and a few columns):")
        # print(X.head())

        # --- 3. Split Data for Training and Testing ---
        # Use stratify=y to ensure train/test sets have similar proportions of 1s and 0s
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Second split: X_train_cal and y_train_cal into 75% (train) and 25% (calibration)
        # This results in: ~60% train, ~20% calibration, ~20% test

        print(f"\nTraining data samples (LightGBM): {len(X_train)}")
        print(f"Test data samples: {len(X_test)}")

        print("\nTraining LightGBM model...")

        # Initialize LightGBM Classifier
        # Using 'binary' objective for binary classification
        # 'is_unbalance' can help with imbalanced datasets (if 0s vastly outnumber 1s)
        # 'metric' can be 'auc' (Area Under ROC Curve) or 'binary_logloss'
        model = lgb.LGBMClassifier(objective='binary', 
                                metric='auc', 
                                is_unbalance=True, # Helps with imbalanced classes, as 0s will be more common
                                random_state=42,
                                n_estimators=200, # Number of boosting rounds
                                learning_rate=0.05,
                                num_leaves=31)

        # Train the model
        model.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(10, verbose=False)]) # Stop if validation AUC doesn't improve for 10 rounds

        print("Model training complete.")

        # Evaluate on test set (optional, but good practice)
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class (1)
        from sklearn.metrics import roc_auc_score

        auc_score = roc_auc_score(y_test, y_pred_proba)
        ap_score = average_precision_score(y_test, y_pred_proba)
        # For precision/recall, you'd need to choose a threshold. For ranking, AUC is more relevant.
        print(f"Test AUC Score: {auc_score:.4f}")
        print(f"Test AP Score: {ap_score:.4f}")
        print(f"Min raw test proba: {y_pred_proba.min():.4f}, Max raw test proba: {y_pred_proba.max():.4f}")
        print(f"Distribution of raw test proba (percentiles):")
        print(np.percentile(y_pred_proba, [0, 10, 25, 50, 75, 90, 100]))

        # You can also inspect feature importances
        # print("\nFeature Importances:")
        # feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        # print(feature_importance.head(10))

        score_rescaler = ScoreRescaler(desired_min=0.01, desired_max=0.99, power=0.5) # Adjust power and range as needed
        score_rescaler.fit(y_pred_proba)

        # --- C. Test the Rescaler on the Test Set (Optional but recommended) ---
        rescaled_test_proba = score_rescaler.transform(y_pred_proba)
        print(f"\nMin rescaled test proba: {rescaled_test_proba.min():.4f}, Max rescaled test proba: {rescaled_test_proba.max():.4f}")
        print(f"Distribution of rescaled test proba (percentiles):")
        print(np.percentile(rescaled_test_proba, [0, 10, 25, 50, 75, 90, 100]))
        trained_feature_columns = X_train.columns.tolist()
    
        version_file = Path("./training/version.txt")

        if version_file.exists():
            # Read and increment version
            current_version = int(version_file.read_text().strip())
            new_version = current_version + 1
        else:
            # Create new version file starting at 1
            new_version = 1

        # Write updated version
        save_model_files(score_rescaler, model, global_one_hot_encoder, trained_feature_columns, new_version)
    except Exception as e:
        print(f"Model training failed with error: {str(e)}")
