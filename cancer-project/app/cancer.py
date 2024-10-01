import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="BREAST CANCER PREDICTION MODEL",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_model_features():
    try:
        with open('../model/model.pkl', "rb") as model_file:
            model = pickle.load(model_file)
        
        # If the model has feature names as an attribute
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        else:
            # Load a sample of your training data to get feature names
            data = pd.read_csv("../model/data.csv")
            return list(data.drop('diagnosis', axis=1).columns)
    except Exception as e:
        st.error(f"Error loading model or feature names: {str(e)}")
        return None

def get_data():
    try:
        data = pd.read_csv("../model/data.csv")
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please check if 'data.csv' exists in the correct path.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def add_side_bar():
    st.sidebar.header("CELL MEASUREMENTS")
    data = get_data()
    model_features = get_model_features()
    
    if data is None or model_features is None:
        return None
    
    input_dict = {}
    
    # Create a mapping of display names to feature names
    feature_display_names = {
        'mean': "Mean",
        'se': "Standard Error",
        'worst': "Worst"
    }
    
    for feature in model_features:
        # Split the feature name to get the measurement type and category
        parts = feature.split('_')
        if len(parts) >= 2:
            category = parts[0]
            measure_type = parts[-1]
            
            # Create a more readable label
            if measure_type in feature_display_names:
                display_name = f"{category.capitalize()} ({feature_display_names[measure_type]})"
            else:
                display_name = feature.replace('_', ' ').title()
            
            # Create the slider
            input_dict[feature] = st.sidebar.slider(
                display_name,
                min_value=float(data[feature].min()),
                max_value=float(data[feature].max()),
                value=float(data[feature].mean())
            )
    
    return input_dict

def get_scaled_values(input_dict):
    data = get_data()
    if data is None or input_dict is None:
        return None
    
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

def get_radar_chart(input_data):
    if input_data is None:
        return None
    
    scaled_data = get_scaled_values(input_data)
    if scaled_data is None:
        return None
    
    # Group features by type (mean, se, worst)
    feature_groups = {
        'mean': [],
        'se': [],
        'worst': []
    }
    
    for feature, value in scaled_data.items():
        for group in feature_groups:
            if feature.endswith(group):
                feature_groups[group].append((feature, value))
    
    # Get unique categories (excluding the type suffix)
    categories = list(set(feature.split('_')[0] for feature in scaled_data.keys()))
    
    fig = go.Figure()
    
    for group, features in feature_groups.items():
        if features:
            values = [0] * len(categories)
            for feature, value in features:
                category = feature.split('_')[0]
                category_index = categories.index(category)
                values[category_index] = value
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=group.upper()
            ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    
    return fig

def add_predictions(input_data):
    if input_data is None:
        return
    
    try:
        with open('../model/model.pkl', "rb") as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        st.error("Model file not found. Please check if 'model.pkl' exists in the correct path.")
        return
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Ensure the input features match the model's expected features
    model_features = get_model_features()
    input_array = np.array([input_data[feature] for feature in model_features]).reshape(1, -1)
    
    try:
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)
        
        st.subheader("Cell Cluster Prediction")
        st.write("The cell cluster is:")
        
        if prediction[0] == 0:
            st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
        else:
            st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
        
        st.write(f"Probability of being benign: {probability[0][0]:.4f}")
        st.write(f"Probability of being malicious: {probability[0][1]:.4f}")
        
        st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

def main():
    with st.container():
        st.title("BREAST CANCER PREDICTION")
        st.write("This is a prediction app which uses a machine learning model to determine whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can update the measurements using the sliders in the sidebar.")
    
    css_path = '../assets/style.css'   
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. The app will continue without custom styling.")
    
    input_data = add_side_bar()
    
    if input_data is not None:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            radar_chart = get_radar_chart(input_data)
            if radar_chart is not None:
                st.plotly_chart(radar_chart)
        
        with col2:
            add_predictions(input_data)

if __name__ == "__main__":
    main()