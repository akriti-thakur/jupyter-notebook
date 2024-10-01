import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os






     
st.set_page_config(
        page_title="BREAST PREDICTION MODEL",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )




def get_data():
    data= pd.read_csv("model\data.csv")
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    corr_matrix = data.corr().abs() 

    mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
    tri_df = corr_matrix.mask(mask)
    
    to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.92)]
    
    df = data.drop(to_drop, axis = 1)
    
    return df
    
    
    
    
def add_side_bar():
    st.sidebar.header("CELL MEASURMENTS")
    data= get_data()
    slider_lab = [
        
        ("Texture (mean)", "texture_mean"),
        
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        
        ("Texture (se)", "texture_se"),
        
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        
        ("Texture (worst)", "texture_worst"),
        
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    input_dict = {}
    
    for label, key in slider_lab:
        input_dict[key] = st.sidebar.slider(
          label,
          min_value=float(0),
          max_value=float(data[key].max()),
          value=float(data[key].mean())
        )
        
    return input_dict



def get_scaled_values(input_dict):
  data = get_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict



def get_radar_chart(input_data):
  
    input_data = get_scaled_values(input_data)
  
    categories = [ 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
            r=[
               input_data['texture_mean'],
               input_data['smoothness_mean'], input_data['compactness_mean'],
               input_data['concave points_mean'], input_data['symmetry_mean'],
              input_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
      ))
    fig.add_trace(go.Scatterpolar(
            r=[
              input_data['texture_se'],  input_data['area_se'],
              input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
              input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
      ))
    fig.add_trace(go.Scatterpolar(
            r=[
               input_data['texture_worst'], 
              input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
              input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
              input_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
      ))
    
    fig.update_layout(
        polar=dict(
          radialaxis=dict(
            visible=True,
            range=[0, 1]
          )),
        showlegend=True
      )
      
    return fig
    


def add_predictions(input_data):
  
  try:
      with open('model\model.pkl', "rb") as model_file:
          model = pickle.load(model_file)
  except FileNotFoundError:
      st.error("Model file not found. Please check the file location.")
      return
  except Exception as e:
      st.error(f"Error loading model: {str(e)}")
      return
  
    
  
  input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
  
  
  
  prediction = model.predict(input_data_reshaped)
  
  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")
  
  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
    
  
  st.write("Probability of being benign: ", model.predict_proba(input_data_reshaped)[0][0])
  st.write("Probability of being malicious: ", model.predict_proba(input_data_reshaped)[0][1])
  
  st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")








def main():
   
    with st.container():
        st.title("BREAST CANCER PREDICTION")
        st.write("This is a prediction app which uses a machine learning model to determine whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")
       
       
    css_path = 'assets\style.css'   
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. The app will continue without custom styling.")
    
    one,two =st.columns([4,1])
    with one:
      radar_chart = get_radar_chart(input_data)
      st.plotly_chart(radar_chart)
        
    with two:
      add_predictions(input_data)


input_data=add_side_bar()







if __name__== "__main__":
      main()

 