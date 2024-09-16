import streamlit as st
import pickle 
import pandas as pd
     

def main():
    st.set_page_config(
        page_title="BREAST PREDICTION MODEL",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    with st.container():
        st.title("BREAST CANCER PREDICTION")
    

if __name__=="__main__":
      main()

 