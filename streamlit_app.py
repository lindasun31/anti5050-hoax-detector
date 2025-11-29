import os
import streamlit as st
from tavily import TavilyClient
import pandas as pd
import plotly.express as px
import RAG_utility
from dotenv import load_dotenv

# Mengatur konfigurasi halaman
st.set_page_config(page_title="Anti5050 Hoax Detector", page_icon=None, layout="wide", initial_sidebar_state=None, menu_items=None)

# Load environment variables
load_dotenv(dotenv_path="secret/.env")
# Fetch the API key
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

if not OPENROUTER_API_KEY or not TAVILY_API_KEY:
    print("Kunci API untuk OpenRouter atau Tavily tidak ditemukan. Harap konfigurasikan di Streamlit secrets.")
    exit()

    
#create tavily client (instance)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
col1, col9, col10 = st.columns([2, 10, 2])
with col1:
        # Top bar with icons and text
    col1, col2, col3= st.columns([0.5, 0.55, 0.5])  # adjust widths
    with st.container():
        with st.container():
            with col1:
                
                st.image("images/logo1.png", width=50)  # first icon
                st.write("Universitas Multimedia Nusantara")
            with col2:
                st.image("images/logo2.png", width=70)  # second icon
                st.write("SYY Hoax Analyzer")
with col9:
    st.markdown("<h1 style='text-align:center'>Selamat Datang di Hoax Detection dengan RAG + LLM</h3>", unsafe_allow_html=True)
    placeholder_text = "Silahkan masukkan klaim"
    with st.form('claim_input_form'):
        col4, col5 = st.columns([7, 1])
        with col4:
            user_claim = st.text_input(
                placeholder_text,        # label argument
                value="",
                placeholder=placeholder_text,
                label_visibility='collapsed'
            )
        with col5:
            submitted = st.form_submit_button("Submit")
        #st.form needs test_input and submit_button. submit_button must be used inside st.form

    claim_length_treshold = 20

    with st.container():
        #i want these belows goes to a new container and being outside the
        if submitted:
            if len(user_claim.strip()) >= claim_length_treshold:
                #.strip() removes accidental spaces at start/end.
                # Menjalankan pipeline dengan spinner
                with st.spinner("Memproses klaim Anda..."):
                    # Memanggil fungsi pipeline paralel
                    pipeline_results, error = RAG_utility.process_rag_pipeline_parallel(
                        user_claim,
                        openrouter_api_key=OPENROUTER_API_KEY,
                        tavily_api_key=TAVILY_API_KEY
                    )
                with st.container():
                    tab1, tab2, tab3 = st.tabs(["Hasil", "Sumber", "Grafik"])
                    with tab1:
                        st.markdown("<h3 style='text-align:center'>Klaim:</h3>", unsafe_allow_html=True)
                        st.write(f"<p style='text-align:center'>{user_claim}</p>",  unsafe_allow_html=True)

                        processed_results = pipeline_results.get("processed_results", {})

                        relevance_scores = []
                        for res in processed_results.get("results", []):
                            relevance_scores.append(res.get("score"))
                            #this is using for append list because processed_results.get("results") is a list, not a dict, so we have to iterate through all rows
                        fact_check_result = pipeline_results.get("fact_check_analysis", {})

                        if fact_check_result.get("status") == "success":
                            #separating datas from analysis 
                            separated_analysis = RAG_utility.analysis_separator(fact_check_result.get("analysis", []), relevance_scores)
                            val_mendukung, val_tidak_mendukung, confidence_level = RAG_utility.confidence_calculator(separated_analysis)
                            #menampilkan
                            st.markdown(f"<h1 style='text-align:center'>{confidence_level}% {separated_analysis.get("kesimpulan", [])}</h3>", unsafe_allow_html=True)
                        else:
                            st.error(f"Gagal menghasilkan analisis: {fact_check_result.get('analysis', 'Error tidak diketahui.')}")

                        #st.markdown(f"<h1 style='text-align:center'>{pipeline_results}></h3>", unsafe_allow_html=True)
                    with tab2:
                        #for table:
                        data_table = []
                        for item in separated_analysis.get("bukti", []):  # default [] to avoid NoneType
                            data_table.append({
                                "Domain": item.get("domain"),
                                "Konklusi": item.get("label"),
                                "Judul": item.get("title"),
                                "Relevansi": item.get("relevance"),
                                "Link": item.get("url")
                                
                            })

                        df_table = pd.DataFrame(data_table)

                        st.markdown("<h3 style='text-align:center'>Sumber</h3>", unsafe_allow_html=True)
                        st.dataframe(df_table, width="content", height="auto")
                    with tab3:
                        #for pie chart:
                        data = {'Category': ['Mendukung', 'Tidak Mendukung'],
                                'Value': [val_mendukung, val_tidak_mendukung]}
                        df_pie = pd.DataFrame(data)
                        # Create pie chart with fixed colors
                        fig = px.pie(
                            df_pie,
                            values='Value',
                            names='Category',
                            title='Persentase Artikel Pendukung',
                            color='Category',
                            color_discrete_map={
                                'Mendukung': 'seagreen',
                                'Tidak Mendukung': 'red'
                            }
                        )
                        st.markdown("<h3 style='text-align:center'>Grafik Hoax vs non Hoax</h3>", unsafe_allow_html=True)
                        st.plotly_chart(fig, theme=None, width="stretch")
            else :
                st.markdown(f"<p style=color:red;> tolong masukkan lebih dari {claim_length_treshold} karakter</p>", unsafe_allow_html=True)
                            
    st.markdown("---")  # horizontal line to separate top bar
with col10:
    st.write("")


