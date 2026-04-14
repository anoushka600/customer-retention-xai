import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import google.generativeai as genai
from utils1 import (
    get_risk, build_input_vector, get_feature_contributions,
    preprocess_uploaded_df, generate_sample_template
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Retention Command Center",
    page_icon="🏢",
    layout="wide",
)

# ── Load artifacts ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = pickle.load(open("model.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    df      = pd.read_csv("data.csv")
    return model, columns, df

model, model_columns, df_full = load_model()

# ── Gemini setup ────────────────────────────────────────────────────────────────
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    gemini = genai.GenerativeModel("gemini-2.5-flash") # Ensure it's 2.5!
else:
    gemini = None

# ── Header ──────────────────────────────────────────────────────────────────────
st.title("🏢 Customer Retention Command Center")
st.caption("End-to-End XAI Dashboard for Business Intelligence · Powered by XGBoost + Gemini AI")
st.divider()

# Tab 2 name updated
tab_single, tab_bulk = st.tabs(["🔍 Single Customer Analysis", "📁 Bulk Upload"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE CUSTOMER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    
    # Moved inputs out of the sidebar and into a clean expander!
    with st.expander("🛠️ Adjust Customer Profile", expanded=True):
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            st.markdown("**📋 Account Info**")
            tenure     = st.slider("Tenure (months)", 1, 72, 12)
            monthly    = st.slider("Monthly Charges ($)", 20, 120, 65)
            total      = st.slider("Total Charges ($)", 20, 8700, min(monthly * tenure, 8700))
            senior     = st.checkbox("Senior Citizen")
            gender     = st.selectbox("Gender", ["Male", "Female"])
            partner    = st.checkbox("Has Partner")
            dependents = st.checkbox("Has Dependents")

        with col_in2:
            st.markdown("**📡 Services**")
            phone_service    = st.selectbox("Phone Service",    ["Yes", "No"])
            multiple_lines   = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
            internet         = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security  = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
            online_backup    = st.selectbox("Online Backup",    ["No", "Yes", "No internet service"])
            device_protect   = st.selectbox("Device Protection",["No", "Yes", "No internet service"])
            tech_support     = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
            streaming_tv     = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col_in3:
            st.markdown("**💳 Billing & Contract**")
            contract  = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.checkbox("Paperless Billing", value=True)
            payment   = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])

    input_vec = build_input_vector(
        senior=int(senior), tenure=tenure, monthly=monthly, total=total,
        gender=gender, partner=int(partner), dependents=int(dependents),
        phone_service=phone_service, multiple_lines=multiple_lines,
        internet=internet, online_security=online_security,
        online_backup=online_backup, device_protect=device_protect,
        tech_support=tech_support, streaming_tv=streaming_tv,
        streaming_movies=streaming_movies, contract=contract,
        paperless=int(paperless), payment=payment
    )
    input_df = pd.DataFrame([input_vec])[model_columns]
    prob      = model.predict_proba(input_df)[0][1]
    risk_label= get_risk(prob)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📊 Churn Risk Assessment")
        c1, c2 = st.columns(2)
        c1.metric("Churn Probability", f"{round(prob * 100, 1)}%")
        c2.metric("Risk Level", risk_label)
        st.progress(float(prob))
        st.divider()

        st.subheader("🔍 Why is this customer at risk?")
        st.caption("Red = pushing toward churn · Green = protecting from churn")

        contributions = get_feature_contributions(model, model_columns, input_df, df_full)

        def prettify(name):
            name = name.replace("_No internet service", " (no internet)")
            name = name.replace("_No phone service", " (no phone)")
            return name.replace("_", " ").title()

        top    = contributions.nlargest(5)
        bottom = contributions.nsmallest(5)
        chart_data = pd.concat([top, bottom]).sort_values()

        chart_df = pd.DataFrame({
            "Feature":      [prettify(i) for i in chart_data.index],
            "Contribution": chart_data.values,
        })
        chart_df["Direction"] = chart_df["Contribution"].apply(
            lambda x: "Increases Risk" if x > 0 else "Decreases Risk"
        )
        fig = px.bar(
            chart_df, x="Contribution", y="Feature",
            color="Direction", orientation="h",
            color_discrete_map={"Increases Risk": "#FF4B4B", "Decreases Risk": "#00C853"},
            height=350,
        )
        fig.update_layout(showlegend=True, margin=dict(l=0, r=0, t=10, b=0),
                          legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns([1, 1], gap="large")    

    with col_right:
        st.subheader("🤖 AI Retention Coach")
        top_risk   = [prettify(f) for f in contributions.nlargest(3).index.tolist()]
        protective = [prettify(f) for f in contributions.nsmallest(2).index.tolist()]

        if not gemini:
            st.warning("⚠️ Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` to enable AI coaching.")
        else:
            coach_prompt = f"""You are a senior customer retention specialist at a telecom company.

Customer Snapshot:
- Tenure: {tenure} months  |  Monthly Bill: ${monthly}  |  Total Spent: ${total}
- Contract: {contract}  |  Internet: {internet}  |  Tech Support: {tech_support}
- Churn Risk: {round(prob * 100, 1)}% → {risk_label}

Top Risk Drivers: {', '.join(top_risk)}
Protective Strengths: {', '.join(protective)}

Write a warm, specific 3-4 sentence retention script for a call center agent.
Include: (1) an exact promotion targeting their biggest pain point,
(2) acknowledgment of their loyalty, (3) a gentle urgency message.
No bullet points. Conversational tone."""

            # Added a unique key here to stop bleeding into Tab 2
            if st.button("Generate Retention Script", use_container_width=True, key="btn_single"):
                with st.spinner("Gemini is crafting your personalized strategy..."):
                    st.session_state["coach_response"] = gemini.generate_content(coach_prompt).text

            if "coach_response" in st.session_state:
                st.info(st.session_state["coach_response"])

        st.divider()
        st.subheader("Dataset Benchmark")
        b1, b2, b3 = st.columns(3)
        avg_churn   = df_full["Churn"].mean()
        avg_monthly = df_full["MonthlyCharges"].mean()
        avg_tenure  = df_full["tenure"].mean()
        b1.metric("Avg Churn Rate",   f"{avg_churn:.1%}",
                  delta=f"{(prob - avg_churn)*100:+.1f}% vs avg", delta_color="inverse")
        b2.metric("Avg Monthly Bill", f"${avg_monthly:.0f}",
                  delta=f"${monthly - avg_monthly:+.0f} vs avg", delta_color="inverse")
        b3.metric("Avg Tenure",       f"{avg_tenure:.0f}mo",
                  delta=f"{tenure - avg_tenure:+.0f}mo vs avg")

    st.divider()
    st.subheader("💬 Chat with Your Data")
    st.caption("Ask Gemini anything about your churn dataset.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant", avatar="🤖"):
            st.write(a)

    # Replaced chat_input with a form so it stays locked inside Tab 1!
    with st.form("chat_form", clear_on_submit=True):
        user_q = st.text_input("Type your question here (e.g., Why are fiber-optic customers churning more?):")
        submitted = st.form_submit_button("Ask AI Analyst")

    if submitted and user_q:
        if not gemini:
            st.warning("Set GEMINI_API_KEY to enable the data analyst.")
        else:
            analyst_prompt = f"""You are an expert telecom data analyst. Answer using ONLY this data.

Dataset: {len(df_full):,} customers | Overall churn rate: {df_full['Churn'].mean():.1%}
Avg Monthly: ${df_full['MonthlyCharges'].mean():.2f} | Avg Tenure: {df_full['tenure'].mean():.1f}mo

Churn by contract — Month-to-month: {df_full[(df_full['Contract_One year']==0)&(df_full['Contract_Two year']==0)]['Churn'].mean():.1%}, One-year: {df_full[df_full['Contract_One year']==1]['Churn'].mean():.1%}, Two-year: {df_full[df_full['Contract_Two year']==1]['Churn'].mean():.1%}
Churn by internet — Fiber optic: {df_full[df_full['InternetService_Fiber optic']==1]['Churn'].mean():.1%}, DSL: {df_full[(df_full['InternetService_Fiber optic']==0)&(df_full['InternetService_No']==0)]['Churn'].mean():.1%}, None: {df_full[df_full['InternetService_No']==1]['Churn'].mean():.1%}
Has tech support churn: {df_full[df_full['TechSupport_Yes']==1]['Churn'].mean():.1%} | No tech support: {df_full[df_full['TechSupport_Yes']==0]['Churn'].mean():.1%}
Senior citizen churn: {df_full[df_full['SeniorCitizen']==1]['Churn'].mean():.1%}

Question: {user_q}
Answer in 3-5 sentences with specific numbers and an actionable business insight."""

            with st.spinner("Analyzing..."):
                resp = gemini.generate_content(analyst_prompt)
                st.session_state.chat_history.append((user_q, resp.text))
                st.rerun() 


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BULK UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

tab_single, tab_bulk = st.tabs(["🔍 Single Customer Analysis", "📁 Bulk Upload"])
with tab_bulk:

    # Template download
    with st.expander("Download Upload Template", expanded=False):
        st.caption("Fill this template with your customers and upload it below.")
        template_df  = generate_sample_template()
        csv_template = template_df.to_csv(index=False).encode()
        st.download_button("⬇Download CSV Template", data=csv_template,
                           file_name="churn_upload_template.csv", mime="text/csv")
        st.dataframe(template_df, use_container_width=True)

    st.divider()
    st.subheader("Upload Your Customer Data")
    uploaded_file = st.file_uploader(
        "Drag and drop your customer CSV or Excel file here",
        type=["csv", "xlsx"],
        help="Use the template above for the correct column format.",
    )

    if uploaded_file is None:
        st.info("Upload a CSV or Excel file to analyze your entire customer base at once.")
        st.stop()

    # Read file
    try:
        raw_df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") \
                 else pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.success(f"Loaded **{len(raw_df):,} customers** from `{uploaded_file.name}`")
    with st.expander("review raw data", expanded=False):
        st.dataframe(raw_df.head(10), use_container_width=True)

    # Preprocess & predict
    with st.spinner("Running XGBoost model on all customers..."):
        try:
            model_ready_df, customer_ids = preprocess_uploaded_df(raw_df, model_columns)
            probs = model.predict_proba(model_ready_df)[:, 1]
        except Exception as e:
            st.error(f"Preprocessing failed — does your file match the template format?\n\n{e}")
            st.stop()

    # Build results table
    results = raw_df.copy()
    if customer_ids is not None:
        results.insert(0, "CustomerID", customer_ids.values)
    results.insert(1, "Churn Probability", (probs * 100).round(1))
    results.insert(2, "Risk Level",        [get_risk(p) for p in probs])
    results = results.sort_values("Churn Probability", ascending=False).reset_index(drop=True)

    high_risk   = (probs >= 0.7).sum()
    medium_risk = ((probs >= 0.3) & (probs < 0.7)).sum()
    low_risk    = (probs < 0.3).sum()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Executive Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(probs):,}")
    k2.metric("🔴 High Risk",    f"{high_risk:,}",
              delta=f"{high_risk/len(probs):.1%} of base", delta_color="inverse")
    k3.metric("🟡 Medium Risk",  f"{medium_risk:,}")
    k4.metric("🟢 Low Risk",     f"{low_risk:,}")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.divider()
    chart1, chart2, chart3 = st.columns(3)

    with chart1:
        st.markdown("**Risk Distribution**")
        fig_donut = go.Figure(go.Pie(
            labels=["High Risk 🔴", "Medium Risk 🟡", "Low Risk 🟢"],
            values=[high_risk, medium_risk, low_risk],
            hole=0.55,
            marker_colors=["#FF4B4B", "#FFC107", "#00C853"],
        ))
        fig_donut.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=260, showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart2:
        st.markdown("**Churn Probability Distribution**")
        fig_hist = px.histogram(x=probs * 100, nbins=20,
                                labels={"x": "Churn Probability (%)"},
                                color_discrete_sequence=["#636EFA"])
        fig_hist.add_vline(x=70, line_dash="dash", line_color="#FF4B4B",
                           annotation_text="High Risk Threshold")
        fig_hist.update_layout(margin=dict(t=10,b=10,l=0,r=0),
                               showlegend=False, height=260, yaxis_title="# Customers")
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart3:
        st.markdown("**Avg Risk by Contract Type**")
        if "Contract" in results.columns:
            seg = results.groupby("Contract")["Churn Probability"].mean().reset_index()
            seg.columns = ["Contract", "Avg Churn %"]
            fig_bar = px.bar(seg, x="Contract", y="Avg Churn %",
                             color="Avg Churn %",
                             color_continuous_scale=["#00C853", "#FFC107", "#FF4B4B"])
            fig_bar.update_layout(margin=dict(t=10,b=10,l=0,r=0),
                                  showlegend=False, height=260, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.caption("Add a `Contract` column to see this breakdown.")

    # ── High-risk table + download ────────────────────────────────────────────
    st.divider()
    st.subheader("🚨 High-Risk Customer List")
    st.caption("Top 50 customers most likely to leave — call these first.")

    priority_cols = ["Churn Probability", "Risk Level"]
    for col in ["CustomerID", "tenure", "MonthlyCharges", "Contract",
                "InternetService", "TechSupport", "PaymentMethod"]:
        if col in results.columns:
            priority_cols.append(col)

    high_risk_df = results[results["Risk Level"] == "High Risk 🔴"][priority_cols].head(50)

    st.dataframe(
        high_risk_df.style.background_gradient(
            subset=["Churn Probability"], cmap="RdYlGn_r"
        ),
        use_container_width=True,
        height=380,
    )

    st.download_button(
        "⬇️ Download Full Scored Dataset",
        data=results.to_csv(index=False).encode(),
        file_name="customers_scored.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Gemini per-customer strategy ──────────────────────────────────────────
    st.divider()
    st.subheader("🤖 AI Retention Strategy — Pick a Customer")

    if not gemini:
        st.warning("⚠️ Set `GEMINI_API_KEY` in `.streamlit/secrets.toml` to unlock AI strategies.")
    else:
        hr = results[results["Risk Level"] == "High Risk 🔴"].head(50)
        if hr.empty:
            st.success("🎉 No high-risk customers found in this dataset!")
        else:
            options = [
                f"{'Customer ' + str(row.get('CustomerID', idx))}  —  {row['Churn Probability']}% churn risk"
                for idx, row in hr.iterrows()
            ]
            selected_label = st.selectbox("Choose a customer:", options)
            selected_idx   = options.index(selected_label)
            selected_row   = hr.iloc[selected_idx]

            row_summary = "\n".join([
                f"- {col}: {selected_row[col]}"
                for col in priority_cols
                if col in selected_row and col not in ["Risk Level"]
            ])

            bulk_prompt = f"""You are a senior customer retention specialist at a telecom company.

High-Risk Customer Profile:
{row_summary}

Write a warm, specific 3-4 sentence call center script for retaining this exact customer.
Include: (1) a personalized opening referencing their specific situation,
(2) an exact targeted promotion addressing their likely pain point,
(3) a soft urgency close. No bullet points. Sound human."""

            # Added a unique key here to prevent UI overlapping
            if st.button("Generate Strategy for this Customer", use_container_width=True, key="btn_bulk"):
                with st.spinner("Generating personalized retention strategy..."):
                    st.session_state["bulk_strategy"] = gemini.generate_content(bulk_prompt).text

            if "bulk_strategy" in st.session_state:
                st.success(st.session_state["bulk_strategy"])