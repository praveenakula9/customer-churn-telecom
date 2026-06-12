import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { color: #94a3b8 !important; font-size: 0.78rem !important; }

/* Metric cards */
.kpi-card {
    background: var(--secondary-background-color);
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid rgba(148,163,184,0.15);
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent, #6366f1);
    border-radius: 4px 0 0 4px;
}
.kpi-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.55; margin-bottom: 6px; }
.kpi-value { font-size: 2rem; font-weight: 700; line-height: 1; }
.kpi-sub   { font-size: 0.75rem; opacity: 0.5; margin-top: 4px; }

/* Risk badge */
.risk-HIGH   { background:#ef444422; color:#ef4444; border:1px solid #ef444455; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.82rem; display:inline-block; }
.risk-MEDIUM { background:#f9731622; color:#f97316; border:1px solid #f9731655; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.82rem; display:inline-block; }
.risk-LOW    { background:#22c55e22; color:#22c55e; border:1px solid #22c55e55; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.82rem; display:inline-block; }

/* Score card */
.score-card {
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    border: 2px solid;
    margin-bottom: 16px;
}
.score-card.churn  { background: linear-gradient(135deg,#ef444415,#ef444408); border-color:#ef4444; }
.score-card.safe   { background: linear-gradient(135deg,#22c55e15,#22c55e08); border-color:#22c55e; }
.score-card.medium { background: linear-gradient(135deg,#f9731615,#f9731608); border-color:#f97316; }
.score-card h1 { font-size: 2.2rem; margin: 0 0 6px; }
.score-card p  { margin: 0; opacity: 0.75; font-size: 0.95rem; }

/* Factor row */
.factor-row {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 10px 0; border-bottom: 1px solid rgba(148,163,184,0.12);
    gap: 12px;
}
.factor-row:last-child { border-bottom: none; }
.factor-label { font-size: 0.88rem; font-weight: 500; }
.factor-detail { font-size: 0.75rem; opacity: 0.55; margin-top: 2px; }

/* Section header */
.section-header {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.45; margin: 18px 0 10px;
}

/* Batch table */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Predict button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }

.tab-content { padding-top: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("best_churn_model.pkl")

model = load_model()

FEATURE_COLS = [
    'gender','SeniorCitizen','Partner','Dependents','tenure',
    'PhoneService','PaperlessBilling','MonthlyCharges','TotalCharges',
    'MultipleLines_No phone service','MultipleLines_Yes',
    'InternetService_Fiber optic','InternetService_No',
    'OnlineSecurity_No internet service','OnlineSecurity_Yes',
    'OnlineBackup_No internet service','OnlineBackup_Yes',
    'DeviceProtection_No internet service','DeviceProtection_Yes',
    'TechSupport_No internet service','TechSupport_Yes',
    'StreamingTV_No internet service','StreamingTV_Yes',
    'StreamingMovies_No internet service','StreamingMovies_Yes',
    'Contract_One year','Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'tenure_group_1-2yr','tenure_group_2-4yr',
    'tenure_group_4-5yr','tenure_group_5-6yr'
]

def get_tenure_group(t):
    if t <= 12:  return '0-1yr'
    elif t <= 24: return '1-2yr'
    elif t <= 48: return '2-4yr'
    elif t <= 60: return '4-5yr'
    else:         return '5-6yr'

def build_row(tenure, monthly_charges, contract, internet, paperless,
              phone_service, multiple_lines, online_sec, online_bk,
              device_prot, tech_sup, streaming_tv, streaming_mv,
              payment, senior, partner, dependents, gender):
    row = {c: 0 for c in FEATURE_COLS}
    row['tenure']          = tenure
    row['MonthlyCharges']  = monthly_charges
    row['TotalCharges']    = tenure * monthly_charges
    row['SeniorCitizen']   = 1 if senior   == 'Yes' else 0
    row['gender']          = 1 if gender   == 'Male' else 0
    row['Partner']         = 1 if partner  == 'Yes' else 0
    row['Dependents']      = 1 if dependents == 'Yes' else 0
    row['PhoneService']    = 1 if phone_service == 'Yes' else 0
    row['PaperlessBilling']= 1 if paperless == 'Yes' else 0
    if multiple_lines == 'No phone service': row['MultipleLines_No phone service'] = 1
    elif multiple_lines == 'Yes':            row['MultipleLines_Yes'] = 1
    if internet == 'Fiber optic': row['InternetService_Fiber optic'] = 1
    elif internet == 'No':        row['InternetService_No'] = 1
    for feat, val in [('OnlineSecurity', online_sec), ('OnlineBackup', online_bk),
                      ('DeviceProtection', device_prot), ('TechSupport', tech_sup),
                      ('StreamingTV', streaming_tv), ('StreamingMovies', streaming_mv)]:
        if val == 'No internet service': row[f'{feat}_No internet service'] = 1
        elif val == 'Yes':               row[f'{feat}_Yes'] = 1
    if contract == 'One year':   row['Contract_One year']  = 1
    elif contract == 'Two year': row['Contract_Two year']  = 1
    pm_map = {'Credit card (automatic)':'PaymentMethod_Credit card (automatic)',
              'Electronic check':'PaymentMethod_Electronic check',
              'Mailed check':'PaymentMethod_Mailed check'}
    if payment in pm_map: row[pm_map[payment]] = 1
    tg = get_tenure_group(tenure)
    tg_map = {'1-2yr':'tenure_group_1-2yr','2-4yr':'tenure_group_2-4yr',
              '4-5yr':'tenure_group_4-5yr','5-6yr':'tenure_group_5-6yr'}
    if tg in tg_map: row[tg_map[tg]] = 1
    return pd.DataFrame([row])[FEATURE_COLS]

def gauge_chart(prob):
    color = '#ef4444' if prob > 0.6 else ('#f97316' if prob > 0.35 else '#22c55e')
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={'suffix': '%', 'font': {'size': 42, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1,
                     'tickcolor': 'rgba(148,163,184,0.4)',
                     'tickfont': {'size': 11}},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  35], 'color': 'rgba(34,197,94,0.12)'},
                {'range': [35, 60], 'color': 'rgba(249,115,22,0.12)'},
                {'range': [60,100], 'color': 'rgba(239,68,68,0.12)'},
            ],
            'threshold': {'line': {'color': color, 'width': 4},
                          'thickness': 0.8, 'value': prob * 100}
        }
    ))
    fig.update_layout(
        height=240, margin=dict(t=20, b=10, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    return fig

def donut_chart(prob):
    color = '#ef4444' if prob > 0.6 else ('#f97316' if prob > 0.35 else '#22c55e')
    fig = go.Figure(go.Pie(
        values=[prob * 100, (1 - prob) * 100],
        labels=['Churn Risk', 'Retention'],
        hole=0.72,
        marker_colors=[color, 'rgba(148,163,184,0.15)'],
        textinfo='none',
        hoverinfo='label+percent'
    ))
    fig.add_annotation(
        text=f"<b>{prob*100:.0f}%</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=30, color=color, family='Inter')
    )
    fig.update_layout(
        height=220, showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def risk_factors(tenure, monthly_charges, contract, internet,
                 online_sec, tech_sup, paperless, payment, streaming_tv,
                 streaming_mv, online_bk, device_prot):
    factors = []
    if contract == 'Month-to-month':
        factors.append(("📋 Month-to-month contract", "HIGH", "43% churn rate vs 3% for 2-year"))
    if tenure <= 12:
        factors.append(("🆕 New customer (≤12 months)", "HIGH", "47.7% of first-year customers churn"))
    if tenure <= 24 and contract == 'Month-to-month':
        factors.append(("⚡ Early + no commitment", "HIGH", "Highest risk combination"))
    if internet == 'Fiber optic':
        factors.append(("🌐 Fiber optic service", "MEDIUM", "Higher churn than DSL customers"))
    if monthly_charges > 70:
        factors.append(("💰 High monthly charges", "MEDIUM", f"${monthly_charges:.0f}/mo — above avg ($64.76)"))
    if online_sec == 'No':
        factors.append(("🔒 No Online Security", "MEDIUM", "Add-on services increase stickiness"))
    if tech_sup == 'No':
        factors.append(("🛠 No Tech Support", "MEDIUM", "Add-on services increase stickiness"))
    if paperless == 'Yes':
        factors.append(("📄 Paperless billing", "LOW", "Slight positive correlation with churn"))
    if payment == 'Electronic check':
        factors.append(("💳 Electronic check", "LOW", "Highest churn rate among payment methods"))
    if contract in ['One year', 'Two year']:
        factors.append(("✅ Long-term contract", "LOW", "Strongly protective against churn"))
    if tenure > 36:
        factors.append(("🏅 Loyal customer (>36 months)", "LOW", "Long tenure = low churn risk"))
    return factors

def retention_plan(prob, contract, tenure, monthly_charges,
                   online_sec, tech_sup, internet):
    recs = []
    if prob > 0.6:
        recs.append(("🚨 Priority Action", "Assign a dedicated retention agent for immediate outreach within 24 hours."))
    if contract == 'Month-to-month':
        discount = 15 if prob > 0.6 else 10
        recs.append(("📋 Contract Upgrade", f"Offer a {discount}% discount to upgrade to a 1-year or 2-year contract. This is the single highest-impact retention lever."))
    if tenure <= 12:
        recs.append(("🎁 New Customer Loyalty Pack", "Enroll in a 3-month loyalty reward program. Early engagement dramatically reduces churn."))
    if online_sec == 'No' and internet != 'No':
        recs.append(("🔒 Free Security Trial", "Offer 2 months of Online Security free. Customers with add-ons churn 30% less."))
    if tech_sup == 'No' and internet != 'No':
        recs.append(("🛠 Tech Support Bundle", "Add Tech Support to the plan for 3 months free. Reduces friction and increases perceived value."))
    if monthly_charges > 80:
        recs.append(("💰 Rate Review", "Offer a personalized rate review — a 10-15% reduction may be more profitable than losing the customer entirely."))
    if prob <= 0.35:
        recs.append(("👍 No Immediate Action", "Customer shows low churn risk. No intervention needed at this time — monitor quarterly."))
    return recs

def batch_predict(df_raw):
    results = []
    for _, row in df_raw.iterrows():
        try:
            inp = build_row(
                tenure=int(row.get('tenure', 12)),
                monthly_charges=float(row.get('MonthlyCharges', 65)),
                contract=str(row.get('Contract', 'Month-to-month')),
                internet=str(row.get('InternetService', 'DSL')),
                paperless=str(row.get('PaperlessBilling', 'No')),
                phone_service=str(row.get('PhoneService', 'Yes')),
                multiple_lines=str(row.get('MultipleLines', 'No')),
                online_sec=str(row.get('OnlineSecurity', 'No')),
                online_bk=str(row.get('OnlineBackup', 'No')),
                device_prot=str(row.get('DeviceProtection', 'No')),
                tech_sup=str(row.get('TechSupport', 'No')),
                streaming_tv=str(row.get('StreamingTV', 'No')),
                streaming_mv=str(row.get('StreamingMovies', 'No')),
                payment=str(row.get('PaymentMethod', 'Mailed check')),
                senior=str(row.get('SeniorCitizen', 'No')),
                partner=str(row.get('Partner', 'No')),
                dependents=str(row.get('Dependents', 'No')),
                gender=str(row.get('gender', 'Male'))
            )
            prob = model.predict_proba(inp)[0][1]
            pred = model.predict(inp)[0]
            risk = 'HIGH' if prob > 0.6 else ('MEDIUM' if prob > 0.35 else 'LOW')
            results.append({
                'Customer':       row.get('customerID', f"C{_+1}"),
                'Churn Prob (%)': round(prob * 100, 1),
                'Prediction':     '⚠️ Churn' if pred == 1 else '✅ Stay',
                'Risk Level':     risk,
                'Tenure':         int(row.get('tenure', 0)),
                'Monthly ($)':    float(row.get('MonthlyCharges', 0)),
                'Contract':       row.get('Contract', '-'),
            })
        except Exception as e:
            results.append({'Customer': row.get('customerID', f"C{_+1}"),
                            'Error': str(e)})
    return pd.DataFrame(results)


st.markdown("<h2 style='text-align: center; font-weight: 700; margin-bottom: 25px;'>📉 Customer Churn Intelligence Dashboard</h2>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯 Single Prediction", "📂 Batch Upload"])

with tab1:
    left_col, right_col = st.columns([1.2, 2.0], gap="large")

    with left_col:
        st.markdown("### Customer Profile")
        ac1, ac2 = st.columns(2)
        with ac1:
            gender      = st.selectbox("Gender",          ["Male", "Female"])
            senior      = st.selectbox("Senior Citizen",  ["No", "Yes"])
        with ac2:
            partner     = st.selectbox("Partner",         ["No", "Yes"])
            dependents  = st.selectbox("Dependents",      ["No", "Yes"])

        st.markdown("<div class='section-header'>Contract & Billing</div>", unsafe_allow_html=True)
        tenure           = st.slider("Tenure (months)", 0, 72, 12)
        cb1, cb2 = st.columns(2)
        with cb1:
            monthly_charges  = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            paperless        = st.selectbox("Paperless Billing", ["Yes", "No"])
        with cb2:
            contract         = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment          = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        st.markdown("<div class='section-header'>Services</div>", unsafe_allow_html=True)
        srv1, srv2 = st.columns(2)
        with srv1:
            phone_service = st.selectbox("Phone Service",    ["Yes", "No"])
            internet      = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_bk     = st.selectbox("Online Backup",    ["No", "Yes", "No internet service"])
            tech_sup      = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
            streaming_mv  = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        with srv2:
            multiple_lines= st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
            online_sec    = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
            device_prot   = st.selectbox("Device Protection",["No", "Yes", "No internet service"])
            streaming_tv  = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])

        st.markdown("")
        predict_btn = st.button("🔍 Analyze Customer", use_container_width=True)

    with right_col:
        if predict_btn:
            input_df = build_row(tenure, monthly_charges, contract, internet,
                                 paperless, phone_service, multiple_lines,
                                 online_sec, online_bk, device_prot, tech_sup,
                                 streaming_tv, streaming_mv, payment,
                                 senior, partner, dependents, gender)

            prob      = model.predict_proba(input_df)[0][1]
            predicted = model.predict(input_df)[0]

            risk_label = "HIGH" if prob > 0.6 else ("MEDIUM" if prob > 0.35 else "LOW")
            score      = max(0, min(100, int((1 - prob) * 100)))

            col_card, col_gauge, col_donut = st.columns([1.1, 1.2, 0.9])

            with col_card:
                card_cls = 'churn' if predicted == 1 else ('medium' if prob > 0.35 else 'safe')
                icon     = '⚠️' if predicted == 1 else ('🟡' if prob > 0.35 else '✅')
                verdict  = 'Likely to Churn' if predicted == 1 else ('Monitor Closely' if prob > 0.35 else 'Likely to Stay')
                st.markdown(f"""
                <div class="score-card {card_cls}">
                    <h1>{icon}</h1>
                    <h2 style="margin:8px 0 4px;font-size:1.4rem">{verdict}</h2>
                    <p>Retention Score: <strong>{score}/100</strong></p>
                    <br>
                    <span class="risk-{risk_label}">{risk_label} RISK</span>
                </div>""", unsafe_allow_html=True)

                tg = get_tenure_group(tenure)
                total_charges = tenure * monthly_charges
                st.markdown(f"""
                <div class="kpi-card" style="--accent:#6366f1; margin-top:10px">
                    <div class="kpi-label">Customer Summary</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px">
                        <div><div style="font-size:0.72rem;opacity:0.5">Tenure Group</div><div style="font-weight:700">{tg}</div></div>
                        <div><div style="font-size:0.72rem;opacity:0.5">Total Charges</div><div style="font-weight:700">${total_charges:,.0f}</div></div>
                        <div><div style="font-size:0.72rem;opacity:0.5">Contract</div><div style="font-weight:700">{contract}</div></div>
                        <div><div style="font-size:0.72rem;opacity:0.5">Internet</div><div style="font-weight:700">{internet}</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            with col_gauge:
                st.markdown("**Churn Probability Gauge**")
                st.plotly_chart(gauge_chart(prob), use_container_width=True, config={'displayModeBar': False})

            with col_donut:
                st.markdown("**Risk Breakdown**")
                st.plotly_chart(donut_chart(prob), use_container_width=True, config={'displayModeBar': False})
                st.markdown(f"""
                <div style="text-align:center;margin-top:-10px">
                    <span style="font-size:0.82rem;opacity:0.55">Model confidence: XGBoost (AUC 0.843)</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            col_rf, col_rp = st.columns(2)

            with col_rf:
                st.markdown("#### 🔎 Risk Factor Analysis")
                factors = risk_factors(tenure, monthly_charges, contract, internet,
                                       online_sec, tech_sup, paperless, payment,
                                       streaming_tv, streaming_mv, online_bk, device_prot)
                if factors:
                    for label, level, detail in factors:
                        badge = f'<span class="risk-{level}">{level}</span>'
                        st.markdown(f"""
                        <div class="factor-row">
                            <div>
                                <div class="factor-label">{label}</div>
                                <div class="factor-detail">{detail}</div>
                            </div>
                            {badge}
                        </div>""", unsafe_allow_html=True)

                if factors:
                    st.markdown("")
                    fdata  = pd.DataFrame(factors, columns=['Factor','Level','Detail'])
                    lvl_map = {'HIGH':3,'MEDIUM':2,'LOW':1}
                    fdata['Score'] = fdata['Level'].map(lvl_map)
                    fdata = fdata.sort_values('Score', ascending=True)
                    color_map = {'HIGH':'#ef4444','MEDIUM':'#f97316','LOW':'#22c55e'}
                    fig_bar = px.bar(
                        fdata, x='Score', y='Factor',
                        orientation='h',
                        color='Level',
                        color_discrete_map=color_map,
                        title='Risk Factor Severity',
                        labels={'Score':'Severity','Factor':''}
                    )
                    fig_bar.update_layout(
                        height=250, showlegend=False,
                        margin=dict(t=36,b=10,l=0,r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(tickfont=dict(size=11)),
                        title_font=dict(size=13)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True,
                                    config={'displayModeBar': False})

            with col_rp:
                st.markdown("#### 💡 Retention Recommendations")
                recs = retention_plan(prob, contract, tenure, monthly_charges,
                                      online_sec, tech_sup, internet)
                for title, detail in recs:
                    with st.container():
                        st.markdown(f"""
                        <div style="background:var(--secondary-background-color);
                                    border-radius:12px;padding:14px 16px;
                                    margin-bottom:10px;
                                    border-left:3px solid #6366f1">
                            <div style="font-weight:600;font-size:0.9rem;margin-bottom:4px">{title}</div>
                            <div style="font-size:0.82rem;opacity:0.7">{detail}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("**Intervention Urgency**")
                urgency = min(100, int(prob * 130))
                urgency_color = '#ef4444' if urgency > 70 else ('#f97316' if urgency > 40 else '#22c55e')
                st.markdown(f"""
                <div style="background:rgba(148,163,184,0.1);border-radius:20px;
                            height:12px;overflow:hidden;margin-top:4px">
                    <div style="width:{urgency}%;height:100%;
                                background:{urgency_color};
                                border-radius:20px;
                                transition:width 0.5s"></div>
                </div>
                <div style="display:flex;justify-content:space-between;
                            font-size:0.72rem;opacity:0.5;margin-top:4px">
                    <span>Low</span><span>Medium</span><span>High</span>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("")
            st.info("👈 **Fill in the customer profile** on the left and click **Analyze Customer** to see the prediction results.")


with tab2:
    st.markdown("### 📂 Batch Churn Prediction")
    st.markdown("Upload a CSV file with customer data to predict churn for multiple customers at once.")

    template_cols = ['customerID','tenure','MonthlyCharges','Contract',
                     'InternetService','PaperlessBilling','PhoneService',
                     'MultipleLines','OnlineSecurity','OnlineBackup',
                     'DeviceProtection','TechSupport','StreamingTV',
                     'StreamingMovies','PaymentMethod','SeniorCitizen',
                     'Partner','Dependents','gender']
    template_data = [
        ['C001',12,75.5,'Month-to-month','Fiber optic','Yes','Yes','Yes','No','No','No','No','Yes','No','Electronic check','No','No','No','Male'],
        ['C002',48,55.0,'Two year','DSL','No','Yes','No','Yes','Yes','Yes','Yes','No','No','Bank transfer (automatic)','No','Yes','Yes','Female'],
        ['C003',6, 90.0,'Month-to-month','Fiber optic','Yes','Yes','Yes','No','No','No','No','Yes','Yes','Electronic check','Yes','No','No','Male'],
    ]
    template_df = pd.DataFrame(template_data, columns=template_cols)
    csv_template = template_df.to_csv(index=False)

    col_dl, col_info = st.columns([1, 3])
    with col_dl:
        st.download_button(
            "⬇️ Download CSV Template",
            data=csv_template,
            file_name="churn_batch_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_info:
        st.markdown("""
        <div style="background:var(--secondary-background-color);border-radius:10px;
                    padding:12px 16px;font-size:0.83rem;opacity:0.8">
        ℹ️ Download the template, fill in your customer data, and upload it below.
        Required columns: <code>tenure</code>, <code>MonthlyCharges</code>, <code>Contract</code>, <code>InternetService</code>.
        All other columns are optional (defaults will be used if missing).
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    uploaded = st.file_uploader("Upload your CSV", type=['csv'],
                                 label_visibility="collapsed")

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.markdown(f"**{len(raw_df)} customers loaded.** Preview:")
        st.dataframe(raw_df.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=False):
            with st.spinner("Predicting..."):
                result_df = batch_predict(raw_df)

            st.markdown("### Results")

            total      = len(result_df)
            n_churn    = (result_df['Prediction'] == '⚠️ Churn').sum()  if 'Prediction' in result_df.columns else 0
            n_high     = (result_df['Risk Level']  == 'HIGH').sum()     if 'Risk Level'  in result_df.columns else 0
            churn_rate = n_churn / total * 100 if total > 0 else 0

            bk1, bk2, bk3, bk4 = st.columns(4)
            batch_kpis = [
                ("#6366f1","Total Customers", str(total),    ""),
                ("#ef4444","Predicted Churn",  str(n_churn), f"{churn_rate:.1f}% churn rate"),
                ("#f97316","High Risk",         str(n_high),  "Needs immediate action"),
                ("#22c55e","Likely to Stay",   str(total-n_churn),""),
            ]
            for col,(accent,label,val,sub) in zip([bk1,bk2,bk3,bk4],batch_kpis):
                with col:
                    st.markdown(f"""
                    <div class="kpi-card" style="--accent:{accent}">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value">{val}</div>
                        <div class="kpi-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("")

            if 'Prediction' in result_df.columns and 'Churn Prob (%)' in result_df.columns:
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    pred_counts = result_df['Prediction'].value_counts().reset_index()
                    pred_counts.columns = ['Prediction','Count']
                    fig_pie = px.pie(pred_counts, values='Count', names='Prediction',
                                     color='Prediction',
                                     color_discrete_map={'⚠️ Churn':'#ef4444','✅ Stay':'#22c55e'},
                                     title='Churn Distribution',hole=0.5)
                    fig_pie.update_layout(height=280,paper_bgcolor='rgba(0,0,0,0)',
                                          margin=dict(t=40,b=10,l=10,r=10))
                    st.plotly_chart(fig_pie,use_container_width=True,
                                    config={'displayModeBar':False})

                with col_chart2:
                    fig_hist = px.histogram(result_df, x='Churn Prob (%)',
                                             color='Risk Level',
                                             color_discrete_map={'HIGH':'#ef4444',
                                                                  'MEDIUM':'#f97316',
                                                                  'LOW':'#22c55e'},
                                             nbins=20, title='Probability Distribution',
                                             labels={'Churn Prob (%)':'Churn Probability (%)'})
                    fig_hist.update_layout(height=280,paper_bgcolor='rgba(0,0,0,0)',
                                           plot_bgcolor='rgba(0,0,0,0)',
                                           margin=dict(t=40,b=10,l=10,r=10),
                                           legend_title='Risk')
                    st.plotly_chart(fig_hist,use_container_width=True,
                                    config={'displayModeBar':False})

            st.markdown("**Full Results Table**")
            st.dataframe(
                result_df.sort_values('Churn Prob (%)', ascending=False)
                         .reset_index(drop=True),
                use_container_width=True, height=320
            )

            csv_out = result_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )



st.markdown("")
st.markdown("""
<div style='text-align:center;opacity:0.35;font-size:0.78rem;padding:16px 0'>
    Customer Churn Intelligence Dashboard · XGBoost · IBM Telco Dataset · Built with Streamlit
</div>""", unsafe_allow_html=True)
