import streamlit as st
import datetime
from pandas.api.types import CategoricalDtype
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import lightgbm as lgb

warnings.filterwarnings("ignore")

cat_type = CategoricalDtype(categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                            ordered=True)


bst = lgb.Booster(model_file='mode.pkl')

# @app.route('/')


def welcome():
    return "Welcome All"


# @app.route('/predict',methods=["Get"])


def predict_load(DateTime):
    df = pd.DataFrame(columns=['DateTime', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear',
                               'dayofmonth', 'weekofyear', 'weekday', 'season'])
    """
    Creates time series features from datetime index.
    """

    if isinstance(DateTime, str):
        df.loc[0] = 0
    else:
        b = len(DateTime)
        df.iloc[:b] = 0
    df['DateTime'] = pd.to_datetime(DateTime)
    df['hour'] = df['DateTime'].dt.hour
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['weekday'] = df['DateTime'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['DateTime'].dt.quarter
    df['month'] = df['DateTime'].dt.month
    df['year'] = df['DateTime'].dt.year
    df['dayofyear'] = df['DateTime'].dt.dayofyear
    df['dayofmonth'] = df['DateTime'].dt.day
    df['weekofyear'] = df['DateTime'].dt.isocalendar().week  # Use isocalendar().week
    df['date_offset'] = (df.DateTime.dt.month * 100 + df.DateTime.dt.day - 320) % 1300

    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300],
                          labels=['Cold', 'Dry Season', 'Raining Season', 'Harmattan'])
    # X = df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'weekday', 'season']]

    df.set_index('DateTime', inplace=True)
    df.drop("date_offset", axis=1, inplace=True)
    df = pd.get_dummies(df, columns=['weekday', 'season'])
    pkl_file = open('scaler.pkl', 'rb')
    scaler = pickle.load(pkl_file)
    pkl_file.close()
    df = scaler.transform(df)

    prediction = bst.predict(df)
    if isinstance(DateTime, str):
        predictions = prediction.item()

    else:
        predictions = pd.DataFrame(columns=['Load Predictions'])
        predictions['Load Predictions'] = prediction

    return predictions


def main():
    # st.title("Junction Traffic Predictor by Team Scipy")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;font-family:'Caveat',cursive;font-weight: 400;max-width: 800px; width: 85%; margin: 0 auto;">Electric Load Predictor</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    date = st.sidebar.date_input(
        'Date', datetime.datetime.today())  # (2011, 1, 28))
    time = st.sidebar.time_input(
        'Time', datetime.datetime.now())  # (hour=18, minute=54, second=30))
    datestr = date.strftime("%Y-%m-%d")
    timestr = time.strftime("%H:%M:%S")
    DateTime = datestr + ' ' + timestr
    DateTime1 = pd.to_datetime(DateTime)

    prediction = predict_load(DateTime)
    result = ""
    if st.button("Predict"):
        result = prediction
    st.success('Successful!!!')
    st.write('The Load Prediction at Date:',
             date, 'and Time:', time, 'is: ', prediction, '\u00B1 3 MWh')
    st.write('OR')

    with st.expander("Upload CSV with DateTime Column"):
        st.write("IMPORT DATA")
        st.write(
            "Import the time series CSV file. It should have one column labelled as 'DateTime'"
        )
        data = st.file_uploader("Upload here", type="csv")
        st.session_state.counter = 0
        if data is not None:
            dataset = pd.read_csv(data)
            dataset["DateTime"] = pd.to_datetime(dataset["DateTime"])
            dataset = dataset.sort_values("DateTime")

            results = predict_load(dataset["DateTime"])
            st.write("Upload Sucessful")
            st.session_state.counter += 1
            if st.button("Predict Dataset"):
                result = results
                result = pd.concat([dataset, result], axis=1)
                st.success("Successful!!!")
                st.write("Predicting Load")
                resulta = result.copy()
                resulta['DateTime'] = resulta['DateTime'].astype(str)
                st.write(resulta)

                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv(index=False).encode("utf-8")

                csv = convert_df(result)
                st.download_button(
                    label="Download Load Predictions as CSV",
                    data=csv,
                    file_name="Load Predictions.csv",
                    mime="text/csv",
                )
                fig = plt.figure(figsize=(12, 10))
                sns.lineplot(
                    x='DateTime', y='Load Predictions', data=result)

                st.write("The following plot shows predicted Load for your provide Datetime Frame:")
                st.pyplot(fig)
                st.session_state.counter += 1

    with st.expander("Real Time Forecasts with Datetime Range"):
        st.write('From:')
        date1 = st.date_input(
            'Date', datetime.date(2017, 7, 1), key='hst%N@&n8&dn2')  # (2011, 1, 28))
        time1 = st.time_input(
            'Time', datetime.time(0, 00), key='hsye^8nyBT@8b2')  # (hour=18, minute=54, second=30))
        datestr = date1.strftime("%Y-%m-%d")
        timestr = time1.strftime("%H:%M:%S")
        DateTime = datestr + ' ' + timestr
        st.write('To:')
        date2 = st.date_input(
            'Date', datetime.datetime.today(), key='dn&@T6thSGSJ6t5T')  # (2011, 1, 28))
        time2 = st.time_input(
            'Time', datetime.datetime.now(), key='HGt73n7bgs6Jsyu&#5$@nysh')  # (hour=18, minute=54, second=30))
        datestr = date2.strftime("%Y-%m-%d")
        timestr = time2.strftime("%H:%M:%S")
        DateTime1 = datestr + ' ' + timestr
        # DateTime1 = pd.to_datetime(DateTime)
        st.write('Real Time Forecasts')

        forecast_junc = pd.date_range(
            start=DateTime, end=DateTime1, freq='H')
        forecast_junc = pd.DataFrame({'DateTime': forecast_junc})
        # if st.button('Forecast'):

        st.write('Real Time Load Forecast from', DateTime, 'to', DateTime1)
        forecast = predict_load(forecast_junc['DateTime'])
        forecast = pd.concat([forecast_junc, forecast], axis=1)

        # st.write(forecast_junc)

        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode("utf-8")

        csv = convert_df(forecast)
        st.download_button(
            label="Download DateTime Range Predictions as CSV",
            data=csv,
            file_name="Load Predictions by DateTime Range.csv",
            mime="text/csv",
        )
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot(
            x='DateTime', y='Load Predictions', data=forecast)
        st.pyplot(fig)

        st.text("Electric Load Prediction Project")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()
