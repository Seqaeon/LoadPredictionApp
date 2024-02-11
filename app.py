import datetime
import pickle
import warnings
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from pandas.api.types import CategoricalDtype
from pandas.errors import EmptyDataError






# # Load an image from a local file and encode it to base64
# with open("Transformer.jpg", "rb") as f:
#     data = base64.b64encode(f.read()).decode("utf-8")
#
# # Display the image at the top of the site using HTML
# st.markdown(f"""
# <div style="align: center;">
#     <img src="data:image/jpg;base64, {data}" alt="This is my header image" width="1000">
# </div>
# """, unsafe_allow_html=True)

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
    <h2 style="color:white;text-align:center;font-family:'Caveat',cursive;font-weight: 400;max-width: 800px; width: 85%; margin: 0 auto;">UNILAG Feeders Load Predictor</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    image = Image.open("Transformer.jpg")

    st.image(image, width=700)
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
             date, 'and Time:', time, 'is: ', prediction, '\u00B1 3 Amps')
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
                st.write(
                    f"Peak Load is {result['Load Predictions'].max():.2f} Amps on Date: {result['DateTime'].loc[result['Load Predictions'].idxmax()].date()}")
                st.write(f"Total Consumption for this period of time is: {result['Load Predictions'].sum():.2f} Amps")

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
                ax = fig.add_subplot(1, 1, 1)

                sns.lineplot(
                    x='DateTime', y='Load Predictions', data=result)
                x1 = result[result["Load Predictions"] == result["Load Predictions"].max()]['DateTime']

                y1 = np.full(len(x1), result['Load Predictions'].max())

                ax.plot(x1, y1, "ko", markersize=20, fillstyle='full', color='red')

                for i in range(len(x1)):
                    plt.annotate(
                        f"Peak Load ({x1.iloc[i].date()}, {y1[i]:.2f} Amps)",
                        xy=(x1.iloc[i], y1[i]),
                        xytext=(x1.iloc[i] + pd.Timedelta(2, unit='D'),
                                y1[i] + 2),
                        arrowprops=dict(arrowstyle="->"), weight='bold', fontsize=15)

                st.write("The following plot shows predicted Load for your provide Datetime Frame:")
                st.pyplot(fig)
                st.session_state.counter += 1

    with st.expander("Real Time Forecasts with Datetime Range"):
        st.write('From:')
        date1 = st.date_input(
            'Date', datetime.date(2020, 12, 3), key='hst%N@&n8&dn2')  # (2011, 1, 28))
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

        try:

            if DateTime > DateTime1:  # check if start date is later than end date
                raise ValueError('Start date cannot be later than end date')


            forecast_junc = pd.date_range(
                start=DateTime, end=DateTime1, freq='H')
            forecast_junc = pd.DataFrame({'DateTime': forecast_junc})
            if forecast_junc.empty:  # check if date range is empty
                raise EmptyDataError('Date range is empty')

            st.write('Real Time Load Forecast from', DateTime, 'to', DateTime1)
            forecast = predict_load(forecast_junc['DateTime'])
        except (ValueError, EmptyDataError) as e:  # catch both types of errors
            st.write(e)
        # if st.button('Forecast'):

        else:

            forecast = pd.concat([forecast_junc, forecast], axis=1)

            st.write(
                f"Peak Load is {forecast['Load Predictions'].max():.2f} Amps on Date: {forecast['DateTime'].loc[forecast['Load Predictions'].idxmax()].date()}")
            st.write(f"Total Consumption for this period of time is: {forecast['Load Predictions'].sum():.2f} Amps")

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
            ax = fig.add_subplot(1, 1, 1)
            sns.lineplot(
                x='DateTime', y='Load Predictions', data=forecast)
            #x1 = forecast['DateTime'].loc[forecast['Load Predictions'].idxmax()]

            x1 = forecast[forecast["Load Predictions"] == forecast["Load Predictions"].max()]['DateTime']
            # y1 = forecast['Load Predictions'].max()

            y1 = np.full(len(x1), forecast['Load Predictions'].max())

            ax.plot(x1, y1, "ko", markersize=20, fillstyle='full', color='red')

            for i in range(len(x1)):

                plt.annotate(
                    f"Peak Load ({x1.iloc[i].date()}, {y1[i]:.2f} Amps)",
                    xy=(x1.iloc[i], y1[i]),
                    xytext=(x1.iloc[i] + pd.Timedelta(2, unit='D'),
                            y1[i] + 2),
                    arrowprops=dict(arrowstyle="->"), weight='bold', fontsize=15)
            st.pyplot(fig)

        st.text("UNILAG Feeders Load Prediction Project")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()
