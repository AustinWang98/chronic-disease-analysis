from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from cleaning import clean_my_data


def get_figures():

    df = clean_my_data()

    return {
        'disability': get_disability_fig(df),
        'vaccination': get_vaccination_fig(df),
        'asthma': get_asthma_fig(df),
        'correlation': get_correlation_fig(df),
        'map': get_map_fig(df),
        'cancer': get_cancer_fig(df),
        'cancer_sex': get_cancer_sex_fig(df),
        'cancer_pie': get_cancer_pie_fig(df),
        'cancer_pie_race': get_cancer_pie_race_fig(df),
        'cancer_bar': get_cancer_bar_fig(df),
        'alcohol_high_sex': get_alcohol_fig(df),
        'alcohol_high_sex_bar': get_alcohol_bar_fig(df)
    }


def get_disability_fig(df):

    df1 = df[(df['Topic'] == 'Disability') &
        (df['DataValueType'] == 'Crude Prevalence') &
        (df['StratificationCategory1'] == 'Overall')]

    years = [2019, 2020, 2021, 2022]

    fig = go.Figure()

    for year in years:
        filtered_df = df1[df1['YearStart'] == year]
        disability_df = filtered_df.groupby('LocationDesc')['DataValue'].mean().reset_index()
        disability_df = disability_df.sort_values(by='DataValue', ascending=False).head(5)

        fig.add_trace(
            go.Bar(
                x=disability_df['LocationDesc'],
                y=disability_df['DataValue'],
                name=str(year),
                visible=(year == years[0])
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{"visible": [year == y for year in years]},
                            {"title": f"Top 5 States by Disability Prevalence in Adults: {y}"}],
                        label=str(y),
                        method="update"
                    ) for y in years
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ],
        title_text=f"Top 5 States by Disability Prevalence in Adults: {years[0]}",
        xaxis_title="State",
        yaxis_title="Prevalence (%)"
    )

    return fig


def get_vaccination_fig(df):

    df1 = df[(df['Question'] == 'Current asthma among adults') &
        (df['DataValueType'] == 'Crude Prevalence') &
        (df['StratificationCategory1'] == 'Overall')]

    fig = go.Figure()

    Locations = df['LocationDesc'].unique().tolist()

    for Location in Locations:
        filtered_df = df1[(df1['LocationDesc'] == Location)]

        grouped_df = filtered_df.groupby('YearStart', as_index=False)['DataValue'].mean()

        fig.add_trace(
            go.Scatter(
                x=grouped_df['YearStart'].astype(str),
                y=grouped_df['DataValue'].astype(float),
                name=Location,
                mode='lines+markers',
                visible=False
            )
        )

    fig.data[0].visible = True

    buttons = [dict(label=Location,
                    method="update",
                    args=[{"visible": [Location == loc for loc in Locations]},
                        {"title": f"Trend of Influenza Vaccination over Years: {Location}"}])
            for Location in Locations]

    fig.update_layout(
        updatemenus=[dict(active=0,
                        buttons=buttons,
                        x=0.1,
                        xanchor="left",
                        y=1.1,
                        yanchor="top")],
        title_text=f"Trend of Influenza Vaccination over Years: {Locations[0]}",
        xaxis_title="Year",
        yaxis_title="Mean Prevalence (%)"
    )

    return fig


def get_asthma_fig(df):

    df1 = df[(df['Question'] == 'Current asthma among adults') &
            (df['DataValueType'] == 'Crude Prevalence') &
            (df['StratificationCategory1'] == 'Overall')]

    fig = go.Figure()

    Locations = ['Georgia', 'Guam', 'Maine', 'Nevada', 'Ohio', 'Oklahoma', 'Virgin Islands', 'West Virginia', 'Alabama', 'Alaska', 'District of Columbia', 'Illinois', 'Kansas', 'New Jersey', 'Pennsylvania', 'South Carolina', 'United States', 'Vermont', 'Washington', 'Wyoming', 'Arizona', 'Arkansas', 'Louisiana', 'Massachusetts', 'Oregon', 'Kentucky', 'Michigan', 'Minnesota', 'Missouri', 'Idaho', 'Colorado', 'New York', 'North Dakota', 'Texas', 'North Carolina', 'Connecticut', 'Mississippi', 'Virginia', 'Wisconsin', 'Delaware', 'Florida', 'Iowa', 'Montana', 'Indiana', 'California', 'Nebraska', 'Hawaii', 'New Mexico', 'South Dakota', 'Rhode Island', 'New Hampshire', 'Utah', 'Maryland', 'Tennessee', 'Puerto Rico']
    for Location in Locations:
        filtered_df = df1[(df1['LocationDesc'] == Location)]

        grouped_df = filtered_df.groupby('YearStart', as_index=False)['DataValue'].mean().reset_index()

        fig.add_trace(
            go.Scatter(
                x=grouped_df['YearStart'].astype(str),
                y=grouped_df['DataValue'].astype(float),
                name=Location,
                mode='lines+markers',
                visible=False
            )
        )

    fig.data[0].visible = True

    buttons = [dict(label=Location,
                    method="update",
                    args=[{"visible": [Location == loc for loc in Locations]},
                          {"title": f"Trend of Current asthma Prevalence among adults over Years: {Location}"}])
               for Location in Locations]

    fig.update_layout(
        updatemenus=[dict(active=0,
                          buttons=buttons,
                          x=0.1,
                          xanchor="left",
                          y=1.1,
                          yanchor="top")],
        title_text=f"Trend of Current asthma Prevalence among adults over Years: {Locations[0]}",
        xaxis_title="Year",
        yaxis_title="Crude Prevalence (%)"
    )

    return fig


def get_correlation_fig(df):

    Diabetes_df = df[(df['Topic'] == 'Diabetes') &
            (df['Question'] == 'Diabetes among adults') &
            (df['DataValueType'] == 'Crude Prevalence') &
            (df['StratificationCategory1'] == 'Overall')]

    Obesity_df = df[(df['Topic'] == 'Nutrition, Physical Activity, and Weight Status') &
            (df['Question'] == 'Obesity among adults') &
            (df['DataValueType'] == 'Crude Prevalence') &
            (df['StratificationCategory1'] == 'Overall')]

    Diabetes_df['DataValue'] = pd.to_numeric(Diabetes_df['DataValue'], errors='coerce')
    Obesity_df['DataValue'] = pd.to_numeric(Obesity_df['DataValue'], errors='coerce')

    Diabetes_df.rename(columns={'DataValue': 'DiabetesPrevalence'}, inplace=True)
    Obesity_df.rename(columns={'DataValue': 'ObesityPrevalence'}, inplace=True)

    merged_df = pd.merge(Diabetes_df[['LocationAbbr', 'DiabetesPrevalence']],
                        Obesity_df[['LocationAbbr', 'ObesityPrevalence']],
                        on='LocationAbbr', how='inner')

    fig = px.scatter(merged_df, x='ObesityPrevalence', y='DiabetesPrevalence',
                    labels={'ObesityPrevalence': 'Obesity Prevalence (%)',
                            'DiabetesPrevalence': 'Diabetes Prevalence (%)'},
                    title='Correlation between Diabetes and Obesity Prevalence among Adults by State',
                    hover_data=['LocationAbbr'])

    return fig

def get_map_fig(df):
    arthritis_df = df[(df['YearStart'] == 2022) &
    (df['Question'] == 'Arthritis among adults') &
    (df['DataValueType'] == 'Crude Prevalence') &
    (df['StratificationCategory1'] == 'Overall')]

    arthritis_df['DataValue'] = pd.to_numeric(arthritis_df['DataValue'], errors='coerce')

    fig = px.choropleth(arthritis_df,
                        locations='LocationAbbr',
                        locationmode="USA-states",
                        color='DataValue',
                        color_continuous_scale="Viridis",
                        scope="usa",
                        labels={'DataValue': 'Arthritis Prevalence (%)'},
                        title='Arthritis Prevalence by State')

    return fig

def get_cancer_fig(df):
    cancer_2015 = df[(df['Topic'] == 'Cancer') &
    (df['Question'] == 'Invasive cancer (all sites combined) mortality among all people, underlying cause') &
    (df['DataValueType'] == 'Crude Rate') &
    (df['Stratification1'] == 'Hispanic') &
    (df['YearStart'] == 2015)]
    cancer_2016 = df[(df['Topic'] == 'Cancer') &
    (df['Question'] == 'Invasive cancer (all sites combined) mortality among all people, underlying cause') &
    (df['DataValueType'] == 'Crude Rate') &
    (df['Stratification1'] == 'Hispanic') &
    (df['YearStart'] == 2016)]
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'choropleth'}, {'type': 'choropleth'}]], subplot_titles=["2015-2019 invasive cancer", "2016-2020 invasive cancer"])

    fig.add_trace(
        go.Choropleth(locations=cancer_2015.LocationAbbr,
                        locationmode="USA-states",
                        z=cancer_2015.DataValue,
                        marker_line_color='white',
                        zmin=0, zmax=120),
        row=1, col=1
    )

    fig.add_trace(
        go.Choropleth(locations=cancer_2016.LocationAbbr,
                        locationmode="USA-states",
                        z=cancer_2016.DataValue,
                        marker_line_color='white',
                        zmin=0, zmax=120),
        row=1, col=2
    )

    fig.update_geos(
        visible=False,
        resolution=50,
        scope="usa",
        showcountries=True,
        countrycolor="Black",
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="White",
        row=1, col=1
    )
    fig.update_geos(
        visible=False,
        resolution=50,
        scope="usa",
        showcountries=True,
        countrycolor="Black",
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="White",
        row=1, col=2
    )

    fig.update_layout(
        title_text='For Hispanic, Comparsion by Year'
    )

    return fig

def get_cancer_sex_fig(df):
    cancer_male = df[(df['Topic'] == 'Cancer') &
    (df['Question'] == 'Invasive cancer (all sites combined) mortality among all people, underlying cause') &
    (df['DataValueType'] == 'Crude Rate') &
    (df['Stratification1'] == 'Male')]
    cancer_female = df[(df['Topic'] == 'Cancer') &
    (df['Question'] == 'Invasive cancer (all sites combined) mortality among all people, underlying cause') &
    (df['DataValueType'] == 'Crude Rate') &
    (df['Stratification1'] == 'Female')]
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'choropleth'}, {'type': 'choropleth'}]], subplot_titles=["male invasive cancer", "female invasive cancer"])

    fig.add_trace(
        go.Choropleth(locations=cancer_male.LocationAbbr,
                        locationmode="USA-states",
                        z=cancer_male.DataValue,
                        marker_line_color='white',
                        zmin=100, zmax=300),
        row=1, col=1
    )

    fig.add_trace(
        go.Choropleth(locations=cancer_female.LocationAbbr,
                        locationmode="USA-states",
                        z=cancer_female.DataValue,
                        marker_line_color='white',
                        zmin=100, zmax=300),
        row=1, col=2
    )

    fig.update_geos(
        visible=False,
        resolution=50,
        scope="usa",
        showcountries=True,
        countrycolor="Black",
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="White",
        row=1, col=1
    )
    fig.update_geos(
        visible=False,
        resolution=50,
        scope="usa",
        showcountries=True,
        countrycolor="Black",
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="White",
        row=1, col=2
    )

    fig.update_layout(
        title_text='For Hispanic, Comparsion by Sex'
    )

    return fig

def get_cancer_pie_fig(df):

    cancer_2015 = df[(df['Topic'] == 'Cancer') &
    (df['Question'] == 'Invasive cancer (all sites combined) mortality among all people, underlying cause') &
    (df['DataValueType'] == 'Crude Rate') &
    (df['Stratification1'] == 'Hispanic') &
    (df['YearStart'] == 2015)]

    state_grouped = cancer_2015.groupby('LocationAbbr')['DataValue'].mean().reset_index()

    fig_cancer_pie = px.pie(state_grouped, names='LocationAbbr', values='DataValue',
                                title='Hispanic Invasive Cancer Rate Pie Plot')

    return fig_cancer_pie

def get_cancer_pie_race_fig(df):

    cancer_2015 = df[(df['Topic'] == 'Cancer') &
    (df['Question'] == 'Invasive cancer (all sites combined) mortality among all people, underlying cause') &
    (df['DataValueType'] == 'Crude Rate') &
    (df['StratificationCategory1'] == 'Race/Ethnicity') &
    (df['YearStart'] == 2015)]

    state_grouped = cancer_2015.groupby('Stratification1')['DataValue'].sum().reset_index()

    fig_cancer_pie = px.pie(state_grouped, names='Stratification1', values='DataValue',
                                title='Invasive Cancer Rate by Races')

    return fig_cancer_pie

def get_cancer_bar_fig(df):
    cancer_2015 = df[(df['Topic'] == 'Cancer') &
    (df['Question'] == 'Invasive cancer (all sites combined) mortality among all people, underlying cause') &
    (df['DataValueType'] == 'Crude Rate') &
    (df['Stratification1'] == 'Hispanic') &
    (df['YearStart'] == 2015)]

    state_grouped = cancer_2015.groupby('LocationAbbr')['DataValue'].mean().reset_index()

    fig_cancer_bar = px.bar(state_grouped, x='LocationAbbr', y='DataValue',
                                title='Hispanic Invasive Cancer Rate Bar Plot')

    return fig_cancer_bar

def get_alcohol_fig(df):
    alcohol = df[(df['Topic'] == 'Alcohol') &
    (df['Question'] == 'Alcohol use among high school students') &
    (df['DataValueType'] == 'Crude Prevalence') &
    (df['StratificationCategory1'] == 'Sex') &
    (df['YearStart'] == 2019)]
    alcohol2 = df[(df['Topic'] == 'Alcohol') &
    (df['Question'] == 'Alcohol use among high school students') &
    (df['DataValueType'] == 'Crude Prevalence') &
    (df['StratificationCategory1'] == 'Sex') &
    (df['YearStart'] == 2021)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=alcohol.LocationAbbr,
        y=alcohol.DataValue,
        name='2019',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=alcohol2.LocationAbbr,
        y=alcohol2.DataValue,
        name='2021',
        marker_color='lightsalmon'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45, title="Compare high school alcohol usage between 2019 and 2021")
    return fig

def get_alcohol_bar_fig(df):
    alcohol = df[(df['Topic'] == 'Alcohol') &
    (df['Question'] == 'Alcohol use among high school students') &
    (df['DataValueType'] == 'Crude Prevalence') &
    (df['StratificationCategory1'] == 'Sex') &
    (df['YearStart'] == 2019)]

    fig = px.bar(alcohol, x="LocationAbbr", y="DataValue", color="Stratification1",
    title="Alcohol use in high school by states and sex")
    return fig
