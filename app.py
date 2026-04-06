import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from cleaning import clean_my_data
from visualizations import get_figures
from model import preprocessor, forest_model, train_df


# Initialize the Dash application
app = dash.Dash(__name__)
server = app.server

# Load and clean data
df_filtered = clean_my_data()

df_filtered['Longitude'] = df_filtered['Geolocation'].apply(lambda x: float(x.split()[1][1:]))
df_filtered['Latitude'] = df_filtered['Geolocation'].apply(lambda x: float(x.split()[2][:-1]))
df_filtered['YearStart'] = df_filtered['YearStart'].astype(str)
df_filtered['YearEnd'] = df_filtered['YearEnd'].astype(str)

# Data preparation for various charts
df_bar = df_filtered[df_filtered['Question'] == 'Diabetes among adults']
df_bar_mean = df_bar.groupby('YearStart')['DataValueAlt'].mean().reset_index()
diabetes_rate_by_year_location = df_bar.groupby(['YearStart', 'LocationAbbr'])['DataValueAlt'].mean().reset_index()

bar_fig = px.bar(
    df_bar_mean,
    x='YearStart',
    y='DataValueAlt',
    labels={'DataValueAlt': 'Average Diabetes Rate(%)', 'YearStart': 'Year'},
    title='Average Diabetes Rate by Year'
)

line_fig = px.line(
    df_bar_mean,
    x='YearStart',
    y='DataValueAlt',
    labels={'DataValueAlt': 'Average Diabetes Rate(%)', 'YearStart': 'Year'},
    title='Trend of Diabetes Rate over Years'
)

heatmap_fig = px.density_heatmap(
    df_filtered[df_filtered['Question'] == 'Asthma mortality among all people, underlying cause'],
    x='Longitude',
    y='Latitude',
    z='DataValueAlt',
    labels={'DataValueAlt': 'Number'},
    nbinsx=5,
    title='Heatmap of Asthma mortality by Location'
)

figs = get_figures()

# App layout
app.layout = html.Div([
    html.H1("Stats 507 Final Project", style={'text-align': 'center'}),
    html.H2("Part 1: EDA", style={'text-align': 'center'}),
    dcc.Graph(figure=bar_fig, id='bar-chart'),
    
    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("What has been the trend in the average diabetes rate over the last four years?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("The average diabetes rate does not fluctuate significantly by year, suggesting that each bar will have roughly the same height in the bar plot."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P([
        "Analysis of the bar plot reveals a steady trend in the average diabetes rates over the years. ",
        "The year 2020 presents the highest average rate, followed by a decrease in 2021. ",
        "Across all four years, the average diabetes rate hovers around 16%, indicating a relatively stable condition without drastic annual fluctuations."
    ]),
], style={'text-align': 'justify'}),


    dcc.Graph(figure=line_fig, id='line-chart'),
   html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("What has been the trend in the average diabetes rate over the last four years?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("On a macro level, there might be little variation between the diabetes rate in different years, but on a micro level in the line plot, we might see the big fluctuation. "),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P([
        "The plot displays a line graph tracking the average diabetes rate from 2019 to 2022. The rate peaks in 2020, followed by a sharp decline in 2021, and a slight uptick again in 2022, suggesting fluctuating diabetes rates over the four-year period."]),
], style={'text-align': 'justify'}),



    dcc.Graph(figure=heatmap_fig, id='heat-chart'),

   html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("How is asthma morality related with longtitude and latitude?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("In habitabal locations, there will be more population, hence higher asthma morality. Therefore, locations with mid longtitude and latitude will have larger asthma morality."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P([
        "The data shows significant mortality clusters in certain longitudinal bands. The most noticeable band with the highest mortality rate is around the -90 to -100 longitude range. Conversely, there are large areas with very low to no asthma-related deaths, indicated by the dominant dark blue color, especially at the extremes of the longitudinal range. This heatmap would be particularly useful for public health officials aiming to identify and target areas with the highest need for asthma-related healthcare resources and interventions."]),
], style={'text-align': 'justify'}),


    html.Div([
        dcc.Dropdown(
            id='year-selector',
            options=[{'label': year, 'value': year} for year in sorted(df_bar['YearStart'].unique())],
            value=sorted(df_bar['YearStart'].unique())[0],
            clearable=False,
        ),
        dcc.Graph(id='top-states-bar-plot')
    ]),

   html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("Which five states have highest diabetes rate within different years?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("Within each year, the top five states with highest diabetes rate will vary. Nontheless, we still expect to see some states always appear on the plot across different years, since it is very hard to eliminate diabetes within only a few years."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P([
        "In 2019, the states NJ, VI, PR, WV, and MS had the highest rates. In 2020, VI remained in the top but was joined by FL, PR, MS, and AL. By 2021, FL had the highest rate, followed by VI, PR, MS, and KY. In 2022, GU appeared as a new entrant with the highest rate, along with PR, VI, WV, and AL. Across the years, while some states like PR and VI consistently appear among the highest, others like NJ and KY only appear once, indicating changes in rankings and diabetes rates across these years."]),
], style={'text-align': 'justify'}),


    html.Div([
        dcc.Dropdown(
            id='state-selector',
            options=[{'label': state, 'value': state} for state in sorted(df_bar['LocationAbbr'].unique())],
            value=sorted(df_bar['LocationAbbr'].unique())[0],
            clearable=False,
        ),
        dcc.Graph(id='state-trend-line-plot')
    ]),

    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("What is the trend of diabetes rate across different states over years"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("The overall diabetes rate across different states over years will vary, but we expect the overall trend of diabetes rate will be decreasing, since people have more opportunities to work out after the COVID-19 pandemic. So the trend is expected to go up in 2020, and go down after 2021."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P([
        "The trend of diabetes rate in different states over the years shows some fluctuations, and the trends vary by state. Nonthless, around half plots show a tranding of decreasing then increasing with a turning point of 2021, indicating our assumption is not baseless."]),
], style={'text-align': 'justify'}),

    
    dcc.Graph(
        id='example-graph',
        figure=figs['disability'],
    ),

    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("Which five states have consistently shown the highest prevalence of disabilities among adults across various years?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("It is presumed that the ranking of the top five states will remain relatively stable over time."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P([
        "The analysis confirms that there has been minimal fluctuation in the rankings. While there are occasional variations in the top five from year to year, the leading states/territories in terms of adult disability prevalence from 2019 to 2022 have been Puerto Rico, West Virginia, Kentucky, Arkansas, Mississippi, and Oklahoma. Puerto Rico consistently exhibits a marginally higher prevalence rate than the other states. "]),
], style={'text-align': 'justify'}),


    dcc.Graph(
        id='example-graph2',
        figure=figs['vaccination'],
    ),

    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("How have influenza vaccination rates among adults varied over time across different states?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("The influenza vaccination rate might remain relatively stable over the years, yet it could vary among different states."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P([
        "The trend of influenza vaccination rates over the years shows some fluctuations, and the trends vary by state. "]),
], style={'text-align': 'justify'}),

    dcc.Graph(
        id='example-graph4',
        figure=figs['asthma'],
    ),

    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("How has asthma prevalence among adults varied over years across different states?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("The asthma prevalence might remain relatively stable over the years, yet it could vary among different states."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P([
        "The trend of asthma prevalence over the years shows some fluctuations, and the trends vary by state. "]),
], style={'text-align': 'justify'}),

    dcc.Graph(
        id='example-graph5',
        figure=figs['correlation'],
    ),

    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("Is there a discernible correlation between the prevalence of diabetes and obesity among the adult populations?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("There is a positive correlation between diabetes and obesity prevalence among adults"),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P([
        "The scatter plot suggests a positive correlation between diabetes and obesity prevalence among adults. As the rate of obesity increases, the prevalence of diabetes tends to rise correspondingly, indicating that higher obesity rates might be a significant predictor of diabetes prevalence in the adult population. "]),
], style={'text-align': 'justify'}),


    dcc.Graph(
        id='example-graph6',
        figure=figs['map'],
    ),

    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("What is the distribution of arthritis prevalence among adults across different states? "),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("There might be regional variations in the prevalence of arthritis."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P([
        "The provided map illustrates the arthritis prevalence by state, indicating that there are indeed regional differences. The color gradient represents the percentage of adults with arthritis, with darker shades signifying higher prevalence rates. The map suggests that some states have a higher percentage of adults reporting arthritis, which aligns with the assumption of regional disparities in arthritis prevalence.  "]),
], style={'text-align': 'justify'}),

    html.Div([
        dcc.Dropdown(
            id='cancer-selector',
            options=[{'label': 'by start year', 'value': '1'}, {'label':'by sex', 'value': '2'}],
            clearable=False,
        ),
        dcc.Graph(id='cancer_1')
    ]),
    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("Does invasive cancer rate depend on regions/sex/time? "),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("Invasive cancer rate may depend on regions but not on sex/time."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P(["The plot shows indeed different cancer rates in different states but there are no notable dependence on sex/time."], style={'text-align': 'justify'}),
    ]),

    # html.H2('''
    #     Invasive Cancer Rate State Distrbution
    # '''),
    
    html.Div([
        dcc.Dropdown(
            id='plot-type-selector',
            options=[{'label': 'bar_plot', 'value':0}, {'label':'pie_plot', 'value':1}],
            clearable=False,
        ),
        dcc.Graph(id='cancer_2')
    ]),
    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("What is the distribution of invasive cancer rate across states?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("Invasive cancer rate may vary across states."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P(["Indeed, according to the pie plot, the highest rate is 4 times the lowest rate."], style={'text-align': 'justify'}),
    ]),

    dcc.Graph(
        id='example-graph10',
        figure=figs['cancer_pie_race'],
    ),
    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("Is cancer rate uniform over races?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("Cancer rates are uniform over races."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P(["White, non-hispanic race has a significantly lower invasive cancer rate."], style={'text-align': 'justify'}),
    ]),
    dcc.Graph(
        id='example-graph11',
        figure=figs['alcohol_high_sex'],
    ),
    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("Is there a sharp change in high-school alcohol usage between 2019 and 2021?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("There are no sharp changes."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P(["The usage has no significant change in all states."], style={'text-align': 'justify'}),
    ]),
    dcc.Graph(
        id='example-graph12',
        figure=figs['alcohol_high_sex_bar'],
    ),
    html.Div([
    html.P("EDA Question:", style={'font-weight': 'bold'}),
    html.P("Are males drinking more than females in high-school?"),
    
    html.P("Assumption:", style={'font-weight': 'bold'}),
    html.P("Males are drinking more."),
    
    html.P("Results:", style={'font-weight': 'bold'}),
    html.P(["In most states, females are drinking more than males."], style={'text-align': 'justify'}),
    ]),

    html.H2("Part 2: Prediction Based on Machine Learning Model", style={'text-align': 'center'}),



    html.Div([
        html.Label("Select YearStart, LocationAbbr, Stratification1:"),
        dcc.Dropdown(id='year-dropdown', options=[{'label': i, 'value': i} for i in train_df['YearStart'].unique()], value=train_df['YearStart'].iloc[0]),
        dcc.Dropdown(id='location-dropdown', options=[{'label': i, 'value': i} for i in train_df['LocationAbbr'].unique()], value=train_df['LocationAbbr'].iloc[0]),
        dcc.Dropdown(id='strat-dropdown', options=[{'label': i, 'value': i} for i in train_df['Stratification1'].unique()], value=train_df['Stratification1'].iloc[0]),
        html.Button('Submit', id='submit-val', n_clicks=0)
    ]),
    html.Div(id='container-button-basic'),

    html.H2("Part 3: Conclusion", style={'text-align': 'center'}),

    html.P('The EDA provides valuable insights into various health-related metrics across the United States over several years. From the visualizations, we observed that diabetes rates have shown slight fluctuations over the years but generally remained stable. A significant finding was in the rates of diabetes in different states, where certain states consistently appeared in the top rankings, indicating persistent public health challenges. Asthma mortality displayed notable geographic clustering, with specific longitudes showing higher rates, which could guide targeted healthcare interventions. The analysis of invasive cancer rates revealed variations across states and a consistent pattern among races, with White, non-Hispanic populations showing notably lower rates compared to other groups. These insights would be helpful for health policy makers to prioritize interventions and resources effectively.'),
    html.P('In the predictive analysis segment, we employed a RandomForestRegressor. It was chosen because of its best performance out of three models we tried, which are linear regression, random forest, and neural networks. This model is part of a pipeline that preprocesses data, incorporating one-hot encoding and standard scaling to optimize inputs for prediction. To refine our model further, we applied feature selection techniques using the correlation matrix, which helped us identify the most influential variables: YearStart, LocationAbbr, and Stratification1. Users can interact with the model by selecting parameters such as year, location, and demographics, which the model uses to predict disease mortality rates. This feature could be useful for public health researchers.')   

])

@app.callback(
    Output('cancer_1', 'figure'),
    Input('cancer-selector', 'value')
)
def update_cancer_view_mode(v):
    if (v == '1'):
        return figs['cancer']
    else:
        return figs['cancer_sex']

@app.callback(
    Output('cancer_2', 'figure'),
    Input('plot-type-selector', 'value')
)
def update_cancer_distribution_plot(v):
    if (v == 1):
        return figs['cancer_pie']
    else:
        return figs['cancer_bar']

@app.callback(
    Output('top-states-bar-plot', 'figure'),
    Input('year-selector', 'value')
)
def update_top_states_bar_plot(selected_year):
    filtered_data = diabetes_rate_by_year_location[diabetes_rate_by_year_location['YearStart'] == selected_year]
    filtered_data = filtered_data.nlargest(5, 'DataValueAlt')
    fig = px.bar(
        filtered_data,
        x='LocationAbbr',
        y='DataValueAlt',
        labels={'DataValueAlt': 'Diabetes Rate(%)', 'LocationAbbr': 'State'},
        title=f'Top 5 States by Diabetes Rate in {selected_year}'
    )
    return fig

@app.callback(
    Output('state-trend-line-plot', 'figure'),
    Input('state-selector', 'value')
)
def update_state_trend_line_plot(selected_state):
    filtered_data = diabetes_rate_by_year_location[diabetes_rate_by_year_location['LocationAbbr'] == selected_state]
    fig = px.line(
        filtered_data,
        x='YearStart',
        y='DataValueAlt',
        labels={'DataValueAlt': 'Diabetes Rate(%)', 'LocationAbbr': 'State'},
        title=f'Trend of Diabetes Rate in {selected_state} Over Years'
    )
    return fig

@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    [Input('year-dropdown', 'value'), Input('location-dropdown', 'value'), Input('strat-dropdown', 'value')]
)
def update_output(n_clicks, year, location, stratification):
    if n_clicks > 0:
        try:
            # Prepare data for prediction
            x = preprocessor.transform(pd.DataFrame([[int(year), location, stratification]], columns=["YearStart", "LocationAbbr", "Stratification1"]))
            pred = forest_model.predict(x)
            return f'The predicted diseases of the heart mortality among all people, underlying cause (cases per 100,000) is: {pred[0]:.2f}'
        except Exception as e:
            return f'Error: {str(e)}'

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
