from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

DATA_PATH = Path('merged_cleaned.xlsx')
df = pd.read_excel(DATA_PATH)
required = {'CLIENT ID','YEAR','REVENUE','HARDWARE','SOFTWARE','MANPOWER','NPS RATING',
            'PRESALES AND PARTNERSHIP','TECHNICAL EXPERTISE','PROJECT DELIVERY','POST-SALES SUPPORT',
            'TYPE','SECTOR','STAFF STRENGTH','COUNTRY'}
missing = sorted(required - set(df.columns))
if missing:
    raise KeyError(missing)

service_cols = ['PRESALES AND PARTNERSHIP','TECHNICAL EXPERTISE','PROJECT DELIVERY','POST-SALES SUPPORT']
service_labels = {
    'PRESALES AND PARTNERSHIP':'Presales & Partnership',
    'TECHNICAL EXPERTISE':'Technical Expertise',
    'PROJECT DELIVERY':'Project Delivery',
    'POST-SALES SUPPORT':'Post-Sales Support'
}
service_order = list(service_labels.values())
category_order = ['Promoter','Passive','Detractor']
colour_map = {'Promoter':'#168aad','Passive':'#f4a261','Detractor':'#e76f51'}
profile_options = [
    {'label':'Industry sector','value':'SECTOR'},
    {'label':'Client type','value':'TYPE'},
    {'label':'Organisation size','value':'STAFF STRENGTH'},
    {'label':'Country','value':'COUNTRY'}
]
he_profile_options = [
    {'label':'Industry sector','value':'Sector'},
    {'label':'Client type','value':'Client_Type'},
    {'label':'Organisation size','value':'Organisation_Size'},
    {'label':'Country','value':'Country'}
]

analysis = df.copy()
analysis['Overall Satisfaction'] = analysis[service_cols].mean(axis=1)
analysis['COGS'] = analysis[['HARDWARE','SOFTWARE','MANPOWER']].sum(axis=1, min_count=1)
analysis['GROSS_PROFIT'] = analysis['REVENUE'] - analysis['COGS']
analysis['GROSS_MARGIN'] = analysis['GROSS_PROFIT'] / analysis['REVENUE'].replace(0, np.nan)
analysis['NPS Category'] = np.select([analysis['NPS RATING'].ge(9), analysis['NPS RATING'].ge(7)], ['Promoter','Passive'], default='Detractor')
analysis['NPS Category'] = pd.Categorical(analysis['NPS Category'], categories=category_order, ordered=True)
years = sorted(analysis['YEAR'].dropna().astype(int).unique())
year_min, year_max = int(min(years)), int(max(years))
client_types = ['All'] + sorted(analysis['TYPE'].dropna().unique().tolist())

print(f'Loaded {len(analysis):,} client-year rows; {analysis["CLIENT ID"].nunique():,} unique clients; years {year_min}–{year_max}.')
print('Duplicate client-year records:', int(analysis.duplicated(['CLIENT ID','YEAR']).sum()))

def source_filter(year_range, client_type, nps_values):
    start, end = year_range
    d = analysis[analysis['YEAR'].between(start, end)].copy()
    if client_type != 'All':
        d = d[d['TYPE'].eq(client_type)]
    if nps_values:
        d = d[d['NPS Category'].astype(str).isin(nps_values)]
    else:
        d = d.iloc[0:0].copy()
    return d

def client_level(d):
    if d.empty:
        return d.copy()
    c = (d.groupby('CLIENT ID', as_index=False, observed=True)
           .agg(Client_Type=('TYPE','first'), Sector=('SECTOR','first'), Organisation_Size=('STAFF STRENGTH','first'),
                Country=('COUNTRY','first'), Revenue=('REVENUE','sum'), Gross_Profit=('GROSS_PROFIT','sum'),
                Average_Satisfaction=('Overall Satisfaction','mean'), Average_NPS=('NPS RATING','mean'),
                First_Observed_Year=('YEAR','min'), Last_Observed_Year=('YEAR','max')))
    c['Gross_Margin'] = c['Gross_Profit'] / c['Revenue'].replace(0, np.nan)
    c['Observed_Longevity'] = c['Last_Observed_Year'] - c['First_Observed_Year'] + 1
    c['NPS Category'] = pd.Categorical(np.select([c['Average_NPS'].ge(9), c['Average_NPS'].ge(7)], ['Promoter','Passive'], default='Detractor'), categories=category_order, ordered=True)
    sat_med = c['Average_Satisfaction'].median()
    margin_med = c['Gross_Margin'].median()
    c['Priority Group'] = np.select([
        (c['Average_Satisfaction'] < sat_med) & (c['Gross_Margin'] >= margin_med),
        (c['Average_Satisfaction'] >= sat_med) & (c['Gross_Margin'] >= margin_med),
        (c['Average_Satisfaction'] >= sat_med) & (c['Gross_Margin'] < margin_med)],
        ['Prioritise','Retain','Improve'], default='Reconsider')
    return c

def profile_summary(d, profile_col):
    p = (d.groupby(profile_col, dropna=False, observed=True)
           .agg(Clients=('CLIENT ID','size'), Revenue=('REVENUE','sum'), Gross_Profit=('GROSS_PROFIT','sum'),
                Gross_Margin=('GROSS_MARGIN','mean'), Average_Satisfaction=('Overall Satisfaction','mean'),
                Median_Longevity=('YEAR','nunique'))
           .reset_index().rename(columns={profile_col:'Profile'}))
    p['Profile'] = p['Profile'].fillna('Unknown').astype(str)
    return p

def kpi_cards(source, clients):
    if source.empty:
        values = [('Records','0'),('Clients','0'),('Gross profit','$0'),('Average NPS','—')]
    else:
        values = [('Records',f'{len(source):,}'),('Clients',f'{source["CLIENT ID"].nunique():,}'),
                  ('Gross profit',f'${source["GROSS_PROFIT"].sum():,.0f}'),('Average NPS',f'{source["NPS RATING"].mean():.2f}/10')]
    return [html.Div([html.Div(label, style={'fontSize':'12px','color':'#718096','fontWeight':'600'}),
                      html.Div(value, style={'fontSize':'22px','fontWeight':'700','color':'#165D7A','marginTop':'4px'})],
                     style={'backgroundColor':'white','padding':'15px 18px','borderRadius':'10px','boxShadow':'0 2px 6px rgba(0,0,0,.06)'}) for label,value in values]

# -------------------- Jhgraphs: profitability and cost drivers --------------------
def jh_fig1(d, profile_col, metric):
    if d.empty:
        return go.Figure().update_layout(title='No data match the selected filters', template='plotly_white')
    p = profile_summary(d, profile_col).sort_values(metric, ascending=True)
    label = 'Gross Profit' if metric == 'Gross_Profit' else 'Revenue'
    fig = px.bar(p, x=metric, y='Profile', orientation='h', color='Gross_Margin',
                 color_continuous_scale='Viridis', text=metric, template='plotly_white',
                 hover_data={'Gross_Profit':':,.0f','Revenue':':,.0f','Gross_Margin':':.1%','Clients':True,'Average_Satisfaction':':.2f'})
    fig.update_traces(texttemplate='$%{x:,.0f}', textposition='outside')
    fig.update_layout(title=f'{label} across {profile_col.lower().replace("_"," ")} profiles', xaxis_title=label, yaxis_title='',
                      height=500, margin=dict(t=70,l=160,r=40,b=60), coloraxis_colorbar=dict(title='Gross margin'))
    fig.update_xaxes(tickformat='$,.0f')
    return fig

def jh_fig2(d, profile_col):
    if d.empty:
        return go.Figure().update_layout(title='No data match the selected filters', template='plotly_white')
    p = profile_summary(d, profile_col)
    fig = go.Figure()
    for profile in p['Profile'].astype(str):
        x = p[p['Profile'].eq(profile)]
        fig.add_trace(go.Scatter(x=x['Average_Satisfaction'], y=x['Gross_Margin']*100, mode='markers+text', text=x['Profile'], name=profile,
            marker=dict(size=np.clip(np.sqrt(x['Gross_Profit'].clip(lower=0))/500,12,48), color=x['Gross_Profit'], colorscale='Viridis', showscale=(profile==p['Profile'].astype(str).iloc[0]),
                        colorbar=dict(title='Gross profit'), line=dict(width=1,color='white')),
            customdata=x[['Profile','Gross_Profit','Revenue','Clients']].to_numpy(),
            hovertemplate='<b>%{customdata[0]}</b><br>Average service rating: %{x:.2f}/5<br>Gross margin: %{y:.1f}%<br>Gross profit: $%{customdata[1]:,.0f}<br>Revenue: $%{customdata[2]:,.0f}<br>Clients: %{customdata[3]:.0f}<extra></extra>'))
    fig.update_layout(title=f'Service quality versus gross margin by {profile_col.lower().replace("_"," ")}', template='plotly_white', height=500,
                      xaxis_title='Average service rating (1–5)', yaxis_title='Average gross margin (%)', showlegend=False,
                      margin=dict(t=70,l=70,r=30,b=60))
    return fig

def jh_fig3(d, profile_col, mode):
    if d.empty:
        return go.Figure().update_layout(title='No data match the selected filters', template='plotly_white')
    p = d.groupby(profile_col, dropna=False, observed=True)[['HARDWARE','SOFTWARE','MANPOWER']].sum().reset_index().rename(columns={profile_col:'Profile'})
    p['Profile'] = p['Profile'].fillna('Unknown').astype(str)
    fig = go.Figure()
    for col,color in [('HARDWARE','#3182CE'),('SOFTWARE','#DD6B20'),('MANPOWER','#38A169')]:
        fig.add_trace(go.Bar(x=p['Profile'], y=p[col], name=col.title(), marker_color=color))
    fig.update_layout(title=f'Cost composition by {profile_col.lower().replace("_"," ")}', barmode=mode.lower(), template='plotly_white', height=500,
                      xaxis_title='', yaxis_title='Total cost ($)', legend_title='Cost component', margin=dict(t=70,l=70,r=30,b=70))
    fig.update_yaxes(tickformat='$,.0f')
    return fig

# -------------------- Kaydengraphs: satisfaction and loyalty --------------------
def kay_fig1(d, period):
    if d.empty:
        return go.Figure().update_layout(title='No data match the selected filters', template='plotly_white')
    if period == 'Overall':
        g = d.groupby('TYPE', as_index=False)[service_cols].mean()
    else:
        g = d[d['YEAR'].eq(int(period))].groupby('TYPE', as_index=False)[service_cols].mean()
    long = g.melt(id_vars='TYPE', var_name='Service', value_name='Average Rating')
    long['Service'] = long['Service'].replace(service_labels)
    fig = px.bar(long, x='TYPE', y='Average Rating', color='Service', barmode='group', text='Average Rating',
                 category_orders={'Service':service_order}, range_y=[0,5.4], template='plotly_white',
                 color_discrete_sequence=['#636EFA','#EF553B','#00CC96','#AB63FA'])
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside', cliponaxis=False)
    fig.update_layout(title=f'Service ratings by client type — {period}', xaxis_title='Client type', yaxis_title='Average rating (1–5)', height=500,
                      margin=dict(t=70,l=70,r=30,b=60))
    return fig

def kay_fig2(d):
    if d.empty:
        return go.Figure().update_layout(title='No data match the selected filters', template='plotly_white')
    p = d.groupby(['TYPE','YEAR'], as_index=False)['NPS RATING'].mean().pivot(index='TYPE',columns='YEAR',values='NPS RATING')
    p = p.reindex(sorted(p.columns), axis=1)
    p['Overall'] = d.groupby('TYPE')['NPS RATING'].mean().reindex(p.index)
    p = p.round(2)
    counts = d.groupby(['TYPE','YEAR']).size().unstack('YEAR').reindex(p.index).reindex(columns=p.columns.drop('Overall'), fill_value=0)
    counts['Overall'] = d.groupby('TYPE').size().reindex(p.index)
    fig = go.Figure(go.Heatmap(x=[str(x) for x in p.columns], y=p.index, z=p.values, text=p.values, texttemplate='%{text:.2f}',
        customdata=counts.values, zmin=7,zmax=9, colorscale=[[0,'#E76F51'],[.5,'#F6D365'],[1,'#2A9D8F']],
        colorbar=dict(title='Average NPS'), hovertemplate='<b>%{y}</b><br>Period: %{x}<br>Average NPS: %{z:.2f}/10<br>Records: %{customdata}<extra></extra>'))
    fig.update_layout(title='Average NPS rating by client type and year', template='plotly_white', height=500, xaxis_title='Year', yaxis_title='Client type', margin=dict(t=70,l=100,r=80,b=60))
    return fig

def kay_fig3(d, segment):
    if d.empty:
        return go.Figure().update_layout(title='No data match the selected filters', template='plotly_white')
    groups = {'Overall':d}
    for t in sorted(d['TYPE'].dropna().unique()): groups[t] = d[d['TYPE'].eq(t)]
    g = groups.get(segment, d)
    corr = g[service_cols + ['NPS RATING']].corr(method='spearman')['NPS RATING'].drop('NPS RATING').rename('Correlation').reset_index().rename(columns={'index':'Service'})
    corr['Service'] = corr['Service'].replace(service_labels)
    corr = corr.sort_values('Correlation')
    fig = go.Figure(go.Bar(x=corr['Correlation'], y=corr['Service'], orientation='h', marker_color=[('#2A9D8F' if x>=0 else '#E76F51') for x in corr['Correlation']],
        text=[f'{x:+.2f}' for x in corr['Correlation']], textposition='outside', customdata=np.full((len(corr),1),len(g)),
        hovertemplate='<b>%{y}</b><br>Spearman correlation: %{x:.3f}<br>Records: %{customdata[0]}<br><i>Association, not causation</i><extra></extra>'))
    fig.add_vline(x=0,line_color='#555',line_width=1)
    fig.update_layout(title=f'Service dimensions associated with NPS — {segment}', template='plotly_white', height=500, xaxis=dict(title='Spearman correlation with NPS',range=[-1.05,1.05]), yaxis_title='', margin=dict(t=80,l=190,r=40,b=60), showlegend=False)
    return fig

# -------------------- Hegraphs: advocacy and retention --------------------
def he_fig1(c, profile_col):
    if c.empty:
        return go.Figure().update_layout(title='No data match the selected filters', template='plotly_white')
    p = c.groupby([profile_col,'NPS Category'], dropna=False, observed=True).agg(Gross_Profit=('Gross_Profit','sum'), Revenue=('Revenue','sum'), Clients=('CLIENT ID','size'), Median_Longevity=('Observed_Longevity','median'), Average_Satisfaction=('Average_Satisfaction','mean')).reset_index().rename(columns={profile_col:'Profile'})
    p['Profile'] = p['Profile'].fillna('Unknown').astype(str); p['Gross_Margin'] = p['Gross_Profit']/p['Revenue'].replace(0,np.nan); p['NPS Category'] = pd.Categorical(p['NPS Category'],categories=category_order,ordered=True)
    p = p.sort_values(['Gross_Profit','NPS Category'])
    p['Label'] = p['Profile'].astype(str)+' — '+p['NPS Category'].astype(str)
    p['Text'] = p['Gross_Profit'].map(lambda x:f'${x/1e6:.1f}M')
    fig = px.bar(p,x='Gross_Profit',y='Label',orientation='h',color='NPS Category',text='Text',category_orders={'NPS Category':category_order},color_discrete_map=colour_map,template='plotly_white',custom_data=['Profile','NPS Category','Revenue','Gross_Profit','Gross_Margin','Clients','Median_Longevity','Average_Satisfaction'])
    fig.update_traces(textposition='outside',hovertemplate='<b>%{customdata[0]} — %{customdata[1]}</b><br>Gross profit: $%{customdata[3]:,.0f}<br>Revenue: $%{customdata[2]:,.0f}<br>Gross margin: %{customdata[4]:.1%}<br>Clients: %{customdata[5]}<br>Median observed longevity: %{customdata[6]:.1f} years<br>Average satisfaction: %{customdata[7]:.2f}/5<extra></extra>')
    fig.update_layout(title=f'Gross-profit contribution by advocacy group and {profile_col.lower().replace("_"," ")}',xaxis_title='Gross profit ($)',yaxis_title='',height=500,margin=dict(t=70,l=190,r=40,b=60),legend_title='NPS category'); fig.update_xaxes(tickformat='$,.0f')
    return fig

def he_fig2(c):
    if c.empty:
        return go.Figure().update_layout(title='No data match the selected filters', template='plotly_white')
    fig = go.Figure()
    for cat in category_order:
        x=c[c['NPS Category'].eq(cat)]
        custom=x[['CLIENT ID','Client_Type','Sector','Country','Revenue','Gross_Profit','Observed_Longevity','Average_NPS','Priority Group']].to_numpy()
        fig.add_trace(go.Scatter(x=x['Average_Satisfaction'],y=x['Gross_Margin']*100,mode='markers',name=cat,marker=dict(size=np.clip(np.sqrt(x['Gross_Profit'].clip(lower=0))/700,8,38),color=colour_map[cat],opacity=.78,line=dict(width=.8,color='white')),customdata=custom,hovertemplate='<b>Client %{customdata[0]}</b><br>Satisfaction: %{x:.2f}/5<br>Gross margin: %{y:.1f}%<br>Gross profit: $%{customdata[5]:,.0f}<br>Revenue: $%{customdata[4]:,.0f}<br>NPS: %{customdata[7]:.1f} (%{fullData.name})<br>Longevity: %{customdata[6]} years<br>Action zone: %{customdata[8]}<extra></extra>'))
    sat=c['Average_Satisfaction'].median(); margin=c['Gross_Margin'].median()*100
    fig.add_vline(x=sat,line_dash='dash',line_color='#4a5568',annotation_text=f'Median satisfaction {sat:.2f}',annotation_position='top right'); fig.add_hline(y=margin,line_dash='dash',line_color='#4a5568',annotation_text=f'Median margin {margin:.1f}%',annotation_position='bottom right')
    fig.update_layout(title='Client priority matrix: satisfaction, profitability and advocacy',template='plotly_white',height=500,xaxis_title='Average overall satisfaction (1–5)',yaxis_title='Client-level gross margin (%)',legend_title='NPS category',margin=dict(t=80,l=70,r=30,b=60))
    return fig

def he_fig3(c, profile_col):
    if c.empty:
        return go.Figure().update_layout(title='No data match the selected filters', template='plotly_white')
    p = c.groupby(profile_col,dropna=False,observed=True).agg(Clients=('CLIENT ID','size'),Revenue=('Revenue','sum'),Gross_Profit=('Gross_Profit','sum'),Median_Longevity=('Observed_Longevity','median')).reset_index().rename(columns={profile_col:'Profile'}); p['Profile']=p['Profile'].fillna('Unknown').astype(str)
    counts=c.assign(_one=1).pivot_table(index=profile_col,columns='NPS Category',values='_one',aggfunc='sum',fill_value=0,observed=False).reset_index().rename(columns={profile_col:'Profile'})
    for cat in category_order:
        if cat not in counts.columns: counts[cat]=0
    p=p.merge(counts[['Profile']+category_order],on='Profile',how='left')
    fig=go.Figure()
    for cat in category_order:
        share=p[cat]/p['Clients']*100
        custom=p[['Profile','Clients','Gross_Profit','Revenue','Median_Longevity',cat]].to_numpy()
        fig.add_trace(go.Bar(x=share,y=p['Profile'],orientation='h',name=cat,marker_color=colour_map[cat],text=share.map(lambda x:f'{x:.0f}%' if x>=8 else ''),textposition='inside',customdata=custom,hovertemplate='<b>%{customdata[0]}</b><br>NPS category: '+cat+'<br>Share: %{x:.1f}%<br>Clients in category: %{customdata[5]:.0f}<br>Total profile gross profit: $%{customdata[2]:,.0f}<br>Clients in profile: %{customdata[1]:.0f}<br>Median longevity: %{customdata[4]:.1f} years<extra></extra>'))
    labels=[dict(x=101.5,y=row['Profile'],xref='x',yref='y',text=f"GP ${row['Gross_Profit']/1e6:.1f}M | n={int(row['Clients'])} | {row['Median_Longevity']:.1f}y",showarrow=False,xanchor='left',font=dict(size=10,color='#4A5568')) for _,row in p.iterrows()]
    fig.update_layout(title=f'NPS composition and commercial value by {profile_col.lower().replace("_"," ")}',template='plotly_white',height=500,barmode='stack',xaxis=dict(title='Share of clients within profile (%)',range=[0,122],ticksuffix='%'),yaxis=dict(title='',categoryorder='array',categoryarray=p['Profile']),annotations=labels,legend_title='NPS category',margin=dict(t=80,l=150,r=220,b=60))
    return fig

# Combined storyboard layout and callbacks
app = Dash(__name__)
app.title = 'Client Value Storyboard'

header = html.Div([
    html.Div([html.Div('DV', style={'width':'48px','height':'48px','borderRadius':'12px','backgroundColor':'#165D7A','color':'white','fontSize':'22px','fontWeight':'800','display':'flex','alignItems':'center','justifyContent':'center'}),
              html.Div([html.Div('CLIENT VALUE STORYBOARD', style={'fontSize':'11px','fontWeight':'800','letterSpacing':'1.8px','color':'#165D7A'}),
                        html.Div('Profitability → Satisfaction → Advocacy', style={'fontSize':'22px','fontWeight':'750','color':'#172B4D'})])], style={'display':'flex','alignItems':'center','gap':'14px'}),
    html.P('A connected view of where value is created, what clients experience and which accounts deserve action.', style={'margin':'10px 0 0','color':'#5E6C84'})
], style={'padding':'22px 30px','backgroundColor':'white','borderBottom':'1px solid #D9E2EC'})

global_controls = html.Div([
    html.Div([html.Label('Observed years',style={'fontWeight':'700'}), dcc.RangeSlider(id='story-years',min=year_min,max=year_max,value=[year_min,year_max],marks={int(y):str(int(y)) for y in years},step=1)] ,style={'gridColumn':'span 2'}),
    html.Div([html.Label('Client type filter',style={'fontWeight':'700'}), dcc.Dropdown(id='story-client-type',options=[{'label':x,'value':x} for x in client_types],value='All',clearable=False)]),
    html.Div([html.Label('NPS categories',style={'fontWeight':'700'}), dcc.Checklist(id='story-nps',options=[{'label':x,'value':x} for x in category_order],value=category_order,inline=True,labelStyle={'display':'inline-block','marginRight':'12px'})])
],style={'display':'grid','gridTemplateColumns':'repeat(2,minmax(0,1fr))','gap':'16px 22px','padding':'18px 30px','backgroundColor':'#F7FAFC','borderBottom':'1px solid #D9E2EC'})

def graph_card(graph_id):
    return html.Div(dcc.Graph(id=graph_id,config={'displaylogo':False,'responsive':True}),style={'backgroundColor':'white','borderRadius':'10px','padding':'8px','boxShadow':'0 2px 6px rgba(0,0,0,.06)'})

app.layout = html.Div([header, global_controls, html.Div(id='story-kpis',style={'display':'grid','gridTemplateColumns':'repeat(4,1fr)','gap':'16px','padding':'18px 30px 2px','backgroundColor':'#EDF2F7'}),
    dcc.Tabs(id='story-tabs',value='profitability',children=[
        dcc.Tab(label='1. Profitability & Cost Drivers',value='profitability',children=[html.Div([html.H2('Where is commercial value created?',style={'color':'#172B4D'}),html.P('Jhgraphs establishes the financial baseline: segment profitability, service quality versus margin and cost composition.',style={'color':'#5E6C84'}),
            html.Div([html.Div([html.Label('Profile breakdown',style={'fontWeight':'700'}),dcc.Dropdown(id='jh-profile',options=profile_options,value='SECTOR',clearable=False)]),html.Div([html.Label('Chart 1 measure',style={'fontWeight':'700'}),dcc.RadioItems(id='jh-metric',options=[{'label':'Gross profit','value':'Gross_Profit'},{'label':'Revenue','value':'Revenue'}],value='Gross_Profit',inline=True)]),html.Div([html.Label('Cost view',style={'fontWeight':'700'}),dcc.RadioItems(id='jh-cost-mode',options=[{'label':'Stacked','value':'stack'},{'label':'Grouped','value':'group'}],value='stack',inline=True)])],style={'display':'grid','gridTemplateColumns':'repeat(3,1fr)','gap':'16px','padding':'12px 0'}),
            html.Div([graph_card('jh-chart-1'),graph_card('jh-chart-2'),graph_card('jh-chart-3')],style={'display':'grid','gridTemplateColumns':'1fr','gap':'18px'})],style={'padding':'20px 30px','backgroundColor':'#EDF2F7'})]),
        dcc.Tab(label='2. Satisfaction & Loyalty',value='satisfaction',children=[html.Div([html.H2('What do clients experience?',style={'color':'#172B4D'}),html.P('Kaydengraphs connects service ratings and NPS over time, then identifies service dimensions associated with advocacy.',style={'color':'#5E6C84'}),
            html.Div([html.Div([html.Label('Service-rating period',style={'fontWeight':'700'}),dcc.Dropdown(id='kay-period',options=[{'label':str(y),'value':str(y)} for y in years]+[{'label':'Overall','value':'Overall'}],value='Overall',clearable=False)]),html.Div([html.Label('Correlation segment',style={'fontWeight':'700'}),dcc.Dropdown(id='kay-segment',options=[{'label':'Overall','value':'Overall'},{'label':'Govt','value':'Govt'},{'label':'NPO','value':'NPO'},{'label':'Private','value':'Private'}],value='Overall',clearable=False)])],style={'display':'grid','gridTemplateColumns':'repeat(2,1fr)','gap':'16px','padding':'12px 0'}),
            html.Div([graph_card('kay-chart-1'),html.Div([graph_card('kay-chart-2'),graph_card('kay-chart-3')],style={'display':'grid','gridTemplateColumns':'repeat(2,minmax(0,1fr))','gap':'18px'})],style={'display':'grid','gridTemplateColumns':'1fr','gap':'18px'})],style={'padding':'20px 30px','backgroundColor':'#EDF2F7'})]),
        dcc.Tab(label='3. Advocacy & Retention Prioritisation',value='advocacy',children=[html.Div([html.H2('Which accounts deserve action?',style={'color':'#172B4D'}),html.P('Hegraphs translates financial and satisfaction evidence into Promoter, Passive and Detractor priorities while showing observed longevity.',style={'color':'#5E6C84'}),
            html.Div([html.Div([html.Label('Advocacy profile breakdown',style={'fontWeight':'700'}),dcc.Dropdown(id='he-profile',options=he_profile_options,value='Sector',clearable=False)])],style={'padding':'12px 0'}),
            html.Div([graph_card('he-chart-1'),html.Div([graph_card('he-chart-2'),graph_card('he-chart-3')],style={'display':'grid','gridTemplateColumns':'repeat(2,minmax(0,1fr))','gap':'18px'})],style={'display':'grid','gridTemplateColumns':'1fr','gap':'18px'})],style={'padding':'20px 30px','backgroundColor':'#EDF2F7'})])
    ]),
    html.Footer('Descriptive storyboard: associations do not prove causation; observed longevity is not confirmed renewal or churn.',style={'padding':'14px 30px','fontSize':'12px','color':'#718096','backgroundColor':'white','borderTop':'1px solid #D9E2EC'})
],style={'fontFamily':'Arial, sans-serif','backgroundColor':'#EDF2F7','minHeight':'100vh'})

@app.callback(
    Output('story-kpis','children'),Output('jh-chart-1','figure'),Output('jh-chart-2','figure'),Output('jh-chart-3','figure'),
    Output('kay-chart-1','figure'),Output('kay-chart-2','figure'),Output('kay-chart-3','figure'),
    Output('he-chart-1','figure'),Output('he-chart-2','figure'),Output('he-chart-3','figure'),
    Input('story-years','value'),Input('story-client-type','value'),Input('story-nps','value'),
    Input('jh-profile','value'),Input('jh-metric','value'),Input('jh-cost-mode','value'),Input('kay-period','value'),Input('kay-segment','value'),Input('he-profile','value'))
def update_storyboard(year_range, client_type, nps_values, jh_profile, jh_metric, jh_cost_mode, kay_period, kay_segment, he_profile):
    src = source_filter(year_range, client_type, nps_values)
    c = client_level(source_filter(year_range, client_type, category_order))
    if nps_values:
        c = c[c['NPS Category'].astype(str).isin(nps_values)]
    else:
        c = c.iloc[0:0].copy()
    return (kpi_cards(src,c), jh_fig1(src,jh_profile,jh_metric), jh_fig2(src,jh_profile), jh_fig3(src,jh_profile,jh_cost_mode),
            kay_fig1(src,kay_period), kay_fig2(src), kay_fig3(src,kay_segment), he_fig1(c,he_profile), he_fig2(c), he_fig3(c,he_profile))

# The app is defined but not started during notebook execution. Run app.run(debug=True) only for presentation.
print('Combined storyboard created with 3 navigable sections and 9 integrated Plotly charts.')
print('Order: Jhgraphs → Kaydengraphs → Hegraphs.')

if __name__ == '__main__':
    app.run(debug=True)
