import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import init_notebook_mode
import seaborn as sns
import datetime as dt
import warnings
%matplotlib inline
import matplotlib.pyplot as plt
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
init_notebook_mode(connected=True)

df = pd.read_csv("/Users/ecemolgun/Desktop/bank_marketing.csv", sep=";")

df.rename(columns={'y':'Deposit'},inplace=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

###### Yaş dağılım grafiği ######

plt.figure(figsize=(10,5))
sns.histplot(x=df['age'],color='Cyan',label='Age')
plt.axvline(x=df['age'].mean(),color='k',linestyle ="--",label='Ortalama Yaş: {}'.format(round(df['age'].mean(),2)))
plt.legend()

plt.title('Yaş Dağılımı')
plt.show()

###### Meslek dağılım grafiği ######

fig=px.bar(df.job.value_counts().reset_index().rename(columns={'index':'Job','job':'Count'}),x='Job',y='Count',color='Job',text='Count',template='simple_white')
fig.update_traces(marker=dict(line=dict(color='#000000', width=1.2)))
fig.update_layout(title_x=0.5,title_text="<b>Meslek Dağılımı" ,font_family="Calibri",title_font_family="Calibri")

###### Medeni Hal dağılım grafiği ######

fig=px.pie(df.marital.value_counts().reset_index().rename(columns={'index':'Marital-Status','marital':'Count'}),names='Marital-Status',values='Count',hole=0.5,template='plotly_white',color_discrete_sequence=['HotPink','LightSeaGreen','SlateBlue'])
fig.update_traces(marker=dict(line=dict(color='#000000', width=1.4)))
fig.update_layout(title_x=0.5,showlegend=True,legend_title_text='<b>Medeni Hal Durumu')
fig.update_traces(textposition='outside', textinfo='percent+label')
fig.update_layout(title_x=0.5,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1))
fig.show()

###### Eğitim durumu dağılım grafiği ######

fig=px.pie(df.education.value_counts().reset_index().rename(columns={'index':'Education','education':'Count'}),names='Education',values='Count',hole=0.5,template='plotly_white')
fig.update_traces(marker=dict(line=dict(color='#000000', width=1.4)))
fig.update_layout(title_x=0.5,showlegend=True,legend_title_text='<b>Eğitim Durumu')
fig.update_traces(textposition='outside', textinfo='percent+label')
fig.update_layout(title_x=0.5,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1))
fig.show()

###### Kredi - Temerrüt analiz grafiği ######

fig=go.Figure()
from plotly.subplots import make_subplots
fig=make_subplots(rows=1,cols=3)
fig = make_subplots(rows=1, cols=3, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],subplot_titles=('Kredi Temerrüt Durumu', 'Konut Kredisi Durumu','Bireysel Kredi Durumu'))
fig.add_trace(go.Pie(values=df.default.value_counts().reset_index().rename(columns={'index':'Default','default':'Count'})['Count'],labels=df.default.value_counts().reset_index().rename(columns={'index':'Default','default':'Count'})['Default'],hole=0.7,marker_colors=['MediumPurple','YellowGreen'],name='Kredi Temerrüdü',showlegend=False),row=1,col=1)
fig.add_trace(go.Pie(values=df.housing.value_counts().reset_index().rename(columns={'index':'Housing','housing':'Count'})['Count'],labels=df.housing.value_counts().reset_index().rename(columns={'index':'Housing','housing':'Count'})['Housing'],hole=0.7,marker_colors=['MediumPurple','YellowGreen'],name='Konut Kredisi',showlegend=False),row=1,col=2)
fig.add_trace(go.Pie(values=df.loan.value_counts().reset_index().rename(columns={'index':'Loan','loan':'Count'})['Count'],labels=df.loan.value_counts().reset_index().rename(columns={'index':'Loan','loan':'Count'})['Loan'],hole=0.7,marker_colors=['MediumPurple','YellowGreen'],name='Bireysel Kredi',showlegend=True),row=1,col=3)

fig.update_layout(title_x=0.5,template='simple_white',showlegend=True,legend_title_text=" ",title_text='<b style="color:black; font-size:100%;">Kredi ve Temerrüt Analizi',font_family="Calibri",title_font_family="Calibri")
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))

###### Vadeli mevduat analiz grafiği ######

fig=px.pie(df.groupby(['deposit'],as_index=False)['age'].count().rename(columns={'age':'Count'}),names='deposit',values='Count',template='ggplot2',hole=0.8)
fig.update_traces(marker=dict(line=dict(color='#000000', width=1.4)))
fig.update_layout(title_x=0.5,showlegend=True,legend_title_text='<b>Deposit')
fig.update_traces(textposition='outside', textinfo='percent+label')
fig.update_layout(title_x=0.5,title_text='<b>Vadeli Mevduat Analizi',font_family="Arial",title_font_family="Arial")
fig.update_layout(title_x=0.5,legend=dict(orientation='v',yanchor='middle',y=1.02,xanchor='right',x=1))
fig.show()

###### Vadeli mevduat vs Bakiye analiz grafiği ######

df['Balance']=df['balance'].apply(lambda x: 'Ortalama Üstü' if x>=df['balance'].mean() else 'Ortalama Altı')
a=df.groupby(['Balance','deposit'],as_index=False)['age'].count().rename(columns={'age':'Count'})
a['percent']=round(a['Count']*100/a.groupby('Balance')['Count'].transform('sum'),1)
a['percent']=a['percent'].apply(lambda x: '{}%'.format(x))
fig=px.bar(a,x='Balance',y='Count',text='percent',color='deposit',barmode='group',template='simple_white',color_discrete_sequence=['MediumPurple','YellowGreen'])
fig.update_layout(legend_title_text='<b>Mevduat:</b>',title_text='<b style="font-family: Times New Roman; font-size:1.2vw">Vadeli Mevduata Göre Bakiye Durumu</b><br><b style="font-family: Times New Roman; font-size:1.01vw">Ortalama Bakiye: 1528.53</b>',font_family="Times New Roman",title_font_family="Times New Roman")
fig.update_layout(legend=dict(orientation='v',yanchor='middle',y=1.02,xanchor='right',x=1))
fig.update_traces(marker=dict(line=dict(color='#000000', width=1.2)))
fig.update_layout(title_x=0.08,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1))
fig.update_traces(textposition='outside')
fig.show()

###### Meslek vs Bakiye analiz grafiği ######

a=df.groupby(['job'],as_index=False)['balance'].mean()
a['balance']=round(a['balance'],1)
fig=px.bar(a.sort_values(by='balance',ascending=False),x='job',y='balance',text='balance',color='job',template='ggplot2')
fig.update_traces(marker=dict(line=dict(color='#000000', width=1.2)))
fig.update_layout(title_x=0.5,title_text='<b>Meslek Türüne Göre Müşterilerin Bakiye Durumu',legend_title_text='Job Type',font_family="Times New Roman",title_font_family="Times New Roman")

###### Mevduat vs Yaş analiz grafiği ######

fig = px.histogram(df, x='age', color='deposit', barmode='group',
                   template='simple_white', color_discrete_sequence=['DeepSkyBlue', 'LightCoral'],
                   title='<b>Vadeli Mevduat Durumuna Göre Yaş Dağılımı')

fig.update_layout(
    title_x=0.5,
    font_family="Times New Roman",
    legend_title_text="<b>Term Deposit",
    xaxis=dict(title="Age"),
    yaxis=dict(title="Count")
)

fig.show()
