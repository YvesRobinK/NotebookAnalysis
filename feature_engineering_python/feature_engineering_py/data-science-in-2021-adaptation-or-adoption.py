#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""

Kaggle Notebook : Data Science in 2021 : Adaptation or Adoption?
URL : https://www.kaggle.com/shivamb/data-science-in-2021-adaptation-or-adoption
Updated : Nov 28th, 2021

"""

## import external libraries 

from IPython.core.display import HTML
import plotly.graph_objects as go
import pandas as pd 
import numpy as np 
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

### Utility Functions ###

## Measure counts (example : ML Tools) by a group (example : country)
def counts_by_group(df, unique_values, col, identifier_col, max_X):
    counts_by_group = {}
    for c in unique_values:
        counts_by_group[c] = {}
        for i in range(1,max_X):
            n = df[df[col] == c][identifier_col.replace("X", str(i))].value_counts()
            counts_by_group[c].update(n.to_dict())
    return counts_by_group

## Measure diversity
def get_diversity(r, cols):
    values = [r[c] for c in cols]
    return np.std(values)

## Additional feature engineering
def add_features(df):
    df["man_to_women_ratio"] = df["Q2_Man"] / df["Q2_Woman"]
    
    cols = ['Q4_Master’s degree', 'Q4_Doctoral degree', 'Q4_Bachelor’s degree', 'Q4_Professional doctorate']    
    df['education_degree_diversity'] = df.apply(lambda x : get_diversity(x, cols)  , axis = 1)
    df['higher_education'] = df['Q4_Master’s degree'] + df['Q4_Doctoral degree'] + df['Q4_Professional doctorate']

    cols = ['Q5_Data Scientist', 'Q5_Program/Project Manager', 'Q5_Research Scientist', 'Q5_Data Analyst', 'Q5_Business Analyst', 'Q5_Software Engineer', 'Q5_Machine Learning Engineer',
            'Q5_Product Manager', 'Q5_Statistician', 'Q5_Data Engineer', 'Q5_Developer Relations/Advocacy', 'Q5_DBA/Database Engineer']
    df['roles_diversity'] = df.apply(lambda x : get_diversity(x, cols)  , axis = 1)
    df['ds_roles'] = df['Q5_Data Scientist'] + df['Q5_Research Scientist'] + df['Q5_Machine Learning Engineer'] + df['Q5_Statistician'] 

    cols = ['Q6_20+ years', 'Q6_10-20 years', 'Q6_1-3 years', 'Q6_5-10 years', 'Q6_< 1 years', 'Q6_3-5 years', 'Q6_I have never written code']
    df['coders_diversity'] = df.apply(lambda x : get_diversity(x, cols)  , axis = 1)

    cols = ['Q8_Python', 'Q8_R', 'Q8_Julia', 'Q8_MATLAB']
    df['language_diversity'] = df.apply(lambda x : get_diversity(x, cols)  , axis = 1)

    cols = ['Q15_Under 1 year', 'Q15_2-3 years', 'Q15_3-4 years', 'Q15_4-5 years', 'Q15_5-10 years', 'Q15_10-20 years', 'Q15_1-2 years', 'Q15_20 or more years']
    df['mlexperience_diversity'] = df.apply(lambda x : get_diversity(x, cols)  , axis = 1)
        
    cols = ['Q21_1000-9,999 employees', 'Q21_50-249 employees', 'Q21_250-999 employees', 'Q21_10,000 or more employees', 'Q21_0-49 employees']
    df['org_scale_diversity'] = df.apply(lambda x : get_diversity(x, cols)  , axis = 1)
    return df

## Clean the country name 
def clean_country(x):
    if x.startswith("United S"):
        return "USA"
    elif x.startswith("United K"):
        return "UK"
    elif x.startswith("Hong"):
        return "HK"
    elif x.startswith("United A"):
        return "UAE"
    elif x.startswith("Iran"):
        return "Iran"
    elif x.startswith("The Netherlands"):
        return "Netherlands"
    return x   


def create_aggregated_frame(df, col, colname, unique_values):
    agg_df = df.groupby(col).agg({"Q23_l" : "mean", "Q26_l" : "mean", "Q1" : "count", "Q22_l" : "mean"}).reset_index()
    agg_df = agg_df.rename(columns = {col : colname, 'Q26_l' : "ML Spendings", "Q23_l" : "ML Levels", "Q1" : "Count", "Q22_l" : "DataScience TeamSize" })

    ml_tools = counts_by_group(df, unique_values, col = col, identifier_col = 'Q38_A_Part_X', max_X = 11)
    automl_tools = counts_by_group(df, unique_values, col = col, identifier_col = 'Q37_A_Part_X', max_X = 7)
    autofe_tools = counts_by_group(df, unique_values, col = col, identifier_col = 'Q36_A_Part_X', max_X = 7)
    ml_products = counts_by_group(df, unique_values, col = col, identifier_col = 'Q31_A_Part_X', max_X = 10)
    cloud_tools = counts_by_group(df, unique_values, col = col, identifier_col = 'Q30_A_Part_X', max_X = 7)
    cloud_adoption1 = counts_by_group(df, unique_values, col = col, identifier_col = 'Q29_A_Part_X', max_X = 4)
    cloud_adoption2 = counts_by_group(df, unique_values, col = col, identifier_col = 'Q27_A_Part_X', max_X = 4)
    BI_tools = counts_by_group(df, unique_values, col = col, identifier_col = 'Q34_A_Part_X', max_X = 16)  
    BigData_products = counts_by_group(df, unique_values, col = col, identifier_col = 'Q32_A_Part_X', max_X = 20)
    
    maps = {
        'ml_tools' : ml_tools,
        'automl_tools' : automl_tools,
        'autofe_tools' : autofe_tools,
        'ml_products' : ml_products,
        'cloud_tools' : cloud_tools,
        'cloud_adoption1' : cloud_adoption1,
        'cloud_adoption2' : cloud_adoption2,
        'bi_tools' : BI_tools,
        'bigdata_tools' : BigData_products,
    }
    for c, doc in maps.items():
        rows = []
        for p in doc.items():
            rows.append( [ p[0],  sum(p[1].values()) ] )
        c1 = pd.DataFrame(rows, columns = [colname, c])
        agg_df = agg_df.merge(c1, on=colname, how = 'left')
        agg_df[c] = agg_df.apply(lambda x : x[c] / x['Count'], axis = 1)
    
    rows = []
    cols = ['Q2', 'Q4', 'Q5', 'Q6', 'Q8', 'Q11', 'Q13', 'Q15', 'Q21', 'Q22', 'Q25']
    for c in agg_df[colname].unique():
        features = {colname : c}
        tempdf = df[df[col] == c]        
        for cx in cols:
            for k, v in tempdf[cx].value_counts().to_dict().items():
                features[cx + "_" + str(k)] = v / len(tempdf)
        rows.append(features)    
    counts_df = pd.DataFrame(rows)
    agg_df = agg_df.merge(counts_df, on=colname, how = 'left')
    
    agg_df = add_features(agg_df)

    return agg_df

## load raw survey dataset
path = "../input/kaggle-survey-2021/kaggle_survey_2021_responses.csv"
raw_df = pd.read_csv(path, low_memory = False)
df = raw_df[1:]

## add custom css fonts 
styling = """@import url('https://fonts.googleapis.com/css2?family=Mochiy+Pop+P+One&display=swap');@font-face { font-family: 'Mochiy Pop P One', sans-serif; }"""
HTML("<style>"+styling+"</style>")


# <div align="center"><p style="font-family: 'Mochiy Pop P One';font-size:32px;color:#f57369">Data Science in 2021 : Adaptation or Adoption?</p></div>
# 
# <font size=4><u><mark>Understanding the Current State of AI / Data Science Adoption<mark></u></font>
# 
# The *COVID-19* pandemic transformed the way organizations work, collaborate, and operate. In just a few months, the crisis brought about years of change in the ways different companies, sectors, and regions do business. In many industries, the changes were damaging, while in others, they were promising and beneficial. The healthcare/pharma sector, for example, became aggressive, supply chains witnessed major disruptions, and the e-commerce sector boomed up. On the other hand, the local business, travel, and tourism industries were critically affected.
# 
# From a technology perspective, organizations felt a need for more stabilization and more innovation in which the *adoption of* / *adapting towards* Data Science and Artificial Intelligence played an important role. To stay competitive, ambitious, and operate just like normal times, organizations were required to make strategic shifts, alter their long-term views and develop a data-first / technology-first mindset. Usually, when organizations need to adjust their technology landscape, they follow one of the following two types of approaches:
# 
# <font size="3" face='Mochiy Pop P One'>"The Adapt Approach" or "The Adopt Approach"</font>
# 
# - The Adapt approach is a conventional idea where companies need to continuously modify their core technology (including the data and the models) to meet the business objectives. This approach only offers a short-term solution which is also inefficient, expensive, and innovation-killing.
# 
# - The Adopt Approach enables the organizations to incorporate long-term solutions with best practices and design principles. Companies need to invest in building specialized teams and start adopting a data-first / technology-first culture. 
# 
# The more incorporation of the "adoption approach" can assist the organizations to accelerate their innovation. Several studies conducted in the pandemic year suggest that digital adoption has taken a quantum leap and the Covid-19 crisis has accelerated the adoption of analytics and AI, This momentum is likely to continue in the 2020s. 
# 
# But a vital question that needs to be answered is: <mark>Are the organizations and governments ready for a data-first culture and AI Adoption?</mark> 
# 
# As part of our regular job as customer-facing data scientists, we interact with several organizations - across different countries and multiple industries. This involves companies of varying sizes, different revenues, and different data maturity levels. Some are at a very early stage, some are highly advanced. A common observation is that "many organizations are still at very early stages in their AI adoption".  If we have to categorize them in different groups, it can be analogous to  stages of plant growth : 
# 
# <br>
# <div align='center'><img src='https://i.imgur.com/bdWhf8o.png' width=600>
# <br>
# This scale ranks the entities (organizations, industries, or countries) at different levels of data science and artificial intelligence adoption. 
# <br>
# 
# <table align='center' style='font-size:16px'><tr><th> </th>
#     <th> Seedling </th>
#     <th> Vegetative </th>
#     <th> Budding </th>
#     <th> Flowering </th>
#     <th> Ripening </th>
#  </tr>
# <tr align="center">
#     <td><b>Stages</b></td>
#     <td>Roots Development</td>
#     <td>Stems Growth</td>
#     <td>Plants Formation</td>
#     <td>Flowers/Fruits</td>
#     <td>More Nutrition</td>
# </tr>
# <tr align='center'>
#     <td><b>Sector / Country</b></td>
#     <td>Lack of good<br>of Data</td>
#     <td>Building a data<br> culuture</td>
#     <td>Supporting the Startups<br> with Talent, Fundings</td>
#     <td>Major Investments<br>in AI Initiatives</td>
#     <td>Qualified Professionals<br>Better Jobs<br>Education Degrees</td>
# </tr>
# <tr align='center'>
#     <td><b>Organizations</b></td>
#     <td>Establishing the need<br>of Data / AI</td>
#     <td>Focus on data <br>procurement</td>
#     <td>Minimal analysis,<br>and basic modelling</td>
#     <td>Use of data, <br>predictive modelling,<br>and machine learning</td>
#     <td>Models in Production <br> with Greater ROI</td>
# </tr>
# </table>
# </div>
# 
# 2021 Kaggle Machine Learning & Data Science Survey is the most comprehensive dataset available on the state of ML and data science. By blending this data with several external datasets, we have analyzed the state of data science and artificial intelligence adoption in 2021. We utilized some of the questions directly related to this topic such as: 
# 
# - Q23: Does your current employer incorporate machine learning methods into their business?
# - Q22: How many individuals are responsible for data science workloads at your place of business?
# - Q26: How much money have you or your team spent on machine learning/data science?  
# 
# and many indirect questions that reflect the state of AI/data science: 
# 
# - Q16-19: Do you use ML, NLP, Computer Vision, etc.?  
# - Q33: Which big data products (relational database, data warehouse, data lake, or similar) do you use?
# - Q38: Do you use tools to help manage machine learning experiments?
# 
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">Methodology</p>
# 
# We aggregated the survey dataset by regions and industries. We then measured the mean, count, diversity among the following: education qualifications, job roles, data science techniques, cloud usage, machine learning platforms usage, data team size, incorporation of machine learning by employers, coding and machine learning experiences, big data tools, auto ml tools, etc. Using the aggregated information, we defined a unified data science adoption index that reflects the current state of maturity levels. We used the simple mean aggregation to derive the adoption index. We then visualized the adoption index, current machine learning incorporation levels by various internal and external data attributes such as - Global innovation index, competitive data science rankings, Industry revenues, job postings by countries, etc. This analysis helped us to identify which groups are leading while which ones are lagging. 
# 
# **Validation:** To make sure that our methodology, insights, and findings are aligned with industry trends, we manually cross-checked a number of online references. This includes online reports, articles, and surveys published by organizations such as IBM, Mckinsey, Dataiku, Stanford, World Intellectual Property Organization, Element AI, etc. The references are added in the last section.  
# 
# **Visual Theme:** We developed a common visualization theme of the plots - HotSpot charts (bubble charts with hotness as the color theme), where something higher in quantity is colored as hot (red), and something lesser is colored as cold (blue). We selected the seed-plant growth stages analogy to reflect different stages of adoption levels. 
# 
# **External Datasets Used:**
# - **[Kaggle Analytics 2021 Survey Data](https://www.kaggle.com/c/kaggle-survey-2021):** We used the entire data of kaggle survey respondents but removed the 'Students', 'Currently Not Employed', from industries - 'Other' and 'Academics/Education', from countries - 'Other', 'I do not wish to disclose my location. For some parts of the analysis, we also removed responses 'I do not know' where respondents were asked about their current employer AI / ML maturity. 
# 
# - **[Kaggle Analytics 2019 Survey](https://www.kaggle.com/c/kaggle-survey-2019):** We selected the Kaggle dataset from the year 2019 to compare a few metrics from pre-pandemic year to post-pandemic year. Most of the questions from this data were similar to the 2021 dataset so it was easier to compare. 
# 
# - **[Data Science Jobs, Revenues by Sectors, Countries](https://www.kaggle.com/shivamb/glassdoor-jobs-data):** This data contains data science/machine learning / AI-related jobs counts from Glassdoor and Linkedin. The Glassdoor data contains information about the companies such as - Revenues, Sectors, and Linkedin Data contains the count of jobs by country. We developed a scraper to obtain Glassdoor jobs data, while we manually obtained the counts of job counts by country from Linkedin. 
# 
# - **[Tech Fundings Since 2020](https://www.kaggle.com/shivamb/tech-company-fundings-2020-onwards):** We used the data of fundings in the technology sector by countries, by sectors, and time. The data is from 2020 and was obtained from GrowthList. The data also contains a specific subset of artificial intelligence fundings.  
# 
# - **[Competitive Data Science by Countries](https://www.kaggle.com/hdsk38/comp-top-1000-data):** This is the dataset of top 1000 users in the four types of Kaggle rankings (i.e. Competition, Datasets, Notebooks, and Discussions) as of October 2021. The dataset was prepared by [tropicbird](https://www.kaggle.com/hdsk38) obtaining the kaggle rankings and identifying the locations of the users. 
# 
# - **[Global Innovation Index 2021](https://www.wipo.int/edocs/pubdocs/en/wipo_pub_gii_2021.pdf):** The Global innovation index (GII) is computed by World Intellectual Property Organization. The GII's overall formula for measuring an economy's innovative capacity and output provides clarity for decision-makers in government, business, and elsewhere as they look forward to creating policies that enable their people to invent and create more efficiently.
# 
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">1. How does AI Adoption looks like in 2021 ?</p>

# In[2]:


def add_annotations(fig, text_cfg):
    for i,text in enumerate(text_cfg['texts']):
        fig.add_annotation(text=text, xref="paper", yref="paper", x=text_cfg['text_xpos'][i], y=0.78, showarrow=False, 
                           font = dict(color = '#f77f77', size=14) )
    
    for i,img in enumerate(text_cfg['imgs']):
        fig.add_layout_image(dict(source=img, x=text_cfg['img_xpos'][i], y=1))
        fig.update_layout_images(dict(xref="paper", yref="paper", sizex=0.1, sizey=0.2, xanchor="center", yanchor="top"))
    
    
    for i, a in enumerate(text_cfg['annos']):
        if 'show_arr' not in a:
            a['show_arr'] = True
        
        fig.add_annotation(x=a['x'], y=a['y'], xref="x", yref="y",
            text=a['text'], showarrow=a['show_arr'],
            font=dict(family="Verdana", size=12, color="#ffffff"),
            align="center", arrowhead=1, arrowsize=1, arrowwidth=1,
            arrowcolor="#222", ax=a['ax'], ay=a['ay'], bordercolor="#c7c7c7", borderwidth=1,
            borderpad=4, bgcolor="#f77f77", opacity=0.7)
    return fig 

temp_df = raw_df[1:]

temp_df = temp_df[~temp_df['Q5'].isin(['Student', 'Currently not employed'])]
temp_df = temp_df[~temp_df['Q3'].isin(['Other', 'I do not wish to disclose my location'])]
temp_df = temp_df[~temp_df['Q20'].isin(['Academics/Education', 'Other'])]
temp_df = temp_df[~temp_df['Q23'].isin(['I do not know'])]
temp_df = temp_df[~temp_df['Q23'].isna()]
temp_df

mapp = {
    'No (we do not use ML methods)' : "0. Not Started<br><sup>(No ML)</sup>" , 
    'We are exploring ML methods (and may one day put a model into production)' : "1. Exploration<br><sup>Only Exploring ML</sup>",
    'We use ML methods for generating insights (but do not put working models into production)' : "2. Beginner Stage<br><sup>Use ML only for Insights</sup>", 
    'We recently started using ML methods (i.e., models in production for less than 2 years)' : "3. Intermediate Stage<br><sup>Recently Started Using ML</sup>",
    'We have well established ML methods (i.e., models in production for more than 2 years)' : "4. Advance Stage<br><sup>Well Established ML</sup>",
}

temp_df['clean_text'] = temp_df['Q23'].apply(lambda x : mapp[x])

agg_df = temp_df['clean_text'].value_counts().to_frame().reset_index()
agg_df = agg_df.sort_values("index")

total = sum(agg_df['clean_text'])
agg_df['percent'] = agg_df['clean_text'].apply(lambda x : round(100*x/ total))


xx = list(agg_df['index'])
pp = list(agg_df['percent'])

xx = [_.split(".")[1].strip()  for i, _ in enumerate(xx)]

yy = [1]*len(xx)
sizes = [_*3.5 for _ in list(agg_df['percent'])]
sizes = [10, 9, 3, 4, 5]
sizes = [_*10 for _ in sizes]
xx_labels = ["<br>"+xx[0], "<br>"+xx[1], "", "", ""]
texts = [str(round(_))  +"%" + str(xx_labels[i]) for i,_ in enumerate(list(agg_df['percent']))]
cols_text = ['#787b82', '#787b82', '#fff', '#fff', '#fff']
trace1 = go.Scatter(x = xx, y = yy, mode='markers+text', 
        text = texts ,textfont=dict(family="Verdana", size=12, color=cols_text), 
        hovertemplate = "%{text}<extra></extra>",
        name="", marker=dict(
        color=[1, 2, 3, 4, 5], 
        size=sizes,
        showscale=False,
        opacity=0.85,
        colorscale = 'reds')
       )

layout = go.Layout(width=1050, height=450, 
                   plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend = False,
                   title = {'text' : "About 24% Organizations have models in production for 2+ years<br> <sup>Q23: Does your current employer incorporate machine learning methods into their business?</sup>", 'x':0.5, 'xanchor': 'center'}, 
                   font = {"color" : '#f27866'})
fig = go.Figure(data=[trace1], layout=layout)
fig.add_vline(x=1.4, line_width=1, line_dash="dash", line_color="#c9c4c3")
fig.add_vline(x=3.5, line_width=1, line_dash="dash", line_color="#c9c4c3")

text_cfg = {
    "texts" : ["Seedling", "Vegetative", "Budding", "Flowering", "Ripening"],
    "text_xpos" : [0.086, 0.28, 0.51, 0.75, 0.94],
    "img_xpos" : [0.12, 0.34, 0.51, 0.71, 0.905],
    "imgs" : [ "https://i.imgur.com/9rs2DVF.png", "https://i.imgur.com/WIusNLF.png", "https://i.imgur.com/j4lAWaU.png", "https://i.imgur.com/VuR8AKW.png", "https://i.imgur.com/dAXbEY2.png"],
    'annos' : [], 
} 
fig = add_annotations(fig, text_cfg)

insight1 = "14% Marketing, 12% Pharma<br>12% Hospitality<br>16% Organizations with <300 Staff"
fig.add_annotation(x=2, y=1.0, text=insight1, showarrow=False, yshift=-50, font=dict(family="Verdana", size=10, color="#787a79"))

insight1 = "Israel (21%) leads this group<br>USA (18%), Netherlands (18%), <br> UK (17%) and Sweeden (17%)"
fig.add_annotation(x=4, y=1, text=insight1, showarrow=False, yshift=-48, font=dict(family="Verdana", size=10, color="#787a79"))

insight1 = "45% Organizations have still not started <br> ML and Data Science adoption !!"
fig.add_annotation(x=0.5, y=1, text=insight1, showarrow=False, yshift=-68, font=dict(family="Verdana", size=10, color="#787a79"))

insight1 = "35% Retail and Online Sales<br>19% Insurance<br>18% Broadcasting"
fig.add_annotation(x=3, y=1, text=insight1, showarrow=False, yshift=-48, font=dict(family="Verdana", size=10, color="#787a79"))

fig.update_yaxes(visible=False)
fig.update_xaxes(showline=False, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', 
                 tickfont=dict(size=14, family='Verdana', color='#f26d63'), 
                 title="", 
                showgrid=False)
fig.show()

# Uncomment Following for Insights Generation
# col = 'Q3'
# idf = temp_df[temp_df['Q23'] == 'No (we do not use ML methods)'][col].value_counts().to_frame().reset_index()
# tots = temp_df[col].value_counts().to_dict()
# idf['percent'] = idf.apply(lambda x : 100*x[col] / tots[x['index']], axis = 1)
# idf.sort_values("percent", ascending = False)


# <br><div align='center'><span class='alert-info' style='padding:20px;color:#222'>&#x25cf; &nbsp;&nbsp; Israel with 21% has the highest proportion of models in production for more than 2 years. </span></div><br>
# 
# The data shows that about 24% have well-established machine learning methods. Some characteristics of people from this group are:  
# - more in technology and engineering focussed job roles (36% Data Scientists, 12% ML Engineers) than the managerial jobs (4% Managers)
# - more coding experiences (5-10 Years: 23%, 3-5 Years: 20%)
# - larger data science teams (20+ team size: 55%)
# 
# **Israel** (with 21% of organizations having models in production for 2 years and more), is no stranger to innovation and is widely acknowledged as a startup nation. Studies suggest that Israel has exceptional levels of time and effort on building a ripe environment in which new tech companies can flourish. Israel Government has played an important role, they even stated in 2019, "We are now making a national plan to make Israel, within five years, one of the top five leading countries in the world in artificial intelligence technology,", It seems that they are right on the track in 2021 and going ahead. 
# 
# <br><div align='center'><span class='alert-danger' style='padding:20px;color:#222'>&#x25cf; &nbsp;&nbsp; <b>45%</b> Organizations have not started exploring data science, machine learning or are in the very early stages of exploring</span></div><br>
# 
# 45% is a big number given that it is 2021. Among this group, 38% are from middle-east countries: <mark>'Saudi Arabia', and 'Iraq'</mark>. Saudi Arabia Government have started making major efforts in improving their Data Science adoption, they recently launched the development of [data science consortium in next decade](https://www.c4isrnet.com/artificial-intelligence/2021/01/15/saudi-arabia-makes-artificial-intelligence-a-cornerstone-of-its-2030-vision/). So we should expect to see this number improve in the coming years. <mark>25% from this group are from Belarus</mark> which is a surprise addition in this group, as most of the other European countries are relatively mature in this field. In general, this 45% has: 
# 
# - respondents with either **no formal education past high school (about 14%)** or they choose not to answer this question (10%)
# - 1 in every 5 individuals (20%) never written code
# 
# The lower adoption of AI and data science can be due to multiple factors. An AI startup called DLabs.AI from Europe compiled the biggest challenges cited during AI / ML adoption. These included: 
# 
# 1. Skillsets, Data, Use-Cases
# 2. Understand the Need for AI 
# 3. Failing to Explain how AI solutions work 
# 4. Lack of Collaboration between AI teams 
# 5. Regulations, Management Issues, Complexity
# 
# <br><div align='center'><span class='alert-info' style='padding:20px;color:#222'>&#x25cf; &nbsp;&nbsp; 35% from Retail, Online Services, and Insurance sectors have well established data science </span></div><br>
# 
# - These three sectors especially 'Insurance' have shown a significant improvement in AI adoption over the last couple of years. 
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">2. The Industry Outlook of Data Science Adoption</p>

# In[3]:


industry_totals = temp_df['Q20'].value_counts().to_dict()
agg_df = temp_df.groupby(['Q20', 'clean_text']).agg({"Q2" : "count"}).reset_index()
agg_df['Q2_percent'] = agg_df.apply(lambda x : x['Q2'] / industry_totals[x['Q20']], axis = 1)
agg_df

trace1 = go.Scatter(
    x=agg_df['Q20'].apply(lambda x : x.split("/")[0]),
    y=agg_df['clean_text'].apply(lambda x : x.split(".")[1]),
    mode='markers', 
    hovertemplate=
        "Industry: %{x}<br>" +
        "AI Adoption Stage: %{y}<br>" +
        "Percentage: %{marker.size:,}" +
        "<extra></extra>",
    marker=dict(
        color=agg_df['Q2_percent'],
        size=agg_df['Q2_percent']*100,
        showscale=True,
        colorbar=dict(title='Percent'),
        opacity=0.7,
        colorscale = 'RdBu_r'
        )
)

layout = go.Layout(width=1050, height=600, plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend = False, 
                   title = {'text' : "Government Sector has least Models in Production<br><sup>Questions Data: Industry (Q20) and ML Adoption Levels (Q23)</sup><br><sup>Size,Color: Percentage of Respondents</sup>", 
                            'x':0.5, 'xanchor': 'center'}, 
                   font = {"color" : '#f27866'})
fig = go.Figure(data = [trace1], layout = layout)
fig.update_xaxes(showline=False, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='#f26d63'), 
                 title="", title_font=dict(size=14, family='Verdana', color='#f57369'), showgrid=False, tickangle=325)
fig.update_yaxes(showline=False, linewidth=1, linecolor='#000', gridcolor='#fff', tickfont=dict(size=14, family='Verdana', color='#699ff5'), 
                 title="", title_font=dict(size=14, family='Verdana', color='#f57369'), showgrid=False)

fig.show()

fig.add_shape(type="rect", x0=2.5, y0=-0.5, x1=11.7, y1=0.5, line=dict(color="#cccac4", width=2, dash='dash' ))
fig.add_shape(type="rect", x0=-0.5, y0=3.5, x1=2.5, y1=4.5, line=dict(color="#cccac4", width=2, dash='dash' ))
fig.add_shape(type="rect", x0=11.5, y0=3.5, x1=13.5, y1=4.5, line=dict(color="#cccac4", width=2, dash='dash' ))
fig.add_shape(type="rect", x0=5.3, y0=-0.7, x1=6.7, y1=4.5, line=dict(color="#cccac4", width=2, dash='dash' ))

insight1 = "Top Job Role: Software Engineering, <br> Data Scientist (~20%)"
fig.add_annotation(x=1, y=4.2, text=insight1, showarrow=False, yshift=48, font=dict(family="Verdana", size=12, color="#5e5656"))
insight1 = "Insurance / Risk Assessment <br> showing high <br> AI/ML Maturity Levels "
fig.add_annotation(x=6, y=4.2, text=insight1, showarrow=False, yshift=48, font=dict(family="Verdana", size=12, color="#5e5656"))
insight1 = "Top Job Role: Others (28.4%)"
fig.add_annotation(x=4.4, y=0, text=insight1, showarrow=False, yshift=48, font=dict(family="Verdana", size=12, color="#5e5656"))
insight1 = "Top Job Role: Data Scientists, ML Engineers (~50%)"
fig.add_annotation(x=13, y=4, text=insight1, showarrow=False, yshift=48, font=dict(family="Verdana", size=12, color="#5e5656"))

fig.update_xaxes(showline=False, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='#f26d63'), 
                 title="", title_font=dict(size=14, family='Verdana', color='#f57369'), showgrid=False)
fig.update_yaxes(showline=False, linewidth=1, linecolor='#000', gridcolor='#fff', tickfont=dict(size=14, family='Verdana', color='#699ff5'), 
                 title="", title_font=dict(size=14, family='Verdana', color='#f57369'), showgrid=False)
fig.show()


## Uncomment following for insights 
# col = 'Q21'
# idf = temp_df[temp_df['Q23'] == 'We have well established ML methods (i.e., models in production for more than 2 years)']
# # idf = idf[idf['Q20'].isin(['Government/Public Service', 'Non-profit/Service'])] 
# idf = idf[idf['Q20'].isin([ 'Online Service/Internet-based Services'])]
# # idf = idf[idf['Q20'].isin([ 'Accounting/Finance'])]
# # idf = idf[idf['Q20'].isin([ 'Insurance/Risk Assessment'])]

# idf = idf[col].value_counts().to_frame().reset_index()
# tot = np.sum(idf[col])
# idf['percent'] = idf[col].apply(lambda x : round(100*x / tot, 1))
# idf.sort_values("percent", ascending = False)

# itotals = raw_df['Q20'].value_counts().to_dict()
# itotals
# for r in temp_df['Q20'].unique():
#     x1 = temp_df
#     x1 = x1[x1['Q20'] == r]

#     for p in [c for c in temp_df.columns if 'Q17' in c]:
#         d =  x1[p].value_counts().to_dict() 
#         m = 'Transformer Networks (BERT, gpt-3, etc)'
#         if m in d:
#             print (r, 100*d[m] / itotals[r])


# <br>
# 
# <div align='center'><span class='alert-danger' style='padding:20px;color:#222'>&#x25cf; &nbsp;&nbsp; 2 out of 5 respondents from <b>Governments or Public Services</b> shared that they have not even started any machine learning or data science </span></div>
# 
# <br>
# 
# A key reason for the lower AI adoption among governments is the established set of practices and processes. In governments, there can be less encouragement for employees to take risks and innovate. In private sectors, employers tend to put a strong focus on experimentation, innovation, and growth. These numbers clearly show a big scope of improvements in the AI and data science vertical. Since all the governments have access to huge amounts of data, they have a very high potential of using data science in a meaningful way. Among this group: 
# 
# - Approx 50% of respondents had <mark>less than or equal to 3 years of coding experience</mark>  
# - 40% shared that there is no data science team, and the other 40% shared max 1 or 2 people are handling the data science workloads
# - About <b>85%</b> has less than 2 years of machine learning experience; among then 27% had never used machine learning 
# 
# 
# Gartner has identified that the most common use case of data science and AI among Governments is the chatbots to assist customer communications.   
# 
# <br>
# <div align='center'><img width='600' src='https://i.imgur.com/hHzqRVC.png'></div>
# <br>
# 
# <mark>But are these so-called productized/deployed chatbots Mature?</mark> Are they good enough to serve the common public? **Maybe not!** 
# 
# A funny incident occurred last month (October 2021) in Singapore, where a chatbot named Jamie of Singapore's Ministry of Health responded with the highly misaligned reply to a Covid-19 Question. A person asked about covid-19 precautions but the generated response was about safe sex practices. It became a comedy meme on social media and the government was criticized, Ultimately they had to shut down Jamie. [Source - Channel News Asia - Singapore MOH ChatBot Funny Response](https://www.channelnewsasia.com/singapore/moh-ask-jamie-covid-19-query-social-media-2222571). 
# 
# Following example shows that Governments not only need a strong focus on data science but also need strong governance and thought leadership. 
# 
# <br>
# <div align="center"><span class='alert-info' style='padding:20px;color:#222'>&#x25cf; &nbsp;&nbsp; Insurance Industry: 1/3rd participants suggested that their current ML landscape are advanced (more models in the production)</span></div>
# <br>
# 
# An interesting find in the analysis shows the better adoption of data science in the insurance industry. One possible intution is that this group contains more actuaries, statisticians, and data scientists. Typical productionable use cases in InsurTech includes - claims forecasting and prediction, underwriting risk assessment, improved customer service and experience, customer acquisition, retention, and churn prediction. Also interesting to note that 64% respondents from Insurance sector have higher education and 44% are in the Data Science roles. 
# 
# 
# <br>
# <div align="center"><span class='alert-info' style='padding:20px;color:#222'>&#x25cf; &nbsp;&nbsp; Online Services / Internet Based Business have significantly higher proportion of more AI / ML adoption</span></div>
# <br>
# 
# - More than **25%** respondents mentioned that they or their teams have spent more than **USD100,000** on ML / Cloud
# - Close to **50%** from this group have larger Data Science / AI teams with more than 20 Individuals part of it. 
# - **One-Third** are from larger Multi-National Corporations (with 10K+ employees)
# - Usage of **Transformer Networks (BERT, GPT-3, etc)** is the highest for Online Services / Internet Based Business: About **15.73%**
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">3. AI Adoption Index - A Unified Index using Kaggle Survey Data</p>
# 
# So far we have explored only the question about the extent of machine learning incorporated by the organizations (Q23). In the Kaggle Survey Data 2021, there is a bunch of other useful and valuable information that reflects the state of data science, machine learning, and artificial intelligence adoption in different groups. There are questions regarding the skills of the people, the technology they use, the job roles they have, and their educational qualifications, etc. All these aspects are the essential elements that define how mature is the adoption of AI in a country/sector. A standard hypothesis regarding a country having more AI data science adoption and maturity can be due to a combination of:  
# - multiple job openings in a country related to data science
# - many highly qualified individuals in a country (with masters, PHDs)
# - several people/organizations using cloud architectures, machine learning management platforms, machine learning tools
# - more fundings, investments in the technology sector 
# - more competitive data science professionals (example - kagglers in top 1000)
# - more organizations with models in production, the larger data science team size, and spend considerate amounts on data related workflows
# - strong focus on innovation, research, and development  
# 
# Using the Kaggle Analytics survey data of 2021, we tried to map the responses of individuals into different categories to derive a standard adoption index. 
# 
# | | Component | Question | What it means | 
# | --- | ---- | ---- | ---- | 
# | 1 | Strategy | Q5: Job Roles, Q22: Team Size, Q26: Money Spent | Organizational Strategies - hiring more diverse job roles ie. a mix of engineers, analysts, scientists, managers, etc.,<br>spending more in building data infrastructures and data science teams  |
# | 2 | Data | Q32, Q33 Data Tools Usage | Organizations using more data tools, databases, data lakes are already in better data / ai adoption stages  |
# | 3 | Technology | Q38, Q23: Current ML Adoption, Tools Used | How advanced is current technology stack  |
# | 4 | People | Q4, Q5: Roles, Qualification | Are there enough qualified people, enough people with relevant job titles  |
# | 5 | Governance | Q31, Q37: Platforms Enabling Governance | Use of Saas platforms to enable Governance - Explainability, Regulations, Controlled Access  |
# | 6 | Automation | Q36, Q37: AutoML Platforms, AutoFE  | Organizations focussing on productivity and automating mundane tasks   |

# In[4]:


factors = [
            '<b>Advance</b> Database Usage<br><sup>Q32: (Databases, DataLakes)</sup><br><b><i>2. Data</i><b>',     
            '<b>More</b> Roles Diversity<br><sup>Q5: (Engineers, Scientists, Managers)</sup><br><b><i>1. Strategy</i><b>', 
            '<b>More</b> Automation<br><sup>Q36: (AutoML, Auto Feature Engineering)</sup><br><b><i>6. Automation</i><b>',            
            '<b>More</b> AI Platforms Usage<br><sup>Q31: (SageMaker, DataRobot, Databricks)</sup><br><b><i>5. Governance</i><b>',            
            '<b>Higher</b> Education Qualification<br><sup>Q4: (Masters, PHDs, Doctorates)</sup><br><b>More</b> Data Science Roles<br><sup>Q5: (Data Scientists, ML Engineers)</sup><br><b><i>4. People</i><b>', 
            '<b>Advanced</b> ML Adoption<br><sup>Q23: (More Models in Production)</sup><br><b>More</b> ML Management Tools<br><sup>Q38: (Tensorboard, MLFlow)</sup><br><b><i>3. Technology</i><b>',
            '<b>Advance</b> Database Usage<br><sup>Q32: (Databases, DataLakes)</sup><br><b><i>2. Data</i><b>',
        ]      

tr1 = go.Scatterpolar(
            r=[4,6,4,6,4,6,4], 
            theta=factors,
            showlegend=False, 
            name='ML Usage Components',
            mode='lines+markers',
            line_color='orange',
            opacity=0.7,
            line_shape='spline',
            line_smoothing=0.6,
            line_width=2.5,
            fill='toself',
        )

tr2 = go.Scatterpolar(
            r=[6,4,6,4,6,4,6], 
            theta=factors,
            showlegend=False, 
            name = 'Education and Roles Components',
            mode='lines+markers',
            line_color='purple',
            opacity=0.2,
            textfont=dict(size=16),
            line_shape='spline',
            line_smoothing=0.6,
            line_width=2.5,
            fill='toself',
        )
fig = go.Figure(data=[tr1, tr2], layout = layout)
title = 'Components of AI Adoption Index'

fig.update_layout(
    title_text = title,
    title_font_color = '#333333',
    title_font_size = 14,    
    polar_bgcolor='white',
    polar_radialaxis_visible=True,
    polar_radialaxis_showticklabels=True,
    polar_radialaxis_tickfont_color='darkgrey',
    polar_angularaxis_color='#f26f66',
    polar_angularaxis_showline=False,
    polar_radialaxis_showline=False,
    polar_radialaxis_layer='below traces',
    polar_radialaxis_gridcolor='#F2F2F2',
    polar_radialaxis_range=(1,7),
    polar_radialaxis_tickvals=[1],
    polar_radialaxis_ticktext=[''],
    polar_radialaxis_tickmode='array',
    
    legend_font_color = 'grey', 
    legend_itemclick = 'toggleothers', 
    legend_itemdoubleclick = 'toggle', 
    height = 500 
)

fig.add_layout_image(dict(source='https://i.imgur.com/xMigaX4.png', x=0.51, y=0.61))
fig.update_layout_images(dict(xref="paper", yref="paper", sizex=0.1, sizey=0.2, xanchor="center", yanchor="top"))

fig.show()


# <br> <div align="center"><img width=700 src='https://i.imgur.com/yZMeKgc.png'><br>Source : DataIku - Key Elements of Enterprise AI </div> <br>
# 
# <div align="center"> These components align with several other interesting metrics and indexes developed by organizations. Using the kaggle data, we developed the index in the following manner: 
#  <br> <br> <img src="https://i.imgur.com/gadPFiT.png"></div> <br>
# 
# First, aggregate the kaggle data by different groups (country/sector), Perform ordinal encoding of questions such as current levels of ML maturity, data science team sizes, current ML spendings. Then measure the counts and mean of tools/technologies used in a group, for example - ml tools used, data tools used, auto ml tools used, technologies used. Also measure the mean and standard deviation of groups such as job roles, education backgrounds, coding experiences, and ML experiences. Finally, the measures are combined and normalized to get a unified index, called the AI adoption index. This index is a mix of people, tools, skills, qualifications, maturity, data, and governance. 
# 
# <br><div align="center"><img src="https://i.imgur.com/76Eb4Sn.png" width=600></div> <br>
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">4. Industry Rankings using AI Adoption Index</p>
# 
# Let's identify the industries which rank higher in the derived AI adoption index. And also how it is related to educational qualifications within the industry. 

# In[5]:


ai_methods = {'We are exploring ML methods (and may one day put a model into production)': 1, 'No (we do not use ML methods)': 0, 'I do not know': 0.1, 'We have well established ML methods (i.e., models in production for more than 2 years)': 4, 'We recently started using ML methods (i.e., models in production for less than 2 years)': 3, 'We use ML methods for generating insights (but do not put working models into production)': 2}
ai_spends = {'$0 ($USD)': 0, '$100-$999': 1, '$1000-$9,999': 3, '$1-$99': 2, '$10,000-$99,999': 4, '$100,000 or more ($USD)': 5}
ai_teamsize = {'1-2': 1, '20+': 6, '0': 0, '3-4': 2, '5-9': 3, '10-14': 4, '15-19': 5}

temp_df['Q22_l'] = temp_df['Q22'].apply(lambda x : ai_teamsize[x] if x in ai_teamsize else x )
temp_df['Q23_l'] = temp_df['Q23'].apply(lambda x : ai_methods[x] if x in ai_methods else x )
temp_df['Q26_l'] = temp_df['Q26'].apply(lambda x : ai_spends[x] if x in ai_spends else x )

industries = temp_df['Q20'].unique()
industry_df = create_aggregated_frame(temp_df, col='Q20', colname='Industry', unique_values = industries)

industry_df['ml_adoption'] = 0
for c in ['ml_tools', 'automl_tools', 'autofe_tools', 'ml_products', 'bigdata_tools']:
    industry_df['ml_adoption'] += industry_df[c] / np.max(industry_df[c])

industry_df['cloud_adoption'] = 0
for c in ['cloud_tools', 'cloud_adoption1', 'cloud_adoption2']:
    industry_df['cloud_adoption'] += industry_df[c] / np.max(industry_df[c])

factors = ["ML Levels", 'ml_tools', 'ml_products', 'higher_education', 'ds_roles', 'roles_diversity', 'bigdata_tools']  
def calculate_ai_maturity(r, max_vals):    
    score = 0
    for i, f in enumerate(factors):
        score += r[f] / (1 + max_vals[i])
    return score

max_vals = []
for f in factors:
    max_vals.append(max(industry_df[f].fillna(0)))

industry_df['AI_Adoption_Index'] = industry_df.fillna(0).apply(lambda x : calculate_ai_maturity(x, max_vals), axis = 1)
industry_df['AI_Adoption_Index'] = industry_df['AI_Adoption_Index'] / max(industry_df['AI_Adoption_Index'])
industry_df['Industry'] = industry_df['Industry'].apply(lambda x : x.split("/")[0].strip())


def add_annotations_ai_adoption(fig, text_cfg):
    for i,text in enumerate(text_cfg['texts']):
        fig.add_annotation(text=text, xref="paper", yref="paper", x=text_cfg['text_xpos'][i], y=text_cfg['text_ypos'], showarrow=False, 
                           font = dict(color = '#f77f77', size=14) )
    
    for i,img in enumerate(text_cfg['imgs']):
        fig.add_layout_image(dict(source=img, x=text_cfg['img_xpos'][i], y=text_cfg['img_ypos']))
        fig.update_layout_images(dict(xref="paper", yref="paper", sizex=0.1, sizey=0.2, xanchor="center", yanchor="top"))
    
    
    for i, a in enumerate(text_cfg['annos']):
        if 'show_arr' not in a:
            a['show_arr'] = True
        
        fig.add_annotation(x=a['x'], y=a['y'], xref="x", yref="y",
            text=a['text'], showarrow=a['show_arr'],
            font=dict(family="Verdana", size=12, color="#ffffff"),
            align="center", arrowhead=1, arrowsize=1, arrowwidth=1,
            arrowcolor="#222", ax=a['ax'], ay=a['ay'], bordercolor="#c7c7c7", borderwidth=1,
            borderpad=4, bgcolor="#f77f77", opacity=0.7)
    return fig


yy = list(industry_df['higher_education'])
xx = list(industry_df['AI_Adoption_Index'])
texts = list(industry_df['Industry'])
sizes = list(industry_df['ML Levels'])

textpos = ['bottom center' if _ not in ['Online Service', 'Marketing'] else 'top center' for _ in texts]

trace1 = go.Scatter(x = xx, y = yy, mode='markers+text', textposition = textpos,
        text = texts ,textfont=dict(family="Verdana", size=12), 
        hovertemplate=
        "Industry: %{text}<br>" +
        "AI Adoption Stage: %{x}<br>" + 
        "% Higher Education Degrees: %{y}",
        name="", marker=dict(
        color=sizes, 
        size=[_*15 for _ in sizes],
        showscale=True,
        colorbar=dict(title='Percent'),
        opacity=1,
        colorscale = 'RdBu_r'
        ))

layout = go.Layout(width=1050, height=700, 
                   plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend = False,
                   title = {'text' : "AI Adoption Index: Manufacturing Sector lags behind, Insurance Leads<br><sup>Size, Color: Percentage of Respondents</sup>", 'x':0.5, 'xanchor': 'center'}, 
                   font = {"color" : '#f27866'}
                  )
fig = go.Figure(data=[trace1], layout=layout)
fig.add_vline(x=0.73, line_width=1, line_dash="dash", line_color="#c9c4c3")
fig.add_vline(x=0.89, line_width=1, line_dash="dash", line_color="#c9c4c3")

text_cfg = {
    "texts" : ["Seedling", "Vegetative", "Budding", "Flowering", "Ripening"],
    "text_xpos" : [0.086, 0.28, 0.51, 0.75, 0.94],
    "text_ypos" : 0.88,
    "img_ypos" : 1.05,
    "img_xpos" : [0.12, 0.34, 0.51, 0.71, 0.905],
    "imgs" : [ "https://i.imgur.com/9rs2DVF.png", "https://i.imgur.com/WIusNLF.png", "https://i.imgur.com/j4lAWaU.png", "https://i.imgur.com/VuR8AKW.png", "https://i.imgur.com/dAXbEY2.png"],
    'annos' : []
} 
fig = add_annotations_ai_adoption(fig, text_cfg)
fig.update_xaxes(showline=True, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', 
                 tickfont=dict(size=14, family='Verdana', color='#f26d63'), 
                 title="Derived AI Adoption Index", ticks='inside', tickwidth=1,ticklen=10,
                showgrid=False)
fig.update_yaxes(showline=False, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', 
                 tickfont=dict(size=14, family='Verdana', color='#f26d63'),
                 ticks='inside', tickwidth=1,ticklen=10, range=(0.5, 0.8),
                 title="Higher Education Degrees", 
                showgrid=False)
# fig.show()

insight1 = "Marketing have highest<br>diversity of education degrees<br>(ie. mode mix of bachelors, masters, PHDs)"
fig.add_annotation(x=0.84, y=0.67, text=insight1, showarrow=False, font=dict(family="Verdana", size=10, color="#787b82"))
insight1 = "Highest ML Platforms Usage<br>(SageMaker, DataBricks etc.)"
fig.add_annotation(x=1, y=0.615, text=insight1, showarrow=False, font=dict(family="Verdana", size=10, color="#787b82"))
insight1 = "All these three sectors<br>have least Spendings<br><10K USD"
fig.add_annotation(x=0.7, y=0.72, text=insight1, showarrow=False, font=dict(family="Verdana", size=10, color="#787b82"))
insight1 = "Non Profit Sector has<br>minimal Data team sizes<br>mostly 1-2 people"
fig.add_annotation(x=0.7, y=0.57, text=insight1, showarrow=False, font=dict(family="Verdana", size=10, color="#787b82"))
insight1 = "Shipping/Energy showing<br>great signs of Improved AI Adoption<br>from the previous years"
fig.add_annotation(x=0.78, y=0.6, text=insight1, showarrow=False, font=dict(family="Verdana", size=10, color="#787b82"))
insight1 = "Group of technology/Online related sectors<br>with highest adoption<br>of data, cloud, platforms, tools"
fig.add_annotation(x=0.95, y=0.66, text=insight1, showarrow=False, font=dict(family="Verdana", size=10, color="#787b82"))
insight1 = "Highest concentration of<br>qualified profiessionals<br>(Master's, PHDs)"
fig.add_annotation(x=0.81, y=0.72, text=insight1, showarrow=False, font=dict(family="Verdana", size=10, color="#787b82"))

fig.show()



# <br>
# 
# <div align="center"><span class='alert-info' style='padding:20px;color:#222'>&#x25cf; &nbsp;&nbsp; Manufacturing Sector shows a huge scope of AI adoption going forward while Insurance Sector Leads </span></div>
# 
# <br>
# 
# Reports suggest that the global market for manufacturing is expected to grow from USD 86.7B in 2020 to USD 117.7B by 2025. These numbers are significant and data science can play a tremendous role in this. However, despite the big market size, the current adoption levels of data science are relatively low. A key reason for the slow adoption is the complexity of edge deployments which allows processing and inferring the data locally on the equipment or machines. Deployments at the edge are not an easy/straightforward task. It requires models to be capable of low latency and high-frequency scoring and can process large chunks of data in near-real-time. In this sector the respondents were: 
# 
# - Majority (18%) are from Japan, which is a hub for manufacturing giants 
# - Majority shared that their day to day job role includes: Analyzing of Understanding data to influence product or business decisions
# - 12% have been the adopters of Google Cloud AutoML 
#     
# Broadcasting/Communication sector ranked 4th among the 16 scored industries in this dataset. With an increased focus on hyper-personalization and an increasing number of data-centric use-cases like "advanced search", "content monetization and moderation", "auto caption, and sub-title generation" becoming common, the industry is growing well in AI adoption. Most numbers of the respondents from the broadcasting sector are from India (14%) where the AI companies struggle to serve a billion-plus population. 34% of respondents shared that at least 20 people are responsible for data science workloads in their teams.   
# 
# Medical/Pharma sector and Military/Defense sector have the highest percentage of qualified professionals (PHDs, Doctorates, Master's). <mark>Military sector had 53% with master's degree while medical had 47%</mark>. Both of these sectors especially are doing much better than public services or governments. 
# 
# The general insights of the AI adoption index align with the machine learning incorporation by organizations (as asked in Q23). All the computers, technology or internet related sectors leads while Governments, Non-Profits lags. The data shows that these sectors have the least spendings on data science workloads. The insurance sector shows the highest levels of AI adoption, they also have the highest usage of machine learning platforms such as - Sagemaker from Amazon, or DataBricks. This particular analysis is interesting since the market size and the average revenues of the Manufacturing Sector are very high, still, AI adoption remains low. In the next section, we focussed on the analysis of sector-wise revenues. 
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">5. Industries (with Data Science Jobs): Revenues and Adoption </p>
# 
# **External Dataset Used:** [Data Science Jobs by Sectors, Countries (2021)](https://www.kaggle.com/shivamb/glassdoor-jobs-data)
# 
# We analyzed an important metric - the average revenue of a sector especially those which are adopting data science or have data science departments. To obtain this data and information, we procured the data about data science-related job roles from Glassdoor and Linkedin along with meta information. The meta-information includes the average revenue of the company, sector, country, and company size. The initial data was very messy and contained text in different languages, we cleaned it manually, used text processing, and translated it all to English. After the preprocessing, we got about 10000 job listings with proper metadata.  

# In[6]:


sectors_data = pd.read_csv("../input/glassdoor-jobs-data/2021-ds-sector-wise-revenues.csv")
totals = sectors_data['Sector'].value_counts().to_dict()
sectors_data['conter'] = [i for i in range(len(sectors_data))]

secdf = sectors_data.groupby(["Sector", "Revenue", "RevenueRank" ]).agg({"conter" : "count"}).reset_index()
secdf['RevPercent'] = secdf.apply(lambda x : 100*x['conter'] / totals[x['Sector']], axis = 1)
secdf['Industry'] = secdf['Sector'].apply(lambda x : str(x).split("/")[0])

c12 = "AI_Adoption_Index" # 'ML Spendings'
small_df = industry_df.merge(secdf, on = 'Industry', how = 'left')[['Industry', c12, 'Revenue', 'RevPercent', 'RevenueRank']]
small_df = small_df.sort_values([c12, 'RevenueRank'])
small_df['RevPercent'] = small_df['RevPercent'].fillna(0)
small_df['Revenue'] = small_df['Revenue'].fillna("")
small_df = small_df[small_df['Revenue'] != ""]


small_df['Revenue1'] = small_df['RevenueRank'].astype(str) +":"+ small_df['Revenue'].astype(str)


xx = list(small_df['Revenue'])
yy = list(small_df['Industry'])
texts = list(small_df['Industry'])
colors = list(small_df[c12])
sizes = list(small_df['RevPercent'])
sizes = [round(_, 1) for _ in sizes]
colors = [round(_, 1) for _ in colors]

trace1 = go.Scatter(x = xx, y = yy, mode='markers+text', textposition = 'bottom center',
        name="",         
        hovertemplate=
        "Industry: %{y}<br>" +
        "Revenue: %{x}<br>" + 
        "Adoption: %{marker.color}<br>" +
        "Percentage of Organizations: %{marker.size}%<br>",
        marker=dict(
        color=colors, 
        size=sizes,
        showscale=True,
        opacity=1,
        colorscale = 'RdBu_r',
            colorbar=dict(title='Adoption')
        ),
       )

layout = go.Layout(width=1050, height=700, 
                   plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend = False,
                   title = {'text' : "Revenue by Sector and AI Adoption<br><sup><b>Size:</b> Number of Organizations</sup>", 'x':0.5, 'xanchor': 'center'}, 
                   font = {"color" : '#f27866'}
                  )

fig = go.Figure(data=[trace1], layout=layout)

insight1 = "Sectors with Very High Revenues<br> - lagging behind in terms of AI Adoption<br> + Relatively more data science jobs"
fig.add_annotation(x=10, y=-0.1, text=insight1, showarrow=False, font=dict(family="Verdana", size=10, color="#787b82"))

insight1 = "More Data Science Jobs<br>by Companies with $10B+ Revenue"
fig.add_annotation(x=10, y=15.5, text=insight1, showarrow=False, font=dict(family="Verdana", size=10, color="#787b82"))

fig.add_vline(x=7.5, line_width=1, line_dash="dash", line_color="#c9c4c3")
fig.add_shape(type="rect", x0=8.5, y0=0.5, x1=11.5, y1=6.5, line=dict(color="#cccac4", width=2, dash='dash' ))

fig.update_yaxes(categoryorder="array")
fig.show()


# <br>
# 
# <div align="center"><span class='alert-info' style='padding:20px;color:#222'>&#x25cf; &nbsp;&nbsp; Shipping, Energy, Manufacturing and Retail sectors are ranked much higher in average revenues but AI adoption is relatively low</span></div>
# 
# <br>
# 
# Accounting/Finance or Military/Defence sectors have higher percentage of companies with more than USD 10B+ USD, but their current AI / data science maturity levels are lower than some of the other sectors such as Insurance. The bigger size of the bubbles shows that there are multiple data science job opportunities in the organizations with higher revenue. The analysis shows the industries/sectors which are leading, such as - Online related, Insurance, Computers and the ones which are lagging such as - Governments, NonProfits, Manufacturing. Hospitality/Entertainment/Sports have also higher revenue but lower percentage of people having machine learning experiences. About 48% shared their current ML experience is less than a year. 
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">6. A Geographical Perspective of AI Adoption </p>
# 
# Respondents were asked about their country/region in the Kaggle Survey. There were respondents from 66 different countries, including all parts of the world - Americas, Asia, Europe, Africa, the Middle East, and Australias. 
# 
# Let's take a look at how different "Countries" stand at different levels for machine learning incorporation. The following visualization lists down the top 20 countries with the most kaggle survey respondents. The y-axis is measured directly using the question (Q23) asked - Do your employers incorporate machine learning methods. The size and the color represent the percentage of respondents from that country who mentioned how much ML is incorporated in their organizations. 

# In[7]:


agg_df = raw_df[1:]

agg_df = agg_df[agg_df['Q23'] != 'I do not know']
agg_df = agg_df[agg_df['Q3'] != 'I do not wish to disclose my location']
agg_df = agg_df[agg_df['Q3'] != 'Other']
agg_df = agg_df[agg_df['Q3'].isin(raw_df['Q3'].value_counts().index[:20])]


agg_df = agg_df[~agg_df['Q23'].isna()]
country_totals = agg_df['Q3'].value_counts().to_dict()
agg_df = agg_df.groupby(["Q3", "Q23"]).agg({"Q21" : "count"}).reset_index()
agg_df['Q21'] = agg_df.apply(lambda x : x['Q21'] / country_totals[x['Q3']], axis = 1 )

mapp = {
    'No (we do not use ML methods)' : "0. Not Started<br><sup>(No ML)</sup>" , 
    'We are exploring ML methods (and may one day put a model into production)' : "1. Exploration<br><sup>Only Exploring ML</sup>",
    'We use ML methods for generating insights (but do not put working models into production)' : "2. Beginner Stage<br><sup>Use ML only for Insights</sup>", 
    'We recently started using ML methods (i.e., models in production for less than 2 years)' : "3. Intermediate Stage<br><sup>Recently Started Using ML</sup>",
    'We have well established ML methods (i.e., models in production for more than 2 years)' : "4. Advance Stage<br><sup>Well Established ML</sup>",
}
agg_df['Q23'] = agg_df['Q23'].apply(lambda x : mapp[x])
agg_df['Q3'] = agg_df['Q3'].apply(clean_country)

agg_df = agg_df.sort_values(['Q3', 'Q23'])
agg_df

trace1 = go.Scatter(
    x=agg_df['Q3'],
    y=agg_df['Q23'].apply(lambda x : x.split(".")[1]),
    mode='markers', 
    hovertemplate=
        "Country: %{x}<br>" +
        "AI Adoption Stage: %{y}<br>" +
        "Percentage: %{marker.color:,}",
    marker=dict(
        color=agg_df['Q21']*100,
        size=agg_df['Q21']*80,
        
        opacity=1,
        colorscale = 'RdBu_r',
        showscale=True, colorbar=dict(title='Percent')
        )
)

layout = go.Layout(width=1050, height=600, plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend = False, 
                   title = {'text' : "USA, UK, Spain, France, Germany are leading in well established ML<br><sup>Q20: Country; Q3: ML Adoption Levels | Size, Color: Percentage of Respondents</sup>", 
                            'x':0.5, 'xanchor': 'center'}, 
                   font = {"color" : '#f27866'})
fig = go.Figure(data = [trace1], layout = layout)



def add_flag(fig, text_cfg):
    for i,img in enumerate(text_cfg['imgs']):
        fig.add_layout_image(dict(source=img, x=text_cfg['img_xpos'][i], y=text_cfg['img_ypos']))
        fig.update_layout_images(dict(xref="paper", yref="paper", sizex=0.04, sizey=0.05, xanchor="center", yanchor="top"))

    return fig

text_cfg = {
    "img_ypos" : 1.05,
    "img_xpos" : [0.07, 0.12, 0.17, 0.215, 0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.55, 0.59, 0.64, 0.69, 0.74, 0.79, 0.84, 0.885,0.93],
    "imgs" : [  "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Flag_of_Bangladesh.svg/2560px-Flag_of_Bangladesh.svg.png", 
                "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Flag_of_Brazil.svg/1280px-Flag_of_Brazil.svg.png",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Flag_of_Canada.svg/1280px-Flag_of_Canada.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Flag_of_the_People%27s_Republic_of_China.svg/2560px-Flag_of_the_People%27s_Republic_of_China.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Flag_of_Egypt.svg/2560px-Flag_of_Egypt.svg.png",
              "https://upload.wikimedia.org/wikipedia/en/thumb/c/c3/Flag_of_France.svg/1200px-Flag_of_France.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Flag_of_Germany.svg/1280px-Flag_of_Germany.svg.png",
              "https://upload.wikimedia.org/wikipedia/en/thumb/4/41/Flag_of_India.svg/1200px-Flag_of_India.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Flag_of_Indonesia.svg/255px-Flag_of_Indonesia.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Flag_of_Japan.svg/2560px-Flag_of_Japan.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Flag_of_Nigeria.svg/1280px-Flag_of_Nigeria.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/1280px-Flag_of_Pakistan.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/d/d4/Flag_of_Russia.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/Flag_of_South_Korea.svg/2560px-Flag_of_South_Korea.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Flag_of_Spain.svg/2560px-Flag_of_Spain.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Flag_of_the_Republic_of_China.svg/320px-Flag_of_the_Republic_of_China.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Flag_of_Turkey.svg/2560px-Flag_of_Turkey.svg.png",
              "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Flag_of_the_United_Kingdom.svg/2560px-Flag_of_the_United_Kingdom.svg.png",
                "https://upload.wikimedia.org/wikipedia/en/thumb/a/a4/Flag_of_the_United_States.svg/1000px-Flag_of_the_United_States.svg.png", 
               ], 
} 
fig = add_flag(fig, text_cfg)

fig.update_xaxes(showline=False, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='#f26d63'), 
                 title="", title_font=dict(size=14, family='Verdana', color='#f57369'), showgrid=False, tickangle=40)
fig.update_yaxes(showline=False, linewidth=1, linecolor='#000', gridcolor='#fff', tickfont=dict(size=14, family='Verdana', color='#699ff5'), 
                 title="", title_font=dict(size=14, family='Verdana', color='#f57369'), showgrid=False)
fig.show()


# c = 'Q4'
# a = temp_df[temp_df['Q20'] == 'Medical/Pharmaceutical'][c].value_counts().to_frame()
# a = a.reset_index()
# a['p'] = a[c] / sum(a[c])
# a

# industry_df[industry_df['Industry'] == 'Insurance']

# 64% have higher education, 44% are in the Data Science roles 
# More 

# cols = [c for c in temp_df.columns if 'Q6' in c]
# for c in cols:
#     x = temp_df[temp_df['Q20'] == 'Medical/Pharmaceutical'][c].value_counts().to_dict()
#     print (list(x.keys())[0], list(x.values())[0] / len(temp_df[temp_df['Q20'] == 'Manufacturing/Fabrication']))

# raw_df['Q20'].value_counts()


# <br>
# 
# <div align="center"><span class='alert-info' style='padding:20px;color:#222'>&#x25cf; &nbsp;&nbsp; Asia is lagging, Europe is leading, Africa,Middle East are growing</span></div>
# 
# <br>
# 
# 
# - The most number of countries from Asia which includes China, Bangladesh, India, Indonesia, Pakistan, South Korea, and Taiwan have comparatively lower levels of machine learning incorporation. On an average, close to 30% of respondents from these countries suggested that they are just exploring machine learning methods, the other <mark>25% suggested they have not even started any machine learning</mark> in their organizations. In total, about 60% suggested that they are just venturing into the machine learning space. Europe on the other hand, shows several countries having well-established methods in the organizations. 
# 
# - African and Middle East countries are showing up more in recent years. Though about 45% from Nigera said they have not yet started any machine learning, The growth of AI and data-related startups in countries like Nigeria have increased and they are improving in the overall machine learning. Due to the latest developments (such as - technology adoption, government policies relaxation, improvements in education) in the continent, Many companies are looking at <mark>Africa as the next place for expansion and investment</mark>. A [report](https://ircai.org/the-growth-of-artificial-intelligence-in-africa-on-diversity-and-representation/), by Internation Research Center of AI (IRCAI), suggested that use AI and related technologies as an opportunity to create and reinforce diversity is the main focus area for companies operating in the African region. This also includes a focus on the skills development of diverse people and to make concerted efforts at leveling the playing field for women and other minorities in the industry.
# 
# In recent years, Government efforts to improve the AI landscape have improved across the globe and it is encouraging to see the countries realizing the true potential of AI and data science. However, The coronavirus pandemic affected all sectors of the economy in 2020, and the AI sector is no different. 
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">7. Pandemic Effect</p>
# 
# As the organizations worldwide manage ongoing pandemic effects and hold a slowly evolving post-pandemic future, a key question became important: <mark>Just how much did COVID impact the AI industry?</mark> 
# 
# Several AI startups emerged during this period that assisted the impacted industries, including accelerating the Covid-19 Drug Discovery, facilitating the supply chains, and optimizing the logistics during lockdowns. Many organizations had to reprioritize their hiring and operational strategies. The virtual formats of meetings, conferences, and meetups lead to significant spikes in attendance too. There were a lot of layoffs too, companies had to cut down their budgets. The following chart shows the change in percentage change of machine learning incorporation by country before the pandemic and after.

# In[8]:


k19 = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

mapp = {
    'No (we do not use ML methods)' : "0. No ML" , 
    'We are exploring ML methods (and may one day put a model into production)' : "1. Exploration Stage",
    'We use ML methods for generating insights (but do not put working models into production)' : "2. Beginner Stage", 
    'We recently started using ML methods (i.e., models in production for less than 2 years)' : "3. Intermediate Stage",
    'We have well established ML methods (i.e., models in production for more than 2 years)' : "4. Advance Stage",
}

k19['ML Adoption'] = k19['Q8'].apply(lambda x : mapp[x] if x in mapp else x)

px = k19.groupby(['Q3', 'ML Adoption']).agg({"Q2" : "count"}).reset_index()
px = px[px['ML Adoption'].isin(['4. Advance Stage', '3. Intermediate Stage'])]
px = px.merge(k19['Q3'].value_counts().to_frame().reset_index().rename(columns = {"index" : "Q3", "Q3" : "total"}), on='Q3', how='left')
px['percent'] = px['Q2'] / px['total']
ad19 = px.sort_values("percent", ascending = False )
ad19 = ad19.rename(columns = {'Q3' : "Country", 'percent' : "2019_percent"})
ad19 = ad19.groupby("Country").agg({"2019_percent" : "sum"}).reset_index()

raw_df['ML Adoption'] = raw_df['Q23'].apply(lambda x : mapp[x] if x in mapp else x)
px = raw_df.groupby(['Q3', 'ML Adoption']).agg({"Q2" : "count"}).reset_index()
px = px[px['ML Adoption'].isin(['4. Advance Stage', '3. Intermediate Stage'])]
px = px.merge(raw_df['Q3'].value_counts().to_frame().reset_index().rename(columns = {"index" : "Q3", "Q3" : "total"}), on='Q3', how='left')
px['percent'] = px['Q2'] / px['total']
a21 = px.sort_values("percent", ascending = False )
a21 = a21.rename(columns = {'Q3' : "Country", 'percent' : "2021_percent"})
a21 = a21.groupby("Country").agg({"2021_percent" : "sum"}).reset_index()


combined = a21.merge(ad19, on='Country', how='left')
combined['difference'] = combined['2021_percent'] - combined['2019_percent']
combined[['Country', '2021_percent', '2019_percent', 'difference']]
combined['abs_difference'] = np.abs(combined['difference'])
c1 = combined.sort_values("2021_percent", ascending = False).head(25)
c1


c = pd.DataFrame()
c['Country'] = list(c1['Country']) + list(c1['Country']) + list(c1['Country'])
c['Country'] = c['Country'].apply(clean_country)
c['x'] = list(c1['2019_percent']) + [0]*len(c1) + list(c1['2021_percent'])
c['y'] = ['2019<br>ML Adoption']*len(c1) + ['2020']*len(c1) + ['2021<br>ML Adoption']*len(c1)
c = c[c['Country'] != 'Other']

xvals = list(c['y'])
yvals = list(c['Country'])[::-1]
svals = list([_*70 for _ in c['x']])[::-1]
cvals = 'Red'
tvals = list([str(round(_*100, 1)) + "%" if _ != 0 else "" for _ in c['x']])[::-1]

trace1 = go.Scatter(x=xvals, y=yvals, mode='markers+text', textposition='middle right', text=tvals, marker_symbol='circle', 
                    textfont=dict(family="Verdana", size=12, color="#a19d9d"),
                    marker = dict(size=svals, color=c['x'], opacity=1, colorscale='RdBu_r', line=dict(width=2, color='#fff'))
                   )

layout = go.Layout(title = {'text' : "Israel and Norway Increased their Machine Learning Adoption by 10%+<br><sup>Majoirty of countries witnessed more incorporation of ML models in production !! </sup>", 
                            'x':0.5, 'xanchor': 'center'}, 
                   font = {"color" : '#f27866'}, 
    height=1000, plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend = False)
fig = go.Figure(data = [trace1], layout = layout)

cnt = 0
for i , r in c1.iterrows():
    change = round(r['difference'] * 100, 1)
    if change < 0:
        c = 'green'
        r = "↑ " + str(change*-1)
    else:
        c = "red"
        r = "↓ " + str(change)
    
    
    fig.add_annotation(x=1, y=25-cnt-0.6, text=r + "%", showarrow=False, yshift=0, font=dict(family="Verdana", size=14, color=c))
    cnt += 1
    
    if cnt == 25:
        break

fig.add_vline(x=0.6, line_width=1, line_dash="dash", line_color="#c9c4c3")
fig.add_vline(x=1.4, line_width=1, line_dash="dash", line_color="#c9c4c3")
insight1 = """Pandamic Year"""
fig.add_annotation(x=1, y=26.0, text=insight1, showarrow=False, yshift=0, font=dict(family="Verdana", size=16, color="red"))

fig.update_xaxes(showline=False, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='#f26d63'), 
                 title="", title_font=dict(size=14, family='Verdana', color='#f57369'), showgrid=False)
fig.show()


# Many countries saw an overall improvement machine learning adoption and incorporation in the organizations. The positive increase is encouraging to see even during the pandemic. Israel, Norway, UK, Australia, and Poland have the highest percentage of change since 2021. Though some countries observed a decline, this includes - Turkey, Greece, Romania, Ireland, and the Netherlands. Varying levels of AI adoption in different regions can be linked to several factors such as:
# 
# <span style="padding:10px;background:orange;margin-left:2px"> 1. Investments </span>
# <span style="padding:10px;background:orange;margin-left:2px"> 2. Jobs</span>
# <span style="padding:10px;background:orange;margin-left:2px"> 3. Research</span>
# <span style="padding:10px;background:orange;margin-left:2px"> 4. Skills</span>
# 
# - Investments: AI / Tech Company Fundings  
# - Jobs: AI Related Hiring / Jobs 
# - Research: Global Innovation Index   
# - Competitive Data Science: More skilled data scientists 
# 
# There could be more factors such as - Education or Government Regulations. In this notebook, we have only explored the above listed factors by using different external datasets and relate them with the AI adoption index using the kaggle survey data. We will use the same methodology to derive the AI adoption index as described in section 3. 
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">8. Investments/Tech Fundings and AI Adoption </p>
# 
# **External Dataset:** [Tech Fundings Since 2020](https://www.kaggle.com/shivamb/tech-company-fundings-2020-onwards)
# 
# This external dataset contains funding information in the technology sector by country, by sector, by date. Original source of the data is GrowthList and the cleaned data is posted on Kaggle. The data contains funding information since January 2020 and clearly shows that more dollars are flowing into the AI sector. 

# In[9]:


ai_methods = {'We are exploring ML methods (and may one day put a model into production)': 1, 'No (we do not use ML methods)': 0, 'I do not know': 0.1, 'We have well established ML methods (i.e., models in production for more than 2 years)': 4, 'We recently started using ML methods (i.e., models in production for less than 2 years)': 3, 'We use ML methods for generating insights (but do not put working models into production)': 2}
ai_spends = {'$0 ($USD)': 0, '$100-$999': 1, '$1000-$9,999': 3, '$1-$99': 2, '$10,000-$99,999': 4, '$100,000 or more ($USD)': 5}
ai_teamsize = {'1-2': 1, '20+': 6, '0': 0, '3-4': 2, '5-9': 3, '10-14': 4, '15-19': 5}

temp_df = raw_df[1:]
temp_df['Q22_l'] = temp_df['Q22'].apply(lambda x : ai_teamsize[x] if x in ai_teamsize else x )
temp_df['Q23_l'] = temp_df['Q23'].apply(lambda x : ai_methods[x] if x in ai_methods else x )
temp_df['Q26_l'] = temp_df['Q26'].apply(lambda x : ai_spends[x] if x in ai_spends else x )

countries = temp_df['Q3'].unique()
country_df = create_aggregated_frame(temp_df, col='Q3', colname='Country', unique_values = countries)

country_df['ml_adoption'] = 0
for c in ['ml_tools', 'automl_tools', 'autofe_tools', 'ml_products', 'bigdata_tools']:
    country_df['ml_adoption'] += country_df[c] / np.max(country_df[c])

country_df['cloud_adoption'] = 0
for c in ['cloud_tools', 'cloud_adoption1', 'cloud_adoption2']:
    country_df['cloud_adoption'] += country_df[c] / np.max(country_df[c])

factors = ["ML Levels", 'ml_tools', 'ml_products', 'higher_education', 'ds_roles', 'roles_diversity', 'bigdata_tools']  
def calculate_ai_maturity(r, max_vals):    
    score = 0
    for i, f in enumerate(factors):
        score += r[f] / (1 + max_vals[i])
    return score

max_vals = []
for f in factors:
    max_vals.append(max(country_df[f].fillna(0)))

country_df['AI_Adoption_Index'] = country_df.fillna(0).apply(lambda x : calculate_ai_maturity(x, max_vals), axis = 1)
country_df['AI_Adoption_Index'] = country_df['AI_Adoption_Index'] / max(country_df['AI_Adoption_Index'])

country_df['Country'] = country_df['Country'].apply(clean_country)

tech_df1 = pd.read_csv("../input/tech-company-fundings-2020-onwards/tech_fundings.csv")
tech_df1 = tech_df1[tech_df1['Company'] != 'WestConnex']
tech_df = tech_df1[tech_df1['Funding Amount (USD)'] != 'Unknown']
tech_df['Funding Amount (USD)'] = tech_df['Funding Amount (USD)'].astype(int)

tech_df['Funding Year'] = tech_df['Funding Date'].apply(lambda x : "2021" if "-21" in x else "2020")
tech_df = tech_df.groupby("Region").agg({"Funding Amount (USD)" : "mean", "Funding Stage" : "count"}).sort_values("Funding Stage", ascending = False).reset_index()

tech_df['Country'] = tech_df['Region'].apply(lambda x : clean_country(x))
tech_df = tech_df[tech_df['Funding Stage'] > 10]

c1 = country_df.merge(tech_df, on = 'Country', how = 'left' )
c1 = c1[~c1['Funding Stage'].isna()]
c1 = c1[c1['Country'] != 'HK']

c1['Funding Amount Log'] = np.log(c1['Funding Amount (USD)'])


def add_annotations_country(fig, text_cfg):
    
    for i,text in enumerate(text_cfg['texts']):
        fig.add_annotation(text=text, xref="paper", yref="paper", x=text_cfg['text_xpos'][i], y=0.88, showarrow=False, 
                           font = dict(color = 'orange', size=12) )
    
    for i,img in enumerate(text_cfg['imgs']):
        fig.add_layout_image(dict(source=img, x=text_cfg['img_xpos'][i], y=1.03))
        fig.update_layout_images(dict(xref="paper", yref="paper", sizex=0.1, sizey=0.2, xanchor="center", yanchor="top"))
    
    for i, a in enumerate(text_cfg['annos']):
        if 'show_arr' not in a:
            a['show_arr'] = True
        
        fig.add_annotation(x=a['x'], y=a['y'], xref="x", yref="y",
            text=a['text'], showarrow=True,
            font=dict(  family="Verdana", size=12, color="#ffffff"),
            align="center", arrowhead=1, arrowsize=1, arrowwidth=1,
            arrowcolor="#222", ax=a['ax'], ay=a['ay'], bordercolor="#c7c7c7", borderwidth=1,
            borderpad=4, bgcolor="#e09834", opacity=0.6)
    
    return fig 

def plot_chart_country3(plot_df, plt):
    ycol, xcol, scol, tcol = plt['ycol'], plt['xcol'], plt['scol'], plt['tcol']

    xvals, yvals = list(plot_df[xcol]), list(plot_df[ycol])
    svals = [_*30 for _ in list(plot_df[scol])]
    cvals = list(plot_df[scol])
    trace1 = go.Scatter(x=xvals, y=yvals, mode='markers+text', textposition='bottom center', text=plot_df[tcol], marker_symbol='circle', 
                        textfont=dict(family="Verdana", size=plt['text_font_size'], color="#a19d9d"),
                            hovertemplate=
                                    "Country: %{text}<br>" +
                                    "AI Adoption: %{x}<br>" +
                                    "Log(Mean Funding Amount): %{y}",
                        marker = dict(size=svals, color=cvals, opacity=1, colorscale='rdbu_r', colorbar=dict(title='Percent'), showscale = True, line=dict(width=2, color='#fff'))
                       )
    
    layout = go.Layout(title = {'text' : "Indonesia, Nigeria, and Mexico attracted huge chunks in recent Tech Fundings <br><sup>Fundings in AI sector increased significantly in last year !! </sup>", 
                                'x':0.5, 'xanchor': 'center'}, 
                       width=1050, height=750, plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend = False, font = {"color" : '#f27866'})
    fig = go.Figure(data = [trace1], layout = layout)
    
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='red'), 
                     title=plt['xtitle'], title_font=dict(size=14, family='Verdana', color='red'), showgrid=False, gridwidth=1, 
                     ticks="inside", tickcolor='red')
    fig.update_yaxes(showline=False, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='red'), 
                     title=plt['ytitle'], title_font=dict(size=14, family='Verdana', color='red'), showgrid=False, gridwidth=1, nticks=10, 
                     ticks="inside", tickcolor='red')

    
    
    
    text_cfg = {
        "texts" : ["Seedling", "Vegetative", "Budding", "Flowering", "Ripening"],
        "text_xpos" : [0.086, 0.28, 0.51, 0.75, 0.94],
        "img_xpos" : [0.12, 0.33, 0.513, 0.72, 0.91],
        "imgs" : [ "https://i.imgur.com/9rs2DVF.png", "https://i.imgur.com/WIusNLF.png", "https://i.imgur.com/j4lAWaU.png", "https://i.imgur.com/VuR8AKW.png", "https://i.imgur.com/dAXbEY2.png"],
        'annos' : []
    }
    fig = add_annotations_country(fig, text_cfg)
    
    fig.add_vline(x=0.81, line_width=1, line_dash="dash", line_color="#c9c4c3")

    insight1 = """GoJek (USD1.2B), Xendit (USD150M)<br>Halodoc (USD80)"""
    fig.add_annotation(x=0.58, y=18.9, text=insight1, showarrow=False, yshift=-50, font=dict(family="Verdana", size=8, color="red"))

    insight1 = """ITeos Therapeutics (125M)<br>Collibra (113M)<br>Silverfin (30M)"""
    fig.add_annotation(x=0.58, y=17.4, text=insight1, showarrow=False, yshift=-50, font=dict(family="Verdana", size=8, color="red"))
    
    insight1 = """SmartNews (230M), Paidy (120M)<br>SmartHR (115M)"""
    fig.add_annotation(x=0.72, y=18.1, text=insight1, showarrow=False, yshift=-50, font=dict(family="Verdana", size=8, color="red"))
    
    insight1 = "Lunar (249M), Pleo (150M)<br>Dixa (105M)"
    fig.add_annotation(x=0.57, y=18, text=insight1, showarrow=False, yshift=-50, font=dict(family="Verdana", size=8, color="red"))
    
    insight1 = "Kavak (485M), Bitso (250M)<br>Clip (250M)"
    fig.add_annotation(x=0.74, y=18.8, text=insight1, showarrow=False, yshift=-50, font=dict(family="Verdana", size=8, color="red"))
    
    insight1 = "Xpeng Motors (300M)<br>4Paradigm (230M)<br>Virtuos(150M)"
    fig.add_annotation(x=0.66, y=18.76, text=insight1, showarrow=False, yshift=-50, font=dict(family="Verdana", size=8, color="red"))
    
    insight1 = "Meesho (570M), ShareChat (502M)<br>PharmEasy (500M)"
    fig.add_annotation(x=0.6, y=18, text=insight1, showarrow=False, font=dict(family="Verdana", size=8, color="red"))
    
    insight1 = "OPay (400M)<br>FairMoney (420M)<br>Kuda (250M)"
    fig.add_annotation(x=0.64, y=17.1, text=insight1, showarrow=False, font=dict(family="Verdana", size=8, color="red"))

    insight1 = "DriveNets (208M)<br>Wiliot (200M)<br>InSightec (150M)"
    fig.add_annotation(x=0.78, y=17.2, text=insight1, showarrow=False, font=dict(family="Verdana", size=8, color="red"))
    
    insight1 = "Northvolt (600M)<br>Einride (110M)<br>Budbee (62M)"
    fig.add_annotation(x=0.78, y=17.6, text=insight1, showarrow=False, font=dict(family="Verdana", size=8, color="red"))
        
    insight1 = "Trax (640M)<br>PatSnap (300M)<br>Nium(200M)"
    fig.add_annotation(x=0.68, y=17.6, text=insight1, showarrow=False, font=dict(family="Verdana", size=8, color="red"))

    text_cfg = {
        "texts" : ["Seedling", "Vegetative", "Budding", "Flowering", "Ripening"],
        "text_xpos" : [0.086, 0.28, 0.51, 0.75, 0.94],
        "img_xpos" : [0.12, 0.33, 0.513, 0.72, 0.91],
        "imgs" : [ "https://i.imgur.com/9rs2DVF.png", "https://i.imgur.com/WIusNLF.png", "https://i.imgur.com/j4lAWaU.png", "https://i.imgur.com/VuR8AKW.png", "https://i.imgur.com/dAXbEY2.png"],
        'annos' : [{ 'x' : 0.65, 'y' : 16, 'text' : "Significant larger Tech / AI fundings<br> in last 2 Years", 'ax' : 0, 'ay' : 0}]
    } 
    fig = add_annotations_country(fig, text_cfg)
    fig.show()

plt = {
    'ycol' : 'Funding Amount Log',
    'xcol' : 'AI_Adoption_Index',
    'scol' : 'AI_Adoption_Index',
    'tcol'   : 'Country',
    'text_font_size' : 10,
    'title' : '',
    'ytitle' : 'Mean Funding Amounts AI Startups in 2021 (USD) (log)',
    'xtitle' : 'AI Adoption Index',
}
plot_chart_country3(c1, plt)


# tech_df1 = pd.read_csv("../input/tech-company-fundings-2020-onwards/tech_fundings.csv")
# tech_df1 = tech_df1[tech_df1['Company'] != 'WestConnex']
# tech_df1 = tech_df1[tech_df1['Funding Amount (USD)'] != 'Unknown']
# tech_df1['Funding Amount (USD)'] = tech_df1['Funding Amount (USD)'].astype(int)

# tech_df1['Funding Year'] = tech_df1['Funding Date'].apply(lambda x : "2021" if "-21" in x else "2020")
# sm = tech_df1[tech_df1['Region'] == 'Singapore'].sort_values("Funding Amount (USD)", ascending = False)
# sm 


# Interesting to see that few countries with a low AI adoption index have had more technology fundings in recent years. 
# 
# This includes countries such as - Indonesia where the data science adoption remains at the exploration stage. However, there a number of tech companies which are making big impacts, example - GoJek, a logistic compnay who recently bagged the funding of USD 1.2B to scale up their technology, AI, and data science operations. Another popular rising startup is the payment platform provider Xendit, which rased USD 150M to incoportate ML and AI driven systems in thier core processes. Other countries include - Belgium, Mexico, and Denmark with more recent fundings in the technology sector. 
# 
# Nigeria is an important regional player in Africa, making about half of West Africa’s population with 202M people and one of the largest youth populations in the world. It has recently become a center of attraction for technology and AI investors. This year SoftBank Vision Fund 2 led a USD 400M funding round for **OPay**, valuing the Nigerian mobile-payments platform at USD 2B and marking the investment vehicle’s first bet in Africa. The other investments in Nigeria included - **FairMoney (USD 42M)** which offers digital banking and instant loans providing collateral-free personal loans. 
# 
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">9. Jobs in AI vs Adoption </p>
# 
# Which countries are hiring in the data-related / AI-related roles?
# 
# **External Data Used:** [Linkedin Data Science Jobs in 2021 by Country](https://www.kaggle.com/shivamb/glassdoor-jobs-data)
# 
# The AI industry witnessed strong hiring growth during the pandemic. While health care, finance, and other service industries have been strong early adopters of AI, manufacturing, retail, and other sectors are expected to grow in hiring and AI-skill penetration as well. Let's see what data is talking about. 
# 

# In[10]:


linkedin_jobs = pd.read_csv("../input/glassdoor-jobs-data/2021-ds-ml-jobs-linkedin.csv")

linkedin_jobs = {'India': 18372.0, 'USA': 172503.0, 'Japan': 1166.0, 'China': 6537.0, 'Brazil': 2337.0, 'Russia': 562.0, 'Nigeria': 200.0, 'UK': 12899.0, 'Pakistan': 262.0, 'Egypt': 277.0, 'Germany': 8580.0, 'Spain': 1974.0, 'Indonesia': 881.0, 'Turkey': 421.0, 'France': 2963.0, 'South Korea': 318.0, 'Taiwan': 499.0, 'Canada': 9001.0, 'Bangladesh': 32.0, 'Italy': 1709.0, 'Mexico': 1469.0, 'Viet Nam': 445.0, 
                 'Australia': 2737.0, 'Kenya': 62.0, 'Colombia': 559.0, 'Poland': 3399.0, 'Ukraine': 607.0, 'Argentina': 613.0, 'Singapore': 4209.0, 'Malaysia': 921.0, 'Netherlands': 3102.0, 'South Africa': 707.0, 'Morocco': 53.0, 'Israel': 2013.0, 'Thailand': 453.0, 'Portugal': 1212.0, 'Peru': 98.0, 'UAE': 320.0, 'Tunisia': 22.0, 'Philippines': 1730.0, 'Sri Lanka': 101.0, 'Chile': 281.0, 'Greece': 421.0, 
                 'Ghana': 26.0, 'Saudi Arabia': 158.0, 'Ireland': 1509.0, 'Sweden': 1118.0, 'HK': 1320.0, 'Nepal': 17.0, 'Switzerland': 1155.0, 'Belgium': 1114.0, 'Czech Republic': 491.0, 'Romania': 1010.0, 'Belarus': 110.0, 'Austria': 455.0, 'Ecuador': 27.0, 'Denmark': 519.0, 'Uganda': 9.0, 'Kazakhstan': 28.0, 'Norway': 300.0, 'Algeria': 4.0, 'Ethiopia': 16.0}
country_df['ds_jobs'] = country_df['Country'].apply(lambda x : linkedin_jobs[x] if x in linkedin_jobs else 0)

mj = max(country_df['ds_jobs']) 
country_df['ds_jobs_log'] = np.log(country_df['ds_jobs'])

def plot_chart_country2(plot_df, plt):
    ycol, xcol, scol, tcol = plt['ycol'], plt['xcol'], plt['scol'], plt['tcol']

    xvals, yvals = list(plot_df[xcol]), list(plot_df[ycol])
    svals = [_*30 for _ in list(plot_df[scol])]
    cvals = list(plot_df[scol])
    trace1 = go.Scatter(x=xvals, y=yvals, mode='markers+text', textposition='bottom center', text=plot_df[tcol], marker_symbol='circle', 
                        textfont=dict(family="Verdana", size=plt['text_font_size'], color="#a19d9d"),                             
                                hovertemplate=
                                    "Country: %{text}<br>" +
                                    "AI Adoption: %{x}<br>" +
                                    "# Data Science Jobs: %{y}",
                        marker = dict(size=svals, color=cvals, opacity=1, colorscale='rdbu_r', showscale=True, colorbar=dict(title='Adoption'), line=dict(width=2, color='#fff'))
                       )
    
    layout = go.Layout(title={'text' : "More Data / AI related Jobs in Developed Countries and more AI Adoption Levels<br><sup>A moderate positive correlation in the adoption and number of jobs</sup>", 
                                'x':0.5, 'xanchor': 'center'}, width=1050, height=750, plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend = False, font = {"color" : '#f27866'})
    fig = go.Figure(data = [trace1], layout = layout)
    
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='red'), 
                     title=plt['xtitle'], title_font=dict(size=14, family='Verdana', color='red'), showgrid=False, gridwidth=1, 
                     ticks="inside", tickcolor='red')
    fig.update_yaxes(showline=False, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='red'), 
                     title=plt['ytitle'], title_font=dict(size=14, family='Verdana', color='red'), showgrid=False, gridwidth=1, nticks=10, 
                    ticks="inside", tickcolor='red', type='log')

    
    text_cfg = {
        "texts" : [], # ["Seedling", "Vegetative", "Budding", "Flowering", "Ripening"],
        "text_xpos" : [0.086, 0.28, 0.51, 0.75, 0.94],
        "img_xpos" : [0.12, 0.33, 0.513, 0.72, 0.91],
        "imgs" : [],# ["https://i.imgur.com/9rs2DVF.png", "https://i.imgur.com/WIusNLF.png", "https://i.imgur.com/j4lAWaU.png", "https://i.imgur.com/VuR8AKW.png", "https://i.imgur.com/dAXbEY2.png"],
        'annos' : []
    }
    fig = add_annotations_country(fig, text_cfg)


    fig.add_hline(y=100000, line_width=1, line_dash="dash", line_color="#c9c4c3")
    insight1 = """175,000+ <br>'Data Science' Job Postings"""
    fig.add_annotation(x=0.4, y=5, text=insight1, showarrow=False, yshift=0, font=dict(family="Verdana", size=10, color="#5e5656"))

    fig.add_hline(y=10000, line_width=1, line_dash="dash", line_color="#c9c4c3")
    insight1 = """10,000+ Job Roles<br>with 'Data Science' requirements"""
    fig.add_annotation(x=0.4, y=4, text=insight1, showarrow=False, yshift=0, font=dict(family="Verdana", size=10, color="#5e5656"))

    fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="#c9c4c3")
    insight1 = """<100 <br>'Data Science' Job Postings"""
    fig.add_annotation(x=0.3, y=2, text=insight1, showarrow=False, yshift=0, font=dict(family="Verdana", size=10, color="#5e5656"))

    fig.show()
    
plt = {
    'ycol' : 'ds_jobs',
    'xcol' : 'AI_Adoption_Index',
    'scol' : 'AI_Adoption_Index',
    'tcol'   : 'Country',

    'text_font_size' : 10,
    'title' : '',
    'ytitle' : 'Data Science Jobs (log scale)',
    'xtitle' : 'AI Adoption',
}
country_df1 = country_df[~country_df['Country'].isin(['Argentina', 'Viet Nam'])]
plot_chart_country2(country_df1, plt)


# ### <p style="font-family: 'Mochiy Pop P One';font-size:20px;color:#f57369">10. Global Innovation Index</p>
# 
# The Global Innovation Index (GII) is developed by World Intellectual Properrty Organization and takes the pulse of the most recent global innovation trends. It ranks the innovation ecosystem performance of economies around the globe each year while highlighting innovation strengths and weaknesses and particular gaps in innovation metrics. The different metrics that the GII offers can be used to monitor performance and benchmark developments against economies within the same region or income group classification. The Global Innovation Index 2021 was released on in September 2021. Switzerland, Sweden, the U.S., and the U.K. continued to lead the innovation ranking, and have all ranked in the top 5 in the past three years. Using this interesting index, we wanted to check if it correlates with our AI adoption index and the relationship between the two. 
# 
# **External Dataset Used:** **[Global Innovation Index 2021](https://www.wipo.int/edocs/pubdocs/en/wipo_pub_gii_2021.pdf)**

# In[11]:


GII_2021 = {'Switzerland': 65.5, 'Sweden': 63.1, 'USA': 61.3, 'UK': 59.8, 'South Korea': 59.3, 'Netherlands': 58.6, 'Finland': 58.4, 'Singapore': 57.8, 'Denmark': 57.3, 'Germany': 57.3, 'France': 55.0, 'China': 54.8, 'Japan': 54.5, 'HK': 53.7, 'Israel': 53.4, 
            'Canada': 53.1, 'Iceland': 51.8, 'Austria': 50.9, 'Ireland': 50.7, 'Norway': 50.4, 'Estonia': 49.9, 'Belgium': 49.2, 'Luxembourg': 49.0, 'Czech Republic': 49.0, 'Australia': 48.3, 
            'New Zealand': 47.5, 'Malta': 47.1, 'Cyprus': 46.7, 'Italy': 45.7, 'Spain': 45.4, 'Portugal': 44.2, 'Slovenia': 44.1, 'UAE': 43.0, 'Hungary': 42.7, 'Bulgaria': 42.4, 'Malaysia': 41.9, 'Slovakia': 40.2, 'Latvia': 40.0, 
            'Lithuania': 39.9, 'Poland': 39.9, 'Turkey': 38.3, 'Croatia': 37.3, 'Thailand': 37.2, 'Viet Nam': 37.0, 'Russia': 36.6, 'India': 36.4, 'Greece': 36.3, 'Romania': 35.6, 'Ukraine': 35.6, 'Montenegro': 35.4, 
            'Philippines': 35.3, 'Mauritius': 35.2, 'Chile': 35.1, 'Serbia': 35.0, 'Mexico': 34.5, 'Costa Rica': 34.5, 'Brazil': 34.2, 'Mongolia': 34.2, 'North Macedonia': 34.1, 
            'Iran': 32.9, 'South Africa': 32.7, 'Belarus': 32.6, 'Georgia': 32.4, 'Republic of Moldova': 32.3, 'Uruguay': 32.2, 'Saudi Arabia': 31.8, 'Colombia': 31.7, 'Qatar': 31.5, 
            'Armenia': 31.4, 'Peru': 31.2, 'Tunisia': 30.7, 'Kuwait': 29.9, 'Argentina': 29.8, 'Jamaica': 29.6, 'Bosnia and Herzegovina': 29.6, 'Oman': 29.4, 'Morocco': 29.3, 'Bahrain': 28.8, 
            'Kazakhstan': 28.6, 'Azerbaijan': 28.4, 'Jordan': 28.3, 'Brunei Darussalam': 28.2, 'Panama': 28.0, 'Albania': 28.0, 'Kenya': 27.5, 'Uzbekistan': 27.4, 'Indonesia': 27.1, 'Paraguay': 26.4, 
            'Cabo Verde': 25.7, 'United Republic of Tanzania': 25.6, 'Ecuador': 25.4, 'Lebanon': 25.1, 'Dominican Republic': 25.1, 'Egypt': 25.1, 'Sri Lanka': 25.1, 'El Salvador': 25.0, 
            'Trinidad and Tobago': 24.8, 'Kyrgyzstan': 24.5, 'Pakistan': 24.4, 'Namibia': 24.3, 'Guatemala': 24.1, 'Rwanda': 23.9, 'Tajikistan': 23.9, 'Bolivia (Plurinational State of)': 23.4, 
            'Senegal': 23.3, 'Botswana': 22.9, 'Malawi': 22.9, 'Honduras': 22.8, 'Cambodia': 22.8, 'Madagascar': 22.5, 'Nepal': 22.5, 'Ghana': 22.3, 'Zimbabwe': 21.9, 'Côte d’Ivoire': 21.0, 
            'Burkina Faso': 20.5, 'Bangladesh': 20.2, 'Lao People’s Democratic Republic': 20.2, 'Nigeria': 20.1, 'Uganda': 20.0, 'Algeria': 19.9, 'Zambia': 19.8, 'Mozambique': 19.7, 
            'Cameroon': 19.7, 'Mali': 19.5, 'Togo': 19.3, 'Ethiopia': 18.6, 'Myanmar': 18.4, 'Benin': 18.0, 'Niger': 17.8, 'Guinea': 16.7, 'Yemen': 15.4, 'Angola': 15.0}

country_df['GII_2021'] = country_df['Country'].apply(lambda x : GII_2021[x] if x in GII_2021 else "NA")


def plot_chart_country1(plot_df, plt):
    ycol, xcol, scol, tcol = plt['ycol'], plt['xcol'], plt['scol'], plt['tcol']

    
    xvals, yvals = list(plot_df[xcol]), list(plot_df[ycol])
    svals = [_*30 for _ in list(plot_df[scol])]
    cvals = list(plot_df[scol])
    
    trace1 = go.Scatter(x=xvals, y=yvals, mode='markers+text', textposition='bottom center', text=list(plot_df[tcol]), marker_symbol='circle', 
                        textfont=dict(family="Verdana", size=plt['text_font_size'], color="red"),
                                hovertemplate=
                                    "Country: %{text}<br>" +
                                    "AI Adoption: %{x}<br>" +
                                    "Global Innovation Index: %{y}",

                        marker = dict(size=svals, color=cvals, opacity=1, colorscale='rdbu_r', showscale=True, colorbar=dict(title='Adoption'), line=dict(width=2, color='#fff'))
                       )
    
    layout = go.Layout(title={'text' : "Moderate Positive (0.6) Correlation with Global Innovation Index<br><sup>Switzerland Wins in both the categories</sup>", 
                                'x':0.5, 'xanchor': 'center'}, width=1050, height=800, plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend = False, font = {"color" : '#f27866'})
    fig = go.Figure(data = [trace1], layout = layout)
    
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='red'), 
                     title=plt['xtitle'], title_font=dict(size=14, family='Verdana', color='red'), showgrid=False, gridwidth=1, 
                     ticks="inside", tickcolor='red')
    fig.update_yaxes(showline=False, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='red'), 
                     title=plt['ytitle'], title_font=dict(size=14, family='Verdana', color='red'), showgrid=False, gridwidth=1, nticks=10, 
                    ticks="inside", tickcolor='red', range = (15,75))

    
    text_cfg = {
        "texts" : ["Seedling", "Vegetative", "Budding", "Flowering", "Ripening"],
        "text_xpos" : [0.086, 0.28, 0.51, 0.75, 0.94],
        "img_xpos" : [0.12, 0.33, 0.513, 0.72, 0.91],
        "imgs" : [ "https://i.imgur.com/9rs2DVF.png", "https://i.imgur.com/WIusNLF.png", "https://i.imgur.com/j4lAWaU.png", "https://i.imgur.com/VuR8AKW.png", "https://i.imgur.com/dAXbEY2.png"],
        'annos' : [
                    { 'x' : 0.99, 'y' : 46, 'text' : "Europe", 'ax' : 0, 'ay' : 0},  
                    { 'x' : 0.76, 'y' : 57, 'text' : "Asian Tigers", 'ax' : 0, 'ay' : 0}
                  ]
    } 
    fig = add_annotations_country(fig, text_cfg)

    fig.add_vline(x=0.4, line_width=1, line_dash="dash", line_color="#c9c4c3")

    insight1 = """77% with Masters, PHDs<br>30% Scientists & Researchers"""
    fig.add_annotation(x=1, y=42, text=insight1, showarrow=False, yshift=0, font=dict(family="Verdana", size=10, color="#5e5656"))

    insight1 = """Moderate Positive<br>Correlation : 0.6"""
    fig.add_annotation(x=0.46, y=60, text=insight1, showarrow=False, yshift=0, font=dict(family="Verdana", size=10, color="#5e5656"))

    insight1 = """70% with 0-5Y Coding Experience<br>21% with 20+ data team size"""
    fig.add_annotation(x=0.65, y=66.5, text=insight1, showarrow=False, yshift=-50, font=dict(family="Verdana", size=10, color="#5e5656"))

    insight1 = """Low Innovation Index<br>But high ML Adoption<br>(50% are startups; More Cloud Usage)"""
    fig.add_annotation(x=0.82, y=25, text=insight1, showarrow=False, yshift=-50, font=dict(family="Verdana", size=10, color="#5e5656"))
    
    fig.add_shape(type="rect", x0=0.7, y0=16, x1=0.74, y1=24, line=dict(color="#cccac4", width=2, dash='dash' ))
    fig.add_shape(type="rect", x0=0.73, y0=40, x1=0.77, y1=45, line=dict(color="#cccac4", width=2, dash='dash' ))

    fig.show()
    
plt = {
    'ycol' : 'GII_2021',
    'xcol' : 'AI_Adoption_Index',
    'scol' : 'AI_Adoption_Index',
    'tcol'   : 'Country',

    'text_font_size' : 10,
    'title' : '',
    'ytitle' : 'Global Innovation Index (2021)',
    'xtitle' : 'AI Adoption Index',
}
country_df1 = country_df[~country_df['Country'].isin(['Uganda', 'Algeria', 'Morocco', 'Ukraine', 'South Africa'])]
plot_chart_country1(country_df1, plt)

# selects = ['Switzerland', 'USA', 'UK', 'Germany', 'Netherlands', 'France', 'Austria', 'Norway']
# selects = ['South Korea', 'Singapore', 'Japan', 'HK', 'China']

# c = 'Q22'
# xx = country_df[country_df['Country'].isin(selects)].mean().reset_index()
# xx = raw_df[raw_df['Q3'].isin(selects)]
# xx = xx[c].value_counts().reset_index()
# xx['p'] = xx[c].apply(lambda x : x / np.sum(xx[c]))
# xx = xx.sort_values("p", ascending = False)
# xx


# <br>
# 
# <div align="center"><span class='alert-info' style='padding:20px;color:#222'>&#x25cf; &nbsp;&nbsp; Switzerland - Most Innovative Country, Highest AI Adoption, Most models in Production !! </span></div>
# 
# <br>
# 
# It is interesting to note that the two indexes show a significant positive correlation, and also the winners of both indexes are the same - Switzerland. The swiss nation is pushed to focus on research and development and has invested in the establishment and funding of a high number of world-renowned universities, which in turn attract multinationals seeking highly qualified employees. In terms of the AI adoption index, Switzerland has:  
# 
# - Very high usage (75%) of Cloud Platforms (GCP, AWS)
# - Very large number of Master's Degree holders (46%)
# - Very large number of PHDs (26%)
# - Large number of individuals with 10+ years of coding experience (44%)
# - The highest percentage of Organizations with Models in production (49%) 
# 
# Other European countries had similar insights - Norway, Austria, France, Netherlands, Germany, along with UK and USA. 
# 
# - United Arab Emirates shows good progress and advancements in terms of both innovation and data science first. More than 50% mentioned they have about 0 - 2 years of the machine learning experience. Mostly mid-sized companies with employee size <250 have been prominent (about 46%). Nepal and Ethiopia are appealing as the innovation index is low but the AI adoption level is better. This is because of the startup culture, startup investments, and people building their data science skills through multiple channels. In these countries, more than 50% of respondents are from non-multinational companies. 
# 
# <br>
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:18px;color:#f57369">11. Competitive Data Science (Top 1000 Kagglers)</p>
# 
# We looked at countries by the number of kagglers in the top 1000 competition ranks. The idea was to understand the presence of skilled people active in competitive data science (by taking kaggle rankings as the proxy). We understand that identifying the numbers by population can give a relative percentage, but we restricted the analysis to absolute counts, to understand which countries lead and is there any correlation between the two. 

# In[12]:


kagglers = {'Japan': 232, 'USA': 123, 'China': 101, 'Russia': 81, 'India': 69, 'Germany': 34, 'France': 29, 'South Korea': 23, 'UK': 23, 'Viet Nam': 21, 'Spain': 17, 'Ukraine': 16, 'Singapore': 15, 'Canada': 14, 'Brazil': 10, 'Taiwan': 9, 'Poland': 8, 
            'Australia': 7, 'Belarus': 7, 'Greece': 7, 'Netherlands': 7, 'Austria': 6, 'Bangladesh': 6, 
            'Turkey': 6, 'Belgium': 5, 'Ireland': 5, 'Italy': 5, 'Malaysia': 5, 'Switzerland': 5, 'HK': 4, 
            'Norway': 4, 'Israel': 3, 'Kazakhstan': 3, 'Pakistan': 3, 'Romania': 3, 'Thailand': 3, 
            'Argentina': 2, 'Egypt': 2, 'Hungary': 2, 'Indonesia': 2, 'Latvia': 2, 'Morocco': 2, 'Portugal': 2, 
            'South Africa': 2, 'Sweden': 2, 'Bulgaria': 1, 'Chile': 1, 'Christmas Island': 1, 'Croatia': 1, 
            'Denmark': 1, 'Finland': 1, 'Lithuania': 1, 'New Zealand': 1, 'Slovenia': 1}
country_df['kagglers'] = country_df['Country'].apply(lambda x : kagglers[x] if x in kagglers else 0)
country_df['kagglers_log']  = np.log(country_df['kagglers'] )


def plot_chart_country3(plot_df, plt):
    ycol, xcol, scol, tcol = plt['ycol'], plt['xcol'], plt['scol'], plt['tcol']

    xvals, yvals = list(plot_df[xcol]), list(plot_df[ycol])
    svals = [_*30 for _ in list(plot_df[scol])]
    cvals = list(plot_df[scol])
    trace1 = go.Scatter(x=xvals, y=yvals, mode='markers+text', textposition='bottom center', text=plot_df[tcol], marker_symbol='circle', 
                        textfont=dict(family="Verdana", size=plt['text_font_size'], color="#f27866"),
                                 hovertemplate=
                                    "Country: %{text}<br>" +
                                    "AI Adoption: %{x}<br>" +
                                    "Competitive Data Scientists: %{y}",
                        marker = dict(size=svals, color=cvals, opacity=1, showscale=True, colorbar=dict(title='Adoption'), colorscale='rdbu_r', line=dict(width=2, color='#fff'))
                       )
    
    layout = go.Layout(title={'text' : "Only Japan, USA, and China have 100+ top competitive data scientists<br><sup>Japan leads with 232 Kagglers in top 1000</sup>", 
                                'x':0.5, 'xanchor': 'center'}, width=1050, height=800, plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend = False, font = {"color" : '#f27866'})
    fig = go.Figure(data = [trace1], layout = layout)
    
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='red'), 
                     title=plt['xtitle'], title_font=dict(size=14, family='Verdana', color='red'), showgrid=False, gridwidth=1, 
                     ticks="inside", tickcolor='red')
    fig.update_yaxes(showline=False, linewidth=1, linecolor='#c9c4c3', gridcolor='#c9c4c3', tickfont=dict(size=14, family='Verdana', color='red'), 
                     title=plt['ytitle'], title_font=dict(size=14, family='Verdana', color='red'), showgrid=False, gridwidth=1, nticks=10, 
                    ticks="inside", tickcolor='red', type="log")


    
    text_cfg = {
        "texts" : ["Seedling", "Vegetative", "Budding", "Flowering", "Ripening"],
        "text_xpos" : [0.086, 0.28, 0.51, 0.75, 0.94],
        "img_xpos" : [0.12, 0.33, 0.513, 0.72, 0.91],
        "imgs" : [ "https://i.imgur.com/9rs2DVF.png", "https://i.imgur.com/WIusNLF.png", "https://i.imgur.com/j4lAWaU.png", "https://i.imgur.com/VuR8AKW.png", "https://i.imgur.com/dAXbEY2.png"],
        'annos' : [{ 'x' : 0.72, 'y' : 2.15, 'text' : "232 Kagglers", 'ax' : 0, 'ay' : 0}],

    }
    fig = add_annotations_country(fig, text_cfg)

    insight1 = """Fewer number of kagglers<br>in top 1000 competition rankings<br>Though Higher AI Adoption Levels"""
    fig.add_annotation(x=0.98, y=0.4, text=insight1, showarrow=False, yshift=0, font=dict(family="Verdana", size=10, color="#5e5656"))
    
    insight1 = """>100 Kagglers in top<br>1000 Competition Ranks"""
    fig.add_annotation(x=0.35, y=2.05, text=insight1, showarrow=False, yshift=0, font=dict(family="Verdana", size=10, color="#5e5656"))

    insight1 = """>10 Kagglers in top<br>1000 Competition Ranks"""
    fig.add_annotation(x=0.35, y=1, text=insight1, showarrow=False, yshift=0, font=dict(family="Verdana", size=10, color="#5e5656"))

    fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="#c9c4c3")
    fig.add_hline(y=10, line_width=1, line_dash="dash", line_color="#c9c4c3")
    fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="#c9c4c3")

    fig.show()
    
plt = {
    'ycol' : 'kagglers',
    'xcol' : 'AI_Adoption_Index',
    'scol' : 'AI_Adoption_Index',
    'tcol'   : 'Country',

    'text_font_size' : 10,
    'title' : '',
    'ytitle' : 'Competitive Data Scientists Count',
    'xtitle' : 'AI Adoption',
}
country_df1 = country_df[country_df['kagglers'] > 0]
plot_chart_country3(country_df1, plt)


# Countries like Japan and Russia, despite lower populations than USA and China, have a lot of competitive data scientists or kagglers in the top 1000 in the competition category. It is in this environment that AI Dynamics is finding increased traction in these countries. The government has also played an important role in the improvement of AI adoption over the years. For example, Russia has adopted a new law <mark>(the "AI Law")</mark> establishing a special legal framework for the development and adoption of AI technologies in Moscow for the period from 1 July 2020 to 1 July 2025 (the "AI Framework"). Enforcing this type of Law straight by the Government shows that many country leaders realize the potential of AI adoption and have to make more efforts to remain competitive. 
# 
# It is interesting to note that all the key European countries where AI adoption levels are higher have fewer than 10 kagglers only in the top 1000. <mark>Yes! the Country population does matter</mark>, but again our aim here is to compare the regions in an absolute sense. This group of countries includes - Norway, Switzerland, Austria, Netherlands. 
# 
# <br>
# 
# <p style="font-family: 'Mochiy Pop P One';font-size:24px;color:#f57369">Conclusion</p>
# 
# Our intention for choosing the theme of AI adoption was to understand the current maturity levels of the data science field across industries, sectors, and regions. The chosen theme and the richness of datasets motivated us to create a unified measure (index) that quantifies the state of AI/data science maturity/adoption. This type of index is analogous to the growth stages in plants, where a plant matures through different stages in its lifetime. This can also be measured by the 'hotness' of something, which means something popular has more hotness and something not popular has less hotness. These reasons inspired us to create a HotSpot chart that measures "hotness" by color and shows different stages of plants to reflect the current stage. 
# 
# In summary, Adopters of AI continue to have confidence in their capabilities to drive value and advantage, they are realizing the competitive advantage and expect AI-powered transformation to happen for both their organization and industry, ultimately for their country. Those following the adaption approach need to seriously consider shifting to adopting side. By following the right practices and aligning them with current trends, current and future AI adopters can place themselves not just to survive but to flourish in the emerging era of pervasive AI.
# 
# From the overall analysis and our research during this analysis, one fact became evident; Beyond the pandemic, the adoption of data science is anticipated to grow both horizontally and vertically. It is likely to move from fast adopters to less tech-focused industries, functions, and geographies in 2022 and onwards. To keep up to date in this period of change and transformations, organizations and leaders need to think of the right strategies to incorporate a data-first culture. While organizations are aware of the importance of adopting this culture, they often fail to approach it from a strategic standpoint. It is important to understand that doing data science to be cool will be disappointing. One needs to have a proper adoption plan along with a purpose.
# 
# To overcome this, organizations definitely need ***K.A.G.G.L.E.*** 
# 
# <font size=4 color='#f57369'><b>K:</b></font> Knowledgable and Skilled Workforce (Skills)   
# <font size=4 color='#f57369'><b>A:</b></font> Adoption of a data-first, data-driven culture (Data)      
# <font size=4 color='#f57369'><b>G:</b></font> Growth-Oriented technology (Strategy)     
# <font size=4 color='#f57369'><b>G:</b></font> Governance and Compliance (Governance)   
# <font size=4 color='#f57369'><b>L:</b></font> Leadership that values a change (Strategy)       
# <font size=4 color='#f57369'><b>E:</b></font> Experienced and Educated Professionals (People)
# 
# <br>
# <p style="font-family: 'Mochiy Pop P One';font-size:24px;color:#f57369">References</p>
# 
# - [1] [SaferBrand Plant Gorwth Stages](https://www.saferbrand.com/articles/plant-growth-stages)
# - [2] [Stanford Human Centered Intelligence](https://hai.stanford.edu/news/how-has-covid-affected-ai-economy)
# - [3] [ElementAI - AI Maturity Framework](https://s3.amazonaws.com/element-ai-website-bucket/AI-Maturity-Framework_White-Paper_EN.pdf)
# - [4] [Oxford Insights AI Readiness Report 2020](https://static1.squarespace.com/static/58b2e92c1e5b6c828058484e/t/5f7747f29ca3c20ecb598f7c/1601653137399/AI+Readiness+Report.pdf)
# - [5] [Sopra Banking Software - Which Approach is best for your business?](https://www.soprabanking.com/insights/adopt-vs-adapt-which-approach-is-best-for-your-business/)
# - [6] [World Global Intellectual Property Organization - Global Innvoation Index 2021](https://www.wipo.int/global_innovation_index/en/2021/)
# - [7] [Dataiku - 5 Elements of Enterprise AI, Insurance Industry](https://www.dataiku.com/stories/how-ai-empowers-the-insurance-industry/)
# - [8] [HBR - AI Adoption in last 18 Months](https://hbr.org/2021/09/ai-adoption-skyrocketed-over-the-last-18-months)
# - [9] [Deloitte AI Adoption Insights](https://www2.deloitte.com/us/en/insights/focus/cognitive-technologies/state-of-ai-and-intelligent-automation-in-business-survey.html)
# - [10] [Tech Fundings in Pandemic](https://theconversation.com/tax-pandemic-profiteering-by-tech-companies-to-help-fund-public-education-155705)
# 

# In[13]:


## !! ThankYou !! ##

