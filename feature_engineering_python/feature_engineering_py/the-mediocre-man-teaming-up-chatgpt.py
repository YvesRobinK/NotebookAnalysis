#!/usr/bin/env python
# coding: utf-8

# Published on June 21, 2023. By Marilia Prata, MPwolke.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <center style="font-family:verdana;"><h1 style="font-size:100%; padding: 20px; background: #90EE90;"><i><b style="color:#006400;">El Hombre Mediocre and ChatGPT teaming up, dreaming up.</b></i></h1></center>

# One hundred and ten years separate The Mediocre Man, book by José Ingenieros published in 1913 and ChatGPT (stable released May 24, 2023).
# 
# I think that many of Ingenieros concepts are still updated and can be applied to our "AI times". Let's check some similarities with ChatGPT's answers.
# 
# Getting ChatGPT's anwers was pretty quick since I've already thought about some questions before. Though our questions should be precise and specific to receive better results.
# 
# What required some time was to read the book, extract some points that are related to this Kaggle Challenge, or just to Kaggle since the topic "Others" could be applied to "anything that does not fall into any other category".
# 
# Since the book was written in spanish, I'm very thankful to Google translate.
# 
# My first thought was to apply the Bard, however that Chatbot is still Not available in my country.
# 
# Sorry Bard, but timing is everything. Maybe, we could team-up on the next time.
# 
# Team up. Dream up!

# #The adaptation of each individual to his environment depends on the balance between what he imitates and what he invents.
# 
# 
# ![](https://pbs.twimg.com/media/BDmSRv1CcAEBeiW.jpg)https://twitter.com/Sandra_Corona/status/304435283410186241

# #Virtue and Honesty 

# <div class="alert alert-block alert-success">
#   <h1 class="display-4">Virtue and Honesty by José Ingenieros - El Hombre Mediocre</h1>
#   <p class="lead">"Honesty is not a virtue, although it is not a vice either. You can be honest without feeling a bit of perfection; enough for it not to display evil, which is not enough to be virtuous. Between vice, which is a scourge, and virtue, which is an excellence, honesty fluctuates." </p>
#   <p>"Virtu rises above ordinary morality: it implies a certain aristocracy of the heart, proper to moral talent; the virtuoso anticipates some form of future perfection and sacrifices to it the automatisms consolidated by habit."</p>
#   <p>"The honest, on the other hand, is passive, a circumstance that assigns a higher moral level to the vicious, although he remains below the one who actively practices some virtue and guides his life some ideal."</p>  
#   <p>"There is no virtue when the actions belie the words, nor is there nobility where the intention creeps."</p>
# </div>

# #What's the difference between virtue and honesty? By ChatGPT
# 
# "Virtue and honesty are related concepts, but they have distinct meanings and can be understood differently depending on the context. Here's a breakdown of the difference between virtue and honesty:"
# 
# "Virtue: Virtue refers to a moral excellence or goodness of character. It encompasses a set of positive qualities, principles, and behaviors that are considered morally upright and desirable in a person. Virtues are often seen as universal ethical standards that guide individuals in making ethical choices and leading a good life. Examples of virtues include honesty, kindness, courage, justice, humility, and patience. While honesty is one virtue, it is not the only virtue and is part of a broader set of moral values."
# 
# "Honesty: Honesty, on the other hand, is a specific virtue that relates to truthfulness, sincerity, and integrity. It involves being truthful in one's words, actions, and dealings with others. Honesty implies refraining from lying, cheating, or deceiving others, and it involves presenting information and expressing oneself in a sincere and genuine manner. Honest individuals are transparent, trustworthy, and have a commitment to truthfulness. While honesty is a highly regarded virtue, it is just one aspect of an individual's overall moral character."
# 
# "In summary, honesty is a virtue that emphasizes truthfulness and integrity in one's actions and interactions, while virtue is a broader concept encompassing a range of positive moral qualities that guide individuals in leading a good and ethical life."
# 
# https://chat.openai.com/?model=text-davinci-002-render-sha

# #The Aurea mediocritas 

# <div class="alert alert-block alert-success">
#   <h1 class="display-4">The Aurea mediocritas - The Golden Mean (Golden middle way) by José Ingenieros</h1>
#   <p class="lead">"The poet Horace did not say Aurea mediocritas in the general and absurd sense that those unable to stand out for their ingenuity, for their virtues or for their works proclaim."</p>
#   <p>"Another was the opinion of the poet: placing the greatest well-being of man in tranquility and independence, he exalted the joys of a simple life that is equally far from opulence and misery, calling that material mediocrity golden."</p>
#   <p>"In a certain epicurean sense, his dictum is true and confirms the remote Arabian proverb: "Tranquil mean well-being is preferable to carefree opulence."</p>  
#   <p>"Mediocribus esse poetis
#  Non di, non homines, non concessere columnae"</p>
# </div>

# ![](https://s2.studylib.es/store/data/000021701_1-075eba44a9c9c83d4b40921a443c35dd-768x994.png)https://studylib.es/doc/21701/documentaci%C3%B3n-sanitaria

# #What is the Golden Mean (Aurea Mediocritas)? By ChatGPT
# 
# "Golden Mean, also known as the Golden Ratio or Divine Proportion, is a mathematical concept that has been recognized and appreciated since ancient times. It is often represented by the Greek letter phi (Φ) and has a numerical value of approximately 1.6180339887."
# 
# "The Golden Mean is derived from a specific geometric ratio. It occurs when the ratio of the sum of two quantities to the larger quantity is equal to the ratio of the larger quantity to the smaller one. Mathematically, this can be expressed as (a + b) / a = a / b, which simplifies to a / b = (sqrt(5) + 1) / 2."
# 
# "What makes the Golden Mean fascinating is its appearance in various natural and aesthetic contexts. It is found in architecture, art, nature, and even human anatomy. The proportions of the Parthenon in Athens and Leonardo da Vinci's Vitruvian Man, for example, are believed to have been influenced by the Golden Mean."
# 
# "In art and design, the Golden Mean is often used to create aesthetically pleasing compositions. It is believed that dividing a line or a rectangle into two sections, with the ratio of the larger section to the smaller one being approximately 1.618, produces a visually appealing result."
# 
# "The concept of the Golden Mean extends beyond mathematics and aesthetics. It has been associated with balance, harmony, and beauty. Philosophically, it has been linked to the idea of moderation and finding a middle ground between extremes. The Golden Mean is a concept that has captured the imagination of many throughout history, and its influence can still be seen in various fields today."
# 
# https://chat.openai.com/
# 
# ChatGPT provided many concepts about the Golden Mean, maybe it's my fault since I should have been more specific writing that I intended the Philosophical meaning only.
# 
# Besides, Leonardo da Vinci is always there too. 

# #Da Vinci, L.

# <div class="alert alert-block alert-success">
#   <h1 class="display-4">FOUR years ago, on Da Vinci's 500th anniversary of death, I wrote a topic: Da Vinci, L. - No explanation needed.</h1>
#   <p class="lead">"Genius, talented in many fields: engineering, physics, mathematics, architecture, medicine and besides he "paints". With great creativity and huge ability he blended ("The" blend of the blends) science and art like no other human being. He recorded everything in notebooks (drafts, thoughts, emotions and reflections). Some notes were written using codes and even backwards."</p>
#   <p>"Thanks for sharing Leo all your incomparable work. Congrats. Leonard, you're still alive!"</p>          <p>"https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadshamperingcsv/discussion/113890"</p>  
#   <p>"Now, look what's written on ChatGPT's URL (davinci). Great choice ChatGPT, since daVinci connects everything to everyrthing else!"</p>
# </div>

# #Why davinci was included on your chat url?
# 
# ChatGPT answer:
# 
# "I apologize for any confusion caused by the reference to "DaVinci" in the chat URL. The term "DaVinci" was the codename used by OpenAI during the development and testing of the GPT-3 model, which powers the ChatGPT system. It is not directly related to Leonardo da Vinci, the renowned Italian polymath. The naming choice was simply an internal reference used by OpenAI during the model's development and training phases."
# 
# https://chat.openai.com/?model=text-davinci-002-render-sha

# #The Vanity and the Envy

# <div class="alert alert-block alert-success">
#   <h1 class="display-4">The vanity and the envy by José Ingenieros</h1>
#   <p class="lead">"The praise of merit is a stimulus for its simulation."</p>
#   <p>"Obsessed with success, and unable to dream of glory, many powerless boast of illusory merits and secret virtues that others do not recognize; believe yourselves actors of the human comedy; they enter life by building themselves a stage, large or small, low or culminating, gloomy or luminous."</p>
#   <p>"They live with perpetual concern about the judgment of others over their shadow."</p>  
#   <p>"Everything is relative . If there are merits, pride is a right; if there are none, it is vanity."</p>
#   <p>"Envy (Envidia) is an admiration of men for shadows, of merit for mediocrity. It is the shackle that the failures drag. It is a poisonous humor that flows from the wounds opened by the disappointment of one's own insignificance."</p>  
# </div>

# ![](https://pbs.twimg.com/media/EKpUmJ8WsAAV53Z.jpg)https://twitter.com/ebalino/status/1200865278231482375

# #Define Vanity ChatGPT
# 
# "Vanity is a term used to describe excessive pride, self-admiration, and a high regard for one's appearance, abilities, or achievements. It refers to an inflated sense of self-importance and an excessive desire for admiration and attention from others. Vanity often involves a preoccupation with physical appearance, social status, and the perception of others."
# 
# "A person who is vain tends to focus on their own appearance, achievements, and qualities, often at the expense of others. They may excessively seek validation and compliments from others, and they may go to great lengths to enhance their physical appearance or social standing."
# 
# "Vanity is often associated with superficiality and a lack of genuine self-worth. It can lead individuals to prioritize external validation and material possessions over more meaningful aspects of life such as personal growth, relationships, and inner fulfillment. Excessive vanity can also make a person overly self-centered and neglectful of the needs and feelings of others."
# 
# "While taking pride in oneself and striving for self-improvement are not inherently negative, vanity becomes problematic when it becomes an obsession and distorts one's perception of oneself and others. It can hinder genuine connections and empathy with others, as well as prevent personal growth and a deeper understanding of oneself."
# 
# https://chat.openai.com/
# 
# It's nice since ChatGPT mentioned : " seek validation and compliments from others" just like Ingenieros had written.

# #Dignity and Upvotes

# <div class="alert alert-block alert-success">
#   <h1 class="display-4">Dignity by José Ingenieros</h1>
#   <p class="lead">I'm pretty sure that Ingenieros had not written anything about Upvotes since his book was published on 1913. However, he wrote a Chapter about Dignity.</p>
#   <p>"Dignity, desire for autonomy, leads to reduce dependence on others to the extent of what is essential, always enormous. The precept that heads the manual of Epictetus:"</p>
#   <p>"Showing off through things that do not depend on others, but on being alone, or renouncing to show off."</p>  
#   <p>I'm suffering since I had to apply Google translate to spanish and to french due the many citations found on the book.</p>
# </div>

# #Is it ethical to ask for upvotes for users that you have never interacted before?
# 
# ChatGPT answer:
# 
# "As an AI language model, I don't have personal opinions or beliefs, but I can offer some insights on the topic."
# 
# "The ethics of asking for upvotes can depend on the context and platform in which it occurs. In general, requesting upvotes from users you have never interacted with could be seen as manipulative or against the norms of certain communities."
# 
# "Platforms such as social media websites or online forums typically have guidelines and policies regarding vote manipulation. These guidelines often discourage or prohibit soliciting upvotes, as it can artificially inflate engagement metrics and undermine the integrity of the platform."
# 
# "Moreover, asking for upvotes without establishing a genuine connection or providing meaningful content may be perceived as spam or self-promotion, which can negatively impact the user experience for others."
# 
# "It's important to respect the rules and norms of the community or platform you are participating in and focus on creating valuable contributions that resonate with others. Engaging with others in a meaningful and authentic way is generally a better approach than seeking upvotes solely for personal gain."
# 
# https://chat.openai.com/?model=text-davinci-002-render-sha

# #Soft Skills

# <div class="alert alert-block alert-success">
#   <h1 class="display-4">Soft Skills</h1>
#   <p class="lead">"José Ingenieros didn't mention anything about soft skills. Though he had a lot of that qualities before people started to talk about it."</p>
#   <p>"José Ingenieros (born Giuseppe Ingegnieri, April 24, 1877 – October 31, 1925) was an Argentine physician, pharmacist, positivist philosopher and essayist."</p>
#   <p>"https://en.wikipedia.org/wiki/Jos%C3%A9_Ingenieros"</p>  
#     <p>This Competition requires an essay. Be inspired by an essayist like Ingenieros.</p> 
# </div>

# #Have people from technological field less soft skills then professionals from the other careers?
# 
# ChatGPT answer:
# 
# "It's important to note that generalizations about any field or profession can be misleading, as individuals within a given field can vary greatly in their skill sets and characteristics. While it is true that the technological field often emphasizes technical expertise and specialized knowledge, it does not necessarily mean that professionals in this field inherently possess fewer soft skills compared to professionals in other careers."
# 
# "Soft skills, such as communication, teamwork, empathy, problem-solving, and adaptability, are valuable in any profession. While technical roles may prioritize certain hard skills, the importance of soft skills is increasingly recognized in the technological field as well. The ability to effectively communicate ideas, collaborate with colleagues, understand user needs, and adapt to changing circumstances can greatly enhance the success of technology projects and the overall functioning of teams and organizations."
# 
# "In fact, many successful professionals in the technological field understand the significance of soft skills and actively work on developing them alongside their technical expertise. They recognize that effective communication and collaboration are crucial for building relationships, fostering innovation, and achieving desired outcomes."
# 
# "While some individuals in the technological field may have a stronger inclination towards technical skills, it is not accurate to assume that they have inherently fewer soft skills. The balance between technical and soft skills may vary from person to person, regardless of their field, and can depend on personal attributes, experiences, and professional development efforts."
# 
# https://chat.openai.com/?model=text-davinci-002-render-sha

# #About Donations to sources?

# 
# 
# <div class="alert alert-block alert-success">
#     
# Since Wikipedia is asking for donations and ChatGPT applies Wikipedia sources, I asked if 
# it had donated for that Platform.
#     
# </div>

# #Have you donated to wikipedia?
# 
# ChatGPT answer:
# 
# "As an AI language model, I don't have the ability to donate to organizations or engage in financial transactions. I don't possess personal funds or the capacity to make donations. My purpose is to provide information and assist users with their inquiries to the best of my abilities. However, if you are interested in making a donation to Wikipedia, you can visit their website and find information on how to contribute and support their mission."
# 
# https://chat.openai.com/?model=text-davinci-002-render-sha

# #Success

# 
# 
# <div class="alert alert-block alert-success">
#     
# Success has many definitions depending on each one point-of-view. Let's check what's a Kaggler beginner success by ChatGPT. And how it could be achieved.    
# 
# 
# </div>

# #How could a beginner be successful on Kaggle platform?
# 
# ChatGPT answer:
# 
# "Becoming successful on the Kaggle platform as a beginner can be a rewarding journey. Here are some steps to help you get started and increase your chances of success:"
# 
# "Understand Kaggle: Familiarize yourself with the Kaggle platform, its competitions, and its community. Explore the competition guidelines, rules, and datasets available. Read through the forums, tutorials, and documentation to gain insights into how things work on Kaggle."
# 
# "Learn Data Science and Machine Learning: Invest time in learning the foundations of data science and machine learning. Understand key concepts, algorithms, and techniques used in the field. There are numerous online courses, tutorials, and resources available, such as Kaggle's Learn platform, which provides interactive courses and notebooks to get you started."
# 
# "Participate in Competitions: Start with beginner-level competitions to gain experience. Read the competition objectives, understand the evaluation metrics, and explore the data provided. Work on building and iterating on models to improve your performance. Don't be discouraged by initial results; learning from failures is a part of the process."
# 
# "Collaborate and Learn from Others: Engage with the Kaggle community through forums, discussions, and code sharing. Collaborate with other participants, ask questions, and seek feedback. Learning from experienced Kagglers and understanding their approaches can be valuable in improving your skills."
# 
# "Practice Exploratory Data Analysis (EDA): EDA is an essential step in any data science project. Learn to explore, visualize, and understand the dataset you are working with. Gain insights into the data distribution, patterns, and relationships that can guide feature engineering and model building."
# 
# "Feature Engineering and Model Selection: Experiment with different feature engineering techniques to extract meaningful information from the data. Explore various models and algorithms suitable for the problem at hand. Learn how to tune hyperparameters and optimize model performance."
# 
# "Share and Document Your Work: Document your approach, code, and insights in notebooks. Share your work with the community, receive feedback, and contribute back to the knowledge pool. Effective communication and sharing can enhance your reputation and help others learn from your experiences."
# 
# "Continuous Learning: Data science is a rapidly evolving field. Stay updated with the latest techniques, algorithms, and tools. Participate in Kaggle discussions, read research papers, and join data science communities outside of Kaggle to broaden your knowledge."
# 
# "Remember, success on Kaggle is not just about winning competitions but also about learning, improving, and building your data science skills. Embrace the challenges, be persistent, and enjoy the process of learning and collaborating with the Kaggle community."
# 
# https://chat.openai.com/?model=text-davinci-002-render-sha

# #Competition topic: Tabular and/or Time series

# <div class="alert alert-block alert-success">
#   <h1 class="display-4">Approaching Competition topic: Tabular and/or Time series data</h1>
#   <p class="lead">One of the topics of this AI 2023 Competition is Tabular and/or Time series data.</p>
#   <p>I asked ChatGPT just an Abstract, though it delivered all the steps that a study should have. It sounds great for anyone to start with that topic.</p>
#     
# </div>

# #How to make an abstract about Tabular and/or time series data?
# 
# ChatGPT answer:
# 
# To create an abstract about tabular and/or time series data, you can follow these general guidelines:
# 
# Introduction: Begin by introducing the topic of tabular and/or time series data. Explain their relevance and importance in various fields, such as finance, healthcare, and business.
# 
# Problem Statement: Clearly define the problem or objective you are addressing with the data analysis. Highlight the challenges or limitations associated with tabular and time series data analysis in relation to the specific problem.
# 
# Methodology: Describe the techniques and methodologies employed to analyze and process the data. This could include data preprocessing, feature engineering, statistical analysis, machine learning algorithms, or time series analysis techniques such as autoregressive models or recurrent neural networks.
# 
# Data Description: Provide a brief overview of the tabular or time series data used in the analysis. Mention the size of the dataset, the number and types of variables or features, and any data sources or collection methods.
# 
# Results and Findings: Summarize the key findings and insights obtained from the data analysis. Highlight any patterns, trends, or correlations discovered in the data. If applicable, discuss the performance metrics or evaluation measures used to assess the accuracy or effectiveness of the analysis.
# 
# Conclusion: Conclude the abstract by summarizing the main contributions or implications of the study. Discuss the potential applications or significance of the findings in the context of the problem statement. You can also mention any limitations or areas for future research.
# 
# Remember to keep the abstract concise, clear, and focused. Use precise language and avoid unnecessary technical jargon to ensure that readers from various backgrounds can understand the essence of your work.
# 
# https://chat.openai.com/

# #Be Precise. ChatGPT simulating MPwolke on Kaggle!

# <div class="alert alert-block alert-success">
#   <h1 class="display-4">While writing on ChatGPT's Prompt be Cristal Clear</h1>
#   <p class="lead">Be precise on your questions. The more specific material that you write on ChatGPT's Prompt, the better will be the output delivered by the Bot.</p>
#   <p>If I incorrectly direct the request to something I already know, the answer may sounds weird.</p>  
# </div>

# #I asked ChatGPT to create a text that simulated Marilia Prata (mpwolke) on Kaggle
# 
# "Hello Kaggle community!"
# 
# "It's Marilia Prata here, excited to be part of this amazing platform once again. For those of you who may not know me, I am a passionate data scientist and researcher with a strong background in machine learning and artificial intelligence."
# 
# "Throughout my journey on Kaggle, I've had the privilege of working on a wide range of projects, tackling diverse problem domains and exploring cutting-edge techniques. Whether it's predictive modeling, natural language processing, computer vision, or deep learning, I'm always eager to dive into new challenges."
# 
# "One of the aspects I love most about Kaggle is the vibrant community. The opportunity to collaborate, learn, and exchange ideas with like-minded data enthusiasts from all around the world is truly inspiring. The collective knowledge and experience we have here are invaluable, and I am constantly amazed by the innovative solutions and insights that emerge from our collaborations."
# 
# "As a Kaggle expert, I strive to share my expertise and contribute back to the community that has given me so much. You can find my notebooks and kernels covering a wide array of topics, from beginner-friendly tutorials to advanced techniques, all aimed at helping others learn and grow in their data science journey."
# 
# "When it comes to competitions, I'm always up for a challenge. The thrill of diving into a new dataset, designing robust models, and optimizing them to achieve top-notch performance is what keeps me hooked. It's not just about the leaderboard rankings but also about the process of continuous improvement and the invaluable lessons learned along the way."
# 
# "Beyond competitions, I believe in the power of open-source collaboration. I actively contribute to Kaggle datasets and discussions, always eager to share my knowledge and learn from others. Data science is a constantly evolving field, and staying connected to the community is crucial to keep up with the latest advancements and best practices."
# 
# "So, whether you're a beginner just starting your data science journey or an experienced practitioner looking for new challenges, feel free to reach out to me. Let's explore the vast world of data together, exchange ideas, and push the boundaries of what's possible."
# 
# "Happy Kaggling!"
# 
# "Marilia Prata"

# #Thoughts after "training" ChatGPT

# <div class="alert alert-block alert-success">
#   <h1 class="display-4">My thoughts about spending some time "training" ChatGPT</h1>
#   <p class="lead">I'm not a Data Scientist, Researcher or whatever on this field. Strong Background on ML and AI?? I don't think so. I don't even like epochs and they don't like me either.</p>
#   <p>That's true: I'm always eager to dive into new challenges. Curiousity to learn could be my Middle Name.</p>
#   <p>Kaggle Expert? I'm not any longer a Kaggle Expert. Though, I'm still a beginner in this whole field. Robust models??? Me? What is it talking about? </p>  
#   <p>Indeed, it's mostly about the process of continuous improvement and the invaluable lessons learned along the way.</p>
#   <p>Wow! "Feel free to reach out to me". Oh Boy! I'm so screwed! If you reach me out, you risk to Unlearn Data Science.</p>
#   <p>It was fun to work with that tool. Mostly if I want to deliver something fast, without searching or reflecting much and even learning.</p>  
# </div>

# #Last words still by José Ingenieros
# 
# "Feeling with the other's heart is equivalent to thinking with a child's head."

# #There are no equal men, there is an inferior man, a mediocre man, and a superior man.
# 
# #Almost a tier position and ranking of mediocrity.
# 
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSthYE9_VhNlujPZJEH9GL4DfMiWzjp3qQFyw&usqp=CAU)https://m.facebook.com/Sociedadefilosofiaplicada/photos/extractos-de-el-hombre-mediocre-1913-de-jos%C3%A9-ingenieros-18771925-la-obra-trata-s/1658837140858203/

# #Acknowledgements:
# 
# Google translate
# 
# HTML:
# Marco Vasquez https://www.kaggle.com/code/marcovasquez/useful-html-for-jupyter-notebook

# In[2]:


submission=pd.read_csv("/kaggle/input/2023-kaggle-ai-report/sample_submission.csv")
submission.head()


# In[3]:


submission.loc[0]['value']='Others'
submission.loc[1]['value']='https://www.kaggle.com/code/mpwolke/the-mediocre-man-teaming-up-chatgpt'
submission.loc[2]['value']='https://www.kaggle.com/code/anthony35813/ai-report-a-first-look-at-the-data-for-text-data/comments'
submission.loc[3]['value']='https://www.kaggle.com/code/bacharacherif/arxiv-evaluation-papers/comments'
submission.loc[4]['value']='https://www.kaggle.com/code/jonasaacampos/artigos-ethics/comments'
submission.head()


# In[4]:


submission.to_csv('submission.csv',index = False)


# In[5]:


#type,value
#essay_category,'Others'
#essay_url,'https://www.kaggle.com/code/mpwolke/the-mediocre-man-teaming-up-chatgpt'
#feedback1_url,'https://www.kaggle.com/code/anthony35813/ai-report-a-first-look-at-the-data-for-text-data/comments'
#feedback2_url,'https://www.kaggle.com/code/bacharacherif/arxiv-evaluation-papers/comments'
#feedback3_url,'https://www.kaggle.com/code/jonasaacampos/artigos-ethics/comments'

