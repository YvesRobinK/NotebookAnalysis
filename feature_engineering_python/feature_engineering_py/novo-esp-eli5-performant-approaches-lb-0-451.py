#!/usr/bin/env python
# coding: utf-8

# <br>
# 
# <br><center><img src="https://www.ingredientsnetwork.com/NZ_Secondary_Purple_RGB-comp249150.jpg" width=100%></center>
# 
# <br><br>
# 
# <h2 style="text-align: center; font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: underline; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><font color="blue">E</font>XPLAIN IT <font color="blue">L</font>IKE <font color="blue">I</font>'M FIVE<font color="blue">(5)</font> <font color="blue">(ELI5)</font><br><br>How Do Those High Scoring Approaches Work?</h2>
# 
# <h5 style="text-align: center; font-family: Verdana; font-size: 12px; font-style: normal; font-weight: bold; text-decoration: None; text-transform: none; letter-spacing: 1px; color: black; background-color: #ffffff;">CREATED BY: DARIEN SCHETTLER</h5>
# 
# <br>
# 
# ---
# 
# <br>
# 
# <center><div class="alert alert-block alert-info" style="margin: 2em; line-height: 1.7em; font-family: Verdana;">
#     <b style="font-size: 18px;">⚠️ &nbsp; NOTE &nbsp; ⚠️</b><br><br><b>This notebook endeavours to explain how other open source approaches are scoring so high. I did this because I wanted to understand more but I hope it helps you too!!</b><br><br>The very nature of this notebook relies on the contributions of others and as such I would greatly appreciate that, if you decide to upvote my work, that you also...<br><br><br><b style="font-size: 20px;">UPVOTE THE ORIGINAL AUTHOR'S WORK!</b><br><br><br>
#     
# <table class="alert alert-block alert-info" style="text-align:center;">
# <thead>
#   <tr>
#     <th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Contributor&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
#     <th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Notebook&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
#   </tr>
# </thead>
# <tbody>
#   <tr>
#       <td><a href="https://www.kaggle.com/kvigly55"><b>kvigly55</b></a></td>
#       <td><a href="https://www.kaggle.com/code/kvigly55/plldt-and-ddg/notebook"><b>pLLDT and DDG</b></a></td>
#   </tr>
#   <tr>
#       <td><a href="https://www.kaggle.com/hengck23"><b>hengck23</b></a></td>
#       <td><a href="https://www.kaggle.com/code/hengck23/lb0-335-deepdgg-server-benchmark"><b>DeepDGG Server Benchmark</b></a></td>
#   </tr>
#   <tr>
#     <td><a href="https://www.kaggle.com/lucasmorin"><b>lucasmorin</b></a></td>
#     <td><a href="https://www.kaggle.com/code/lucasmorin/nesp-changes-eda-and-baseline"><b>NESP: Changes EDA and Baseline</b></a></td>
#   </tr>
#   <tr>
#     <td>placeholder 4</td>
#     <td>notebook 4</td>
#   </tr>
# </tbody>
# </table>
#     
# <br>
#     
# </div></center>
# 
# 
# 
# <center><div class="alert alert-block alert-danger" style="margin: 2em; line-height: 1.7em; font-family: Verdana;">
#     <b style="font-size: 18px;">🛑 &nbsp; WARNING:</b><br><br><b>THIS IS A WORK IN PROGRESS</b><br>
# </div></center>
# 
# 
# <center><div class="alert alert-block alert-warning" style="margin: 2em; line-height: 1.7em; font-family: Verdana;">
#     <b style="font-size: 18px;">👏 &nbsp; IF YOU FORK THIS OR FIND THIS HELPFUL &nbsp; 👏</b><br><br><b style="font-size: 22px; color: darkorange">PLEASE UPVOTE!</b><br><br>This was a lot of work for me and while it may seem silly, it makes me feel appreciated when others like my work. 😅
# </div></center>
# 
# 
# 

# <p id="toc"></p>
# 
# <br><br>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: navy; background-color: #ffffff;">TABLE OF CONTENTS</h1>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><a href="#introduction">1&nbsp;&nbsp;&nbsp;&nbsp;INTRODUCTION & JUSTIFICATION</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><a href="#background_information">2&nbsp;&nbsp;&nbsp;&nbsp;BACKGROUND INFORMATION</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><a href="#imports">3&nbsp;&nbsp;&nbsp;&nbsp;IMPORTS</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><a href="#setup">4&nbsp;&nbsp;&nbsp;&nbsp;SETUP & HELPER FUNCTIONS</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><a href="#step_by_step">5&nbsp;&nbsp;&nbsp;&nbsp;STEP-BY-STEP WALKTHROUGH</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><a href="#observations">6&nbsp;&nbsp;&nbsp;&nbsp;OBSERVATIONS</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><a href="#next_steps">7&nbsp;&nbsp;&nbsp;&nbsp;NEXT STEPS</a></h3>
# 
# ---

# <br>
# 
# <a id="introduction"></a>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #ffffff; color: navy;" id="introduction">1&nbsp;&nbsp;INTRODUCTION & JUSTIFICATION&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>
# 
# <br>
# 
# This notebook assumes the reader has a basic understanding of:
# * The problem we are trying to solve
# * The provided data
#     * train.csv
#     * test.csv
#     * sample_submission.csv
#     * wildtype_structure_prediction_af2.pdb <b><i>(If you are a bit confused by this one that's ok!)</i></b>
# * The evaluation technique used (Spearman Correlation)
# 
# If any of this is news to you, please feel free to read my <b><a href="https://www.kaggle.com/code/dschettler8845/novo-esp-eda-baseline">Exploratory Data Analysis (EDA)</a></b> prior to attempting to understand this notebook

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">1.1 <b>WHAT</b> IS THIS?</h3>
# 
# ---
# 
# This notebook will attempt to "<b>E</b>xplain It <b>L</b>ike <b>I</b>'m <b>F</b>ive". Where <b>"IT"</b> refers to the leading open-source approaches in this competition.
# 
# <sup><b>REMINDER:</b> Please don't forget to upvote the original authors work!</sup>
# 
# <table class="alert alert-block alert-info" style="text-align:center;">
# <thead>
#   <tr>
#     <th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Contributor&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
#     <th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Notebook&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
#   </tr>
# </thead>
# <tbody>
#   <tr>
#       <td><a href="https://www.kaggle.com/kvigly55"><b>kvigly55</b></a></td>
#       <td><a href="https://www.kaggle.com/code/kvigly55/plldt-and-ddg/notebook"><b>pLLDT and DDG</b></a></td>
#   </tr>
#   <tr>
#       <td><a href="https://www.kaggle.com/hengck23"><b>hengck23</b></a></td>
#       <td><a href="https://www.kaggle.com/code/hengck23/lb0-335-deepdgg-server-benchmark"><b>DeepDGG Server Benchmark</b></a></td>
#   </tr>
#   <tr>
#     <td><a href="https://www.kaggle.com/lucasmorin"><b>lucasmorin</b></a></td>
#     <td><a href="https://www.kaggle.com/code/lucasmorin/nesp-changes-eda-and-baseline"><b>NESP: Changes EDA and Baseline</b></a></td>
#   </tr>
#   <tr>
#     <td>placeholder 4</td>
#     <td>notebook 4</td>
#   </tr>
# </tbody>
# </table>

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">1.2 <b>WHY</b> IS THIS?</h3>
# 
# ---
# 
# <b>The purpose of this notebook is to explore some of the leading open source approaches to solving this problem</b>. 
# * The approaches mentioned previously revolve arounds similar concepts/approaches that, due to the domain complexity, may not be as easy to understand as approaches found in other competitions. 

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">1.3 <b>WHO</b> IS THIS FOR?</h3>
# 
# ---
# 
# The primary purpose of this notebook is to educate <b>MYSELF</b>, however, my review/learning might be beneficial to others. Hence this notebook.
# 
# If you've looked at the <b>CODE</b> or <b>DISCUSSION</b> section of this competition and thought any of the following:
# * <i>"What is $t_m$... and why do people keep saying stability if this is about temperature?"</i>
# * <i>"What is this --> $\Delta \Delta G$? How come it looks like we are predicting this not $t_m$?"</i>
# * <i>"What is a B Factor and how do I use it?"</i>
# * <i>"What is ... Wildtype? Single-Point Mutation? SNP? PDB File? AlphaFold2? How do we use any of this?"</i>
# * <i>"What is a substitution matrix?"</i>
# 
# Then maybe this notebook can help a little. No promises though!

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">1.4 <b>HOW</b> WILL THIS WORK?</h3>
# 
# ---
# 
# I'm going to assemble some markdown cells (like this one) at the beginning of the notebook to go over some concepts/details/etc.
# 
# Following this, I will attempt to walk through a/some high-scoring approach(es) and explain them in-depth step-by-step

# <br>
# 
# <a id="background"></a>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #ffffff; color: navy;" id="background">2&nbsp;&nbsp;BACKGROUND INFORMATION&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>
# 
# <br>
# 
# <b>I already mentioned that you had to know particular things prior to this notebook, so I will keep this section brief and will structure it mostly as a Glossary that can be referred back to.</b> 
# 
# If you want a more in-depth exploration of the basics please see my <b><a href="https://www.kaggle.com/code/dschettler8845/novo-esp-eda-baseline">Exploratory Data Analysis (EDA)</a></b> or those provided by other wonderful Kagglers.
# 
# 

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">2.1 <b>VERY QUICK</b> OVERVIEW</h3>
# 
# ---
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">PRIMARY TASK DESCRIPTION</b>
# 
# The goal of this competition is to <b><mark>predict the thermal stability of enzyme variants</mark></b>. In this competition, you are asked to develop models that can predict the ranking of protein stability (as measured by melting point, $t_m$) after single-point amino acid mutation and deletion.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">BASIC BACKGROUND INFORMATION</b>
# 
# <b><a href="https://www.britannica.com/science/enzyme">Enzymes</a></b> are <b><a href="https://www.britannica.com/science/protein">proteins</a></b> that act as <b><a href="https://www.britannica.com/science/catalyst">catalysts</a></b> in the chemical reactions of living organisms. 
# 
# <b><a href="https://www.novozymes.com/en">Novozymes</a></b> finds enzymes in nature and optimizes them for use in industry. 
# * In industry, enzymes replace chemicals and accelerate production processes. 
# * They help our customers make more from less, while saving energy and generating less waste. 
# * Enzymes are widely used in laundry and dishwashing detergents where they remove stains and enable low-temperature washing and concentrated detergents. 
# * Other enzymes improve the quality of bread, beer and wine, or increase the nutritional value of animal feed. 
# * Enzymes are also used in the production of biofuels where they turn starch or cellulose from biomass into sugars which can be fermented to ethanol. 
# 
# These are just a few examples as we sell enzymes to more than <b>40 different industries</b>. Like enzymes, microorganisms have natural properties that can be put to use in a variety of processes. 
# * Novozymes supplies a range of microorganisms for use in agriculture, animal health and nutrition, industrial cleaning and wastewater treatment.
# 
# <b><mark>However, many enzymes are only marginally stable, which limits their performance under harsh application conditions.</mark></b> 
# * Instability also decreases the amount of protein that can be produced by the cell. 
# * Therefore, the development of efficient computational approaches to predict protein stability carries enormous technical and scientific interest. 
# 
# Computational protein stability prediction based on physics principles have made remarkable progress thanks to advanced physics-based methods such as <b><a href="https://foldxsuite.crg.eu/">FoldX</a></b>, <b><a href="https://www.rosettacommons.org/software">Rosetta</a></b>, and others. Recently, many machine learning methods were proposed to predict the stability impact of mutations on protein based on the pattern of variation in natural sequences and their three dimensional structures. More and more protein structures are being solved thanks to the recent breakthrough of <b><a href="https://www.deepmind.com/research/highlighted-research/alphafold">AlphaFold2</a></b>. <b><mark>However, accurate prediction of protein thermal stability remains a great challenge.</mark></b>
# 
# <br>
# 
# ---
# 
# <br>
# 
# In this competition, <b><a href="https://www.novozymes.com/en">Novozymes</a></b> invites you to <b><mark>develop a model to predict/rank the thermal stability of enzyme variants based on experimental melting temperature data</mark></b>, which is obtained from <b><a href="https://www.novozymes.com/en">Novozymes'</a></b> high throughput screening lab. 
# * You’ll have access to data from previous scientific publications. 
# * You'll also have access to available thermal stability data spans from natural sequences to engineered sequences with single or multiple mutations upon the natural sequences. 
# 
# <br>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">COMPETITION IMPACT INFORMATION</b>
# 
# Understanding and accurately predict protein stability is a fundamental problem in biotechnology. Its applications include enzyme engineering for addressing the world’s challenges in sustainability, carbon neutrality and more. <b>Improvements to enzyme stability could lower costs and increase the speed scientists can iterate on concepts.</b>
# 
# If successful, you'll help tackle the fundamental problem of improving protein stability, making the approach to design novel and useful proteins, like enzymes and therapeutics, more rapidly and at lower cost.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">COMPETITION HOST INFORMATION</b>
# 
# <b><a href="https://www.novozymes.com/en">Novozymes</a></b> is the world’s leading biotech powerhouse. Our growing world is faced with pressing needs, emphasizing the necessity for solutions that can ensure the health of the planet and its population. At Novozymes, we believe biotech is at the core of connecting those societal needs with the challenges and opportunities our customers face. Novozymes is the global market leader in biological solutions, producing a wide range of enzymes, microorganisms, technical and digital solutions which help our customers, amongst other things, add new features to their products and produce more from less.
# 
# Together, we find biological answers for better lives in a growing world. Let’s Rethink Tomorrow. This is <b><a href="https://www.novozymes.com/en">Novozymes'</a></b> purpose statement. <b><a href="https://www.novozymes.com/en">Novozymes</a></b> strives to have great impact by balancing good business for our customers and our company, while spearheading environmental and social change. In 2021, <b><a href="https://www.novozymes.com/en">Novozymes</a></b> enabled savings of 60 million tons of CO2 in global transport.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">VISUAL EXPLANATION</b>
# 
# <center><img src="https://i.ibb.co/V32S3LT/Screen-Shot-2022-09-22-at-12-01-52-PM.png" width=100%></center>
# 
# <br>

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">2.2 GLOSSARY</h3>
# 
# ---
# 
# <br>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; font-size: 130%; text-transform: uppercase;">QUICK GLOSSARY NAVIGATION LINKS</b>
# 
# <br>
# 
# <ul>
#     <li><a href="#protein"><b style="font-size: 110%;">PROTEIN (GENERAL)</b></a>
#         <ul>
#             <li><a href="#protein_definition">Textbook Definition</a></li>
#             <li><a href="#protein_eli5">Competition (ELI5) Definition</a></li>
#             <li><a href="#protein_visual">Visual Definition/Helpers</a></li>
#         </ul>
#     </li><br>
#     <li><a href="#protein_stability"><b style="font-size: 110%;">PROTEIN STABILITY (CONFORMATIONAL STABILITY)</b></a>
#         <ul>
#             <li><a href="#protein_stability_definition">Textbook Definition</a></li>
#             <li><a href="#protein_stability_eli5">Competition (ELI5) Definition</a></li>
#             <li><a href="#protein_stability_visual">Visual Definition/Helpers</a></li>
#         </ul>
#     </li><br>
#     <li><a href="#tm"><b style="font-size: 110%;">TEMPERATURE OF MELTING $(T_m)$</b></a>
#         <ul>
#             <li><a href="#tm_definition">Textbook Definition</a></li>
#             <li><a href="#tm_eli5">Competition (ELI5) Definition</a></li>
#             <li><a href="#tm_visual">Visual Definition/Helpers</a></li>
#         </ul>
#     </li><br>
#     <li><a href="#ddg"><b style="font-size: 110%;">CHANGE IN GIBBS FREE ENERGY ($\Delta \Delta G$)(ddG)</b></a>
#         <ul>
#             <li><a href="#ddg_definition">Textbook Definition</a></li>
#             <li><a href="#ddg_eli5">Competition (ELI5) Definition</a></li>
#             <li><a href="#ddg_visual">Visual Definition/Helpers</a></li>
#         </ul>
#     </li><br>
#     <li><a href="#alphafold"><b style="font-size: 110%;">ALPHAFOLD2</b></a>
#         <ul>
#             <li><a href="#alphafold_definition">Textbook/Paper Definition</a></li>
#             <li><a href="#alphafold_eli5">Competition (ELI5) Definition</a></li>
#             <li><a href="#alphafold_visual">Visual Definition/Helpers</a></li>
#         </ul>
#     </li><br>
#     <li><a href="#deepddg"><b style="font-size: 110%;">DEEP-DDG ALGORITHM</b></a>
#         <ul>
#             <li><a href="#deepddg_definition">Textbook/Paper Definition</a></li>
#             <li><a href="#deepddg_eli5">Competition (ELI5) Definition</a></li>
#             <li><a href="#deepddg_visual">Visual Definition/Helpers</a></li>
#         </ul>
#     </li>
# </ul>
# 
# <br><br>
# 
# ---
# 
# <br>
# 
# <a id="protein"></a><br><b style="text-decoration: underline; font-family: Verdana; font-size: 120%; text-transform: uppercase;">Protein</b>
# 
# <a id="protein_definition"></a><br>
# 
# <b>Textbook Definition (Key Points)</b>
# 
# * Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. 
# * Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another. 
# * Proteins differ from one another primarily in their sequence of amino acids, which is dictated by the nucleotide sequence of their genes, and which usually results in protein folding into a specific 3D structure that determines its activity.
# * A single amino acid monomer may also be called a residue indicating a repeating unit of a polymer.
# * A linear chain of amino acid residues is called a polypeptide. 
# * A protein contains at least one long polypeptide. 
# * The individual amino acid residues are bonded together by peptide bonds and adjacent amino acid residues. 
# * The sequence of amino acid residues in a protein is defined by the sequence of a gene, which is encoded in the genetic code. 
# * In general, the genetic code specifies 20 standard amino acids
# * Shortly after or even during synthesis, the residues in a protein are often chemically modified by post-translational modification, which alters the physical and chemical properties, folding, stability, activity, and ultimately, the function of the proteins. 
# * Once formed, proteins only exist for a certain period and are then degraded and recycled by the cell's machinery through the process of protein turnover. A protein's lifespan is measured in terms of its half-life and covers a wide range. They can exist for minutes or years with an average lifespan of 1–2 days in mammalian cells
# * Many proteins are enzymes that catalyse biochemical reactions and are vital to metabolism. 
# * Proteins also have structural or mechanical functions, such as actin and myosin in muscle and the proteins in the cytoskeleton, which form a system of scaffolding that maintains cell shape. 
# * Other proteins are important in cell signaling, immune responses, cell adhesion, and the cell cycle. 
# * In animals, proteins are needed in the diet to provide the essential amino acids that cannot be synthesized. Digestion breaks the proteins down for metabolic use.
# * Proteins are polymers – specifically polypeptides – formed from sequences of amino acids, the monomers of the polymer. 
# * Proteins form by amino acids undergoing condensation reactions, in which the amino acids lose one water molecule per reaction in order to attach to one another with a peptide bond. 
# * To be able to perform their biological function, proteins fold into one or more specific spatial conformations driven by a number of non-covalent interactions such as hydrogen bonding, ionic interactions, Van der Waals forces, and hydrophobic packing. 
# * To understand the functions of proteins at a molecular level, it is often necessary to determine their three-dimensional structure. 
# * This is the topic of the scientific field of structural biology, which employs techniques such as X-ray crystallography, NMR spectroscopy, cryo electron microscopy (cryo-EM) and dual polarisation interferometry to determine the structure of proteins.
# * Protein structures range in size from tens to several thousand amino acids. By physical size, proteins are classified as nanoparticles, between 1–100 nm
# * <b><a href="https://www.wikiwand.com/en/Protein_structure">Reference - Protein Structure (Wikipedia)</a> & <a href="https://www.wikiwand.com/en/Protein">Reference - Protein (Wikipedia)</a></b>
# 
# <a id="protein_eli5"></a><br>
# 
# <b>ELI5 Competition Definition</b>
# 
# * This is a big one and I recommend you don't just read this to understand proteins... use the internet, youtube, google, textbooks, etc. This stuff is important and it won't hurt you to learn more... that being said... here we go...
# * Biomolecules are one of the most important things in Biology. The four major types of BIOLOGICAL MOLECULES (BIOMOLECULES) are listed below.
#     * Carbohydrates
#     * Lipids
#     * <b>Proteins</b> 
#     * Nucleic Acids
# * Proteins are responsible for <b>doing things</b>. When you think of stuff in your body doing stuff, protein is most definitely involved.
#     * Horomones are proteins
#     * Enzymes are proteins
#     * Catalysts are proteins
#     * Hemoglobin (oxygen carrier in blood)
#     * Keratin (hair)
#     * Muscle (mostly)
#     * etc.
# * Proteins are made up of chains of <b>Amino Acids</b> (may also be referred to as residues)
#     * There are 20 unique Amino Acids (difference comes from R groups... probably beyond the scope here).
#     * Note that we use the word chain because an Amino Acid unit can only be connected to two other Amino Acid units (i.e. a chain)
# * Amino Acids are made up of a central carbon atom linked together with a basic amino group, a carboxylic acid group, a hydrogen atom and an R-group, or side-chain group. 
# * Amino Acid structure (particularly the R-Group) is determined by a particular <b>codon</b> (triplet of Nucleotides).
#     * There are only 5 types of nucleotides
#         * 3 are found in both DNA and RNA (Adenine (A), Cytosine (C), Guanine (G))
#         * Thymine (T) is also used in DNA while Uracil (U) is used in RNA
# * SO to make a protein we use <b>instructions (<i>DNA/RNA, Nucleotides</i>)</b>, to build up the protein <b>chain</b> by <b>adding one Amino Acid at a time</b> (in our instructions each <b>codon <i>(triplet of nucleotides)</i></b> tells us what Amino Acid comes next). At some point the instructions will also tell us when to stop (stop codon)!
# * An important note here is that since there are 64 combinations of 4 nucleotides (ACTG in DNA and ACUG in RNA) taken three at a time and only 20 amino acids, the code is <b>degenerate</b> (more than one codon per amino acid, in most cases)
#     * This means that <b>different instructions</b> can create the <b>same amino acid</b>
#     * This also means that information can only flow forward not backwards
#         * Nucleotides --> Amino Acids ✅ &nbsp;&nbsp;&nbsp;(i.e. CTC and CTT both code for Leucine... this is ok)
#         * Amino Acids --> Nucleotides ❌ &nbsp;&nbsp;&nbsp;(i.e. Leucine could be translated into CTC or CTT... but we don't know which was really what it came from)
# * Now we know how the chains of amino acids are made... but that's not really a protein. To make a protein this long chain needs to fold up into a very complicated structure. This structure is comprised of 4 distinct levels of complexity
#     * <b>Primary Structure</b> is the exact ordering of amino acids forming their chains (pretty much what we've described up to this point)
#     * <b>Secondary Structure</b> refers to local folded structures that form within a polypeptide due to interactions between atoms of the backbone.
#         * They are found to exist in two different types of structures α – helix and β – pleated sheet structures.
#     * <b>Tertiary Structure</b> arises from further folding of the secondary structure of the protein.
#         * H-bonds, electrostatic forces, disulphide linkages, and Vander Waals forces stabilize this structure.
#         * Tertiary folding gives rise to two major molecular shapes called <b>fibrous and globular</b>.
#     * <b>Quaternary Structure</b>
#         * The spatial arrangement of various tertiary structures gives rise to the quaternary structure.
# * SO an addendum to our "make a protein" line from above would be to understand that as our chain is being created it will be kinking/folding/spiralling/etc and forming it's 3D structure (up to a tertiary or quaternary level)
# * One final note, the final shape, often called the <b>protein conformation</b>, adopted by a newly synthesized protein is typically the most energetically favorable one. 
#     * A protein is often said to be <b>active</b> when existing in it's <b>conformation</b>. 
#     * <b>Active</b> means that the protein now does the things it was created to do (i.e. insulin can now regulate the metabolism of carbohydrates, fats and protein by promoting the absorption of glucose from the blood into liver, fat and skeletal muscle cells. WOO!)
# * SO, reading all of this ( which is by NO MEANS all of it), we should now understand roughly how proteins are made, what they are made of, why they are important, and that to be useful (<b>active</b>) they must be folded into a specific structure (<b>conformation</b>).
# 
# <a id="protein_visual"></a><br>
# 
# <b>Explain With Pictures</b>
# 
# <b><sub>Image Comparing Proteins with Language</sub></b>
# 
# <img src="https://www.ptglab.com/media/3301/1503677_complexity-of-proteins-blog-diagram_v1.jpg">
# 
# <br>
# 
# <b><sub>Image Showing 20 Amino Acids</sub></b>
# 
# <img src="https://bio.libretexts.org/@api/deki/files/14208/clipboard_e1e6db9e752224bced0f37cd93a0642b3.png?revision=1">
# 
# <br>
# 
# <b><sub>Image Showing How It All Works (sorta)</sub></b>
# 
# <img src="https://cdn.britannica.com/80/780-050-CC40AEDF/Synthesis-protein.jpg">
# 
# <br>
# 
# ---
# 
# <a id="protein_stability"></a><br>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; font-size: 125%; text-transform: uppercase;">Protein (Conformational) Stability</b>
# 
# <a id="protein_stability_definition"></a><br>
# 
# <b>Textbook Definition</b>
# 
# * Thermodynamic stability of proteins represents the <b>free energy difference between the folded and unfolded protein states.</b>
# * <b>This free energy difference is very sensitive to temperature</b>, hence a change in temperature may result in unfolding or denaturation. 
# * Protein denaturation may result in loss of function, and loss of native state. 
# * Taking into consideration the large number of hydrogen bonds that take place for the stabilization of secondary structures, and the stabilization of the inner core through hydrophobic interactions, the free energy of stabilization emerges as small difference between large numbers.
# * <b><a href="https://www.wikiwand.com/en/Protein_structure#/Protein_stability">Reference - Protein Structure <sub>Stability Section</sub> (Wikipedia)</a></b>
# 
# <a id="protein_stability_eli5"></a><br>
# 
# <b>ELI5 Competition Definition</b>
# 
# * Go watch the Khan Academy Video (it's short) --> <b><a href="https://www.khanacademy.org/test-prep/mcat/biomolecules/amino-acids-and-proteins1/v/conformational-stability-protein-folding-and-denaturation-2">link</a></b>
#     * In fact go watch the whole playlist if you have time --> <b><a href="https://www.khanacademy.org/test-prep/mcat/biomolecules#amino-acids-and-proteins1">link</a></b>
# * The Protein Conformational Stability refers to the resistance of a particular protein conformation (active state) to resist <b>denaturation</b>
#     * Remember a protein has 4 levels of structure (primary, secondary, tertiary, and quaternary), and these all contribute to the stability of the protein
#     * Remember that conformation refers to the final <b>actived</b> structure that the protein ends up in. Only in this state can the protein function properly.
#     * Denaturation simply refers to the process of "breaking" a protein (breaking bonds/forces) such that the protein is no longer in it's active conformation and subsequently can no longer function properly.
# * Proteins may become denatured in various ways... however these are the most common (bolded terms are important to us)
#     * <b>Heat</b>
#     * <b>Acid (pH)</b>
#     * High Salt Concentrations
#     * Alcohol
#     * Mechanical Agitation
# * <b>Important Side Note: <mark>Proteins do not exist by themselves, you will have many of the same protein in a solution</mark></b>
#     * Think of everything that isn't the protein as <b>the environment</b>
#     
# <a id="protein_stability_visual"></a><br>
# 
# <b>Explain With Pictures</b>
# 
# <img src="https://i.ibb.co/V32S3LT/Screen-Shot-2022-09-22-at-12-01-52-PM.png">
# 
# <br>
# 
# ---
# 
# <br>
# 
# <a id="tm"></a><br><b style="text-decoration: underline; font-family: Verdana; font-size: 125%; text-transform: uppercase;">Temperature of Melting $(T_m)$</b>
# 
# <a id="tm_definition"></a><br>
# 
# <b>Textbook Definition</b>
# 
# * The melting temperature ($T_m$) is defined as the temperature at which half of the DNA strands are in the random coil or single-stranded (ssDNA) state. 
# * $T_m$ depends on the length of the DNA molecule and its specific nucleotide sequence. 
# * DNA, when in a state where its two strands are dissociated (i.e., the double-stranded DNA (dsDNA) molecule exists as two independent strands), is referred to as having been <b>denatured by the high temperature</b>.
# * <b><a href="https://www.wikiwand.com/en/Nucleic_acid_thermodynamics">Reference - Nucleic Acid Thermodynamics (Wikipedia)</a></b>
# 
# <a id="tm_eli5"></a><br>
# 
# <b>ELI5 Competition Definition</b>
# 
# * Remember we need to think of protein as a <b>population</b> of that particular protein instead of a single <b>individual protein</b>
# * Therefore, temperature of Melting $(T_m)$ refers to the <b>Temperature</b> for which <b>50%</b> of the <b>protein population</b> will be <b>denatured</b> (unfolded, active conformation is lost)
#     * Therefore if the temperature is HIGHER than the specified $T_m$: MORE THAN <b>50%</b> of the <b>protein population</b> will be <b>denatured</b>
#     * Therefore if the temperature is LOWER than the specified $T_m$: LESS THAN <b>50%</b> of the <b>protein population</b> will be <b>denatured</b>
# * So the ELI5 Definition of Temperature of Melting $(T_m)$ therefore is:
#     * <b>The temperature at which more than 50% of the proteins in a population stop functioning (due to denaturation/unfolding)</b>
# * Side note: 
#     * It is REALLY hard to determine experimentally the $T_m$ for a given protein in a given environment and as such efforts to predict this value given the Amino Acid sequence and environmental conditions is a topic of a lot of interest in the field.
#     * Note also from the picture below that near the ground truth $T_m$ value even small changes in Temperature will have a large impact. This means that it's quite easy to be wrong and very difficult to be exactly right.
# 
# <a id="tm_visual"></a><br>
# 
# <b>Explain With Pictures</b>
# 
# <img src="https://europepmc.org/articles/PMC2931665/bin/cbe0031002450004.jpg">
# 
# <br>
# 
# ---
# 
# <br>
# 
# <a id="ddg"></a><br><b style="text-decoration: underline; font-family: Verdana; font-size: 125%; text-transform: uppercase;">Change in Gibbs Free Energy $( \Delta \Delta G )$</b>
# 
# <a id="ddg_definition"></a><br>
# 
# <b>Textbook Definition</b>
# * The Change in Gibbs Free Energy is a metric for predicting how a single point mutation will affect protein stability. 
# * The Change in Gibbs Free Energy is often referred to as 𝚫𝚫G or DDG. 
# * DDG is a measure of the change in energy between the folded and unfolded states (𝚫Gfolding) and the change in 𝚫Gfolding when a point mutation is present. 
#     * This has been found to be an excellent predictor of whether a point mutation will be favorable in terms of protein stability.
# * <b><a href="https://cyrusbio.com/wp-content/uploads/DDG-v2.pdf">Reference - A Deeper Look at DDG For Proteins (CyrusBio.com)</a></b>
# 
# <a id="ddg_eli5"></a><br>
# 
# <b>ELI5 Competition Definition</b>
# 
# * Since it is very difficult to calculate/measure $T_m$ another approach is to find a metric that is highly correlated with it. This is why we are interested in $\Delta \Delta G$
# * In the context of this competition, the unfolding free energy difference $( \Delta \Delta G )$ between the wild type and mutant protein, i.e., ΔΔG = ΔGwildtype - ΔGmutant (measured in kcal/mol)
# * The Change in Gibbs Free Energy $( \Delta \Delta G )$ is analagous to how much energy is needed to fold up the protein (i.e. the energy difference between folded and unfolded states).
# * Generally speaking, if you need $\Delta \Delta G$ to fold something up, this will tell you a lot about how much energy is needed to <b>unfold/denature/break</b> that same thing
# 
# <a id="ddg_visual"></a><br>
# 
# <b>Explain With Pictures</b>
# 
# <img src="https://www.molsoft.com/gui/eqn2.png">
# 
# <br>
# 
# <img src="https://cdn.kastatic.org/ka-perseus-images/d5b999fb65f5902a61d25e1c466060c35e8db6da.png">
# 
# <br>
# 
# <img src="https://i.ytimg.com/vi/2KuNzB0cZL4/maxresdefault.jpg">
# 
# <br>
# 
# ---
# 
# <br>
# 
# <a id="alphafold"></a><br><b style="text-decoration: underline; font-family: Verdana; font-size: 125%; text-transform: uppercase;">AlphaFold2</b>
# 
# <a id="alphafold_definition"></a><br>
# 
# <b>Textbook</b> <b><a href ="https://www.nature.com/articles/s41586-021-03819-2">[Paper]</a> Definition</b>
# * Proteins are essential to life, and understanding their structure can facilitate a mechanistic understanding of their function. 
#     * Through an enormous experimental effort the structures of around 100,000 unique proteins have been determined, but this represents a small fraction of the billions of known protein sequences. 
#     * Structural coverage is bottlenecked by the months to years of painstaking effort required to determine a single protein structure. 
#     * Accurate computational approaches are needed to address this gap and to enable large-scale structural bioinformatics. 
# * Predicting the three-dimensional structure that a protein will adopt based solely on its amino acid sequence —– the structure prediction component of the ‘protein folding problem’ –– has been an important open research problem for more than 50 years. 
#     * Despite recent progress, existing methods fall far short of atomic accuracy, especially when no homologous structure is available. 
# * [AlphaFold2] provides the first computational method that can regularly predict protein structures with atomic accuracy even in cases in which no similar structure is known. 
#     * [DeepMind] validated an entirely redesigned version of our neural network-based model, AlphaFold, in the challenging 14th Critical Assessment of protein Structure Prediction (CASP14), demonstrating accuracy competitive with experimental structures in a majority of cases and greatly outperforming other methods. 
#     * Underpinning the latest version of AlphaFold is a novel machine learning approach that incorporates physical and biological knowledge about protein structure, leveraging multi-sequence alignments, into the design of the deep learning algorithm.
# * <b><a href="https://www.nature.com/articles/s41586-021-03819-2">Reference - Highly Accurate Protein Structure Prediction With AlphaFold</a></b>
# 
# <a id="alphafold_eli5"></a><br>
# 
# <b>ELI5 Competition Definition</b>
# 
# * As we know from the previous definitiions, proteins are made up of a chain of amino acids that has been composed into a conformation characterized by 4 levels of structural complexity (primary, secondary, tertiary, quaternary).
# * The determination of the actual 3D structure (angles of bonds, etc.) is extremely cumbersome to determine experimentally
# * DeepMind have created a tool (v2 of the tool actually) that can predict with unprecedented accuracy the details of how a protein will fold (including angles, and all the other details) from only the Amino Acid sequence
# * This is important in the competition for two reasons
#     1. For the molecule the test set is concerned with we are provided the AlphaFold2 predicted 3D structure (.pdb file). This .pdb file can be used with existing solutions/models to predict how mutations might impact stability and/or $t_m$ and/or $ \delta \delta G $.
#     2. AlphaFold2 is publically available (opensource) and could be pivotal in the creation of training or supplementary data that may be required to win this competition
# * We will see the structure of the AlphaFold2 prediction for our test molecule further down (and in the EDAs mentioned previously). The basics are:
#     * We get information at the Atomic level (1 row per atom)
#     * We get position of each atom (xyz)
#     * We get the atom name
#     * We get residue number (which atom is part of which amino acid)
#     * We get B-factor (which is actually NOT the typical B-factor)
#         > AlphaFold produces a per-residue estimate of its confidence on a scale from 0 - 100.<br>
#         > This confidence measure is called pLDDT and corresponds to the model’s predicted score on the lDDT-Cα metric.<br>
#         > It is stored in the B-factor fields of the mmCIF and PDB files available for download (although unlike a B-factor, higher pLDDT is better).<br>
#         > pLDDT is also used to colour-code the residues of the model in the 3D structure viewer.<br>
# 
# <a id="alphafold_visual"></a><br>
# 
# <b>Explain With Pictures</b>
# 
# <img src="https://i0.wp.com/www.blopig.com/blog/wp-content/uploads/2021/07/image-3.png?ssl=1">
# 
# <br>
# 
# <center><img src="https://raw.githubusercontent.com/cdeotte/Kaggle_Images/main/Sep-2022/test-image.png"></center>
# 
# <br>
# 
# ---
# 
# <br>
# 
# <a id="deepddg"></a><br><b style="text-decoration: underline; font-family: Verdana; font-size: 125%; text-transform: uppercase;">DeepDDG</b>
# 
# <a id="deepddg_definition"></a><br>
# 
# <b>Textbook</b> <b><a href ="https://www.nature.com/articles/s41586-021-03819-2">[Paper]</a> Definition</b>
# * Accurately predicting changes in protein stability due to mutations is important for protein engineering and for understanding the functional consequences of missense mutations in proteins. 
# * [Hence,] DeepDDG, a neural network-based method, for use in the prediction of changes in the stability of proteins due to point mutations. 
# * The neural network was trained on more than <b>5700 manually curated experimental data points and was able to obtain a Pearson correlation coefficient of 0.48–0.56</b> for three independent test sets, which <b>outperformed 11 other methods</b>. 
# * Detailed analysis of the input features shows that the <b>solvent accessible surface area of the mutated residue is the most important feature</b>, which suggests that <b>the buried hydrophobic area is the major determinant of protein stability</b>. 
# * The neural network is freely available to academic users at <b>http://protein.org.cn/ddg.html</b>
# * <b><a href="https://pubs.acs.org/doi/10.1021/acs.jcim.8b00697">Reference - DeepDDG: Predicting the Stability Change of Protein Point Mutations Using Neural Networks</a></b>
# 
# <a id="deepddg_eli5"></a><br>
# 
# <b>ELI5 Competition Definition</b>
# 
# * The <b><a href="http://protein.org.cn/ddg.html">DeepDDG Server</a></b> predicts the stability change of protein point mutations using neural networks.
# * The server requires:
#     * The 3D structure of the molecule (the .PDB file we are provided by the competition hosts)
#     * The single point mutations represented in the test set (substitutions only!) OR simply generate EVERY POSSIBLE SINGLE POINT MUTATION effect and then read the respective values after.
# * Since $\Delta \Delta G$ is highly correlated with $T_m$ this represents a very usable off the shelf solution that can (and will further down in this notebook) be leveraged<br>
# 
# <a id="deepddg_visual"></a><br>
# 
# <b>Explain With Pictures</b>
# 
# **DeepDDG Architecture**
# 
# <center><img src="https://pubs.acs.org/cms/10.1021/acs.jcim.8b00697/asset/images/medium/ci-2018-00697p_0002.gif"></center>
# 
# <br>
# 
# **DeepDDG Performance Compared With Other Approaches**
# 
# <center><img src="https://pubs.acs.org/cms/10.1021/acs.jcim.8b00697/asset/images/medium/ci-2018-00697p_0005.gif"></center>
# 
# <br>
# 
# ---
# 
# <br>
# 
# <b><font color="red">LET'S PAUSE THE GLOSSARY FOR NOW AND MOVE ON... WE WILL COME BACK TO THIS AND FIX/UPDATE IT AS WE NEED TO LATER</font></b>
# 
# <br>

# <br>
# 
# <a id="imports"></a>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #ffffff; color: navy;" id="imports">3&nbsp;&nbsp;IMPORTS&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>

# In[1]:


print("\n... PIP INSTALLS STARTING ...\n")
# Pip installs for biology specific libraries
get_ipython().system('pip install -q biopandas')
get_ipython().system('pip install -q biopython')
print("\n... PIP INSTALLS COMPLETE ...\n")


print("\n... IMPORTS STARTING ...\n")
print("\n\tVERSION INFORMATION")

# Biology Specific Imports (You'll see why we need these later)
import Bio; print(f"\t\t– BioPython VERSION: {Bio.__version__}");
from Bio import SeqIO
from io import StringIO
import biopandas; from biopandas.pdb import PandasPdb; print(f"\t\t– BioPandas VERSION: {biopandas.__version__}"); pdb = PandasPdb();

# Machine Learning and Data Science Imports (basics)
import tensorflow as tf; print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}");
import pandas as pd; pd.options.mode.chained_assignment = None; pd.set_option('display.max_columns', None);
import numpy as np; print(f"\t\t– NUMPY VERSION: {np.__version__}");
import sklearn; print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}");

# Built-In Imports (mostly don't worry about these)
from kaggle_datasets import KaggleDatasets
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from glob import glob
import Levenshtein
import warnings
import requests
import hashlib
import imageio
import IPython
import sklearn
import urllib
import zipfile
import pickle
import random
import shutil
import string
import json
import math
import time
import gzip
import ast
import sys
import io
import os
import gc
import re

# Visualization Imports (overkill)
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm; tqdm.pandas();
import plotly.express as px
import tifffile as tif
import seaborn as sns
from PIL import Image, ImageEnhance; Image.MAX_IMAGE_PIXELS = 5_000_000_000;
import matplotlib; print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}");
from matplotlib import animation, rc; rc('animation', html='jshtml')
import plotly
import PIL
import cv2

import plotly.io as pio
print(pio.renderers)

def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_it_all()

print("\n\n... IMPORTS COMPLETE ...\n")


# <br>
# 
# <a id="setup"></a>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #ffffff; color: navy;" id="setup">4&nbsp;&nbsp;SETUP AND HELPER FUNCTIONS&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">4.1 HELPER FUNCTIONS</h3>
# 
# ---
# 
# <br>
# 
# Don't worry about these for now. I've hidden them in the notebook viewer to not add complexity. I will explain any functions that are important in-line later.

# In[2]:


def get_mutation_info(_row, _wildtype="VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQ" \
                                      "RVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGT" \
                                      "NAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKAL" \
                                      "GSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"):
    terminology_map = {"replace":"substitution", "insert":"insertion", "delete":"deletion"}
    req_edits = Levenshtein.editops(_wildtype, _row["protein_sequence"])
    _row["n_edits"] = len(req_edits)
    
    if _row["n_edits"]==0:
        _row["edit_type"] = _row["edit_idx"] = _row["wildtype_aa"] = _row["mutant_aa"] = pd.NA
    else:
        _row["edit_type"] = terminology_map[req_edits[0][0]]
        _row["edit_idx"] = req_edits[0][1]
        _row["wildtype_aa"] = _wildtype[_row["edit_idx"]]
        _row["mutant_aa"] = _row["protein_sequence"][req_edits[0][2]] if _row["edit_type"]!="deletion" else pd.NA
    return _row

def revert_to_wildtype(protein_sequence, edit_type, edit_idx, wildtype_aa, mutant_aa):
    if pd.isna(edit_type):
        return protein_sequence
    elif edit_type!="insertion":
        new_wildtype_base = protein_sequence[:edit_idx]
        if edit_type=="deletion":
            new_wildtype=new_wildtype_base+wildtype_aa+protein_sequence[edit_idx:]
        else:
            new_wildtype=new_wildtype_base+wildtype_aa+protein_sequence[edit_idx+1:]
    else:
        new_wildtype=protein_sequence[:edit_idx]+wildtype_aa+protein_sequence[edit_idx:]
    return new_wildtype

def flatten_l_o_l(nested_list):
    """ Flatten a list of lists """
    return [item for sublist in nested_list for item in sublist]

def print_ln(symbol="-", line_len=110):
    print(symbol*line_len)
    
# Note mutation edit_idx is offset by 1 as many tools require it to be 1 indexed to 0 indexed.
def create_mutation_txt_file(_test_df, filename="/kaggle/working/AF70_mutations.txt", return_mutation_list=False, include_deletions=False):
    if return_mutation_list: mutation_list = []        
    with open(filename, 'w') as f:
        for _, _row in _test_df[["protein_sequence", "edit_type", "edit_idx", "wildtype_aa", "mutant_aa"]].iterrows():
            if not include_deletions and (pd.isna(_row["edit_type"]) or _row["edit_type"]=="deletion"): continue
            f.write(f'{_row["wildtype_aa"]+str(_row["edit_idx"]+1)+(_row["mutant_aa"] if not pd.isna(_row["mutant_aa"]) else "")}\n')
            if return_mutation_list: mutation_list.append(f'{_row["wildtype_aa"]+str(_row["edit_idx"]+1)+(_row["mutant_aa"] if not pd.isna(_row["mutant_aa"]) else "")}')
    if return_mutation_list: return mutation_list
        
def create_wildtype_fasta_file(wildtype_sequence, filename="/kaggle/working/wildtype_af70.fasta"):
    with open(filename, 'w') as f: f.write(f">af70_wildtype\n{wildtype_sequence}")
        
def uniprot_id2seq(uniprot_id, _sleep=3):
    base_url = "http://www.uniprot.org/uniprot"
    full_url = os.path.join(base_url, str(uniprot_id)+".fasta")
    _r = requests.post(full_url)
    if _r.status_code!=200: print(_r.status_code)
    if _sleep!=-1: time.sleep(_sleep)
    return ''.join(_r.text.split("\n")[1:])


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">4.2 LOAD DATA</h3>
# 
# ---
# 
# Much of this may not be needed. 
# * If anything is unclear (and needed) I will make sure to add in additional explanation

# In[3]:


# Define the path to the root data directory
DATA_DIR = "/kaggle/input/novozymes-enzyme-stability-prediction"


print("\n... BASIC DATA SETUP STARTING ...\n")
print("\n\n... LOAD TRAIN DATAFRAME FROM CSV FILE ...\n")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
display(train_df)

print("\n\n... LOAD TEST DATAFRAME FROM CSV FILE ...\n")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
display(test_df)

print("\n\n... LOAD SAMPLE SUBMISSION DATAFRAME FROM CSV FILE ...\n")
ss_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
display(ss_df)

print("\n\n... LOAD ALPHAFOLD WILDTYPE STRUCTURE DATA FROM PDB FILE ...\n")
pdb_df = pdb.read_pdb(os.path.join(DATA_DIR, "wildtype_structure_prediction_af2.pdb"))

print("ATOM DATA...")
atom_df   = pdb_df.df['ATOM']
display(atom_df)

print("\nHETATM DATA...")
hetatm_df = pdb_df.df['HETATM']
display(hetatm_df)

print("\nANISOU DATA...")
anisou_df = pdb_df.df['ANISOU']
display(anisou_df)

print("\nOTHERS DATA...")
others_df = pdb_df.df['OTHERS']
display(others_df)

print("\n\n... SAVING WILDTYPE AMINO ACID SEQUENCE...\n")
wildtype_aa = "VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"
print(wildtype_aa)

print("\n\n... DEFINE AMINO ACID SHORTFORM DICTIONARY MAPPING...\n")
aa_map = dict(Alanine=("Ala", "A"), Arginine=("Arg", "R"), Asparagine=("Asn", "N"), Aspartic_Acid=("Asp", "D"),
              Cysteine=("Cys", "C"), Glutamic_Acid=("Glu", "E"), Glutamine=("Gln", "Q"), Glycine=("Gly", "G"),
              Histidine=("His", "H"), Isoleucine=("Ile", "I"), Leucine=("Leu", "L"), Lysine=("Lys", "K"),
              Methionine=("Met", "M"), Phenylalanine=("Phe", "F"), Proline=("Pro", "P"), Serine=("Ser", "S"),
              Threonine=("Thr", "T"), Tryptophan=("Trp", "W"), Tyrosine=("Tyr", "Y"), Valine=("Val", "V"))
n_aa = len(aa_map)
aa_chars_ordered = sorted([v[1] for v in aa_map.values()])
aa_long2tri = {k:v[0] for k,v in aa_map.items()}
aa_long2char = {k:v[1] for k,v in aa_map.items()}
aa_tri2long = {v:k for k,v in aa_long2tri.items()}
aa_char2long = {v:k for k,v in aa_long2char.items()}
aa_char2int = {_aa:i for i, _aa in enumerate(aa_chars_ordered)}
aa_int2char = {v:k for k,v in aa_char2int.items()}

# Get data source map
print("\n\n... DEFINE DATASOURCE DICTIONARY MAPPING...\n")
ds_str2int = {k:i for i,k in enumerate(train_df["data_source"].unique())}
ds_int2str = {v:k for k,v in ds_str2int.items()}

for k,v in aa_map.items(): print(f"'{k}':\n\t3 LETTER ABBREVIATION --> '{v[0]}'\n\t1 LETTER ABBREVIATION --> '{v[1]}'\n")
    
print("\n\n... FOR FUN ... HERE IS THE ENTIRE WILDTYPE WITH FULL AMINO ACID NAMES (8 AA PER LINE) ...\n")
for i, _c in enumerate(wildtype_aa): print(f"'{aa_char2long[_c]}'", end=", ") if (i+1)%8!=0 else print(f"{aa_char2long[_c]}", end=",\n")

print("\n\n... ADD COLUMNS FOR PROTEIN SEQUENCE LENGTH AND INDIVIDUAL AMINO ACID COUNTS/FRACTIONS ...\n")
train_df["n_AA"] = train_df["protein_sequence"].apply(len)
test_df["n_AA"] = test_df["protein_sequence"].apply(len)
for _aa_char in aa_chars_ordered: 
    train_df[f"AA_{_aa_char}__count"] = train_df["protein_sequence"].apply(lambda x: x.count(_aa_char))
    train_df[f"AA_{_aa_char}__fraction"] = train_df[f"AA_{_aa_char}__count"]/train_df["n_AA"]
    test_df[f"AA_{_aa_char}__count"] = test_df["protein_sequence"].apply(lambda x: x.count(_aa_char))
    test_df[f"AA_{_aa_char}__fraction"] = test_df[f"AA_{_aa_char}__count"]/test_df["n_AA"]
    
print("\n... ADD COLUMNS FOR DATA SOURCE ENUMERATION ...\n")
train_df["data_source_enum"] = train_df['data_source'].map(ds_str2int)
test_df["data_source_enum"] = test_df['data_source'].map(ds_str2int)

print("\n... DO TEMPORARY pH FIX BY SWAPPING pH & tm IF pH>14 ...\n")
def tmp_ph_fix(_row):
    if _row["pH"]>14:
        print(f"\t--> pH WEIRDNESS EXISTS AT INDEX {_row.name}")
        _tmp = _row["pH"]
        _row["pH"] = _row["tm"]
        _row["tm"] = _tmp
        return _row
    else:
        return _row

print(f"\t--> DOES THE  STILL EXIST: {train_df['pH'].max()>14.0}")
train_df = train_df.progress_apply(tmp_ph_fix, axis=1)
test_df = test_df.progress_apply(tmp_ph_fix, axis=1)
    
print("\n\n\n... BASIC DATA SETUP FINISHED ...\n\n")


# <br>
# 
# <a id="step_by_step"></a>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #ffffff; color: navy;" id="step_by_step">5&nbsp;&nbsp;STEP-BY-STEP WALKTHROUGH – <b><a href="https://www.kaggle.com/code/kvigly55/plldt-and-ddg">PLLDT AND DDG</a></b>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>
# 
# <br>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; font-size: 110%; text-transform: uppercase;">LINKS – Remember To Upvote The Original Author!</b>
# * <b><a href="https://www.kaggle.com/code/kvigly55/plldt-and-ddg">Notebook Link – pLLDT and DDG</a></b>
# * <b><a href="https://www.kaggle.com/kvigly55">Author - KVIGLY55</a></b>
# 
# <br>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; font-size: 110%; text-transform: uppercase;">Notebook Overview</b>
# 
# * This notebook/section leverages:
#     * The provided AlphaFold2 generated pdb file
#         * The pdb file is used by DeepDDG
#         * The pdb file contains a column <b>b_factor</b> that actually contains a value called <b>pLLDT</b> that represents the AlphaFold2 model <b><mark>confidence of prediction (per residue/Amino-Acid)</mark></b> which has been found to correlate with protein stabillity
#     * An existing model known as <a href="http://protein.org.cn/ddg.html"><b>DeepDDG</b></a>
#     * The <b><a href="https://www.wikiwand.com/en/BLOSUM">BloSum</a>100</b> substitution matrix
#     * Various algorithms to combine and rank the important values
#         * $\Delta \Delta G$ as produced by DeepDDG
#         * The b-factor (pLLDT) value
#         * The BLOSUM100 substitution matrix values

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.1 PREREQUISITE DATA FILES</h3>
# 
# ---
# 
# <br>
# 
# The first thing we do before even look at this notebook is to examine the attached data sources that are provided in addition to the default competition data files.
# * DataSet (clickable): <a href="https://www.kaggle.com/datasets/hengck23/submit-novozymes-00"><b><code>submit-novozymes-00</code></b></a>
#     * Specifically we care about the <b><code>wildtype_structure_prediction_af2.deepddg.ddg.txt</code></b> file which contains $ \Delta \Delta G $ predictions for every single point mutation in our test protein
#    
# To ensure I understood everything I regenerated this data file on my own via the DeepDDG server. It is identical to the one attached to the original notebook. Use either as you see fit or make your own.

# In[4]:


# 1.1 – Define the path to the model output text file (I'm using mine but you can use the original)
DEEPDDG_PRED_TXT_PATH = "/kaggle/input/novoesp-deepddg-server-predictions-sub-only/wildtype_structure_prediction_af2.ddg.txt"

# 1.2 – Read the txt file into a pandas dataframe specifying a space as the delimiter
#       --> This introduces some weirdness with the number of columns... so we have to drop some and rename them
#       --> We also drop the #chain column as it contains no usable information (all rows are the same --> A)
#       --> We also rename the columns to be more user friendly (note ddg stands for ΔΔG)
deepddg_pred_df = pd.read_table(DEEPDDG_PRED_TXT_PATH, sep=" ").drop(columns=["#chain", "ddG", "is", "stable,", "is.1", "unstable)", "<0"])
deepddg_pred_df.columns = ["wildtype_aa", "residue_id", "mutant_aa", "ddg", "ddg_"]

# 1.3 – Coerce all the ddg values to be in the right column
deepddg_pred_df["ddg"] = deepddg_pred_df["ddg"].fillna(0.0)+deepddg_pred_df["ddg_"].fillna(0.0)
deepddg_pred_df = deepddg_pred_df.drop(columns=["ddg_"])

# 1.4 – Change edit location string name and change from 1-indexed to 0-indexed
deepddg_pred_df.loc[:,'location'] = deepddg_pred_df["residue_id"]-1
deepddg_pred_df = deepddg_pred_df.drop(columns=["residue_id"])

# 1.5 – Create a new column that contains the entire mutation as a string
#   --> This mutation string has the format   <wildtype_aa><location><mutant_aa>
deepddg_pred_df.loc[:,'mutant_string'] = deepddg_pred_df["wildtype_aa"]+deepddg_pred_df["location"].astype(str)+deepddg_pred_df["mutant_aa"]

# 1.6 – Display the newly created datafarme containing predictions (and describe float/int based columns)
display(deepddg_pred_df.describe().T)
display(deepddg_pred_df)


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.2 UPDATE WILDTYPE ATOM DATAFRAME INFORMATION</h3>
# 
# ---
# 
# <br>
# 
# The wildtype information generated by AlphaFold2 needs minor updates
# * Subtract 1 from the residue number so that it will be 0-indexed instead of 1-indexed
# * Create mapping between residue number and the 'b-factor' which is really just pLLDT (prediction confidence ranging from 0 to 100)

# In[5]:


# 2.1 – Adjust the residue column to be 0-indexed instead of 1-indexed
atom_df['residue_number'] -= 1

# 2.2 – Create a mapping from residue to b-factor 
#   --> b-factor is always the same for a particular residue
residue_to_bfactor_map = atom_df.groupby('residue_number').b_factor.first().to_dict()


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.3 UPDATE TEST DATAFRAME</h3>
# 
# ---
# 
# <br>
# 
# Using the AlphaFold2 outputs and other information we can now update the test dataframe information
# * Update the test dataframe with information about the type of mutation, position of the mutation, wildtype amino acid and mutant amino acid
# * Update the test dataframe with information about AlphaFold2 prediction confidence (b-factor --> pLLDT)
# * Update some string information in the test dataframe related to the mutations

# In[6]:


# 3.1 – Add mutation information about test set
#   --> Type of Mutation ['substitution'|'deletion'|'no_change'|'insertion'] (insertion never occurs)
#   --> Position of Mutation (Note this is now 0 indexed from the previous cell/section)
#   --> Wildtype Amino Acid (Single Letter Short Form)
#   --> Mutant Amino Acid (Single Letter Short Form – Will be NaN in 'deletion' mutations)
test_df = test_df.progress_apply(get_mutation_info, axis=1)

# 3.2 – Add b-factor to the test dataframe using previously created dictionary
test_df['b_factor'] = test_df["edit_idx"].map(residue_to_bfactor_map)

# 3.3 – Change edit_type from NaN to 'no_change'
#   --> this will allow the entire column to be a string as NaN is considered a float
test_df["edit_type"] = test_df["edit_type"].fillna("no_change")

# 3.4 – Change mutant_aa from NaN to '+' or '-' if edit_type is 'insertion' or 'deletion' respectively 
#   --> this will allow the entire column to be a string as NaN is considered a float
test_df.loc[test_df['edit_type']=='deletion', 'mutant_aa'] = '-'
test_df.loc[test_df['edit_type']=='insertion', 'mutant_aa'] = '+'

# 3.5 – Create a new column that contains the entire mutation as a string
#   --> This mutation string has the format   <wildtype_aa><edit_idx><mutant_aa>
test_df.loc[:, 'mutant_string'] =  test_df["wildtype_aa"]+test_df["edit_idx"].astype(str)+test_df["mutant_aa"]
test_df[["seq_id", "edit_type", "edit_idx", "wildtype_aa", "mutant_aa", "b_factor","mutant_string"]].head()

# 3.6 – Drop columns we don't need for this notebook
test_df = test_df[[_c for _c in test_df.columns if (("__" not in _c) and ("data_source" not in _c))]]

# 3.7 – Display the updated dataframe (and describe float/int based columns)
display(test_df.describe().T)
display(test_df)


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.4 COMBINE TEST DATA WITH DEEPDDG PREDICTIONS</h3>
# 
# ---
# 
# <br>
# 
# As the two dataframes share the <b><code>mutant_string</code></b> column, we can simply merge them together so that we have the <b>DeepDDG</b> prediction information available in the test dataset.
# * We also use this opportunity to fill in any $\Delta \Delta G$ (DDG) values that are missing
#     * This occurs for the wildtype variation which can simply have a ddG of 0 (as it IS the thing we are measuring from)
#     * This occurs for all deletions as the DeepDDG server cannot predict on these.
#         
# <br>
# 
# <b style="color: darkred">In the original notebook all missing values are replaced with <mark>-0.25</mark></b>. We know this isn't perfect (probably a best guess), and we can leave it be for now... however, we DO know that the one row in the test dataset that contains the wildtype should have a DDG value of 0.0! Therefore I have updated this cell accordingly... this is probably what is resulting in the very slightly higher score of this notebook over the original.

# In[7]:


# 4.1 – Merge the two dataframes together 
test_df = test_df.merge(deepddg_pred_df[['ddg','mutant_string']], on='mutant_string', how='left')

# 4.2 – Fill in the missing ddg values with predetermined values
#   --> We set the default value for deletion to be equivalent to the bottom quartile value
#       of all substitutions... this is because it is more deleterious than simple substitutions
#   --> The default no_change value is simply 0.0 because this is the wildtype
#   --> What I would have thought is better: 
#            ----> DEFAULT__DELETION__DDG  = test_df[test_df["edit_type"]=="substitution"]["ddg"].quantile(q=0.25)
DEFAULT__DELETION__DDG  = -0.25
DEFAULT__NO_CHANGE__DDG = 0.0   # THIS IS DIFFERENT THAN THE ORIGINAL NOTEBOOK
test_df.loc[test_df['edit_type']=="deletion", 'ddg'] = DEFAULT__DELETION__DDG
test_df.loc[test_df['edit_type']=="no_change", 'ddg'] = DEFAULT__NO_CHANGE__DDG

# 4.3 – Display the updated dataframe (and describe float/int based columns)
display(test_df.describe().T)
display(test_df)


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.5 PERFORM MATRIX SUBSTITUTION (BLOSUM-100)</h3>
# 
# ---
# 
# <br>
# 
# **Substitution Matrices** are a concept in biology that is used to <b>determine the impact of changes between two sequences</b> (amino acids or nucleotides). 
# 
# **Here are the key points:**
# * The 20 amino acids translated by the genetic code vary greatly by the physical and chemical properties of their side chains.
# * However, these amino acids can be categorised into groups with similar physicochemical properties.
# * Substituting (single point mutation like in this competition) an amino acid with another from the <b>same category</b> is more likely to have a <b>smaller impact</b> on the structure and function of a protein than replacement with an amino acid from a different category.
# * Sequence alignment is a fundamental research method for modern biology. 
#     * The most common sequence alignment for protein is to look for similarity between different sequences in order to infer function or establish evolutionary relationships. 
#     * This helps researchers better understand the origin and function of genes through the nature of homology and conservation. 
# * Substitution matrices are utilized in algorithms to calculate the similarity of different sequences of proteins; 
#     * The utility of Dayhoff PAM Matrix has decreased over time due to the requirement of sequences with a similarity more than 85%. 
#     * In order to fill in this gap, Henikoff and Henikoff introduced <b>BLOSUM</b> (<b>BLO</b>cks <b>SU</b>bstitution <b>M</b>atrix) matrix which led to marked improvements in alignments and in searches using queries from each of the groups of related proteins.
# * Several sets of BLOSUM matrices exist using different alignment databases, named with numbers. 
#     * BLOSUM matrices with high numbers are designed for comparing closely related sequences, while those with low numbers are designed for comparing distant related sequences. 
#     * We use the BLOSUM matrix with the highest possible number as our sequences are only different by a single mutation.
# 
# <br>
# 
# ---
# 
# <br>
# 
# **BLOSUM SUBSTITUTION MATRIX**
# 
# <br>
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Blosum62-dayhoff-ordering.svg/1200px-Blosum62-dayhoff-ordering.svg.png" width=100%>
# 
# <br>
# 
# ---
# 
# <br>
# 
# Let's look at a trivial simple example (not exactly the same as a real substitution matrix) to understand. (Please note this is just using dummy values and English characters not Amino Acids).
# * Let's assume that we have the following scores for substitutions
#     * **`A <--> B = 1`**
#     * **`A <--> C = 2`**
#     * **`B <--> C = 1`**
# * For the following two sequences:
#     * Sequence 1 --> **`AAABBCC`**
#     * Sequence 2 --> **`ACABBBB`**
# * Therefore we would see that between the two sequences we have the following scores:
#     * **`A --> A = 0`**
#     * **`A --> C = 2`**
#     * **`A --> A = 0`**
#     * **`B --> B = 0`**
#     * **`B --> B = 0`**
#     * **`C --> B = 1`**
#     * **`C --> B = 1`**
#     * **`SUM IS  = 4`**
# * If we sum these scores up we see that the score for these two strings is **`4`**. If we do this again with the same amount of substitutions but in different places we can get a different score because the **`A <--> C`** substitution is represents a larger difference than the **`A <--> B`** substitution
# * For the following two sequences:
#     * Sequence 1 --> **`AAABBCC`**
#     * Sequence 2 --> **`CCCBBCC`**
# * Therefore we would see that between the two sequences we have the following scores:
#     * **`A --> C = 2`**
#     * **`A --> C = 2`**
#     * **`A --> C = 2`**
#     * **`B --> B = 0`**
#     * **`B --> B = 0`**
#     * **`C --> C = 0`**
#     * **`C --> C = 0`**
#     * **`SUM IS  = 6`**
# * Therefore we can see that this combination of strings results in a difference that is 1.5 times greater even though the same number of substitutions is used.

# In[8]:


# from Bio.Align import PairwiseAligner
# from Bio.Align import substitution_matrices


## substitution_matrices.load()
## len(list(MatrixInfo.blosum100.values()))
## algorithm Smith-Waterman
## matrix Blosum 100
## gap penalty normalized and weighted
## gap opening penalty 5
## gap extension penalty 0.5


# test_df[test_df["edit_type"]=="deletion"].protein_sequence.values[0]
# aligner = PairwiseAligner()
# aligner.substitution_matrix = substitution_matrices.load("BLOSUM60")
# alignment = aligner.align(wildtype_aa, test_df[test_df["edit_type"]=="deletion"].protein_sequence.values[0])
# alignment


# In[9]:


# 5.1 – Define a function to return the substitution matrix (backwards and forwards)
from Bio.SubsMat import MatrixInfo
def get_sub_matrix(matrix_name="blosum100"):
    sub_matrix = getattr(MatrixInfo, matrix_name)
    sub_matrix.update({(k[1], k[0]):v for k,v in sub_matrix.items() if (k[1], k[0]) not in list(sub_matrix.keys())})
    return sub_matrix
sub_matrix = get_sub_matrix()

# 5.2 – Conduct matrix substitution
#   --> First we create a tuple that has the wildtype amino acid and the 
#       mutant amino acid to access the substitution matrix
#   --> Second we access the substitution matrix and replace with the respective score
#       and in cases where no respective score is found we mark it to be updated later
test_df["sub_matrix_tuple"] = test_df[["wildtype_aa", "mutant_aa"]].apply(tuple, axis=1)
test_df["sub_matrix_score"] = test_df["sub_matrix_tuple"].progress_apply(lambda _mutant_tuple: sub_matrix.get(_mutant_tuple, "tbd"))

# 5.3 – Fill in the missing data with default values for now
#   --> We set the default value for matrix sub to be equivalent to the bottom quartile value
#       of all substitutions... this is because it is more deleterious than simple substitutions (larger difference)
#   --> The default no_change value is 1 higher than the max score because a higher score means more similarity
#DEFAULT__DELETION__MATRIXSCORE  = test_df[test_df["edit_type"]=="substitution"]["sub_matrix_score"].quantile(q=0.25)
DEFAULT__DELETION__MATRIXSCORE = -10.0
#DEFAULT__NO_CHANGE__MATRIXSCORE = test_df[test_df["edit_type"]=="substitution"]["sub_matrix_score"].max()+1.0
DEFAULT__NO_CHANGE__MATRIXSCORE = 0.0
test_df.loc[test_df['edit_type']=="deletion", 'sub_matrix_score'] = DEFAULT__DELETION__MATRIXSCORE
test_df.loc[test_df['edit_type']=="no_change", 'sub_matrix_score'] = DEFAULT__NO_CHANGE__MATRIXSCORE
test_df["sub_matrix_score"] = test_df["sub_matrix_score"].astype(float)

# 5.4 – Display the updated dataframe (and describe float/int based columns)
display(test_df.describe().T)
display(test_df)


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.6 ADJUST THE MATRIX SUBSTITUTION AND B_FACTOR VALUES</h3>
# 
# ---
# 
# **The key takeaways are as follows:**
# * The matrix substitution score (which tells us how much of an impact a particular single point substitution has) is normalized with a flattened inverted version of a sigmoid fn.
#     * Very large negative scores will normalize to approaching 1
#         * So changes resulting in a large impact have a score approaching 1
#     * Scores approaching 0 will normalize to approaching 0.5
#         * So changes resulting in a moderate/small impact have a score approaching 0.5
#     * Very large positive scores will normalize to approaching 0
#         * So changes resulting in a negligible impact have a score approaching 0.0
#     * The change from 0.5 to -1 or 0.5 to 0 is flatter when compared to a traditional sigmoid meaning that changes will have a smaller impact than normal (this may be to account for the relatively tight distribution).
#     * So tl;dr the matrix sub score will be normalized such that:
#         * It spans from 0-1
#         * Closer to 0 means negligible impact of mutation while 1 means extreme impact of mutation
# * We then multiply this normalized matrix substitution score with the provided b-factor
#     * Since the b-factor is analagous to AlphaFold2 model confidence, this operation can be thought of as refining the matrix substitution score by including a factor that changes it based on the confidence of the structural prediction of AlphaFold2 
#     * i.e. The impact of matrix substitution might be 0.95 (very high) but if the b-factor is 30 (very low), then the resulting value is only 28.5
#     * i.e. The impact of matrix substitution might be 0.6 (slightly higher than average) but if the b-factor is 95 (very high), than the resulting value is 57 (double the previous value!)
#     
# <br><br>
# 
# The above being said, this is the most abstruse cell I've found in this entire notebook. The information below attempts to delve deeper by looking at the equations and plots generated by them. 
# 
# The code provided is as follows:
# 
# <b>1. Capping The Matrix Substitution Score</b>
# 
# > ```python
# > test_df.loc[test_df['sub_score'] > 0, 'sub_score'] = 0
# > ```
# 
# <b>2. Creating An 'Adjusted' Matrix Substritution Score</b>
# 
# > ```python
# > test_df['score_adj'] = [
# >    1 - (1 / (1+np.exp(-x/sigmoid_norm_factor))) for x in sub_scores
# > ]
# > ```
# 
# <b>3. Combining the 'Adjusted' Matrix Substution Score with AlphaFold2 Predicted B-Factor</b>
# 
# > ```python
# > test_df['b_factor_adj'] = test_df['b_factor'] * test_df['score_adj']
# > ```
# 
# <br>
# 
# ---
# 
# <br>
# 
# Let's view this as Latex to see if it makes things clearer
# 
# <br>
# 
# <b>1. Capping The Matrix Substitution Score As A Function</b>
# 
# $$f(x) = min(x, 0)$$
# 
# <b>2. Creating An 'Adjusted' Matrix Substritution Score</b>
# 
# $$
# f(x) = 1 - \dfrac{1}{1+e^{\dfrac{-x}{s_f}}}
# $$
# 
# **NOTE:** 
# * This is quite similar to the equation for the sigmoid/logistic equation 
# * The difference is just an adjustment factor ($s_f$) and inversion (`1-`)
# 
# $$
# f(x) = \dfrac{1}{1+e^{-x}}
# $$
# 
# <br>
# 
# <b><sub>Here's a photo comparing the provided equation (dark-red) with the default $s_f$ of $3$ and that of the traditional sigmoid (dark-blue). The inversion is simply flipping the delta from positive to a negative around 0.5 ... and the impact of the $s_f$ term is that the curve is flattened. i.e. at $x=1$ $y=0.417$ ( a change of 0.83 ) in the provided equation and $y=0.731$  ( a change of 0.231 ) in the classic sigmoid equation.</sub></b>
# 
# <center><img src="https://i.ibb.co/Y7n2qKk/desmos-graph.png"></center>
# 
# <br>
# 
# <b>3. Combining the 'Adjusted' Matrix Substution Score with AlphaFold2 Predicted B-Factor</b>
# 
# No latex here... this is just elementwise multiplication of the AlphaFold2 B-Factor and the Normalized/Adjusted Matrix Substitution Score.
# 
# <br>
# 

# In[10]:


# 6.1 – If the flag is set, reduce all positive matrix substitution scores to 0
CAP_SUB_MATRIX_SCORE = True
CAP_ABOVE_VAL, CAP_VAL = 0.0, 0.0
if CAP_SUB_MATRIX_SCORE:
    test_df.loc[test_df['sub_matrix_score'] > CAP_ABOVE_VAL, 'sub_matrix_score'] = CAP_VAL

# 6.2 – Normalize the matrix substitution score with adjusted sigmoid function
SIGMOID_ADJUSTMENT_CONSTANT = 3
def sigmoid_w_adjustment(x, adjustment_factor=3.0):
    return 1-(1/(1+np.exp(-x/adjustment_factor)))
test_df['sub_matrix_score_normalized'] = test_df['sub_matrix_score'].apply(lambda x: sigmoid_w_adjustment(x, SIGMOID_ADJUSTMENT_CONSTANT))
test_df['b_factor_matrix_score_adjusted'] = test_df['b_factor']*test_df['sub_matrix_score_normalized'] 

# 6.3 – Display the updated dataframe (and describe float/int based columns)
display(test_df.describe().T)
display(test_df)


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.7 CALCULATE RANKS OF IMPORTANT COLUMNS</h3>
# 
# ---
# 
# For the important columns we determine the rankings (from largest to smallest for instance) and store those as new columns. The important columns are as follows (the +/- indicates if the ranking is ascending or descending)
# * **`+`**`ddg`
# * **`+`**`sub_matrix_score`
# * **`-`**`b_factor`
# * **`-`**`b_factor_matrix_score_adjusted`
# 
# To do this we will use the **`scipy.stats`** library, specifically the **`scipy.stats.rankdata`** method.

# In[11]:


from scipy import stats

# 7.1 – Assign ranks to data, dealing with ties appropriately.
test_df['ddg_rank'] = stats.rankdata(test_df['ddg'])
test_df['sub_matrix_score_rank'] = stats.rankdata(test_df['sub_matrix_score'])
test_df['b_factor_rank'] = stats.rankdata(-test_df['b_factor'])
test_df['b_factor_matrix_score_adjusted_rank'] = stats.rankdata(-test_df['b_factor_matrix_score_adjusted'])

# 7.2 – Display the updated dataframe (and describe float/int based columns)
display(test_df.describe().T)
display(test_df)


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.8 COMBINE THE RANKS OF THE IMPORTANT COLUMNS</h3>
# 
# ---
# 
# This is weird... because we seemingly ignore the calculations we just performed (prior to ranking) in favour of the non-normalized values. 
# 
# Basically the combination is the following equation
# 
# $$
# f(x,y,z) = (x*y*z)^{\dfrac{1}{3}}
# $$
# 
# <br>
# 
# <img src="https://i.stack.imgur.com/Ljk9m.png">
# 
# 

# In[12]:


# 8.1 – Identify the columns to be combined
combo_cols = ['b_factor_rank', 'sub_matrix_score_rank', 'ddg_rank']

# 8.2 – Combine the columns by multiplying them together
test_df["combined_val"] = test_df[combo_cols].apply(np.prod, axis=1)

# 8.3 – Plot the newly created column to see the distribution of values
#       prior to any type of manipulation (i.e. raising to 1/3 power)
plt.figure(figsize=(14,6))
test_df["combined_val"].hist()
plt.title("Distribution of Values Prior To Being Raised to 1/3 Power")
plt.xlabel("b_factor_rank * sub_matrix_score_rank * ddg_rank", fontweight="bold")
plt.ylabel("frequency of occurence in dataset", fontweight="bold")
plt.show()

# 8.4 – 'Normalize' the combined value column by raising to a particular power
COMBO_NORM_PWR = 1/3
test_df["norm_combined_val"] = test_df["combined_val"]**COMBO_NORM_PWR

# 8.5 – Plot the newly created column to see the distribution of values
#       after manipulation (i.e. raising to 1/3 power)
plt.figure(figsize=(14,6))
test_df["norm_combined_val"].hist()
plt.title("Distribution of Combined Normalized Values After Being Raised to 1/3 Power")
plt.xlabel("(b_factor_rank * sub_matrix_score_rank * ddg_rank)^(1/3)", fontweight="bold")
plt.ylabel("frequency of occurence in dataset", fontweight="bold")
plt.show()

# 8.6 – Display the updated dataframe (and describe float/int based columns)
display(test_df.describe().T)
display(test_df)


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.9 CREATE SUBMISSION FILE USING COMBINED NORMALIZED RANK VLAUE</h3>
# 
# ---
# 
# Straightforward

# In[13]:


# 9.1 – Check that no NaN values slipped through
assert not test_df["norm_combined_val"].isna().any()

# 9.2 – Update the sample submission dataframe by creating a mapping and
#       applying it to the previously created sample_submission.csv
seqid_2_tmrank_mapping = test_df.groupby("seq_id")["norm_combined_val"].first().to_dict()
ss_df["tm"] = ss_df["seq_id"].apply(lambda x: seqid_2_tmrank_mapping[x])

# 9.3 – Save properly for submission
ss_df.to_csv("submission.csv", index=False)

# 9.4 – Display the sample submission dataframe and it's details
display(ss_df.describe().T)
display(ss_df)


# <br>
# 
# <a id="observations"></a>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #ffffff; color: navy;" id="observations">6&nbsp;&nbsp;OBSERVATIONS&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">6.1 GENERAL OBSERVATIONS</h3>
# 
# ---
# 
# I'll come back to this
# 
# <b>Here are some key takeaways</b>
# * These notebooks leverage existing models and/or provided files
# * These approaches rely on the correlation between $\Delta \Delta G$ and $T_m$
# * These approaches rely on the correlation between $pLLDT$ (b_factor) and $T_m$

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">6.2 OBSERVATIONS REGARDING <b><a href="https://www.kaggle.com/code/kvigly55/plldt-and-ddg">PLLDT AND DDG</a></b></h3>
# 
# ---
# 
# <br><b style="text-decoration: underline; font-family: Verdana; font-size: 110%; text-transform: uppercase;">POSITIVES</b>
# * Strong performance
# * Lots of knobs/dials that could/can be tweaked or replaced with drop-in alternatives <b><a href="https://demask.princeton.edu/">(demask</a><i> - as suggested by </i><a href="https://www.kaggle.com/hengck23"><i>hengck23</i></a>)</b>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; font-size: 110%; text-transform: uppercase;">OPEN QUESTIONS</b>
# * <b>Why were some default values chosen??</b>
#     * <i>? default replacement value of -0.25 for missing ddG values ?</i>
#     * <i>? default replacement value of -10 for missing substitution matrix values ?</i>
# * <b>Why were certain algorithms and certain constants used?</b>
#     * <i>? the modified sigmoid algorithm ?</i>
#     * <i>? multiplying the 3 columns and raising to the power of 1/3 ?</i>
#         * <font color="darkgreen">In this instance raising the output to the power of 0.333 doesn't accomplish anything as the values are scored based on rank order and raising them to 0.33333 maintains the order... I think</font>
# * <b>Why were certain values created and never used?</b>
#     * <i>? the matrix substitution normalized/adjusted columns ?</i>
#     

# <br>
# 
# <a id="next_steps"></a>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #ffffff; color: navy;" id="next_steps">7&nbsp;&nbsp;NEXT STEPS&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>
# 
# ---
# 
# * DeepDDG <b>+</b> BLOSUM100 <b>+</b> b_factor
#     * <b>Score --> 0.399 LB</b>
# * DeepDDG <b>+</b> DeMaSk Matrix Substitution <b>+</b> b_factor <b>+</b> BLOSUM100
#     * * <b>Score --> <mark>0.450 LB</mark></b>
# * DeepDDG <b>+</b> DeMaSk Matrix Substitution <b>+</b> b_factor
#     * <b>Score --> 0.425 LB</b>
# * DeepDDG
#     * <b>Score --> 0.345 LB</b>
# * DeMaSk Matrix Substitution
#     * <b>Score --> 0.363 LB</b>
# * DeepDDG <b>+</b> DeMaSk Matrix Substitution
#     * <b>Score --> 0.410 LB</b>

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">7.1 DEMASK DROP-IN</h3>
# 
# ---
# 
# <b>As suggested by <a href="https://www.kaggle.com/hengck23"><i>hengck23</i></a></b>
# 
# <br>
# 
# I have generated both generic and specific substitution matrices using DeMaSk but the position specific substitution works better.
# * I'm not 100% certain this is the best way to do this so forgive any mistakes/errors that might reduce performance.
# 

# In[14]:


# 1.1 – Create substitution matrix for each position from DeMaSk data (apologies for the dump file names)
demask_sub_matrix = pd.read_csv("../input/demask-directional-substitution-matrix/inpsseq_mutation_ddg - Sheet2.csv")
demask_tuple_sub_map = {}
for _, _row in demask_sub_matrix.iterrows():
    demask_tuple_sub_map[(_row["pos"]-1, _row["WT"], _row["var"])] = _row["score"]
    demask_tuple_sub_map[(_row["pos"]-1, _row["var"], _row["WT"])] = _row["score"]

# 1.2 – Conduct matrix substitution
#   --> First we create a tuple that has the wildtype amino acid and the 
#       mutant amino acid to access the substitution matrix
#   --> Second we access the substitution matrix and replace with the respective score
#       and in cases where no respective score is found we mark it to be updated later
test_df["demask_sub_matrix_tuple"] = test_df[["edit_idx", "wildtype_aa", "mutant_aa"]].apply(tuple, axis=1)
test_df["demask_sub_matrix_score"] = test_df["demask_sub_matrix_tuple"].progress_apply(lambda _mutant_tuple: demask_tuple_sub_map.get(_mutant_tuple, "tbd"))

# 1.3 – Fill in the missing data with default values for now
#   --> We aim to make this as similar to the -10.0 that was previously used as possible
#       the most similar value considering the different scale would be --> -0.75
#   --> No change remains as 0.0
DEMASK_DEFAULT__DELETION__MATRIXSCORE = -0.75 # This is about 1.25 times larger than the minimum value (just like -10 in the original)
DEMASK_DEFAULT__NO_CHANGE__MATRIXSCORE = 0.0
test_df.loc[test_df['edit_type']=="deletion", 'demask_sub_matrix_score'] = DEMASK_DEFAULT__DELETION__MATRIXSCORE
test_df.loc[test_df['edit_type']=="no_change", 'demask_sub_matrix_score'] = DEMASK_DEFAULT__NO_CHANGE__MATRIXSCORE
test_df["demask_sub_matrix_score"] = test_df["demask_sub_matrix_score"].astype(float)

# 1.4 – Calculate the column rank similar to what we did previously
test_df['demask_sub_matrix_score_rank'] = stats.rankdata(test_df['demask_sub_matrix_score'])

# 1.5 – Plot the delta between the assigned rank values between the two approaches for raw sub_matrix_score
plt.figure(figsize=(18, 6))
(test_df["sub_matrix_score_rank"]-test_df["demask_sub_matrix_score_rank"]).abs().plot(kind="hist", bins=100)
plt.title("Disagreement Amount Between DeMaSk and BloSUM100", fontweight="bold")
plt.xlabel("Delta Change in Rank Between DeMaSk & BloSUM100", fontweight="bold")
plt.ylabel("Frequency of Occurence", fontweight="bold")
plt.grid(which="both")
plt.show()

# 1.6 – Identify the columns to be combined
combo_cols = ['b_factor_rank', 'demask_sub_matrix_score_rank', 'ddg_rank']

# 1.7 – Combine the columns by multiplying them together
test_df["demask_combined_val"] = test_df[combo_cols].apply(np.prod, axis=1)

# 1.7/ – To be fully added later... convert both outputs to rank
test_df["combined_val_rank"] = stats.rankdata(test_df['combined_val'])
test_df["deamsk_combined_val_rank"] = stats.rankdata(test_df['demask_combined_val'])

# 1.8 – Plot the newly created column to see the distribution of values
#       prior to any type of manipulation (i.e. raising to 1/3 power)
plt.figure(figsize=(14,6))
test_df["demask_combined_val"].hist()
plt.title("Distribution of Values Prior To Being Raised to 1/3 Power")
plt.xlabel("b_factor_rank * demask_sub_matrix_score_rank * ddg_rank", fontweight="bold")
plt.ylabel("frequency of occurence in dataset", fontweight="bold")
plt.show()

# 1.9 – 'Normalize' the combined value column by raising to a particular power (as we are using Spearman this does nothing)
COMBO_NORM_PWR = 1/3
test_df["demask_norm_combined_val"] = test_df["demask_combined_val"]**COMBO_NORM_PWR

# 1.10 – Plot the newly created column to see the distribution of values
#       after manipulation (i.e. raising to 1/3 power)
plt.figure(figsize=(14,6))
test_df["demask_norm_combined_val"].hist()
plt.title("Distribution of Combined Normalized Values After Being Raised to 1/3 Power")
plt.xlabel("(b_factor_rank * demask_sub_matrix_score_rank * ddg_rank)^(1/3)", fontweight="bold")
plt.ylabel("frequency of occurence in dataset", fontweight="bold")
plt.show()

# 1.11 – Plot the delta between the assigned rank values between the two approaches after combination and normalization
plt.figure(figsize=(18, 6))
(test_df["norm_combined_val"]-test_df["demask_norm_combined_val"]).abs().plot(kind="hist", bins=100)
plt.title("Disagreement Amount Between DeMaSk and BloSUM100", fontweight="bold")
plt.xlabel("Delta Change in Rank Between DeMaSk & BloSUM100", fontweight="bold")
plt.ylabel("Frequency of Occurence", fontweight="bold")
plt.grid(which="both")
plt.show()

# 1.12 – Create the submission file
demask_seqid_2_tmrank_mapping = test_df.groupby("seq_id")["demask_norm_combined_val"].first().to_dict()
ss_df["tm"] = ss_df["seq_id"].apply(lambda x: demask_seqid_2_tmrank_mapping[x])
ss_df.to_csv("demask_submission.csv", index=False)

# 1.13 – Display the sample submission dataframe and it's details
display(ss_df.describe().T)
display(ss_df)

# 1.14 – Display the updated test dataframe (and describe float/int based columns)
display(test_df.describe().T)
display(test_df)


# In[15]:


# 1.11 – Plot the delta between the assigned rank values between the two approaches after combination and normalization
plt.figure(figsize=(18, 6))
(test_df["demask_combined_val"]-test_df["demask_norm_combined_val"]).abs().plot(kind="hist", bins=100)
plt.title("Disagreement Amount Between DeMaSk and BloSUM100", fontweight="bold")
plt.xlabel("Delta Change in Rank Between DeMaSk & BloSUM100", fontweight="bold")
plt.ylabel("Frequency of Occurence", fontweight="bold")
plt.grid(which="both")
plt.show()


# <br>
# 
# **Submission**
# * Equal Weighting of 4 Rank Columns

# In[21]:


cols_to_ensemble = ["ddg_rank", "b_factor_rank", "sub_matrix_score_rank", "demask_sub_matrix_score_rank", ]
ss_df["tm"] = test_df[cols_to_ensemble].mean(axis=1)
ss_df.to_csv("submission.csv", index=False)


# <br>
# 
# **What About the Wildtype Ranking?**

# In[28]:


test_df[test_df["seq_id"]==32559][[_c for _c in test_df.columns if 'rank' in _c]].T.rename(columns={1169:"wildtype_relative_ranking"})


# <br>
# 
# **Experiment w/ Asymmetry**
# * blosum100

# In[27]:


# 5.1 – Define a function to return the substitution matrix (backwards and forwards)
from Bio.SubsMat import MatrixInfo
def get_sub_matrix(matrix_name="blosum100"):
    sub_matrix = getattr(MatrixInfo, matrix_name)
    sub_matrix.update({(k[1], k[0]):v for k,v in sub_matrix.items() if (k[1], k[0]) not in list(sub_matrix.keys())})
    return sub_matrix
sub_matrix = get_sub_matrix()

# 5.2 – Conduct matrix substitution
#   --> First we create a tuple that has the wildtype amino acid and the 
#       mutant amino acid to access the substitution matrix
#   --> Second we access the substitution matrix and replace with the respective score
#       and in cases where no respective score is found we mark it to be updated later
test_df["sub_matrix_tuple_asym"] = test_df[["mutant_aa", "wildtype_aa"]].apply(tuple, axis=1)
test_df["sub_matrix_score_asym"] = test_df["sub_matrix_tuple_asym"].progress_apply(lambda _mutant_tuple: sub_matrix.get(_mutant_tuple, "tbd"))

# 5.3 – Fill in the missing data with default values for now
#   --> We set the default value for matrix sub to be equivalent to the bottom quartile value
#       of all substitutions... this is because it is more deleterious than simple substitutions (larger difference)
#   --> The default no_change value is 1 higher than the max score because a higher score means more similarity
#DEFAULT__DELETION__MATRIXSCORE  = test_df[test_df["edit_type"]=="substitution"]["sub_matrix_score"].quantile(q=0.25)
DEFAULT__DELETION__MATRIXSCORE = -10.0
#DEFAULT__NO_CHANGE__MATRIXSCORE = test_df[test_df["edit_type"]=="substitution"]["sub_matrix_score"].max()+1.0
DEFAULT__NO_CHANGE__MATRIXSCORE = 0.0
test_df.loc[test_df['edit_type']=="deletion", 'sub_matrix_score_asym'] = DEFAULT__DELETION__MATRIXSCORE
test_df.loc[test_df['edit_type']=="no_change", 'sub_matrix_score_asym'] = DEFAULT__NO_CHANGE__MATRIXSCORE
test_df["sub_matrix_score_asym"] = test_df["sub_matrix_score_asym"].astype(float)
test_df["sub_matrix_score_asym_rank"] = stats.rankdata(test_df["sub_matrix_score_asym"])

# 5.4 – Display the updated dataframe (and describe float/int based columns)
display(test_df.describe().T)
display(test_df)


# In[33]:


cols_to_ensemble = ["ddg_rank", "ddg_rank", "ddg_rank", "b_factor_rank", "b_factor_rank", "b_factor_rank", "sub_matrix_score_rank", "sub_matrix_score_rank", "sub_matrix_score_asym_rank", "demask_sub_matrix_score_rank", "demask_sub_matrix_score_rank", "demask_sub_matrix_score_rank"]
ss_df["tm"] = test_df[cols_to_ensemble].mean(axis=1)
ss_df.to_csv("submission.csv", index=False)


# In[ ]:




