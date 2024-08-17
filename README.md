# NotebookAnalysis
This repo contains the code used to process jupyter notebooks,
and to perform ast analysis on the resulting python code.
Further it includes code to evaluete the resulting data for 
occurance and frequenzy of python functions and modules.

It consist of two parts.

## Part one: convert_and_ast.ipynb
###
The function `convert_to_python` converts jupyter notebooks to python code.

###
The function `process_python` takes a python file and extracts the syntax tree
using the AST module. From the syntax tree it exracts all the symbols for funtion 
invokation e.g. `df.sum()` >>> `map`and and collects them. It then appends these
stats to a file.


## Part two: data_analysis.ipynb
This notebook first reads from the file produced by `convert_to_python` and aggregates
the data form the seperate notebooks into combined dictionaries. 
These dictionaries can then be analysed or used for plotting.
 
