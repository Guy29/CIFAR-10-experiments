###
### To use this word counting tool, issue this command in the command line:
### 
### python3 count_jupyter_nb_words.py XXXXXX.ipynb
### 
### where XXXXXX.ipynb is the name of your Notebook file, and XXXXXX is your candidate number.
### 
import nbformat
import sys
import re

def count_words_in_markdown(notebook_file):
    # Load the notebook
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Iterate through each cell and count words in markdown cells
    limit = 2500
    words_to_be_replaced = 80
    word_count = 0

    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            # Extract text from the markdown cell
            cell_content = cell.source
            if '<ignore>' in cell_content:
                continue
            else:
                # Remove frmatting that should not contribute towards the word count.
                cell_content = re.sub('\<(lt)\>\$[^\$]+\$','',cell_content) # Remove in-line Latex (flagged by <lt>$.$)
                cell_content = re.sub('\<(lt)\>(\$){2}[^\$]+(\$){2}','',cell_content) # Remove newline Latex (flagged by <lt>$$.$$)
                cell_content = re.sub('\<(figure)\>.*\<(\/figure)\>','',cell_content) # Remove figures (flagged by <figure>.</figure>)
                # Count words remaining in this cell
                cell_count = len(cell_content.split())
                # Update word count
                word_count += cell_count
                # Check whether word count exceeds limit
                if (word_count > limit) and ((word_count - cell_count)<limit):
                    print("stop marking from here: {}" .format(' '.join(cell['source'].split()[(-(word_count - limit)-5):(-(word_count - limit))])))
            
    return word_count

fname = sys.argv[1] # Provide notebook file name
word_count = count_words_in_markdown(fname)
print(f"Total number of words in markdown cells (ignoring LaTeX and figures): {word_count}")
