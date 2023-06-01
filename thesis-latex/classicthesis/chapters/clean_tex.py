import sys
 
in_file = sys.argv[1]
 
text = open(in_file, "r").read()
clean_text = text
replacements = {
    r"\{cite:t\}\texttt{": r"\citet{",
    r"\{cite\}\texttt{": r"\cite{",
    r"\{cite:p\}\texttt{": r"\citep{",
    r"\subsection{": r"\section{",
    r"\{numref\}\texttt{": r"\Cref{",
    r"\{ref\}\texttt{": r"\Cref{",
    }
for val_to_replace, replace_val in replacements.items():
    clean_text = clean_text.replace(val_to_replace, replace_val)
with open(in_file, "w") as cleaned_file:
    cleaned_file.write(clean_text)

#print(text[:3000])
