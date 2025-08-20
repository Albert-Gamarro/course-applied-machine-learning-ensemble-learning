# Applied Machine Learning: Ensemble Learning (Personal Workspace)

This repository is my **personal workspace** for the LinkedIn Learning course
"Applied Machine Learning: Ensemble Learning" by Matt Harrison.  

All code, notes, and experiments are set up using my **own Conda environment** and 
Jupyter notebooks in **VSCode**, not the default Codespaces setup.  




💡 Analysing results of our modles:

Combo A → mean F1 = 0.78, std = 0.02 → consistent across folds, reliable.

Combo B → mean F1 = 0.80, std = 0.10 → slightly better mean, but unstable.

Why it matters:

In real life, you don’t just chase the highest mean — you also care about stability (low std).

A high std suggests your model might perform very differently depending on the split → riskier in production.

