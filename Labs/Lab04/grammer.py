import nltk
from nltk import CFG
from nltk.parse import EarleyChartParser

grammar = CFG.fromstring(r"""
S    -> NP AUX VP
S    -> NP VP
S    -> NP AUX VGER              

NP   -> DET NOM
NP   -> NOM
NP   -> NOM COMP
NP   -> NP CONJ NP               
NP   -> VGER NP                  

NOM  -> ADJ NOM
NOM  -> N
NOM  -> NOM PP                                    

VP   -> V ADJ                   
VP   -> V NP                    
VP   -> VP COMP                 

PP   -> P NP
COMP -> 'more' 'than' NP

DET  -> 'the' | 'The'
CONJ -> 'and'
P    -> 'of'

N    -> 'planes' | 'parents' | 'bride' | 'groom'
ADJ  -> 'dangerous' | 'flying' | 'Flying'
V    -> 'loves' | 'be'
AUX  -> 'can' | 'were'
VGER -> 'flying' | 'Flying'
""")

parser = EarleyChartParser(grammar)

sentences = [
    "Flying planes can be dangerous".split(),
    "The parents of the bride and the groom were flying".split(),
    "The groom loves dangerous planes more than the bride".split()
]

for sent in sentences:
    print(f"\n=== Sentence: {' '.join(sent)} ===")
    parses = list(parser.parse(sent))
    print(f"Total parses found: {len(parses)}")

    if parses:
        for i, tree in enumerate(parses[:2], 1):
            print(f"\n--- Tree #{i} ---")
            print(tree)
            tree.pretty_print()
    else:
        print("No parse found.")
