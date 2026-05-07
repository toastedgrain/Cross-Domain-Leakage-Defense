# Cross-Domain Query/Memory Domain Relationship

- Samples: 200
- Chi-square statistic: 79.20
- Cramer's V: 0.222
- Interpretation: Cramer's V near 0 means weak association; near 1 means strong association.

## Most Overrepresented Domain Pairs

- Work memory -> Health query: n=5, lift=3.03x
- Social memory -> Journals query: n=4, lift=2.86x
- Work memory -> Finance/Legal query: n=4, lift=2.80x
- Work memory -> Beliefs query: n=3, lift=2.73x
- Journals memory -> Social query: n=5, lift=2.60x
- Identity memory -> Romantic query: n=3, lift=2.22x
- Finance/Legal memory -> Beliefs query: n=2, lift=2.11x
- Social memory -> Health query: n=3, lift=2.00x

## Suggested Visualizations

- Count heatmap: best first view for absolute dataset composition.
- Row-normalized heatmap: best for asking where each memory domain tends to be queried.
- Lift heatmap: best for finding relationships beyond marginal domain frequency.
- Bipartite/alluvial diagram: useful in a paper or slide when emphasizing flows from memory domains to query domains.
- Clustered heatmap: useful if domain order should be data-driven rather than semantic.
