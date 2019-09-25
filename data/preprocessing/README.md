Data from:

- Taur, Ying, et al. "Reconstitution of the gut microbiota of antibiotic-treated patients by autologous fecal microbiota transplant." Science translational medicine 10.460 (2018): eaap9489.

The files ```taur-otu-table.csv```
```taur-events.csv``` contain the relevant tables for analysis.

The script ```pipeline.sh``` runs all the preprocessing steps. Call

```
./pipeline.sh
```
to re-generate the data tables.

The files

```
antibiotics.csv
otu_couts.csv
otu_taxonomy.csv
patient_table.csv
```
are tables from the original Excel file of Taur *et al.*