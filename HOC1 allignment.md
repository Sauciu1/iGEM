We wanted to visually compare whether the active site was conserved for HOC1 full and Hoc1 truncated version.
Note that both of these structures were folded with Alphafold, hence the sequence might not be representative of reality.


# Proteins:
#### Truncated HOC1:
* Putative mannosyltransferase
* https://alphafold.ebi.ac.uk/entry/F2QVW2
* Pichia pastoris 
* 286AA
* Uniprot: F2QVW2



#### Full HOC1:
* Alpha-1,6-mannosyltransferase
* https://alphafold.ebi.ac.uk/entry/Q71A36
* 402 AA
* Uniprot: Q71A36


# Workflow
Launch pyMOL 
Install plugin https://github.com/APAJanssen/Alphafold2import

#### Sequence alignment  
``` pyMOL
fetchAF2 F2QVW2, HOC1_truncated
fetchAF2 Q71A36, HOC1_full
align HOC1_truncated, HOC1_full
color blue, HOC1_full
color red, HOC1_truncated

```
The sequences should now be aligned.

To show as ribbon:
```
show_as ribbon, HOC1_full HOC1_truncated

```

![Wireframe as ribbon](ribbon.png)

We can see that the truncated C terminus occurs at the spheric of the protein. It well explains decreased functionality of the protein.


```
show_as surface, HOC1_full HOC1_truncated

```

![Wireframe as ribbon](ribbon.png)