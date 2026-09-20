---
name: biomni-molbio-plasmid-orf
description: Use to annotate an open-reading frame on a sequence, design primers for a region, or characterize a plasmid sequence (linear or circular).
x-scimas-role: molecular-biologist
x-scimas-server: biomni-molecular_biology
x-scimas-tools:
  - annotate_open_reading_frames
  - annotate_plasmid
  - find_restriction_sites
  - find_restriction_enzymes
---

# Plasmid & ORF annotation

## When to use
- User has a DNA sequence and wants ORF / start-codon annotation.
- User has a circular plasmid map (GenBank / FASTA) and wants a feature table.
- User wants to enumerate restriction-enzyme cut sites on a given sequence, or pick enzymes that cut a given sequence.

## Limitations
- Uses Biopython Restriction / SeqIO; circular / linear flag matters.
- ``find_restriction_enzymes`` returns Biopython's REBASE subset, not the full commercial catalog.
