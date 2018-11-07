#!/bin/bash
PATH=/gsc/btl/linuxbrew/Cellar/abyss/1.9.0/bin/:$PATH
PATH=/gsc/btl/linuxbrew/bin/:$PATH
samtools view /projects/btl/cjustin/ecoliReadSubsets/SRR959238.bam "gi|170079663|ref|NC_010473.1|:1000000-1000400" | abyss-tofastq > ~/Desktop/Dump/ecoli-0-400.fastq
samtools view /projects/btl/cjustin/ecoliReadSubsets/SRR959238.bam "gi|170079663|ref|NC_010473.1|:1000400-1000600" | abyss-tofastq > ~/Desktop/Dump/ecoli-400-600.fastq
samtools view /projects/btl/cjustin/ecoliReadSubsets/SRR959238.bam "gi|170079663|ref|NC_010473.1|:1000600-1001000" | abyss-tofastq > ~/Desktop/Dump/ecoli-600-1000.fastq
