#!/bin/bash
BLUE='\033[0;34m'
NC='\033[0m' 
for CUMULUS in 92 95 32 13

do
    printf "${BLUE}Runing soundscapes-spectrograms-video pipeline for cumulus $CUMULUS ${NC}\n"
    python sound-scape-nod-rec-dep-rs7g6.kale_parameterized.py --cumulus $CUMULUS --pagesize 5000
done