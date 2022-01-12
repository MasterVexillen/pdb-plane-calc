#!/bin/bash

python3 pdb-plane-calc.py -e h11-h4-rbd-simp.pdb 490 52 \
	--chainA=E \
	--chainB=F \
	--nameA=cg,cd1,ce1,cz,ce2,cd2 \
	--nameB=nh2,ne,nh1 \
	-np=1000000
