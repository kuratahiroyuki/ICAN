#!/bin/bash

encode_method=fcs # fcs smiles selfies
DB=DAVIS # DAVIS BindingDB BIOSNAP/full_data
algorithm=CA_P

python traintest.py --encode_method ${encode_method} --algorithm ${algorithm} --DB ${DB} --epochs 50 --iter_num 5


