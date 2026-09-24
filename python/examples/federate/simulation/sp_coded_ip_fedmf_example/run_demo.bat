@echo off
python coded_ip_fedmf_demo.py --client-num 8 --recovery-threshold 5 --dropout 2 --dimension 16
python run_experiments.py --output results\coded_ip_fedmf_sweep.csv
