### Data Preparation

To prepare the data, first set the Python path and then run the data processing module.

export PYTHONPATH=/ceph/lfertig/Thesis

python -m data.data_processing
--dataset HelpDesk
--raw_log_file /ceph/lfertig/Thesis/data/HelpDesk/raw/df_helpdesk.csv.gz
--dir_path /ceph/lfertig/Thesis/data
--task next_activity
--sort_temporally True
