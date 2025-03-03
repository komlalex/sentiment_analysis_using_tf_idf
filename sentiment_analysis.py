import pandas as pd 

raw_df = pd.read_csv("./data/train.tsv", sep="\t") 
test_df = pd.read_csv("./data/test.tsv", sep="\t")
sub_df = pd.read_csv("./data/sampleSubmission.csv") 

print(sub_df.head())