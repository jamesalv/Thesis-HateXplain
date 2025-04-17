from DataPreprocessing.dataCollector import collect_data
import json

def splitDataset(params):
  data = collect_data(params)
  with open('Data/post_id_divisions.json', 'r') as f:
    post_division_dict = json.load(f)
    
  train = data[data['post_id'].isin(post_division_dict['train'])]
  val = data[data['post_id'].isin(post_division_dict['val'])]
  test = data[data['post_id'].isin(post_division_dict['test'])]
  
  # For now we only take text, attention, and final label
  # TODO: look if we can add additional features that might help to mitigate the bias or enhancing model performance
  # print(train.head())
  features = ['text_vector', 'attention', 'final_label']
  train = train[features]
  val = val[features]
  test = test[features]
  
  return train, val, test