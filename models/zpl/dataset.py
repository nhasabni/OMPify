import os
import math
import pickle
import json
import logging
from pqdm.threads import pqdm
from statistics import mean, geometric_mean

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# def load_sample(sample_path):
# 		omp_pragma = None
# 		#with open(os.path.join(sample_path, 'ast.pickle'), 'rb') as f:
# 		#    asts = pickle.load(f)
# 		with open(os.path.join(sample_path, f"code.{'cpp' if 'cpp_loops' in sample_path else 'c'}"), 'r') as f:
# 			code = f.read()
# 		try:
# 			with open(os.path.join(sample_path, 'pragma.pickle'), 'rb') as f:
# 				omp_pragma = pickle.load(f)
# 		except FileNotFoundError:
# 			pass
# 		return omp_pragma, code
		#return omp_pragma, asts.ast_loop, code
		
# def create_sample(sample_path, update_ratio=0):
#     '''
#         given path to sample folder and the ratio of the variable names that will be replaced
#         return tuple (code, label)
#     '''
#     find_variables = CounterIdVisitor()
#     generator = c_generator.CGenerator()
#     omp_pragma, ast_loop, code = load_sample(sample_path)
#     find_variables.visit(ast_loop)
#     replacor = ReplaceIdsVisitor(find_variables.ids, find_variables.array, find_variables.struct, find_variables.func, update_ratio)
#     replacor.visit(ast_loop)
#     try:
#         code = generator.visit(ast_loop)
#     except:
#         pass
#     return code, 0 if omp_pragma is None else 1

def create_dataset(data_path, use_positive_examples_only=False, update_ratio=0):
		def get_single_record(line):
			json_line = json.loads(line.strip())
			#path=json.loads(line.strip())['path']
			#omp_pragma, code = load_sample(path)
			code = json_line['code']
			# so far, we only use code directly. We don't use AST.
			# We also filter out negative examples.
			omp_pragma_exists = json_line['exist']
			return (code, omp_pragma_exists)

		with open(data_path, 'r') as f:
			lines=f.readlines()
		# If it is a training dataset, remove negative examples.
		# If it is a test dataset, preserve both positive and negative examples.
		dataset = pqdm(lines, get_single_record, n_jobs=4)
		if (use_positive_examples_only):
			dataset = [(code, omp_pragma_exists) for (code, omp_pragma_exists)
	      				  in dataset
									if omp_pragma_exists != 0]
		return dataset

def get_torch_dataloader(dataset_name, use_positive_examples_only=False, tokenizer_max_len=256, batch_size=32, shuffle=False):
  print('-' * 100)
  print('Loading dataset', dataset_name)

  dataset_path = os.path.join('/nfs_home/nhasabni/other/openmp_transformer/ompify/models/dataset/json_c_cpp', f'{dataset_name}.jsonl')
  dataset = create_dataset(dataset_path, use_positive_examples_only)
  print(f'The size of {dataset_name} set: {len(dataset)}')

  logger.info('Datasets loaded successfully')

  # --------------------------------------------------
  # Tokenizer
  # --------------------------------------------------
  code_examples = [code for (code, _) in dataset]
  if (use_positive_examples_only == False):
    labels = [omp_pragma_exists for (_, omp_pragma_exists) in dataset]

  def print_dataset_stats(dataset_name, tokenizer_output):
    print("Stats of %s dataset", dataset_name)
    lengths = [len(x) for x in tokenizer_output['input_ids']]
    print("max_len:", max(lengths))
    print("min_len", min(lengths))
    print('mean:', mean(lengths))
    print('geomean', geometric_mean(lengths))

  tokenizer = AutoTokenizer.from_pretrained("NTUYG/DeepSCC-RoBERTa")
  # max_len is obtained by checking stats of tokenized training data.
  codes_examples_tokenized = tokenizer(code_examples, truncation=True, max_length=tokenizer_max_len,
                                       pad_to_max_length=True)

  # Remove length and trucation params to tokenizer to get actual data for datasets.
  # train_codes_tokenized = tokenizer(train_codes)
  # test_codes_tokenized = tokenizer(test_codes)
  #print_dataset_stats("train", train_codes_tokenized)
  #print_dataset_stats("test", test_codes_tokenized)

  # We normalize word IDs by dividing them by the maximum word id.
  # This is needed to keep loss values small.
  #print('tokenizer.vocab_size:', tokenizer.vocab_size)
  def normalize_token_ids(tokenized_code):
    return [x / tokenizer.vocab_size for x in tokenized_code]
  code_examples_normalized = torch.tensor([normalize_token_ids(l) 
                                           for l in codes_examples_tokenized['input_ids']],
                                           dtype=torch.float32)

  # DataLoader is used to load the dataset for training
  # We get tensor of shape [32, 256] as a single batch.
  if use_positive_examples_only:
    return DataLoader(dataset = code_examples_normalized,
                      batch_size = batch_size,
                      shuffle = shuffle)
  else:
    code_examples_normalized_with_labels = [(code_examples_normalized[i],
                                            labels[i]) for i in range(len(labels))]
    return DataLoader(dataset = code_examples_normalized_with_labels,
                      batch_size = batch_size,
                      shuffle = shuffle)
