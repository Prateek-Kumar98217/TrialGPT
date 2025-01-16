import os
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm  #to add progress bars
num_process= 6  #number of cpu cores//2 or less i guess
num_process_load_dataset= 6
tokenizer=tiktoken.get_encoding("gpt2")
if __name__=='__main__':
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset("openwebtext", num_proc=num_process_load_dataset, trust_remote_code=True)
    #to create a validation split
    split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=1234, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 7212392
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 801377
    #     })
    # })
    #now to define encoder funtion
    def process(example):
        ids=tokenizer.encode_ordinary(example['text'])
        ids.append(tokenizer.eot_token)
        out={'ids': ids, 'len': len(ids)}
        return out
    #tokenize the dataset
    tokenised_dataset=split_dataset.map(
        process,
        remove_columns=['text'],
        desc='tokenizing the splits',
        num_proc=num_process,
    )
    #concatinate all the ids of datasets to form a single file to use
    for split, dset in tokenised_dataset.items():
        arr_len=np.sum(dset['len'], dtype=np.uint64)
        file_name=os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype=np.uint16  #can be done cause the max token value is 50256 which is less than 2**16
        arr=np.memmap(file_name, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches=1024

        idx=0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {file_name}"):
            #batching samples together to reduce time taken for writing
            batch =dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch=np.concatenate(batch['ids'])
            #write it to mmap
            arr[idx : idx+len(arr_batch)] =arr_batch
            idx+= len(arr_batch)
        arr.flush()
# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')