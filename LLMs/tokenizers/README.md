## Summary
In this folder you'll find code for different types of tokenizers, 2 of which are recreated from Karpathy's tutorial on tokenizer. The classes covered are: **BPE** (Byte Pair Encoding): Basic tokenizer utilizing unicode encoder and regex tokenizer, Sentencepiece tokenizer and WordPiece tokenizer. All of them apply  

### Basic tokenizer
The text is split into tokens using python unicode encoder and each token has an id. We define the number of desired merges and start iterating over the tokens until the max number of possible merges is reached. Steps in each iteration:
1. Get pair with highest occurence
2. Create new index for that pair
3. Replace the pair in the list of tokens with its new index
4. Add the index to the vocabulary 
5. keep track of merged by appending the merges to a dictionary  

**Encoder**
- Pass sequence to method
- the sequence is encoded using python unicode
- list of ids get compressed until at least 2 elements are left
    1. get stats of pair occurence
    2. get pair with lowest merge idx that also appears in the merges from training 
    3. if the pair doesn't appear in the merges generated during train -> return inf
    4. if the pair is present in the merges generated during train -> apply the merge to the ids and replace the pair with the index from the merges

**Decoder**
- The output from the encoder is passed (ids)
- For each id find the corresponding text from the vocabulary
- join elements as byte string
- apply python .decode(utf-8)


### Regex tokenizer

During training we use a regex pattern to split the text before passing each chunk through the unicode encoder. We apply BPE to each chunk the concatenate all results. The steps would be:
1. iterate in range of possible merges
2. For each chunk get stats about most common pair
    - do this while updating stats about most common pair
    - after iterating through each chunk, the stats should represent a dictionary with pair ids as keys and occurences as values.
    - get pair with highest occurence and add to vocabulary
    - replace pair in each chunk with new idx

**Adding special tokens**

Additional tokens identify end_of_text, prompt, etc. When these are passed, their predefined ids are added to the dictionary in case they're present in the text during training.

### Sentencepiece