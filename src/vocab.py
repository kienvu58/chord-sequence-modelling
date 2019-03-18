class Vocab:
    def __init__(self):
        self._token_to_index = {"<s>": 0}
        self._index_to_token = ["<s>"]

    def add(self, word):
        index = self._token_to_index.get(word, None)
        if index is None:
            index = len(self._index_to_token)
            self._index_to_token.append(word)
            self._token_to_index[word] = index
        return index

    def from_dataset(self, dataset):
        flatten_dataset = [word for sent in dataset for word in sent]
        dataset_unique = set(flatten_dataset)
        for word in dataset_unique:
            self.add(word)

    def encode_sentence(self, sent):
        sent = [self[word] for word in sent]
        sent = [0] + sent 
        return sent

    def __len__(self):
        return len(self._token_to_index)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._index_to_token[key]
        else:
            return self._token_to_index[key]