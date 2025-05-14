from torch.utils.data import Dataset
from transformers import AutoTokenizer
# from data_preprocessing import load_data, data_preprocessing

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer_name="bert-base-multilingual-cased", max_length=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # source와 target 문장 토크나이징 (항상 max_length 길이로 고정)
        tokenized_src = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized_tgt = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'src_input_ids': tokenized_src['input_ids'].squeeze(0),
            'src_attention_mask': tokenized_src['attention_mask'].squeeze(0),
            'tgt_input_ids': tokenized_tgt['input_ids'].squeeze(0),
            'tgt_attention_mask': tokenized_tgt['attention_mask'].squeeze(0)
        }

# if __name__ == '__main__':
#     train_data, valid_data = load_data()
#     src_sentences, tgt_sentences = data_preprocessing(train_data)
#
#     dataset = TranslationDataset(src_sentences, tgt_sentences, tokenizer_name="bert-base-multilingual-cased")
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
#
#     # 첫번째 샘플의 길이 확인 (패딩 적용 후 길이)
#     print('length of each data after adding pad :', len(dataset[0]['src_input_ids']))
