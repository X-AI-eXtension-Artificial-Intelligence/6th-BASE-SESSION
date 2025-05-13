from datasets import load_dataset

def load_data(data_path="wmt/wmt19", data_name="lt-en"):
    # 리투아니아어(lt, Lithuanian)와 영어(en, English) 번역 데이터
    datas = load_dataset(path=data_path, name=data_name)

    _, data = datas['train'], datas['validation']['translation']

    # train data 품질이 valid 대비 구린 느낌이 있고, valid data가 2000개라 valid를 split해서 사용함
    train_data = data[:1000]
    valid_data = data[1000:1250]
    test_data = data[1250:]

    return train_data, valid_data, test_data


def data_preprocessing(data):
    tgt_sentences = []
    src_sentences = []
    for d in data:
        tgt_sentences.append(list(d.values())[0])
        src_sentences.append(list(d.values())[1])

    return src_sentences, tgt_sentences




if __name__ == '__main__':
    train_data, valid_data, test_data = load_data()
    src_sentences, tgt_sentences = data_preprocessing(train_data)
    # print("train_data : \n", train_data[:10])
    # print("valid_data : \n", valid_data[:10])