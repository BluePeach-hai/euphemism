import os
import os.path as osp
import json
import jieba
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from functools import partial
from tqdm import tqdm

# pytorch_lightning 兼容
try:
    import pytorch_lightning as pl
    LightningDataModule = None
    try:
        from pytorch_lightning import LightningDataModule
    except ImportError:
        try:
            from pytorch_lightning.core.datamodule import LightningDataModule
        except ImportError:
            class LightningDataModule: pass
except ImportError:
    pl = None
    class LightningDataModule: pass

ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
DATA_ROOT = osp.join(ROOT, 'data')


class EuphemismDataset(Dataset):
    def __init__(self, items, split):
        super().__init__()
        self.items = items
        self.split = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class EuphemismDataModule(LightningDataModule):
    def __init__(self,
                 root=DATA_ROOT,
                 text_input='text',
                 use_definitions=False,
                 use_images=False,
                 use_hallucinations=True,
                 batch_size=64,
                 num_workers=0,
                 tokenizer='microsoft/deberta-base',
                 seed=42,
                 val_percent=0.2,
                 use_chinese_tokenization=True):
        super().__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.text_input = text_input
        self.use_definitions = use_definitions
        self.use_images = use_images
        self.use_hallucinations = use_hallucinations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.torch_rng = torch.Generator().manual_seed(seed)
        self.val_percent = val_percent
        self.use_chinese_tokenization = use_chinese_tokenization

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

        # 中文分词
        self.nlp = jieba  # 中文用jieba，不需要spacy

        # 初始化特征字典
        self.image_features = {}
        self.term_features = {}
        self.desc_features = {}

    def tokenize_chinese(self, text):
        return ' '.join(jieba.cut(text))

    def prepare_split(self, split='train', file_path=None):
        import csv
        input_file = osp.join(self.root, f'{split}.csv')
        data = []
        if osp.exists(input_file):
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(tqdm(reader, desc=f"处理 {split} 数据集")):
                    text = row.get('text', '')
                    segmented_text = self.tokenize_chinese(text)
                    keywords = row.get('keywords', '')
                    segmented_keywords = self.tokenize_chinese(keywords) if keywords else ''
                    data.append({
                        'index': i,
                        'text': text,
                        'segmented_text': segmented_text,
                        'is_drug_related': int(row.get('is_drug_related', 0)),
                        'original_keyword': row.get('原始关键词', ''),
                        'final_keyword': row.get('最终合并关键词', ''),
                        'keywords': keywords,
                        'segmented_keywords': segmented_keywords,
                        'main_type': row.get('main_type', ''),
                    })
        if file_path is not None:
            with open(osp.abspath(file_path), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def prepare_data(self):
        self.prepare_split('dataset_text', osp.join(self.root, 'dataset_text.json'))

    def setup_features(self, features_path):
        features = dict()
        if not osp.exists(features_path):
            return features
        for file_name in os.listdir(features_path):
            file_path = osp.join(features_path, file_name)
            term = osp.splitext(file_name)[0].replace('_', ' ')
            features[term] = torch.load(file_path)
        return features

    def setup(self, stage='fit'):
        # 1. 加载中文术语表
        self.terms = {}
        terms_file = osp.join(self.root, 'terms.tsv')
        if osp.exists(terms_file):
            try:
                with open(terms_file, 'r', encoding='utf-8') as f:
                    lines = [line.strip().split('\t') for line in f.readlines()][1:]
                    for term, definition in lines:
                        self.terms[term] = definition
            except Exception as e:
                print(f"加载中文术语表失败: {e}")

        # 2. 加载完整 JSON 数据集
        dataset_file = osp.join(self.root, 'dataset_text.json')
        if osp.exists(dataset_file):
            with open(dataset_file, 'r', encoding='utf-8') as f:
                labeled = json.load(f)

            num_total = len(labeled)
            num_test = int(0.1 * num_total)  # 例如 10% 作为测试集
            num_trainval = num_total - num_test

            # 随机划分测试集和剩余集
            remaining_data, test_data = random_split(
                labeled, [num_trainval, num_test], generator=self.torch_rng
            )

            # 再划分训练集和验证集
            num_train = int((1 - self.val_percent) * num_trainval)
            num_val = num_trainval - num_train
            train_data, val_data = random_split(
                remaining_data, [num_train, num_val], generator=self.torch_rng
            )

            self.train_data = EuphemismDataset(train_data, 'train')
            self.val_data = EuphemismDataset(val_data, 'val')
            self.test_data = EuphemismDataset(test_data, 'test')

        # 3. 加载图像特征
        if self.use_images:
            feature_dir = osp.join(self.root, 'features_dfs_en')
            self.image_features = self.setup_features(feature_dir)

        # 4. 加载幻觉特征
        self.term_features = dict()
        self.desc_features = dict()
        if self.use_hallucinations:
            features_dir = osp.join(self.root, 'features_dfs_en')
            term_file = osp.join(features_dir, 'term_en.pt')
            desc_file = osp.join(features_dir, 'desc_en.pt')
            if osp.exists(term_file):
                self.term_features = torch.load(term_file)
            if osp.exists(desc_file):
                self.desc_features = torch.load(desc_file)

    # ---------------- DataLoader ----------------
    def _dataloader(self, dataset, split, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=create_collate_fn(
                split=split,
                tokenizer=self.tokenizer,
                text_input=self.text_input,
                use_definitions=self.use_definitions,
                terms=self.terms,
                use_images=self.use_images,
                image_features=self.image_features,
                use_hallucinations=self.use_hallucinations,
                term_features=self.term_features,
                desc_features=self.desc_features
            ),
        )

    def train_dataloader(self):
        return self._dataloader(self.train_data, 'train', shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_data, 'val')

    def predict_dataloader(self):
        return [self._dataloader(self.test_data, 'test')]

# ---------------- collate_fn ----------------

def _helper(batch, key):
    return [x.get(key) for x in batch]

def _get_sentences_with_definitions(batch, terms, text_input, use_definitions=True):
    sentences = []
    for item in batch:
        term = item.get('final_keyword', item.get('original_keyword', '未知'))
        sent = item[text_input]
        if use_definitions:
            desc = terms.get(term, '无定义')
            prompt = f"术语: {term}。定义: {desc}。原句: {sent}"
        else:
            prompt = sent
        # 中文分词
        prompt = ' '.join(jieba.cut(prompt))
        sentences.append(prompt)
    return sentences

def _get_features(batch, features_dict):
    features = []
    batch_size = len(batch)
    # 处理 features_dict 是张量的情况
    if isinstance(features_dict, torch.Tensor):
        # 为每个样本复制相同的特征
        for _ in batch:
            features.append(features_dict)
    else:
        # 处理 features_dict 是字典的情况
        for item in batch:
            term = item.get('final_keyword', item.get('original_keyword', ''))
            feat = features_dict.get(term, torch.zeros(1, 1024))
            features.append(feat)
    return torch.cat(features, dim=0)

def _collate_fn(batch, split, tokenizer, text_input, use_definitions, terms,
                use_images, image_features, use_hallucinations, term_features, desc_features):
    indexes = torch.tensor(_helper(batch, 'index'))
    sentences = _get_sentences_with_definitions(batch, terms, text_input, use_definitions)
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=256)
    labels = torch.tensor(_helper(batch, 'is_drug_related')).long() if split != 'test' else None

    batch_image_features = _get_features(batch, image_features) if use_images else None

    batch_term_features = batch_desc_features = None
    if use_hallucinations:
        batch_term_features = _get_features(batch, term_features)
        batch_desc_features = _get_features(batch, desc_features)

    return {
        'indexes': indexes,
        'inputs': inputs,
        'labels': labels,
        'image_features': batch_image_features,
        'term_features': batch_term_features,
        'desc_features': batch_desc_features
    }

def create_collate_fn(split, tokenizer, text_input, use_definitions, terms,
                      use_images, image_features, use_hallucinations,
                      term_features, desc_features):
    return partial(_collate_fn,
                   split=split,
                   tokenizer=tokenizer,
                   text_input=text_input,
                   use_definitions=use_definitions,
                   terms=terms,
                   use_images=use_images,
                   image_features=image_features,
                   use_hallucinations=use_hallucinations,
                   term_features=term_features,
                   desc_features=desc_features)
