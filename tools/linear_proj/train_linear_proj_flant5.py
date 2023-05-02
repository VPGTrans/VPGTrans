import json
import os.path
import re
import json, sys, os

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch.nn as nn
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers import T5TokenizerFast
from tqdm import tqdm
import sys

from omegaconf import OmegaConf
from lavis.common.utils import (
    get_abs_path, get_cache_path
)

def get_flant5_word_embed(t5_model):
    t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
    # opt_tokenizer.padding_side = "left"
    t5_config = T5Config.from_pretrained(t5_model)
    t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
    )

    word_embed = t5_model.encoder.embed_tokens
    print(word_embed.weight.size())
    # if t5_model.model.decoder.project_in is not None:
    #     word_embed.weight = nn.Parameter(t5_model.model.decoder.project_in(word_embed.weight.data))
    return word_embed, t5_tokenizer

def pre_caption(caption):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")
    return caption

class CaptionDataset(Dataset):
    def __init__(
            self, caption_data_paths,
    ):
        caption_data = [json.load(open(p)) for p in caption_data_paths]
        caption_data = [b for a in caption_data for b in a]
        print(len(caption_data))
        if os.path.exists("tmp/cached_caption_flant5.json"):
            self.captions = json.load(open("tmp/cached_caption_flant5.json", "r"))
        else:
            cids = []
            opt_embed, opt_tokenizer = get_flant5_word_embed("google/flan-t5-base")
            # preprocess
            for c in tqdm(caption_data):
                c = c['caption']
                c = pre_caption(c)
                t5_c = opt_tokenizer(c).input_ids
                cids.append([t5_c])
            self.captions = cids
            json.dump(self.captions, open("tmp/cached_caption_flant5.json", "w"))

    def __getitem__(self, item):
        caption = self.captions[item]  # opt, t5
        return caption

    def __len__(self):
        return len(self.captions)

    def collate(self, batch):
        left = batch
        left = [a for b in left for a in b]
        left = [a for b in left for a in b]
        return torch.tensor(left, dtype=torch.long)


class BaseModel(torch.nn.Module):
    def __init__(self, from_wrd_model, to_wrd_model):
        super().__init__()
        from_word_embed_data, t5_tokenizer = get_flant5_word_embed(from_wrd_model)
        to_word_embed_data, _ = get_flant5_word_embed(to_wrd_model)
        self.from_word_embed_data = from_word_embed_data
        self._freeze(self.from_word_embed_data)
        self.to_word_embed_data = to_word_embed_data
        self._freeze(self.to_word_embed_data)
        self.tokenizer = t5_tokenizer
        self.max_txt_len = 32

    def _freeze(self, model):
        for p in model.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return list(self.parameters())[0].device

    def to_wrd_embed(self, t5_tokens):
        from_embeds = self.from_word_embed_data(t5_tokens)
        to_embeds = self.to_word_embed_data(t5_tokens)
        return from_embeds, to_embeds


class LinearModel(BaseModel):
    def __init__(self, from_wrd_model, to_wrd_model):
        super().__init__(from_wrd_model, to_wrd_model)
        self.linear = nn.Linear(self.from_word_embed_data.weight.size(-1), self.to_word_embed_data.weight.size(-1))

    def collect(self, embeds, attention_mask):
        to_merge = [emb[:msk.sum()] for emb, msk in zip(embeds, attention_mask)]
        to_merge = torch.cat(to_merge, 0)
        return to_merge

    def forward(self, captions):
        from_embeds, to_embeds = self.to_wrd_embed(captions)
        output = self.linear(from_embeds)
        # MSE Loss
        # criterion = nn.MSELoss()
        criterion = nn.CosineSimilarity()
        if self.training:
            loss = (1 - criterion(output, to_embeds)).mean()
            return output, loss
        else:
            return output

import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0)

device = "cuda:0"
num_epoch = 5
lr = 5.
batch_size = 4096

config_path = get_abs_path("configs/datasets/coco/defaults_cap.yaml")
coco_path = OmegaConf.load(
    config_path
).datasets.coco_caption.build_info.annotations.train.storage
coco_path = get_cache_path(coco_path)

config_path = get_abs_path("configs/datasets/sbu_caption/defaults.yaml")
sbu_path = OmegaConf.load(
    config_path
).datasets.sbu_caption.build_info.annotations.train.storage[0]
sbu_path = get_cache_path(sbu_path)


dataset = CaptionDataset([coco_path, sbu_path])

dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)

from_size, to_size = str(sys.argv[1]), str(sys.argv[2])
model = LinearModel(f"google/flan-t5-{from_size}", f"google/flan-t5-{to_size}")
model = model.to(device)

optimizer = torch.optim.SGD(model.linear.parameters(), lr=lr)

for epoch in range(num_epoch):
    model.train()
    print(f"epoch {epoch}:")
    for i, captions in enumerate(tqdm(dataloader)):
        _, loss = model(captions.to(device))
        loss.backward()
        # adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)  # Update learning rate schedule
        optimizer.step()
        model.zero_grad()
        if i%20==0:
            print(loss)
    if (epoch+1)%10 == 0:
        for pg in optimizer.param_groups:
            pg['lr'] = pg['lr']/10
# torch.save(model.linear.state_dict(), "linear_125m_to_1.3b_cos.pth")

# merge with trained proj
from_model = torch.load(sys.argv[3], "cpu")['model']
weight = from_model['t5_proj.weight']
bias = from_model['t5_proj.bias']

trans_model = model.linear.to("cpu").state_dict()
trans_weight = trans_model['weight']
trans_bias = trans_model.get("bias", 0)

print(trans_weight.size(), weight.size())
weight = trans_weight@weight
bias = trans_weight@bias+trans_bias

from_model = {}
from_model["t5_proj.weight"]= weight
from_model['t5_proj.bias'] = bias
save_dir = sys.argv[4]
print("save", f"linear_flant5_{from_size}_to_{to_size}.pth to {save_dir}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save({"model": from_model}, os.path.join(save_dir, f"linear_flant5_{from_size}_to_{to_size}.pth"))
