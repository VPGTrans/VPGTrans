import torch
import torch.nn as nn
from tqdm import tqdm
import re
from transformers import AutoTokenizer
from transformers import T5TokenizerFast
import json, sys, os
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from lavis.models.blip2_models.modeling_opt import OPTForCausalLM
from lavis.models.blip2_models.modeling_llama import LlamaConfig, LlamaForCausalLM

from omegaconf import OmegaConf
from lavis.common.utils import (
    get_abs_path, get_cache_path
)

def get_llama_word_embed(llama_model):
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
    # opt_tokenizer.padding_side = "left"
    llama_model = LlamaForCausalLM.from_pretrained(
        llama_model, torch_dtype=torch.float
    )
    word_embed = llama_model.model.embed_tokens
    return word_embed, llama_tokenizer


def get_opt_word_embed(opt_model):
    opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
    # opt_tokenizer.padding_side = "left"
    opt_model = OPTForCausalLM.from_pretrained(
        opt_model, torch_dtype=torch.float
    )
    word_embed = opt_model.model.decoder.embed_tokens
    if opt_model.model.decoder.project_in is not None:
        word_embed.weight = nn.Parameter(opt_model.model.decoder.project_in(word_embed.weight.data))
    return word_embed, opt_tokenizer

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
        self, caption_data_paths, llama_model
    ):
        caption_data = [json.load(open(p)) for p in caption_data_paths]
        caption_data = [b for a in caption_data for b in a]
        print("dataset items:", len(caption_data))
        if os.path.exists("tmp/cached_caption_opt_llama.json"):
            self.captions = json.load(open("tmp/cached_caption_opt_llama.json", "r"))
        else:
            n_matched = 0
            cids = []
            llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
            opt_embed, opt_tokenizer = get_opt_word_embed("facebook/opt-125m")
            # preprocess
            for c in tqdm(caption_data):
                c = c['caption']
                c = pre_caption(c)
                opt_c = opt_tokenizer(c).input_ids
                llama_c = llama_tokenizer(c).input_ids
                # print(opt_tokenizer.convert_ids_to_tokens(opt_c), t5_tokenizer.convert_ids_to_tokens(t5_c))
                if len(opt_c) == len(llama_c):
                    n_matched += 1
                    cids.append([opt_c, llama_c])
            self.captions = cids
            print(f"matched {n_matched} sentences.")
            json.dump(self.captions, open("tmp/cached_caption_opt_llama.json", "w"))

    def __getitem__(self, item):
        caption = self.captions[item] # opt, t5
        return caption

    def __len__(self):
        return len(self.captions)

    def collate(self, batch):

        left, right = list(zip(*batch))
        left = [a for b in left for a in b]
        right = [a for b in right for a in b]
        return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)



class BaseModel(torch.nn.Module):
    def __init__(self, from_wrd_model, to_wrd_model):
        super().__init__()
        print(f"from {from_wrd_model} to {to_wrd_model}")
        def get_word_embed(wrd_model):
            if "opt" in wrd_model:
                word_embed_data, _= get_opt_word_embed(wrd_model)
            elif "llama" in wrd_model:
                word_embed_data, _ = get_llama_word_embed(wrd_model)
            return word_embed_data
        if "opt" in from_wrd_model:
            self.mode = "opt_to_llama"
        else:
            self.mode = "llama_to_opt"
        from_word_embed_data = get_word_embed(from_wrd_model)
        to_word_embed_data = get_word_embed(to_wrd_model)

        self.from_word_embed_data = from_word_embed_data
        self._freeze(self.from_word_embed_data)
        self.to_word_embed_data = to_word_embed_data
        self._freeze(self.to_word_embed_data)
        self.max_txt_len = 32

    def _freeze(self, model):
        for p in model.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return list(self.parameters())[0].device

    def to_wrd_embed(self, captions):
        if self.mode == "opt_to_llama":
            from_embeds = self.from_word_embed_data(captions[0])
            to_embeds = self.to_word_embed_data(captions[1])
        else:
            from_embeds = self.from_word_embed_data(captions[1])
            to_embeds = self.to_word_embed_data(captions[0])
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
            loss = (1-criterion(output, to_embeds)).mean()
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
num_epoch = 8
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

from_size, to_size = str(sys.argv[1]), str(sys.argv[2])
model = LinearModel(from_size, to_size)
model = model.to(device)
optimizer = torch.optim.SGD(model.linear.parameters(), lr=lr)

dataset = CaptionDataset([coco_path, sbu_path], llama_model=from_size if "opt" in to_size else to_size)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)

for epoch in range(num_epoch):
    model.train()
    print(f"epoch {epoch}:")
    for i, captions in enumerate(tqdm(dataloader)):
        captions = [captions[0].to(device), captions[1].to(device)]
        _, loss = model(captions)
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
if "opt" in from_size:
    from_key = "opt_proj"
    to_key = "llama_proj"
    assert "opt" in sys.argv[3]
else:
    from_key = "llama_proj"
    to_key = "opt_proj"
    assert "llama" in sys.argv[3]
from_model = torch.load(sys.argv[3], "cpu")['model']
weight = from_model[f'{from_key}.weight']
bias = from_model[f'{from_key}.bias']

trans_model = model.linear.to("cpu").state_dict()
trans_weight = trans_model['weight']
trans_bias = trans_model.get("bias", 0)

weight = trans_weight@weight
bias = trans_weight@bias+trans_bias

to_model = {}
to_model[f"{to_key}.weight"]= weight
to_model[f'{to_key}.bias'] = bias
save_dir = sys.argv[4]
print("save", f"linear_{from_size.split('/')[-1]}_to_{to_size.split('/')[-1]}.pth to {save_dir}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save({"model": to_model}, os.path.join(save_dir, f"linear_{from_size.split('/')[1]}_to_{to_size.split('/')[1]}.pth"))