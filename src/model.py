import config
import torch
import transformers
import torch.nn as nn 

# cross entropy loss since it's a classification problem
def loss_fn(output): # See 34:20 in the video
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1 # what is this?
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target) # the ignore index is -100, why is it ignored?
    )
    loss = lfn(active_logits, active_labels)

# "Our model will calculate the loss, so we dont need to calculate it inside the engine"

# "To change to roberta, just change the model used from transformers package, and change the tokenizer inside config.py"

class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH, return_dict=False)
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3) # How would you know to do 2 and waht values to give them?
        self.out_tag = nn.Linear(768, self.num_tag) # What is 768? (it comes from BERT base) What is self.out_tag?
        self.out_pos = nn.Linear(768, self.num_pos)

    # "Something something sequence outputs"
    def forward(self, ids, mask, token_type_ids, target_pos, target_tag): # should have the same names as the keys in the dictionary returned by the dataset class
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids) # What is o1 and o2? (we dont use the latter here - why?)
        # what is an attention mask?    
        bo_tag = self.bert_drop_1(o1)
        bo_pos = self.bert_drop_2(o1)

        tag = self.out_tag(bo_tag)
        pos = self.out_pos(bo_pos) # He called self.out_tag() again - a mistake I think. 

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss_tag = (loss_tag + loss_pos) / 2

        return tag, pos, loss