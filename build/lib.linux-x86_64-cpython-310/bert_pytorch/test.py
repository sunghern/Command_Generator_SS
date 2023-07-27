import torch
from model.bert import BERT
bert = BERT(128)
print(torch.cuda.is_available())
inputs = torch.rand([1,128])
inputs = torch.arange(128, dtype=torch.long).unsqueeze(0)
inputs.expand(1,-1)
inputs = inputs.expand(1, -1)
seg = torch.zeros_like(inputs)
output = bert(inputs, seg)
print(output)