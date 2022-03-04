import torch
batch_size, feat = 64 ,21
pos = torch.randn(batch_size,1)
neg = torch.randn(batch_size,batch_size)
cri = torch.nn.CrossEntropyLoss()
logits=torch.cat([pos,neg],dim=1)
labels = torch.zeros(logits.shape[0],dtype=torch.long)
print(cri(logits,labels))

pos_exp = pos.exp()
neg_exp = neg.exp().sum(dim=-1)
hcl_loss = (- torch.log(pos_exp / (pos_exp + neg_exp) )).mean()
print(hcl_loss)


