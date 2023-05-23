import torch

a = torch.tensor([[1,3,5,7,9,11],[2,4,6,8,10,12]]).float().to('cuda:0')
print(a)

b = a.max(dim=1)
c = a.min(dim=1)

#b = a.topk(3, dim=1, largest=False)
#b = a.kthvalue(3, dim=1)
#b = torch.eq(a,a)
#b = torch.all(torch.eq(a,a))
#b = torch.equal(a,a)

t = torch.tensor([[5,4,3,2],[2,3,4,5]]).to('cuda:0')
b = torch.gather(a, dim=1, index=t)
c = a.t()[[1]]

print('b:',b)
print('c:',c)

d = a.argmin(dim=1)
print(d)

f = a.norm(2)
print(f)

g = a.mean(dim=1)
print(g)

g = a.prod(dim=1)
print(g)