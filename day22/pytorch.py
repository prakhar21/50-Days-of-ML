import torch # importing library

x = torch.Tensor(5, 3) # create a 2-D array
y = torch.rand(5, 3) # creates a 2-D array with values b/w 0-1

print y.size() # returns the dimension of y
print y[:,0]  # slicing and indexing methods are same as that of NumPy

z1 = x + y # adding two tensors  # method-1
print z1

z2 = torch.add(x, y) # adding two tensors  # method-2
print z1==z2

z3 = x - y # subtracting two tensors  # method-1
print z3

z4 = torch.add(x, -y) # subtracting two tensors  # method-2
print z3==z4

z1 += z2 # supports inplace addition/subtraction
print z1

#####
# P.S. PyTorch gives you the portability to convert torch tensors to numpy arrays and vice-versa ;)
#####


from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
print x
