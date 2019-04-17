
a=input[0]
print(a.shape)
a = PaddedSameConv2d(3, 64, kernel_size=4, stride=2)(a)
print(a.shape)
a = PaddedSameConv2d(64, 128, kernel_size=4, stride=2)(a)
print(a.shape)
a = PaddedSameConv2d(128, 1024, kernel_size=16, stride=8)(a)
print(a.shape)


a=input[0]
print(a.shape)
a = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=padding_same(4, 2))(a)
print(a.shape)
a = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=padding_same(4, 2))(a)
print(a.shape)
a = nn.Conv2d(128, 1024, kernel_size=16, stride=8, padding=padding_same(16, 8))(a)
print(a.shape)
