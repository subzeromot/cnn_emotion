import numpy as np

# dim = 64*64*100
# imgs = np.arange(dim).reshape(100, 64,64,1)
# print(imgs[1,0,0,:])
# y = np.stack((imgs[:,:,:,0],)*3, axis=-1)
# print(y[1,0,0,:])
# print(y.shape)

num_blocks=[2, 5, 5, 2]
count = 0
for i in range(len(num_blocks)):
    for j in range(num_blocks[i]):
        count += 1
print(count)