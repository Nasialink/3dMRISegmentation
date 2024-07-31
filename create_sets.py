import os
from os.path import join
from dataset import MRIDataset
import numpy as np
from tqdm import tqdm



try:
    os.mkdir("./sets")
    os.mkdir("./sets/zoom_in")
    os.mkdir("./sets/zoom_out")
except OSError as error:
    print(error)

data_dir = "/home/superteam/3dMRISegmentation/data/"
size = [128, 128, 71]
assert len(size) == 3

train_dir = join(data_dir, 'train')
train_dat = MRIDataset(train_dir, size, sampling_mode='center_val', deterministic=True)

val_dir = join(data_dir, 'val')
val_dat = MRIDataset(val_dir, size, sampling_mode='center_val', deterministic=True)

test_dir = join(data_dir, 'test')
test_dat = MRIDataset(test_dir, size, sampling_mode='center_val', deterministic=True)


train_dat.create_dataset()
val_dat.create_dataset()
test_dat.create_dataset()

print()
print("Zoom-in sets:")
print(train_dat.x.shape, train_dat.y.shape)
print(val_dat.x.shape, val_dat.y.shape)
print(test_dat.x.shape, test_dat.y.shape)

np.save("./sets/zoom_in/train_x", train_dat.x)
np.save("./sets/zoom_in/train_y", train_dat.y)

np.save("./sets/zoom_in/val_x", val_dat.x)
np.save("./sets/zoom_in/val_y", val_dat.y)

np.save("./sets/zoom_in/test_x", test_dat.x)
np.save("./sets/zoom_in/test_y", test_dat.y)



size = [144, 172, 71]
assert len(size) == 3

train_dir = join(data_dir, 'train')
train_dat = MRIDataset(train_dir, size, sampling_mode='center_val', deterministic=True)

val_dir = join(data_dir, 'val')
val_dat = MRIDataset(val_dir, size, sampling_mode='center_val', deterministic=True)

test_dir = join(data_dir, 'test')
test_dat = MRIDataset(test_dir, size, sampling_mode='center_val', deterministic=True)


train_dat.create_dataset()
val_dat.create_dataset()
test_dat.create_dataset()

print()
print("Zoom-out sets:")
print(train_dat.x.shape, train_dat.y.shape)
print(val_dat.x.shape, val_dat.y.shape)
print(test_dat.x.shape, test_dat.y.shape)

np.save("./sets/zoom_out/train_x", train_dat.x)
np.save("./sets/zoom_out/train_y", train_dat.y)

np.save("./sets/zoom_out/val_x", val_dat.x)
np.save("./sets/zoom_out/val_y", val_dat.y)

np.save("./sets/zoom_out/test_x", test_dat.x)
np.save("./sets/zoom_out/test_y", test_dat.y)