from library import *

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

batch_size = 12

save_path = './weight_trained_3.pth'