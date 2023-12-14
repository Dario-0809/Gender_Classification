from library import *
from config import *
from network import CNN
from utils import *
from image_transform import ImageTransform


# predict with new data
class_index = ['female', 'male']

class Predictor():
    def __init__(self, class_index):
        self.clas_index = class_index
    
    def predict_max(self, output):
        max_id = np.argmax(output.detach().numpy())
        # print(max_id)
        predicted_label = self.clas_index[max_id]
        return predicted_label

predictor = Predictor(class_index)
img = Image.open('./test6.jpg')
img = img.convert("RGB")

def predict(img):
    # prepare network
    model = CNN()
    model.eval()

    # prepare model
    trained_model = load_model(model, save_path)

    # prepare input img
    transform = ImageTransform(resize, mean, std)
    img = transform(img, phase="test")
    img = img.unsqueeze_(0)

    # predict
    output = trained_model(img)
    response = predictor.predict_max(output)

    return response
