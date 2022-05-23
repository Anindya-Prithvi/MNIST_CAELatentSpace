trained_model = CAutoEncoder()
from torch.autograd import Variable
trained_model.load_state_dict(torch.load('Cautoencoder.torch'))
dummy_input = Variable(torch.randn(10, 7, 1, 1))
torch.onnx.export(trained_model.decoder, dummy_input, "Cautoencoder_dec.onnx")

from onnx_tf.backend import prepare
model = onnx.load('Cautoencoder_dec.onnx')
tf_rep = prepare(model)

output = tf_rep.run(torch.randn(1,7,1,1))

plt.imshow(np.array(output).reshape(28,28))
#works like a charm