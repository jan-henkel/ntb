# ntb
ntb, which stands for neural toy box (or neural toolbox if you prefer), is a small deep learning framework. It is written in Python 3 and uses NumPy.
It was initially inspired by the cs231n assignments from Stanford University (by now little remains of the original code).
Syntax and usage is similar to Google's TensorFlow, however this project has a much smaller scope and puts less emphasis on computational efficiency.

This projects aim is to provide a relatively concise and readable implementation of backpropagation in computational graphs (i.e. automatic differentiation) and some deep learning building blocks like convolutional layers, lstms, dropout layers, batch normalization etc.

The capabilities of ntb are on display in a few jupyter notebooks:

<a href="mnist_demo.ipynb">MNIST demo</a> (MNIST files have to be present in ./ntb/datasets to run this)<br>
<a href="cifar10_demo.ipynb">CIFAR10 demo</a> (CIFAR-10 files have to be present in ./ntb/datasets to run this)<br>
<a href="textdata_demo.ipynb">Text data demo</a> (akin to Andrej Karpathy's char-rnn)<br>
<a href="textdata_demo_tf.ipynb">Text data demo in tensorflow</a> (pretty much the same but implemented in tensorflow)<br>

The general procedure is to add nodes to a computational graph and run the ones of interest (during training usually the loss node, optimization node and some performance metrics like accuracy). Nodes represent input placeholders, learnable variables and all sorts of transformations.
