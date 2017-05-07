# ntb
ntb, which stands for neural toy box (or neural toolbox if you prefer), is a small deep learning framework. It is written in Python 3 and uses NumPy.
It was initially inspired by the cs231n assignments from Stanford University (by now little remains of the original code).
Syntax and usage is similar to Google's TensorFlow, however this project has a much smaller scope and puts less emphasis on computational efficiency.

This projects aim is to provide a relatively concise and readable implementation of backpropagation in computational graphs (i.e. automatic differentiation) and some deep learning building blocks like convolutional layers, lstms, dropout layers, batch normalization etc.

The core backpropagation algorithm is implemented in ntb/graph.py, the various building blocks in ntb/nodes/*.py

## Examples

Here are a couple of small instructive examples:

* [Linear fit demo](linear_fit_demo.ipynb)  
  Fit a linear function to some artificial data

* [Minimal MNIST demo](mnist_minimal_demo.ipynb)  
  Classify MNIST images using a neural network with 1 hidden layer  
  (MNIST files have to be present in ./ntb/datasets to run this)

To see some more involved examples, run the following notebooks:

* [MNIST convnet demo](mnist_convnet_demo.ipynb)  
  Classify MNIST images using a convolutional neural network  
  (MNIST files have to be present in ./ntb/datasets to run this)

* [CIFAR10 convnet demo](cifar10_convnet_demo.ipynb)  
  Classify CIFAR-10 images using a convolutional neural network  
  (CIFAR-10 files have to be present in ./ntb/datasets to run this)

* [RNN demo](textdata_demo.ipynb)  
  Train a character level RNN on an input text and generate samples (akin to Andrej Karpathy's char-rnn)

* [RNN in tensorflow](textdata_demo_tf.ipynb)  
  Pretty much the same but implemented in tensorflow rather than ntb

## Usage

The general procedure is to add nodes to a computational graph and run the ones of interest (during training usually the loss node, optimization node and some performance metrics like accuracy). Nodes represent input placeholders, learnable variables and all sorts of transformations.

A graph object `g` is created by invoking
```python
g = ntb.ComputationGraph()
```
Various nodes may be added to it, e.g.:
```python
x = ntb.Placeholder(graph=g)
y = ntb.Variable(value=5.0,graph=g)
```
The above code creates a placeholder node `x`, and a variable node `y`.
Placeholders need to be assigned a value when they or anything depending on them is run by the graph.
Variables can be used as a learnable parameter (or just hold constants if `learnable=False` is passed).
The `graph=g` part can be skipped by running a `with` statement:
```python
with ntb.default_graph(g):
     #...
```
Nodes may be combined in various ways to create new nodes, for instance:
```python
z = x+y
w = x*y
```
We can assign a value, say `2.0` to the placeholder `x` and run the nodes `z` and `w` as follows:
```python
result_z,result_w = g.run([z,w],assign_dict={x:2.0})
```
The interesting part, namely automatic differentiation, comes into play with optimization nodes.
Let's define another placeholder `r` of shape `(10,)` and the sum of `(r-y)**2`:
```python
r = ntb.Placeholder(shape=[10],graph=g)
l = ((r-y)**2).sum()
```

We can minimize `l` by adding an optimization node ...
```python
opt = ntb.Optim(l,lr=1e-1) #learning rate 1e-1
```
... and running it repeatedly (with some fixed random input `rr` for the node `r`)
```python
rr = np.random.normal(size=10,loc=6.5,scale=.5)
for i in range(100):
  _, = g.run([opt],assign_dict={r:rr})
```

`y` will be adjusted to minimize `l` (you can run `g.run([l,y],{r:rr})` to verify this). Voila, we have implemented a needlessly elaborate way to find the mean of `rr` (or whatever is fed into `r`).

For slightly more sensible applications, check the Examples section of the readme.

If you've come into contact with tensorflow the syntax should look familiar to you.
