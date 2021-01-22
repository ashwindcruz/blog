---
layout: posts
title: "Gotcha! TF Saver Subset Initialization"
date: 2018-08-14
---

While working on the first part of my [style-transfer project](/blog/2018/07/30/content-reconstruction), I dealt with two main variable groups: 
* The input variable which was the image I was optimizing. 
* The _VGG-19_ network, whose weights were frozen. 

To get both of these interacting properly and functioning correctly within a Tensorflow (TF) environment, the input variable would need to be initialized to an initial value while the _vgg19_ network would need to be initialized with pre-trained weights. 

In the past, I have either initialized the entire network with pre-trained weights or with new, initial values but not a combination. I knew that a TF Saver object would be used to restored the weights like so: 
```python
saver.restore(sess, 'vgg_19.ckpt')
```

Where _sess_ refers to a TF Session and the filepath points to the checkpoint file. Since I had a new input variable that didn't correspond to one of the pre-trained weights, I assumed that the Saver would skip restoring this and rely on me to initialize it myself later. I was wrong: 
```python
NotFoundError: Tensor name "input_var" not found in checkpoint files vgg_19.ckpt
```

The Saver noted down all the variables present in the graph and expected to find all those variables in the checkpoint. When it didn't, it complained. I found some guidance using this [link](https://stackoverflow.com/questions/45179556/key-variable-name-not-found-in-checkpoint-tensorflow/47917561#47917561) which helped me better understand the _var\_list_ parameter of the [Saver](https://www.tensorflow.org/api_docs/python/tf/train/Saver) initialization method. This parameter specificied which variables would be saved _and_ restored. The former was a lot easier to grasp than the latter. It struck me as a little counterintutive that in initializing the Saver object, I would be specifying up-front which variables I would be restoring. Having thought about it for awhile now and tinkering with it, it makes a little more sense to me. Or perhaps I've just got used to this quirk. 

So given my new knowledge, I knew that I had to specify which variables would be loaded from the checkpoint (all the _vgg19_ variables) and which would not be the responsiblity of the Saver (the new input variable). To get the names of all the variables present in the graph, I used the following: 
```python
all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
```

Printing that out, I could see that the first variable was the new input variable and so _all\_variables[1:]_ provided me with all the _vgg19_ variables. So the Saver would be created as such: 
```python
saver = tf.train.Saver(var_list=all_variables[1:])
```

Running the restore method of this object caused no errors! 

In my graph, the input variable was at the very start so I could specify the _var\_list_ the way I did. Perhaps in yours, the new variable might exist somewhere else in the list or maybe you even have several variables you want the Saver to ignore. In that case, just have a look at _all\_variables_ and extract the necessary variables. Maybe you could write a _for_ loop to go through the list and ignore variables with certain names. 

So when mixing around new and pre-trained variables, be sure create the Saver object appropriately. 
If you'd like to play around more, here's a [notebook](https://github.com/ashwindcruz/style-transfer/blob/master/gotchas/restore_subset_of_variables.ipynb) for you to tinker with! 