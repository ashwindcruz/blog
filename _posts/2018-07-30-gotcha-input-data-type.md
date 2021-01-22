---
layout: posts
title: "Gotcha! TF Input Data Type"
date: 2018-08-15
---

While working on the first part of my [style-transfer project](/blog/2018/07/30/content-reconstruction), I found out the hard way that TF is very sensitive to the network's input's data type. 

I was tinkering around, trying to trace down a particular bug and in my frustration and tiredness, accidentally changed: 
```python
input_var = np.asarray(dummy_data, dtype=np.float32)
```
to: 
```python
input_var = np.asarray(dummy_data, dtype=np.float64)
```

I fed this value as an array directly into the _vgg19_ network, bypassing the use of a placeholder or variable. When I tried to restore the weights of the variables in the pre-trained _vgg19_ network, I was met with the following glaring error: 
> InvalidArgumentError: Expected to restore a tensor of type double, got a tensor of type float instead: tensor_name = vgg_19/conv1/conv1_1/biases 	 

And a lot more information that I won't copy here. Suffice it to say, the rest of the information was as unhelpful as this message. It took me a really long time to figure this out. Thanks to the message, I spent most of that time digging into the documenation for Savers and their restoring functionality. Eventually, with almost all other options exhausted, I accidentally reset the input to _float32_ and the error disappeared. 

*Always check that your input data is of type _float32_.* 

If you tried _float16_, the error wouldn't be exactly the same but it would have a similar flavour: 
> InvalidArgumentError: Expected to restore a tensor of type half, got a tensor of type float instead: tensor_name = vgg_19/conv1/conv1_1/biases

_int_ types would throw a different type of error earlier on, which is much more informative: 
> TypeError: Cannot create initializer for non-floating point type.

I'm not sure exactly why different _floats_ return these weird errors but they do so watch out! If you're not exactly sure of the type of your input data, be safe and cast it to _float32_. 

If you'd like to experience these oddities for yourself, here's a [notebook](https://github.com/ashwindcruz/style-transfer/blob/master/gotchas/wrong_input_type.ipynb) for you to tinker with! 