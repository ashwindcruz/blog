---
layout: posts
title: "Gotcha! TF Variable Initialization Order"
date: 2018-08-15
---

While working on the first part of my [style-transfer project](/blog/2018/07/30/content-reconstruction), I had: 
* A new input variable which would have to be initialized from scratch. 
* The _VGG-19_ network, whose weights would be initialized using pre-trained values from the checkpoint file. 

In most use cases, the input would be a placeholder and hence not require initialization. However, since the input was the image I was optimizing, it needed to be a variable and being a variable, it needed initialization. 

The initialization methods were as follows: 
* Input variable: ```python sess.run(tf.global_variables_initializer())```
* _VGG-19_: ```python saver.restore(sess, 'vgg_19.ckpt')```

On my first pass through, I hadn't considered that the order of these mattered. I assumed that the if the global initializer was run first, all variables would be initializer and then the saver would restore the appropriate subset corresponding to the checkpoint file. On the other hand, I assumed that if the saver restored the subset first, the global initializer would only focus on the remaining variables, in this case, the newly introduced input variable. 

After playing around a little, I found that my second assumption was incorrect! I stumbled across this experimentally. Basically, if both my assumptions were correct, then passing a fixed input variable through the pre-trained network should give the same output always. This happened consistently when the global initializer was run before the saver but not the other way around. Repeated initializations followed by passes through the network gave different outputs. 

This clued me in to the fact the the correct way to initialize a combination of new variables and pre-trained variables was to run the global initializer before the saver. 

This subtle difference can have insidious effects! If your _vgg19_ network was not initialized with pre-trained weights, it's possible that the effects aren't seen! For example, in my case, I was trying to find a new input image that had the same network response as another input image. In this instance, whether the weights are pre-trained or random, the optimization procedure will still attempt to match the reponses! The incorrect initialization would have eventually caught up to me when I started to make stronger assumptions about the network such as what deeper layers were encoding. If I wasn't using pre-trained weights, the network's layers would not have been capturing the correct abstractions of the image (e.g. objects, concepts, etc).

If you'd like to see for yourself the difference in output between correct and incorrect initialization orders, here's a [notebook](https://github.com/ashwindcruz/style-transfer/blob/master/gotchas/init_order.ipynb) for you to tinker with! 