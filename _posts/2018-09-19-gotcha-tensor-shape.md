---
layout: posts
title: "Gotcha! Tensor Shape"
date: 2018-09-19
---

While working on the second part of my [style-transfer project](/blog/2018/09/08/style-reconstruction), I needed to obtain the shape of a tensor. I decided to try using the [*tf.shape*](https://www.tensorflow.org/api_docs/python/tf/shape) function. 

Reading the example provided on the documentation, it seemed like it would do what I needed, nevermind the odd:
> This operation returns a 1-D integer tensor representing the shape of input.

When I tried this in a notebook: 

```python
dummy_data = np.random.rand(1,4,2)
dummy_tensor = tf.constant(dummy_data)
print("TF shape function outside a session: {}".format(tf.shape(dummy_tensor)))
```

The behaviour was unexpected: 
>> TF shape function outside a session: Tensor("Shape:0", shape=(3,), dtype=int32)

I had a hunch about what I was doing wrong so I tried the following: 

```python
with tf.Session() as sess:
    dummy_tensor_shape_ = sess.run(dummy_tensor_shape)
    print("TF shape function inside a session: {}".format(dummy_tensor_shape_))
 ```

 Which produced the following: 
 >> TF shape function inside a session: [1 4 2]

 As expected! 

 Once again, I was reminded that tensor operations will only provide what you want within a session! Before that, the tensor is simply an operation waiting to be executed. The *3* noted in the shape outside the session seems to be the length of the list containing the actual shape. 

Thankfully, TF offers another function: [*get_shape()*](https://www.tensorflow.org/api_docs/python/tf/Tensor#shape)
Note that *get_shape* is just an alias for *shape*. This method infers the static shape but can fail in some cases, for example where the input data shape isn't known until runtime. This [explanation](https://stackoverflow.com/a/37096395) provides a clear example of a failure case. 

The [TF documentation](https://www.tensorflow.org/api_docs/python/tf/Tensor#shape) notes that this function can provide debugging information and early warnings. My use case was slightly different in that I actually used the shape information in a loss calcuation. So perhaps I should have figured out how to use *tf.shape* properly within the graph. However, *get_shape* worked fine and it was easier to understand at the time so it is what I ended up using. 

If you'd like to see for yourself the difference ways of getting a tensor's shape, here's a [notebook](https://github.com/ashwindcruz/style-transfer/blob/master/gotchas/tensor_shape.ipynb) for you to tinker with! 