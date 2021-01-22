---
layout: posts
title: "Gotcha! CV2 JPEG vs PNG"
date: 2018-08-10
---

While working on the first part of my [style-transfer project](/blog/2018/07/30/content-reconstruction), I ran into lots of image issues. 
One of the issues was that cv2 uses a BGR channel order instead of RGB, the latter is more common. This resulted in a lot of my images seeming to have a weird tint. I talk about that [here](/blog/2018/08/10/gotcha-cv2-pyplot-channel-order.html). This post will be focusing on a different issue I found while trying to solve the channel issue. 

Before I found the correct fix, I did get desperate enough to try looking at some of the raw pixel values and found something very suprising: I was specifying a particular array to be saved but upon loading the saved image, the loaded array had different pixel values! 
To examine this effect, you can play with this [notebook](https://github.com/ashwindcruz/style-transfer/blob/master/gotchas/cv2_extension_saving.ipynb).

I found out that the whole time I was saving images with the _.jpg_ extension, cv2 was carrying out JPEG compression on the array and saving the compressed array. So when it was loaded back in, the values would look extremely different! 

There are two things to note here: 
* While the pixel values seem to look very different, there is no visually perceptible change. 
* If you load a previously saved _jpg_, then you'll find that the values don't change much. Since you're already working with an array that has had its values compressed, no further compression is run. 

I found that PNG offered lossless compression and indeed if you look at the notebook and the values of the _png_ image, they are almost the same as the original array. The difference comes about just from rounding floats to ints. 
Interestingly, the small differences do add up which you can see in the summed difference of arrays. However, the difference between the original image and the _jpg_ one is 10x more than between the original image and the _png_ one. 

The takeaway here is to be aware that depending on the type of extension you choose to use with _[cv2.imwrite](https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#imwrite)_, you'll end up with different arrays. I spent an embarrasingly large amount of time being baffled by why the arrays looked so different! Hopefully that doesn't happen to you. 

If you'd like to play around more, here's a [notebook](https://github.com/ashwindcruz/style-transfer/blob/master/gotchas/cv2_extension_saving.ipynb) for you to tinker with! 