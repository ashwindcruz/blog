---
layout: posts
title: "Gotcha! Pyplot Image Displays"
date: 2018-08-10
---

While working on the first part of my [style-transfer project](/blog/2018/07/30/content-reconstruction), I used pyplot's [imshow](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html) to diplay images in the notebook. However, it took me a little bit of playing around before the images looked as I expected them to. For reference, here is how I might load the original image and what it would look like:
```python
image = cv2.imread('../coastal_scene.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
<img src="/assets/images/2018-07-30-content-reconstruction/coastal_scene.jpg" width="45%">

To better simulate the kind of data I was working with, I'll cast this image to a float, keeping the values the same though. The reason I'm doing this is because the input to my network, the image I was optimizing and trying to view, was of the type _float32_. Here's the same image but in float format: 
```python
image_float = np.array(image, dtype=np.float32)
```
<img src="/assets/images/2018-07-30-content-reconstruction/pyplot_formatting/original_float.png" width="45%">

So it's obviously wrong but at least it's still possible to tell that the underlying array does have some relationship to the original images. However, this is only happening because the values are exactly the same even though the data type is different. During the image optimization procedure, the values only start to look similar a little later during training. To simulate what it might look like earlier on, I'll add some noise to the image and show you what it looks like: 
```python
height, width, channels = np.shape(image)
image_noise = np.random.rand(height, width, channels) * 50 
image_invalid = image + image_noise
```
<img src="/assets/images/2018-07-30-content-reconstruction/pyplot_formatting/noisy_float.png" width="45%">

I saw images like this so many times during training and it drove me crazy! Thankfully I was able to use cv2 and tensorboard to view images and could see images that made more sense. Looking a little closer at the [documenation for imshow](), I found this not-so-fine-print: 
> Elements of RGB and RGBA arrays represent pixels of an MxN image. All values should be in the range [0 .. 1] for floats or [0 .. 255] or integers. Out-of-range values will be clipped to these bounds.

I was using floats in the range [0 .. 255] and there were being clipped to [0 .. 1]. The rubbish I was seeing made some sense. So let's cast the array to _uint8_ so the values aren't clipped: 
```python
image_invalid_int = np.array(image_invalid, dtype=np.uint8)
```
<img src="/assets/images/2018-07-30-content-reconstruction/pyplot_formatting/noisy_int_overflow.png" width="45%">

Better but it's still not quite right. The final issue comes from what happens when you cast from _float32s_ to _uint8s_. The maximum value that _uint8_ can have is 255. Go any higher, as I did when I added noise to the image in _float_ format and the value will wrap around! So 256 becomes 0, 257 becomes 1, and so on. 

So before casting the image to _uint8_, we should clip the values to be within the [0 .. 255] range. Here's one way to go about it:
```python
image_clipped = np.clip(initial_image, 0, 255)
image_rounded = np.rint(image_clipped)
formatted_image = np.asarray(image_rounded, dtype=np.uint8)
```

The _rint_ function round _floats_ to _ints_ properly. If I skipped that, values like 1.5 are _truncated_ to 1. Honestly, the images would likely look the same whether the _floats_ are rounded or truncated but the former is more 'correct' so I chose that instead. Plus, once this image formatting is wrapped up in a separate function, you won't even have to go through the hassle of typing that extra line each time you want to format an image so why not go all the way?

For comparison, I've shown the original image on the left and the noisy, formatted image on the right:
<img src="/assets/images/2018-07-30-content-reconstruction/pyplot_formatting/corrected.png" width="45%"> <img src="/assets/images/2018-07-30-content-reconstruction/pyplot_formatting/noisy_int.png" width="45%">

If you look carefully, you'll notice that the images aren't quite the same. This is because there's still the additive noise introduced and the formatting doesn't get rid of that completely. Lots of previously smaller pixel channel values are now at 255. However, in the image optimization context, as training progresses, fewer pixels will be out of bounds and this effect will be less noticeable. 

So when working with images and pyplot, be wary of the little subtleties! I advise you to create a formatting function (it would probably contain the 3 lines from the last snippet) and use that on an image before displaying it. 

If you'd like to play around more, here's a [notebook](https://github.com/ashwindcruz/style-transfer/blob/master/gotchas/plt_image_formatting.ipynb) for you to tinker with! It's even got a copy of the formatting function that I mentioned if you would like.  