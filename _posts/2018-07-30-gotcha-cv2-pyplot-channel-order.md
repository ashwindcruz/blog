---
layout: posts
title: "Gotcha! CV2 and Pyplot Channel Order"
date: 2018-08-10
---

While working on the first part of my [style-transfer project](/blog/2018/07/30/content-reconstruction), I used Open CV's [cv2.imwrite](https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#imwrite) to save images to disk. However, the images seemed to have a weird tint to them. In the figure below, I show the original image (left) alongside the result of loading the image using cv2 and displaying it with pyplot (right): 

<img src="/assets/images/2018-07-30-content-reconstruction/coastal_scene.jpg" width="45%"> <img src="/assets/images/2018-07-30-content-reconstruction/cv2_no_convert/bgr.jpg" width="45%">

After reading the documentation for _imwrite_ more carefully, I realised that cv2 was interpreting the channel order as BGR instead of RGB which was what pyplot was expecting. Converting between the two was fairly straightforward: 
```python
image = cv2.imread('dummy_image.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

Then when working with pyplot, you can use _rgb\_image_. 

And here is the result: 

<img src="/assets/images/2018-07-30-content-reconstruction/coastal_scene.jpg" width="45%"> <img src="/assets/images/2018-07-30-content-reconstruction/cv2_no_convert/rgb.jpg" width="45%">

Note that if you had an image in the RGB space and wanted to save it to disk using cv2, you'd need to convert it to BGR first as cv2 saves images in that format. 
In general, whether displaying images in a notebook or saving them to disk, if the colour scheme seems off, check the channel order. 

If you'd like to play around more, here's a [notebook](https://github.com/ashwindcruz/style-transfer/blob/master/gotchas/cv2_loading.ipynb) for you to tinker with! 
