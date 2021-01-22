---
layout: posts
title: "Style Transfer"
date: 2018-09-26
---

If you're reading this, I'm assuming that you've read the paper _[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)_ and have some familiarity with it. 

Here are the images I'll be working with for this section, the content images are on the left, the style image is on the right:


<img src="/assets/images/2018-09-26-style-transfer/coastal_scene.jpg" width="30%"> <img src="/assets/images/2018-09-26-style-transfer/perth_skyline.jpg" width="30%"> <img src="/assets/images/2018-09-26-style-transfer/starry_night.jpg" width="30%">

(Here they're shown in the original resolution but I resize them to 224x224 when using them for styling.)

The focus of this post is on the _Style Transfer_ section of the paper, Section 2.3 to be precise. This article is accompanied by a [notebook](https://github.com/ashwindcruz/style-transfer/blob/master/style_transfer.ipynb "Style Transfer"). 

This work builds of my previous ones on _Content Representation/Reconstruction_ and _Style Representation/Reconstruction_. I tie these two components in to create the final product. Before reading this post, I'd recommend reading the previous ones on [content](/blog/2018/07/30/content-reconstruction) and [style](/blog/2018/09/08/style-reconstruction). Additionally, the notebook I build here utilizes the code I wrote when building the previous two notebooks; again [content](https://github.com/ashwindcruz/style-transfer/blob/master/content_recs.ipynb "Content Reconstruction") and [style](https://github.com/ashwindcruz/style-transfer/blob/master/style_recs.ipynb "Style Reconstruction").

Unsurprisingly, this final section was the easiest. Working through this paper, the programming became easier as I went along. There were no real surprises or *gotchas* for this final piece. I probably took less than an hour to write the notebook up. 

Structuring the code took some time. This involved deciding on which parts of the code should be abstracted and what the progression of the code should be when reading the notebook top to bottom. I ended up moving the gram matrix and style loss computation into a separate utility module titled *loss*. I think the style loss computation definitely belongs here but perhaps not the gram matrix. However, since I didn't have any other maths functions, I didn't want to create a new module entitled *maths* just for the gram matrix. Since I wrote the content reconstruction notebook before the style reconstruction, this notebook details the content reconstruction before the style.  

Aside from combining the notebooks, there were a couple of new additions here. Firstly, there are now two real images to utilize: the content image and the style image. This meant setting up a new image tensor and ensuring I was getting the response (content representation or style representaiton) for the correct image. Additionally, there's also the weighting on each of the loss components, content and style. The authors provide the ratio between the weights but not the actual weights and I tried out a couple. 

After playing around a little, here are two images I obtained (that I liked, or a the very least, didn't dislike too much):

<img src="/assets/images/2018-09-26-style-transfer/starry_coastal.png" width="45%"> <img src="/assets/images/2018-09-26-style-transfer/starry_perth.png" width="45%">


To produce these images, both the content and style images were resized to 224x224. For the content representation, I used layer 2 of the network and for the style representation, I used the first 2 layers. I fiddled around with these choices and ended on these because they looked pleasing but also because I could iterate a little faster compared to using more layers. Also, I wanted the content image to come through strongly so I decided to use a ratio of *1e-2* between the weights for the content and the style loss. 

Overall, I think the I've achieved my main objective of transferring style as described in the paper but there's still room for improvement. I think my naive/lazy use of Adam instead of L-BFGS (what the authors used) probably hurt. I did some searching but couldn't find a clean, easy-to-use TF implementation of it so I stuck with Adam. Some have noted that using Adam means some tuning is required but I didn't have the patience or the resources which brings me to the main bottleneck I've faced in this implementation: my hardware. As I mentioned in my previous [post](/blog/2018/09/08/style-reconstruction), I don't have a great GPU or power supply. If my memory requirements got too big, for example if I used an image that was too large, my machine would reboot. I would have been happy to play around with the image size to figure out how far I could push the model if all I got was an OOM error. Having the machine reboot each time is a big drain on my resources, disconcerting, and disheartening. Working with a small image meant that it looked quite grainy. Not tweaking the hyperparameters (content-style ratio, gradient optimizer used, initial random seed) meant that there are probably lots of small improvements I could eke out to produce nicer images. Additionally, I didn't use any particular learning schedule which is probably why it looks like the image could still stand to be optimized more. 

Resources aside, I also didn't make these commitments to improve the model because I view this work as just the starting point on this journey. Since their work, the field of neural art has been rapidly improving and I want to get more involved. Perhaps if I was implementing the state of the art, I would invest more time trying to squeeze every little bit of performance out of it. 

That is not to say that I'm definitely done or satisfied with this particular implementation however. I'm hoping that I can upgrade my hardware sometime in the near future and then I hopefully I can attain some better results. Additionally if I learn of implementation hacks as I work through other papers, I might come back and apply them here. 

## References
[1] [Gatys, L.A., Ecker, A.S. and Bethge, M., 2016. Image style transfer using convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2414-2423).](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)