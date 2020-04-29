# hotdog_classifier
Lab 1 of Deep Learning course in HSE. Need to classify images to two classes: hotdog and non-hotdog :)

Test accuracy (0.2 of full train data):
- CNN - 84.5% (20 epochs, 256 batch, 7 min to learn)
- SVM - 74% (20 min to learn)
- VGG16 - 60% (Too long to learn (> 40 min). Only 10 epochs were processed, 32 batch)
- VGG13 + RF - 95.5% (10000 trees)
- VGG13 + SVM - 94-95.5% (10000 trees)
- VGG16 + RF - 94-96% (10000 trees, 64 batch, pre-trained Net, 2 min to learn)
- VGG16 + SVM - 94.6-97.6% (64 batch, pre-trained Net, 10 secs to learn :))
- VGG19 + RF - 94-96% (64 batch, pre-trained Net, 10 secs to learn :))
- VGG19 + SVM - 94-96.4% (64 batch, pre-trained Net, 10 secs to learn :))

So I think that the most efficient approach to solve that task - VGG + SVM (SVC) or any another ML classifier (for example RF).
Also I tried to change VGG16 to VGG19 it made result worse. I think that it's some kind of overfit.
I suppose that VGG16 is optimal net for pre-processing features to ML classifier