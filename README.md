<p align="center">
  <a href="" rel="noopener">
 <img width=500px height=200px src="http://statics.scnu.edu.cn/statics/css/scnuportal/contentlogo2.png" alt="SCNU logo"></a>
</p>

<h3 align="center">学校创新周泰迪人脸分类任务</h3>
---

<p align="center"> 结合Faceboxes以及ResNet-50的人脸分类任务
    <br> 
</p>

## 📝 Table of Contents
- [Getting Started](#getting_started)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)
- [Instruction](#instruction)


## 🏁 Getting Started <a name = "getting_started"></a>

### Requirement

- Python 3.6
- Tensorflow 1.12
- Tensorboard
- OpenCV-Python 3.4.0.12

### For Task1:

```
cd Task1
Python get_face.py
```

### For Task2:

```
cd Task2
Python task2.py
```

### For Task3:

```
cd Task3
Python task3.py
```

### For Task4:

```
cd Task4
Python task4.py
```

### For Task5:

```
cd Task5
Python model.py
```

## ✍️ Authors <a name = "authors"></a>
- [@qs956](https://github.com/qs956) 

  email:qs956@163.com

  QQ:510733503

See also the list of [contributors](https://github.com/qs956/Taidi-face-classify/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

### Inspiration

- [@TropComplique](https://github.com/TropComplique/FaceBoxes-tensorflow)
- [@tensorflow](https://github.com/tensorflow/)


### References

- [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/abs/1708.05234)
- [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)

## ⛏️Instruction <a name = "instruction"></a>

### Task1

Use OpenCV`s 'VideoCapture' function to get photo from the camera and save as 'JPG' format.

### Task2

Face detection by using Faceboxes on tensorflow with the pretrained weights on FDDB.

### Task3

Use OpenCv to grayscale and scale the image to (256,256). The results are saved as a numpy.array

### Task4

Build ResNet-50 by silm in Tensorflow's Model.Use the cross entropy loss function and the AdamOptimizer for optimize.Tensorboard visualization and save the weights will work automatically.You can enter the tensorboard by using:

```powershell
tensorboard --logdir=log
```

### Task5

Real-time prediction.Press 'q' to exit.