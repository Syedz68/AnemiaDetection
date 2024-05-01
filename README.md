---Anemia Detection---

This is a web application to detect anemia from eye image. This web application was built using Flask framework. There are two trianed deep learning models used for this project.

// Segmentation Task
For the segmentation task, Unet was used as a pre trained model on a custom dataset. It is used to predict the conjunctiva mask of an eye image. Then by overlapping the prdeicted mask over the original image, the conjuctiva gets segmneted from the image.

// Classification Task
For the classification task, CNN was used as a pre trained model on a publicly availabe dataset which containts the conjunctiva images labeled as Anemic or Nonanemic. Then the segmented conjuctiva from the segmentation part mentioned above is passed to this model and the probabilty of the image having Anemia is then determined.

// Images of the web-app
![soft1](https://github.com/Syedz68/AnemiaDetection/assets/107263740/2419f5b3-db6f-43a6-bbab-ea61ae5668ec)

![soft2](https://github.com/Syedz68/AnemiaDetection/assets/107263740/eae0263f-caaa-4f78-b389-9efe8ec6e21e)
