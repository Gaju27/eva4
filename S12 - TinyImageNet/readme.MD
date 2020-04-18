# ![Explaination of 50 Images dogs annotations](https://github.com/Gaju27/eva4/tree/master/S12%20-%20TinyImageNet/DogImageSet)
```
"images":[
      {
           "id":0,
          "width":500,
         "height":375,
         "file_name":"dog_01.jpeg",
         "license":1,
         "date_captured":""
      } 
 ```
>> - [x]  images: Key of json file for each dog images

>> - [x] id: Unique key value for each image_id

>> - [x] width, height: represent image size

>> - [x] file_name: image name 

```
annotations":[
      {
         "id":0,
         "image_id":"0",
         "segmentation":[
            9,
            7,
            458,
            7,
            458,
            369,
            9,
            369
         ],
         "area":162538,
         "bbox":[
            9,
            7,
            449,
            362
         ],
         "iscrowd":0
      }
```
- [x] annotations: Key of dog images annotations

- [x] id: Unique key values of annotations

- [x] image_id: Unique key values which matched with each image file_name  (dog_01.jpeg)

- [x] bbox: repsents x,y and height and width respectively as per representation above
	 
```
categories":[
      {
         "id":1,
         "name":"human",
         "supercategory":"type"
      },
      {
         "id":2,
         "name":"unknown",
         "supercategory":"type"
      },
      {
         "id":3,
         "name":"Dog",
         "supercategory":"type"
      },
      {
         "id":4,
         "name":"Board",
         "supercategory":"type"
      },
      {
         "id":5,
         "name":"tub",
         "supercategory":"type"
      },
      {
         "id":6,
         "name":"Belt",
         "supercategory":"type"
      },
      {
         "id":7,
         "name":"Bag",
         "supercategory":"type"
      },
      {
         "id":8,
         "name":"Jacket",
         "supercategory":"type"
      },
      {
         "id":9,
         "name":"Sandles",
         "supercategory":"type"
      },
      {
         "id":10,
         "name":"Watch",
         "supercategory":"type"
      },
      {
         "id":11,
         "name":"camera",
         "supercategory":"type"
      },
      {
         "id":12,
         "name":"blur",
         "supercategory":"image_quality"
      },
      {
         "id":13,
         "name":"good_illumination",
         "supercategory":"image_quality"
      },
      {
         "id":14,
         "name":"frontal",
         "supercategory":"image_quality"
      },
```	
>> - [x] categories : Key represent the classes of each annotations done

>> - [x] id: Unique key of the each classes

>> - [x] name: class name

>> - [x] supercategory: parent class	
