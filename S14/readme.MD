<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo, twitter_handle, email
-->




<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Mentor][mentor-shield]][mentor-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/github_username/repo">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Preparation of Foreground Backgroudn Image Dataset</h3>

  <p align="center">
    Stored my zipped data set in
    <br />
    <a href="https://drive.google.com/uc?id=1-08m0zJ1-PNcfFcM1VRdBq1r6C7R6raJ&export=download/"><strong>Gdrive Store »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Gaju27/eva4/blob/master/S14/fg_bg.ipynb">Code for Fg_bg images </a>
    ·
    <a href="https://github.com/Gaju27/eva4/issues">Report Bug</a>
    ·
   </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents



* [Dataset Preparation](#preparing-dataset)
    * [Download Images](#download-images)
    * [Foreground Images](#foreground-images)
    * [Background Images](#backgroud-images)
* [Prerequisites](#prerequisites)
* [Code Changes](#code-changes)
* [License](#license)
* [Contact](#contact)
* [Mentor](#mentor)



<!-- DATASET PREPARATION -->
## Preparing Dataset    
   -  #### Download Images 
         Downloaded Human pose images as Foreground (100) and shopping mall as background Image (100).
   
      <!-- FOREGROUND IMAGES -->
   -  #### Foreground Images 
         1. Remove [Duplicate](https://github.com/Gaju27/eva4/blob/master/S14/remove_dup_images.ipynb) Images if any
         2. The downloaded images are of different formats (.jfif,jpeg,.png,.webp), hence I [Renamed and converted](https://github.com/Gaju27/eva4/blob/master/S14/Convert_png_rename.py) them to .png. 
         3. I used [GIMP](https://www.gimp.org/downloads/) tool to remove backgroud in these images. Please see [Tutorial](https://www.google.com/search?safe=active&rlz=1C1GCEU_enIN883IN883&sxsrf=ALeKk030ZpJhWSmJ1h5Xn54q3aUjptyRJQ%3A1589105142609&ei=9tG3XtLnJMS4sgXB16nICg&q=gimp+remove+background&oq=gimp+remo&gs_lcp=CgZwc3ktYWIQAxgAMgQIIxAnMgIIADICCAAyAggAMgIIADICCAAyAggAMgIIADICCAAyAggAOgQIABBHOgcIABAUEIcCOgoIABCDARAUEIcCOgUIABCDAToFCAAQkQJQivoDWJmIBGD4kARoAHABeACAAeMCiAG6GpIBBDMtMTCYAQCgAQGqAQdnd3Mtd2l6&sclient=psy-ab#kpvalbx=_O9K3XvqZOc_EswX12ZSADQ51) how to remove background from images
         4. I have [Flipped](https://github.com/Gaju27/eva4/blob/master/S14/Flip_Images.py) these 100 Images Horizantally. Now I have [200 Foreground](https://github.com/Gaju27/eva4/tree/master/S14/Foreground) Images.
         5. I have generated [Mask Images](https://github.com/Gaju27/eva4/tree/master/S14/Foreground_Mask) for these 200 Foreground Images. [mask](https://github.com/Gaju27/eva4/blob/master/S14/Flip_Images.py) 

      <!-- BACKGROUND IMAGES -->
   -  #### Backgroud Images
         1. Remove [Duplicate](https://github.com/Gaju27/eva4/blob/master/S14/remove_dup_images.ipynb) Images if any
         2. The downloaded images are of different formats (.jfif,jpeg,.png,.webp), hence I [Renamed and converted](https://github.com/Gaju27/eva4/blob/master/S14/Convert_png_rename.py) them to .png. 
         3. [Resized](https://github.com/Gaju27/eva4/blob/master/S14/resize_250-250.py) Images to 250x250
         

## Prerequisites

* [Jupyter](https://jupyter.org/install) notebook
* [Colab](http://colab.research.google.com/)
* [Python 3](https://www.python.org/downloads/) and Above
* [PIL](https://pypi.org/project/Pillow/2.2.1/)
* [GIMP](https://www.gimp.org/downloads/) 
* [zip-files](https://pypi.org/project/zip-files/)

<!-- CODE CHANGES -->
## Code Changes

Till now we have just prepared  [Foregroud](), [foreground Mask]() and [Background]() Images. :zzz: Here is the [fg_bg.ipynb](https://github.com/Gaju27/eva4/blob/master/S14/fg_bg.ipynb) big fun starts. I have to generate 400k Images by randomly placing foreground images over background images. `Background (100) X Foreground (200) x Random Placing (20) times= 400k Images ~2gb size`
Now i have to generate using [google colab](http://colab.research.google.com/) becasue google colab gives us GPU for free. Learn More about colab [here](https://www.tutorialspoint.com/google_colab/index.htm)
    
   ### Lets Understand code 
      1. Import all the required packages
      2. Mount gdrive
      3. Read Background Images from Gdrive folder
      4. Now Open zip with `append` for every 4080 file because I have foreground images 204x20=4080
      5. Read Foreground Images (204) and resize to 100x100
      6. Overlay Randomly Foreground Images over Background and save for 20 Iterations
      7. Create Mask for Foregroudn Images for same randomly called positions and save for 20 Iterations
      8. Now Increament zip index for 1 for each background Image
      9. Close zip file after completion for 100 background Images.
  
  :boom: Note-->> If you forget to close zip file it will be corrupted :poop:
      
     

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Gajanana Ganjigatti - [Gaju_on_twitter](https://twitter.com/Gajucg)

Project Link: [gaju27/eva4](https://github.com/gaju27/eva4)



<!-- MENTOR -->
## Mentor

* [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[mentor-shield]: https://img.shields.io/badge/Mentor-mentor-yellowgreen
[mentor-url]: https://www.linkedin.com/in/rohanshravan/
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/Gaju27/EVA/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/gajanana-ganjigatti/
[convert-rename]: https://github.com/Gaju27/eva4/blob/master/S14/Convert_png_rename.py
