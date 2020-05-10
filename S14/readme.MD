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
* [Mentors](#mentors)



<!-- DATASET PREPARATION -->
## Preparing Dataset    
   -  #### Download Images 
            Downloaded Human pose images as Foreground (100) and shopping mall as background Image (100).
   
      <!-- FOREGROUND IMAGES -->
   -  #### Foreground Images 
         1. The downloaded images are of different formats (.jfif,jpeg,.png,.webp), hence I [Renamed and converted](https://github.com/Gaju27/eva4/blob/master/S14/Convert_png_rename.py) them to .png. 
         2. I used [GIMP](https://www.gimp.org/downloads/) tool to remove backgroud in these images. Please see [Tutorial](https://www.google.com/search?safe=active&rlz=1C1GCEU_enIN883IN883&sxsrf=ALeKk030ZpJhWSmJ1h5Xn54q3aUjptyRJQ%3A1589105142609&ei=9tG3XtLnJMS4sgXB16nICg&q=gimp+remove+background&oq=gimp+remo&gs_lcp=CgZwc3ktYWIQAxgAMgQIIxAnMgIIADICCAAyAggAMgIIADICCAAyAggAMgIIADICCAAyAggAOgQIABBHOgcIABAUEIcCOgoIABCDARAUEIcCOgUIABCDAToFCAAQkQJQivoDWJmIBGD4kARoAHABeACAAeMCiAG6GpIBBDMtMTCYAQCgAQGqAQdnd3Mtd2l6&sclient=psy-ab#kpvalbx=_O9K3XvqZOc_EswX12ZSADQ51) how to remove background from images
         3. I have [Flipped](https://github.com/Gaju27/eva4/blob/master/S14/Flip_Images.py) these 100 Images Horizantally. Now I have [200 Foreground](https://github.com/Gaju27/eva4/tree/master/S14/Foreground) Images.
         4. I have generated [Mask Images](https://github.com/Gaju27/eva4/tree/master/S14/Foreground_Mask) for these 200 Foreground Images. [mask](https://github.com/Gaju27/eva4/blob/master/S14/Flip_Images.py) 
   
      <!-- BACKGROUND IMAGES -->
   -  #### Backgroud Images

[![Pysource][product-screenshot]](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)


## Prerequisites

* Jupyter notebook
* Colab
* Python 3 and Above
* PIL
* GIMP

<!-- CODE CHANGES -->
## Code Changes

For more examples, please refer to the [Documentation](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Gajanana Ganjigatti - [Gaju_on_twitter](https://twitter.com/Gajucg)

Project Link: [gaju27/eva4](https://github.com/gaju27/eva4)



<!-- MENTORS -->
## Mentors

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
