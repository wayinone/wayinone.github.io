# Landing Page Jekyll theme

Jekyll theme based on [landing-page bootstrap theme ](http://startbootstrap.com/templates/landing-page/)

## How to use
 - Place a image in `/img/services/`
 - Create posts in `/_post` to display your services.  Use the follow as an example:

```txt
---
layout: default
img: ipad.png
category: Services
title: The service title
---
The description of this service
```
 - Note that the file name of each post is "YYYY-MM-DD-..." 

## Using jekyll server to speed up the build
 - I am using Windows, so I used powershell to type the command.
 - First you should clone all the repository to the local folder.
 - In powershell, go to the folder and type
 ```txt
 ---
 jekyll serve
 ---
 ```
 - Then go to your browers and type: http://localhost:4000/


## License
The contents of this repository are licensed under the [Apache
2.0](http://www.apache.org/licenses/LICENSE-2.0.html).

## Version
1.0.0
