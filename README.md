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
 - First, you need to be sure that jekyll works on your machine. I have to install ruby and some other things for it to work.
 - I am using Windows, so I used powershell to type the command.
 - First you should clone all the repository to the local folder.
 - In powershell, go to the folder and type
 ```txt
 jekyll serve
 ```
 - Then go to your browers and go to: http://localhost:4000/
 - Every time you make change in the local repository, the jekyll server will update the code and the local webpage will display your work.
 - Keey an eye on powershell since sometime there is bug and you wont see any change in the site.
 - If the website is not updating and no error message, just delete the `/_site` folder, and build again by: (remember to ctrl+C to end the current session.
 ```txt
 jekyll serve
 ```
 


## License
The contents of this repository are licensed under the [Apache
2.0](http://www.apache.org/licenses/LICENSE-2.0.html).

## Version
1.0.0
