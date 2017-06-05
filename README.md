# Wei-Ying's Personal Website

I am using Jekyll theme based on [landing-page bootstrap theme ](http://startbootstrap.com/templates/landing-page/)

## Using Github
 - I am new to Github. What I do to create this site is:
   1. create a repository in Github names "YOUR-GITHUB-Username.github.io"
   2. Pick jekyll theme template at https://jekyllthemes.io/
   3. Put the theme you choosed in your newly created repository.
   4. Now you should be able to see your template website online. Go to "YOUR-GITHUB-Username.github.io"
   
 - I am using Github Desktop(GD) to sync my repository with local repository (local means in my computer. I will use "repo" as the short for "repository")
   1. Once you connect via GD, everything you changed in the local repo should be visible in the GD. 
	  - To syn (when you change thing in local repo and want to upload): in GD, type something in the "summary", then press "commit to master", then "Pull origin"
   2. If you encounter an error message: please tell me who you are, do the following in command line[solution from here](https://stackoverflow.com/questions/11656761/git-please-tell-me-who-you-are-error)
	 ```
	 1.git init
     2.git config user.name "someone"
  	 3.git config user.email "someone@someplace.com"
	 4.git add *
	 ```

## Using jekyll server to speed up the build
 - I am using Windows, so I used "powershell" to type the command. I think it is better than "cmd"
 - First, you need to be sure that jekyll works on your machine. 
   -I have to install ruby and use (descibed [here](https://jekyllrb.com/docs/installation/))
   ```
   gem install jekyll
   ``` 
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

## How to add post
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

## Change the font css file	
 - To change the font of your own .md file (e.g. Resume.md, and I am NOT talking about anything in the front page), you need to change css file: Go to "/css/mycss.css" to edit them.
 - Since 
 1. these .md articles will have layout: default2 (default2.html is in folder:"/_include")
 2. default2 will quote "header2.html" (also in folder:"/_include"
 3. header2.html will quote "mycss.css" 
 

## Using Notepad++
 - Note that you want to install 32bit version since it includes Plugin Manager (So that you have spell checker.)
 - After install, open Notepad++, then go to "Plugin->Plugin Manager", install spell checker and Dspell checker. (you will be prompt to install Aspell dictionary.) Practically, open "Plugin->Dspell" and choose the option of auto detect, and use Aspell dictionary.
 - If you want to change theme: go to "Settings-> style configurator -> select theme."
 - To easily edit web content, download style file (.xml) from [here](https://github.com/Edditoria/markdown-plus-plus), and go to "Language-> Define your language -> import" to import the xml file. I love to use "Blackboard Style"

## License
The contents of this repository are licensed under the [Apache
2.0](http://www.apache.org/licenses/LICENSE-2.0.html).

## Version
1.0.0
