#!/bin/sh

wget http://www.kasrl.org/jaffeimages.zip
unzip jaffeimages.zip
mkdir output
rm jaffeimages.zip
rm -R __MACOSX
rm jaffe/README
rm jaffe/.DS_Store
