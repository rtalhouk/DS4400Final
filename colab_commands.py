from colab.google import files

files.upload()

! mkdir ~/.kaggle
! mv kaggle.json ~/.kaggle/
! rm -rf ./images
! mkdir ./images
! kaggle datasets download grassknoted/asl-alphabet
! unzip asl-alphabet.zip -d ./images/
! rm asl-alphabet.zip