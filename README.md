# Poznan_classification
The goal of the project was to recognize most popular places from Poznań using Bag of Visual Words.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Wie%C5%BCowiec_Ba%C5%82tyk_w_Poznaniu.jpg/1024px-Wie%C5%BCowiec_Ba%C5%82tyk_w_Poznaniu.jpg" width="350">
<img src="https://static.polskieszlaki.pl/zdjecia/wycieczki/2017-08/poznan-8.jpg" width="350">
<img src="https://amu.edu.pl/__data/assets/image/0016/22390/Colegium_Minus_Zegar_600_dni_2017_fot._lukasz_Wozny-1.jpg" width="350">
<img src="https://cdn.galleries.smcloud.net/t/galleries/gf-HYF4-Nw2A-DyEQ_teatr-wielki-w-poznaniu-664x442-nocrop.jpg" width="350">
<img src="https://bi.im-g.pl/im/22/b1/f6/z16167202V,Okraglak.jpg" width="350">

The training dataset was provided by a lecturer. It contained 32 images per class -> 160 images. I've used only those
 images and scored 96% on a evaluation set and 85% on test set. 

## Project requirements
- Python 3.8
- see requirements.txt

## Usage
In the `train` folder you can find jupyter notebook and python script. You can use any of them to train your vocabulary
 model and classificator.
In the main folder you can find vocabulary model and classificator that was already trained by me.
Run `main.py` to test the model and classificator. 