# Kanji Generator

## Install
```shell
# Pillow and opencv are required
python -m pip install Pillow
python -m pip install opencv-python

# Use pip to install KanjiG
python -m pip install -U KanjiG -i https://test.pypi.org/simple 
```

## Usage
```python3
from PIL import Image
from KanjiG import Generator

g = Generator()
g.init(mode='single')
for i in range(10):
    img, label = g.gen_sample()
    img = Image.fromarray(img)
    word = label['word']
    img.save(f'{word}.jpg')

g.init(mode='multiple', grid=[5,5])
for i in range(10):
    img, label = g.gen_sample()
    img = Image.fromarray(img)
    img.save(f'{i}.jpg')
    for i in label['words']:
        print(i['word'])
        print(i['location'])
```

<img src="./doc/single.jpg">
<img src="./doc/multiple/0.jpg">