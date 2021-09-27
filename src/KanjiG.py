from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Generator():
    def __init__(self):
        pass

    def init(self, size=64, bg_color='white', word_color='black', mode='single', grid=[1, 1]):
        self.SIZE = size
        self.BG_COLOR = bg_color
        self.WORD_COLOR = word_color
        self.MODE = mode
        self.GRID = grid # row, column

        # width, height
        self.BG = np.array(Image.new('L', (self.SIZE,self.SIZE), self.BG_COLOR))
        self.BG_MULTI = np.array(Image.new('L', (self.SIZE*self.GRID[1], self.SIZE*self.GRID[0]), self.BG_COLOR))

        self.DATA_PATH = Path(__file__).resolve().parents[0] / 'data'
        with (self.DATA_PATH/'3755.txt').open(encoding='utf-8') as f:
            self.KANJI_LIST = f.readline()

        self.FONTS_PATH = self.DATA_PATH / 'fonts'
        self.FONT_ENCODING = 'unic'
        self.FONT_SIZE = size
        self.FONTS = []
        for font_path in self.FONTS_PATH.glob('*.ttf'):
            self.FONTS.append(
                ImageFont.truetype(font_path.as_posix(), self.FONT_SIZE, encoding=self.FONT_ENCODING)
            )
        self.TEXT_LOCATION = (0, 0)
        self.LINE_COLOR = [144] * 3

        self.ROTATE_ANGLES = [-7, -5, -3, -1, 0, 1, 3, 5, 7]

    def get_font(self):
        index = np.random.randint(len(self.FONTS))
        font = self.FONTS[index]
        return font

    def noise(self, image):
        noise = np.random.randint(0,256,(self.SIZE,self.SIZE))
        # _, noise = cv2.threshold(noise.astype('uint8'), 240, 255, image, cv2.THRESH_BINARY)
        t = np.random.randint(200, 250)
        image[noise > t] = 255 - image[noise > t]
        return image

    def rotate(self, image):
        h, w = image.shape
        m = cv2.getRotationMatrix2D((h*0.5, w*0.5), np.random.choice(self.ROTATE_ANGLES), 1)
        if self.BG_COLOR == 'white':
            image = 255 - image
        image = cv2.warpAffine(image, m, (h, w))
        if self.BG_COLOR == 'white':
            image = 255 - image
        return image

    def line(self, image):
        p1 = np.random.randint(0, self.SIZE, (2,))
        p2 = np.random.randint(0, self.SIZE, (2,))
        thickness = np.random.randint(1, 5)
        image = cv2.line(image, p1, p2, self.LINE_COLOR, thickness=thickness)
        return image

    def gen_sample(self):
        gen = {'single': self.gen_single, 'multiple': self.gen_multiple}[self.MODE]
        image, label = gen()
        return image, label

    def gen_single(self):
        index = np.random.randint(len(self.KANJI_LIST))
        word = self.KANJI_LIST[index]
        image = Image.fromarray(self.BG)
        draw = ImageDraw.Draw(image)
        draw.text(self.TEXT_LOCATION, word, font=self.get_font(), fill=self.WORD_COLOR)
        image = np.array(image)
        image = self.noise(image)
        image = self.rotate(image)
        for _ in range(np.random.randint(3)):
            image = self.line(image)
        label = {
            'word': word,
            'index': index
        }
        return image, label

    def gen_multiple(self):
        image = self.BG_MULTI
        cv2.imwrite('a.jpg', image)
        words = []
        for i in range(self.GRID[0]):
            for j in range(self.GRID[1]):
                img, label = self.gen_single()
                location = i*self.SIZE, j*self.SIZE
                image[location[0]:location[0]+self.SIZE, location[1]:location[1]+self.SIZE] = img
                word = label['word']
                index = label['index']
                size = self.SIZE, self.SIZE
                words.append({
                    'word': word,
                    'index': index,
                    'size': size,
                    'location': location
                })
        label = {
            'words': words
        }
        return image, label


if __name__ == '__main__':
    g = Generator()

    # g.init(mode='single')
    # for i in range(100):
    #     img, label = g.gen_sample()
    #     img = Image.fromarray(img)
    #     word = label['word']
    #     img.save(f'test/{word}.jpg')

    g.init(mode='multiple', grid=[5,5])
    for i in range(10):
        img, label = g.gen_sample()
        img = Image.fromarray(img)
        img.save(f'test_mul/{i}.jpg')
        for i in label['words']:
            print(i['word'])
            print(i['location'])



