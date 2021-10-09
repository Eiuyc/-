from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Generator():
    def __init__(self):
        self._DATA_PATH = Path(__file__).resolve().parents[0] / 'data'
        with (self._DATA_PATH/'3755.txt').open(encoding='utf-8') as f:
            self.Hanzi_LIST = f.readline()

        self._FONTS_PATH = self._DATA_PATH / 'fonts'
        self._FONT_ENCODING = 'unic'
        self._TEXT_LOCATION = (0, 0)

        self._NOISE_THRESHOLD_RANGE = [200, 240]

        self._ROTATE_ANGLES = [-9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9]

        self._LINE_COLORS = [100, 150, 200]
        self._LINE_COUNT_RANGE = [1, 3]
        self._LINE_THICKNESS_RANGE = [1, 3]

        self._SCALE_SIZE_RANGE = [1, 0.8, 0.7, 0.6]

        self._AUGMENT_METHODS = {
            'noise': self._noise,
            'rotate': self._rotate,
            'line': self._line,
            'scale': self._scale,
        }

        self.config()

    def config(self, size=64, bg_color='white', word_color='black', mode='single', grid=[1, 1]):
        self._SIZE = size
        self._BG_COLOR = bg_color
        self._WORD_COLOR = word_color
        self._MODE = mode
        self._GRID = grid # row, column

        # width, height
        self._BG = np.array(Image.new('L', (self._SIZE,self._SIZE), self._BG_COLOR))
        self._BG_MULTI = np.array(Image.new('L', (self._SIZE*self._GRID[1], self._SIZE*self._GRID[0]), self._BG_COLOR))

        self._BIAS = size / 4
        self._FONT_SIZE = size
        self._FONTS = []
        for font_path in self._FONTS_PATH.glob('*.ttf'):
            self._FONTS.append(ImageFont.truetype(font_path.as_posix(), self._FONT_SIZE, encoding=self._FONT_ENCODING))

    def _get_font(self, font_name=None, size=None):
        font = self._FONTS[0]
        if font_name is None:
            index = np.random.randint(len(self._FONTS))
            font = self._FONTS[index]
        else:
            for f in self._FONTS:
                if Path(f.path).name == font_name:
                    font = f
                    break
        if size:
            font = ImageFont.truetype(font.path, size, encoding=self._FONT_ENCODING)
        return font

    def _get_text_image(self, word):
        image = Image.fromarray(self._BG)
        draw = ImageDraw.Draw(image)
        draw.text(self._TEXT_LOCATION, word, font=self._get_font(), fill=self._WORD_COLOR)
        image = np.array(image)
        return image

    def _noise(self, image, threshold=None):
        noise = np.random.randint(0, 256, image.shape[:2])
        t = np.random.randint(*self._NOISE_THRESHOLD_RANGE) if threshold is None else threshold
        m = noise > t
        # image[m] = 255 - image[m]
        image[m] = 144
        return image

    def _rotate(self, image):
        hw = np.array(image.shape[:2])
        m = cv2.getRotationMatrix2D(0.5*hw, np.random.choice(self._ROTATE_ANGLES), 1)
        if self._BG_COLOR == 'white':
            image = 255 - image
        image = cv2.warpAffine(image, m, hw)
        if self._BG_COLOR == 'white':
            image = 255 - image
        return image

    def _line(self, image):
        # for _ in range(np.random.randint(*self._LINE_COUNT_RANGE)):
        for _ in range(max(image.shape[:2])//64):
            p1, p2 = map(tuple, np.random.randint(0, image.shape[:2], (2,2)))
            thickness = np.random.randint(*self._LINE_THICKNESS_RANGE)
            image = cv2.line(image, p1, p2, int(np.random.choice(self._LINE_COLORS)), thickness=thickness)
        return image

    def _scale(self, image):
        hw = np.array(image.shape[:2])
        s = np.random.choice(self._SCALE_SIZE_RANGE)
        image = cv2.resize(image, list(map(int, s*hw)))
        return image

    def gen_sample(self):
        gen = {'single': self._gen_single, 'multiple': self._gen_multiple}[self._MODE]
        image, label = gen()
        return image, label

    def _augment(self, image, *args):
        for method in args:
            image = self._AUGMENT_METHODS[method](image)
        return image

    def _gen_single(self, blank=False, index=None, augment=True):
        if index is None: index = np.random.randint(len(self.Hanzi_LIST))
        word = self.Hanzi_LIST[index]
        if blank:
            index = 0
            word = ''
        image = self._get_text_image(word)
        if augment:
            image = self._augment(image, 'rotate', 'line', 'noise')
        label = {
            'word': word,
            'index': index
        }
        return image, label

    def _calc_location(self, i, j, size):
        try:
            yx_bias = np.random.randint(0, self._SIZE - size, (2,))
        except:
            yx_bias = 0
        if i*j and i != self._GRID[0]-1 and j != self._GRID[1]-1:
            yx_bias += np.random.randint(-self._BIAS, self._BIAS)
        location = np.array([i, j], dtype=int) * self._SIZE + yx_bias
        return location

    def _gen_multiple(self):
        image = self._BG_MULTI
        words = []
        for i in range(self._GRID[0]):
            for j in range(self._GRID[1]):
                blank = np.random.randint(2)
                if blank: continue
                # generate
                img, label = self._gen_single(blank=blank, augment=0)
                img = self._augment(img, 'rotate', 'line', 'scale')
                size = np.array(img.shape[:2])
                # location
                location = self._calc_location(i, j, size)
                # area
                area = image[location[0]:location[0]+size[0], location[1]:location[1]+size[1]]
                m = area > 128
                area[m] = img[m]
                image[location[0]:location[0] + size[0], location[1]:location[1] + size[1]] = area
                # label
                word = label['word']
                index = label['index']
                words.append({
                    'word': word,
                    'index': index,
                    'size': size,
                    'location': location
                })
        label = {
            'words': words
        }
        image = self._augment(image, 'noise', 'line')
        return image, label

    def label2cxywh(self, sample_size, label):
        sample_size = np.array(sample_size)
        words = label['words']
        def word2line(word):
            c = word['index']
            hw = np.array(word['size'])
            y, x = (hw / 2 + word['location']) / sample_size
            h, w = hw / sample_size
            # s = f'{c} {x:.4f} {y:.4f} {w:.4f} {h:.4f}\n'
            return c, x,y,w,h
        l = list(map(word2line, words))
        return l

    def cxywh2cxyxy(self, cxywh, sample_size):
        sample_size = np.array(sample_size)
        c, x, y, w, h = cxywh
        yx = sample_size * (y, x)
        hw = sample_size * (h, w)
        yx1 = yx - hw // 2
        yx2 = yx + hw // 2
        y1, x1, y2, x2 = map(int, (*yx1, *yx2))
        return c, x1, y1, x2, y2

    def label_visualize(self, image, label):
        img = Image.fromarray(image)
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        font_size = self._FONT_SIZE//3
        font = self._get_font('simkai.ttf', font_size)
        font_color = (255,0,0)
        box_color = 'red'
        # box_thickness = 3
        hw = image.shape[:2]
        l = self.label2cxywh(hw, label)
        cuts = []
        for i, box in enumerate(l):
            c, x1, y1, x2, y2 = self.cxywh2cxyxy(box, hw)
            # cv2.rectangle(img, (x1,y1), (x2,y2), (10,10,10), 3)
            word = self.Hanzi_LIST[c]
            draw.text((x1,max(0,y1-font_size)), word, font=font, fill=font_color)
            draw.rectangle((x1,y1,x2,y2), outline=box_color)
        return img

if __name__ == '__main__':
    g = Generator()

    # g.init(mode='single')
    # for i in range(100):
    #     img, label = g.gen_sample()
    #     img = Image.fromarray(img)
    #     word = label['word']
    #     img.save(f'test/{word}.jpg')

    g.config(mode='multiple', grid=[100,80])
    img, label = g.gen_sample()
    cv2.imwrite('../test/a.jpg', img)
    img = g.label_visualize(img, label)
    img.save('../test/b.jpg')



