from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.lib.function_base import angle

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

        # self._LINE_COLORS = [(100, 150, 200), (10, 70, 180), (200, 20, 160)]
        self._LINE_COLORS = [(0,0,0), (114,114,114), (200,0,0)]
        self._LINE_COUNT_RANGE = [1, 10]
        self._LINE_THICKNESS_RANGE = [1, 3]

        self._SCALE_SIZE_RANGE = [1, 0.8, 0.7, 0.6]

        self._AUGMENT_METHODS = {
            'noise': self._noise,
            'rotate': self._rotate,
            'line': self._line,
            'scale': self._scale,
        }

        self.config()

    def config(self, size=64, size_stamp=384, bg_color='white', word_color='black', mode='single', grid=[1, 1]):
        self._SIZE = size
        self._SIZE_STAMP = size_stamp
        self._BG_COLOR = bg_color
        self._WORD_COLOR = word_color
        self._MODE = mode
        self._GRID = grid # row, column

        # width, height
        self._BG = np.array(Image.new('RGB', (self._SIZE,self._SIZE), self._BG_COLOR))
        self._BG_MULTI = np.array(Image.new('RGB', (self._SIZE*self._GRID[1], self._SIZE*self._GRID[0]), self._BG_COLOR))
        self._BG_STAMP = np.array(Image.new('RGB', (self._SIZE_STAMP, self._SIZE_STAMP), self._BG_COLOR))

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
        image = np.array(image)[...,::-1]
        return image

    def _noise(self, image, threshold=None):
        noise = np.random.randint(0, 256, image.shape[:2])
        t = np.random.randint(*self._NOISE_THRESHOLD_RANGE) if threshold is None else threshold
        m = noise > t
        # image[m] = 255 - image[m]
        image[m] = 144
        return image

    def _rotate(self, image, rotate_angle=None):
        hw = np.array(image.shape[:2])
        if rotate_angle is None:
            rotate_angle = np.random.choice(self._ROTATE_ANGLES)
        m = cv2.getRotationMatrix2D(tuple(0.5*hw), rotate_angle, 1)
        if self._BG_COLOR == 'white':
            image = 255 - image
        image = cv2.warpAffine(image, m, tuple(hw))
        if self._BG_COLOR == 'white':
            image = 255 - image
        return image

    def _line(self, image):
        # for _ in range(np.random.randint(*self._LINE_COUNT_RANGE)):
        for _ in range(max(image.shape[:2])//64):
            p1, p2 = map(tuple, np.random.randint(0, image.shape[:2], (2,2)))
            thickness = np.random.randint(*self._LINE_THICKNESS_RANGE)
            line_color = self._LINE_COLORS[np.random.randint(len(self._LINE_COLORS))]
            image = image.astype(np.uint8)
            image = cv2.line(image, p1, p2, color=line_color, thickness=thickness)
        return image

    def _scale(self, image, s_factor=None):
        hw = np.array(image.shape[:2])
        if s_factor is None:
            s_factor = np.random.choice(self._SCALE_SIZE_RANGE)
        image = cv2.resize(image, tuple(list(map(int, tuple(s_factor*hw)))))
        return image

    def _augment(self, image, methods):
        for method in methods:
            image = self._AUGMENT_METHODS[method](image)
        return image

    def gen_sample(self):
        gen = {'single': self._gen_single, 'multiple': self._gen_multiple, 'stamp': self._gen_stamp}[self._MODE]
        image, label = gen()
        return image, label

    def _gen_single(self, blank=False, index=None, augment=True):
        if index is None: index = np.random.randint(len(self.Hanzi_LIST))
        word = self.Hanzi_LIST[index]
        if blank:
            index = 0
            word = ''
        image = self._get_text_image(word)
        if augment:
            image = self._augment(image, ['rotate', 'line', 'noise'])
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
 
    def _gen_multiple(self, rotate_anno=0):
        image = self._BG_MULTI.copy()
        words = []
        for i in range(self._GRID[0]):
            for j in range(self._GRID[1]):
                blank = np.random.randint(2)
                if blank: continue
                # generate
                img, label = self._gen_single(blank=blank, augment=0)
                img = self._augment(img, ['rotate', 'line', 'scale'])
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
        image = self._augment(image, ['noise', 'line'])
        return image, label
        
    def _gen_stamp(self):
        debug = 0
        upsidedown = 0 # np.random.randint(2)
        image = self._BG_STAMP.copy()
        words = []

        # 2 circles
        r = self._SIZE_STAMP//2
        center = np.array(image.shape[:2])//2
        image = cv2.circle(image, tuple(center), r, color=(0,0,255), thickness=2)
        image = cv2.circle(image, tuple(center), int(r*0.97), color=(0,0,255), thickness=1)
        sr = int(r*0.7)
        image = cv2.circle(image, tuple(center), sr, color=(0,0,255), thickness=2)
        cr = np.mean((sr,r))
        locs_up = []
        locs_down = []

        for i in range(10,1,-1):
            theta = np.pi*i/12
            d = np.cos(theta),-np.sin(theta)
            p = (np.array(d)*cr).astype(int)
            loc = center+p
            locs_up.append((theta, loc, 'up'))

            d = np.cos(theta),np.sin(theta)
            p = (np.array(d)*cr).astype(int)
            loc = center+p
            locs_down.append((theta+np.pi*upsidedown, loc, 'down'))
        locs = locs_up+locs_down
        
        # center words
        # center up
        self.CENTER_GRID_SIZE=32
        a = sr/np.sqrt(2)*2
        s = a//self.CENTER_GRID_SIZE
        s = int(s)
        for i in range(s//2):
            for j in range(s):
                loc = np.array((j,i))-s//2
                loc *= self.CENTER_GRID_SIZE
                loc += center-np.array((self.CENTER_GRID_SIZE,self.CENTER_GRID_SIZE))//4
                locs.append((np.pi/2, loc, 'center_up'))

        # center down
        for i in range(s//2+2, s):
            for j in range(s):
                loc = center+np.array((j,i))*self.CENTER_GRID_SIZE - np.array((2,2))*self.CENTER_GRID_SIZE
                locs.append((np.pi/2, loc, 'center_down'))
            break

        self.STAMP_WORD_SCALES={
            'up':0.7,
            'down':0.7,
            'center_up':0.5,
            'center_down':0.3,
        }
        # bias
        self.STAMP_WORD_BIASES={
            'up': np.array([0, 0]),
            'down': np.array([0, 0]),
            'center_up': np.random.randint(0,30,(2,)),
            'center_down': np.random.randint(-30,0,(2,)),
        }

        for i, (theta, loc, tag) in enumerate(locs):
            # bias
            bias = self.STAMP_WORD_BIASES[tag]
            # generate
            # blank = np.random.randint(2)
            blank = 0
            if 'center' not in tag:
                blank=0
                
            img, lb = self._gen_single(blank=blank, augment=0)
            img = self._augment(img, ['line'])
            img = self._scale(img, self.STAMP_WORD_SCALES[tag])
            hw = np.array(img.shape[:2])
            pts = np.array([[0,0],[hw[1],0],[hw[1],hw[0]],[0,hw[0]]])
            
            angle = theta/np.pi*180-90
            if tag=='down':
                angle *= -1
            if abs(angle)>1e-6:
                img = self._rotate(img, angle)
                m = cv2.getRotationMatrix2D(tuple(0.5*hw), angle, 1)
                pts = pts.reshape([-1,2])
                pts = np.hstack([pts, np.ones([len(pts), 1])]).T
                pts = np.dot(m,pts)
                pts = np.array(list(zip(pts[0],pts[1])))
            
            pts += loc-hw[::-1]//2 + bias
            size = np.array(img.shape[:2])
            # location
            location = (loc-size//2)[::-1] + bias[::-1]
            # area
            area = image[location[0]:location[0]+size[0], location[1]:location[1]+size[1]]
            m = area[...,0] > 128
            area[m] = img[m]
            image[location[0]:location[0] + size[0], location[1]:location[1] + size[1]] = area
            if debug:
                cv2.imwrite(f'../test/debug_{i}.jpg',image)
            # label
            if blank: continue
            word = lb['word']
            index = lb['index']
            words.append({
                'word': word,
                'index': index,
                'size': size,
                'points': pts,
                'angle': angle
            })
        

        label = {
            'words': words
        }
        image = self._augment(image, ['noise', 'line'])
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
        font_size = self._FONT_SIZE//3
        font = self._get_font('simkai.ttf', font_size)
        font_color = (255,0,0)
        box_color = 'red'
        if self._MODE in ['simple', 'multiple']:
            img = Image.fromarray(image[...,::-1])
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
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
        else:
            for i, anno in enumerate(label['words']):
                pts = anno['points']
                pts = pts.reshape((1, -1, 1, 2)).astype(int)
                color = [np.random.randint(255) for _ in range(3)]
                img = cv2.polylines(image, pts=pts, isClosed=1, color=color, thickness=1)

            img = Image.fromarray(image[...,::-1])
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            font_color = (0,0,0)
            for i, anno in enumerate(label['words']):
                x1, y1 = min(anno['points'][...,0]), min(anno['points'][...,1])
                word = anno['word']
                angle = anno['angle']
                txt = word
                if abs(angle)>1e-6:
                    txt+=f'_{angle:.0f}'
                draw.text((x1,max(0,y1-font_size)), txt, font=font, fill=font_color)

        return np.array(img)[...,::-1]

def ellipse(img, lb):
    if not isinstance(img, type(np.array([]))):
        img = np.array(img)
    hw = img.shape[:2]
    img = cv2.resize(img, (hw[1],int(hw[0]*0.5)))
    return img, lb

def label_cvt(lb):
    upwords = lb['words'][:9]
    downwords = lb['words'][9:9+9]
    center_up1 = lb['words'][18:18+7]
    center_up2 = lb['words'][18+7:18+7*2]
    center_up3 = lb['words'][18+7*2:18+7*3]
    center_down = lb['words'][-7:]
    lines = []
    for i, words in enumerate([upwords, downwords, center_up1, center_up2, center_up3, center_down]):
        # 1,2,3,4,5,6,7,8,9,10,11,...,txt,1,2,3,4,5,6,7,8,w1,...,1,2,3,4,5,6,7,8,wn
        line = []
        up_pts = []
        down_pts = []
        txt = ''
        for anno in words:
            word = anno['word']
            index = anno['index']
            size = anno['size']
            points = anno['points'].reshape((8,))
            angle = anno['angle']

            up_pts += list(points[:4])
            d = points[4:]
            down_pts += list(d[:2:-1])+list(d[2::-1])
            
            txt += word
            line += list(points)
            line.append(word)
        
        head_fmt = '%g,'*(len(up_pts)+len(down_pts))+'%s,'

        group_num = len(line)//9
        fmt = head_fmt+('%g,'*8+'%s,')*group_num
        line = up_pts+down_pts[::-1]+[txt]+line
        line = fmt[:-1] % tuple(line)
        lines.append(line)

    return '\n'.join(lines)

if __name__ == '__main__':
    g = Generator()

    # g.init(mode='single')
    # for i in range(100):
    #     img, label = g.gen_sample()
    #     img = Image.fromarray(img)
    #     word = label['word']
    #     img.save(f'test/{word}.jpg')

    # g.config(mode='multiple', grid=[8,32])
    # img, label = g.gen_sample()
    # cv2.imwrite('../test/words.jpg', img)
    # img = g.label_visualize(img, label)
    # img.save('../test/words_vis.jpg')

    g.config(mode='stamp', size_stamp=512, bg_color='white', word_color='red')
    for i in range(10):
        img, label = g.gen_sample()
        cv2.imwrite(f'../test/{i}.jpg', img)
        img_vis = g.label_visualize(img, label)
        cv2.imwrite(f'../test/{i}_vis.jpg', img_vis)
        label = label_cvt(label)
        with open(f'../test/{i}.jpg.txt', 'w') as f:
            print(label, file=f)


    # img_ellipse = ellipse(img).astype(np.uint8)
    # cv2.imwrite('../test/stamp_ellipse.jpg', img_ellipse)

    # img_ellipse = Image.fromarray(img_ellipse)
    # img_ellipse_vis = ellipse(img_vis)
    # img_ellipse_vis = Image.fromarray(img_ellipse_vis)
    # img_ellipse_vis.save('../test/stamp_ellipse_vis.jpg')