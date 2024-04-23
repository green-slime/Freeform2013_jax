class cases_dict():
    def __init__(self, imgName, half_width, half_height, rx, ry):
        self.imgName = imgName
        self.half_width = half_width
        self.half_height = half_height
        self.rx = rx    
        self.ry = ry
    def get(self):
        return './image/'+self.imgName, self.half_width, self.half_height, self.rx, self.ry
        
einstein = cases_dict('einstein.jpg', 1, 168/128, 128, 168)
mao = cases_dict('mao.png', 886/442, 1, int(128*886/442), 128)
zju = cases_dict('zju.png', 5, 5, 128, 128)
blbl = cases_dict('blbl.jpg', 1, 1, 128, 128)
zju_inv = cases_dict('zju_inverse.png',1, 1, 128, 128)

