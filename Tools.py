import random
import sys

import cv2
import numpy as np
import math

from numpy.core._multiarray_umath import ndarray

# 灰度变换
class GrayTransform():
    def __init__(self, img):
        self.img = img
        self.imgHeight, self.imgWidth, _ = self.img.shape
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)  # get HSV from RGB
        self.GrayNum = self.img.max() + 1

    def LinearTransform(self, min, max):
        # img_new = np.zeros(self.img.shape, np.uint8)
        img_new = self.img.copy()
        a = self.img[:,:,2].min()
        b = self.img[:,:,2].max()
        for i in range(0, self.imgHeight):
            for j in range(0, self.imgWidth):
                value = (max - min) / (b.astype(np.float) - a.astype(np.float)) * (self.img[i][j][2].astype(np.float) - a.astype(np.float)) + min
                if value > 255:
                    value =255
                if value<0:
                    value = 0
                img_new[i][j][2] = value
        img_new = cv2.cvtColor(img_new, cv2.COLOR_HSV2BGR)
        return img_new

    def SegLinearTransform(self, s1, s2, t1, t2):
        img_new = self.img.copy()
        for i in range(0, self.imgHeight):
            for j in range(0, self.imgWidth):
                if self.img[i][j][2] >= 0 and self.img[i][j][2] < s1:
                    img_new[i][j][2] = self.img[i][j][2] * t1 / s1
                elif s1 <= self.img[i][j][2] and self.img[i][j][2] <= s2:
                    img_new[i][j][2] = (t2 - t1) / (s2 - s1) * (self.img[i][j][2] - s1) + t1
                elif s2 <= self.img[i][j][2] and self.img[i][j][2] <= 255:
                    N = 255
                    img_new[i][j][2] = (N - t2) / (N - s2) * (self.img[i][j][2] - s2) + t2
        img_new = cv2.cvtColor(img_new, cv2.COLOR_HSV2BGR)
        return img_new

    def LogTransform(self, c=255 / math.log(256)):
        img_new = self.img.copy()
        for i in range(0, self.imgHeight):
            for j in range(0, self.imgWidth):
                img_new[i][j][2] = c * math.log(self.img[i][j][2] + 1) + 0.5
        img_new = cv2.cvtColor(img_new, cv2.COLOR_HSV2BGR)
        return img_new

    def ExpTransform(self, a=5, b=0.8, c=1.05):
        img_new = self.img.copy()
        for i in range(0, self.imgHeight):
            for j in range(0, self.imgWidth):
                img_new[i][j][2] = math.pow(b, c * (self.img[i][j][2].astype(np.int) - a)) - 1
                # img_new[i][j] =a * math.pow(self.img[i][j],c)
        img_new = cv2.cvtColor(img_new, cv2.COLOR_HSV2BGR)
        return img_new

    def HistorgramEqual(self):
        img_new = self.img.copy()
        gray = np.zeros(256)
        fPro = np.zeros(256)
        gPro = np.zeros(self.GrayNum)
        f = np.zeros(self.GrayNum)

        for i in range(self.imgHeight):
            for j in range(self.imgWidth):
                gray[self.img[i][j][2]] += 1

        for i in range(self.GrayNum):
            fPro[i] = gray[i] / (self.imgWidth * self.imgHeight)

        for i in range(self.GrayNum):
            for j in range(i + 1):
                gPro[i] += fPro[j]

        for i in range(self.GrayNum):
            f[i] = i / (self.GrayNum - 1)

        for i in range(self.imgHeight):
            for j in range(self.imgWidth):
                dis = 0
                for m in range(self.GrayNum):
                    if gPro[self.img[i][j][2]] > f[m]:
                        dis = gPro[self.img[i][j][2]] - f[m]
                    else:
                        if dis <= f[m] - gPro[self.img[i][j][2]]:
                            index = m - 1
                        else:
                            index = m
                        break
                img_new[i][j][2] = index
        img_new = cv2.cvtColor(img_new, cv2.COLOR_HSV2BGR)
        return img_new

    def MiddleSmooth(self, size=3):
        img_new = self.img.copy()
        step = int(size / 2)
        sum = math.pow(size, 2)
        for i in range(self.imgHeight):
            for j in range(self.imgWidth):
                if i < step or i > (self.imgHeight - step - 1) or j < step or j > (self.imgWidth - step - 1):
                    img_new[i][j][2] = self.img[i][j][2]
                else:
                    pixel = []
                    pixel.append(self.img[i][j][2])
                    for m in range(1, step + 1):
                        pixel.append(self.img[i - m][j][2])  # top
                        pixel.append(self.img[i + m][j][2])  # bottom
                        pixel.append(self.img[i][j - m][2])  # left
                        pixel.append(self.img[i][j + m][2])  # right

                        pixel.append(self.img[i - m][j - m][2])  # up-left
                        pixel.append(self.img[i - m][j + m][2])  # up-right
                        pixel.append(self.img[i + m][j - m][2])  # bottom-left
                        pixel.append(self.img[i + m][j + m][2])  # bottom-right
                    pixel.sort()
                    img_new[i][j][2] = pixel[int(sum / 2)]
        img_new = cv2.cvtColor(img_new, cv2.COLOR_HSV2BGR)
        return img_new

    def AverageSmooth(self, size=3):
        img_new = self.img.copy()
        step = int(size / 2)
        sum = math.pow(size, 2)
        for i in range(self.imgHeight):
            for j in range(self.imgWidth):
                average = 0
                if i < step or i > (self.imgHeight - step - 1) or j < step or j > (self.imgWidth - step - 1):
                    img_new[i][j][2] = self.img[i][j][2]
                else:
                    average += self.img[i][j][2]
                    for m in range(1, step + 1):
                        average = average + self.img[i - m][j][2]  # up
                        average = average + self.img[i + m][j][2]  # bottom
                        average = average + self.img[i][j - m][2]  # left
                        average = average + self.img[i][j + m][2]  # right
                        average = average + self.img[i - m][j - m][2]  # up-left
                        average = average + self.img[i - m][j + m][2]  # up-right
                        average = average + self.img[i + m][j - m][2]  # bottom-left
                        average = average + self.img[i + m][j - m][2]  # bottom-right
                    img_new[i][j][2] = average / sum
        img_new = cv2.cvtColor(img_new, cv2.COLOR_HSV2BGR)
        return img_new

    def Sharpen(self, mode='g=G'):
        img_new = self.img.copy()
        G = np.zeros(self.img.shape, np.uint8)
        for i in range(self.imgWidth):
            for j in range(self.imgHeight):
                if i + 1 == self.imgWidth or j + 1 == self.imgHeight:
                    G[i][j][2] = self.img[i][j][2]
                    continue
                tmp = abs(self.img[i][j][2] - self.img[i + 1][j + 1]) + abs(self.img[i + 1][j] - self.img[i][j + 1])
                G[i][j][2] = tmp
        if mode == 'g=G':
            return G

    def RGB2HSI(rgb_img):
        """
        这是将RGB彩色图像转化为HSI图像的函数
        :param rgm_img: RGB彩色图像
        :return: HSI图像
        """
        # 保存原始图像的行列数
        row = np.shape(rgb_img)[0]
        col = np.shape(rgb_img)[1]
        # 对原始图像进行复制
        hsi_img = rgb_img.copy()
        # 对图像进行通道拆分
        B, G, R = cv2.split(rgb_img)
        # 把通道归一化到[0,1]
        [B, G, R] = [i / 255.0 for i in ([B, G, R])]
        H = np.zeros((row, col))  # 定义H通道
        I = (R + G + B) / 3.0  # 计算I通道
        S = np.zeros((row, col))  # 定义S通道
        for i in range(row):
            den = np.sqrt((R[i] - G[i]) ** 2 + (R[i] - B[i]) * (G[i] - B[i]))
            thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)  # 计算夹角
            h = np.zeros(col)  # 定义临时数组
            # den>0且G>=B的元素h赋值为thetha
            h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
            # den>0且G<=B的元素h赋值为thetha
            h[G[i] < B[i]] = 2 * np.pi - thetha[G[i] < B[i]]
            # den<0的元素h赋值为0
            h[den == 0] = 0
            H[i] = h / (2 * np.pi)  # 弧度化后赋值给H通道
        # 计算S通道
        for i in range(row):
            min = []
            # 找出每组RGB值的最小值
            for j in range(col):
                arr = [B[i][j], G[i][j], R[i][j]]
                min.append(np.min(arr))
            min = np.array(min)
            # 计算S通道
            S[i] = 1 - min * 3 / (R[i] + B[i] + G[i])
            # I为0的值直接赋值0
            S[i][R[i] + B[i] + G[i] == 0] = 0
        # 扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
        hsi_img[:, :, 0] = H * 255
        hsi_img[:, :, 1] = S * 255
        hsi_img[:, :, 2] = I * 255
        return hsi_img

    def HSI2RGB(hsi_img):
        """
        这是将HSI图像转化为RGB图像的函数
        :param hsi_img: HSI彩色图像
        :return: RGB图像
        """
        # 保存原始图像的行列数
        row = np.shape(hsi_img)[0]
        col = np.shape(hsi_img)[1]
        # 对原始图像进行复制
        rgb_img = hsi_img.copy()
        # 对图像进行通道拆分
        H, S, I = cv2.split(hsi_img)
        # 把通道归一化到[0,1]
        [H, S, I] = [i / 255.0 for i in ([H, S, I])]
        R, G, B = H, S, I
        for i in range(row):
            h = H[i] * 2 * np.pi
            # H大于等于0小于120度时
            a1 = h >= 0
            a2 = h < 2 * np.pi / 3
            a = a1 & a2  # 第一种情况的花式索引
            tmp = np.cos(np.pi / 3 - h)
            b = I[i] * (1 - S[i])
            r = I[i] * (1 + S[i] * np.cos(h) / tmp)
            g = 3 * I[i] - r - b
            B[i][a] = b[a]
            R[i][a] = r[a]
            G[i][a] = g[a]
            # H大于等于120度小于240度
            a1 = h >= 2 * np.pi / 3
            a2 = h < 4 * np.pi / 3
            a = a1 & a2  # 第二种情况的花式索引
            tmp = np.cos(np.pi - h)
            r = I[i] * (1 - S[i])
            g = I[i] * (1 + S[i] * np.cos(h - 2 * np.pi / 3) / tmp)
            b = 3 * I[i] - r - g
            R[i][a] = r[a]
            G[i][a] = g[a]
            B[i][a] = b[a]
            # H大于等于240度小于360度
            a1 = h >= 4 * np.pi / 3
            a2 = h < 2 * np.pi
            a = a1 & a2  # 第三种情况的花式索引
            tmp = np.cos(5 * np.pi / 3 - h)
            g = I[i] * (1 - S[i])
            b = I[i] * (1 + S[i] * np.cos(h - 4 * np.pi / 3) / tmp)
            r = 3 * I[i] - g - b
            B[i][a] = b[a]
            G[i][a] = g[a]
            R[i][a] = r[a]
        rgb_img[:, :, 0] = B * 255
        rgb_img[:, :, 1] = G * 255
        rgb_img[:, :, 2] = R * 255
        return rgb_img

# 几何变换
class Arithmetic:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def Add(self,x,y,weight):
        img_new = np.zeros(self.img1.shape, np.uint8)
        for i in range(self.img1.shape[0]):
            for j in range(self.img1.shape[1]):
                if j < x or j > self.img2.shape[1]+x or i < y or i > self.img2.shape[0]+y:
                    img_new[i][j] = self.img1[i][j]
                else:
                    img_new[i][j] = weight * self.img1[i][j].astype(np.float) + (1-weight)*self.img2[i-y-1][j-x-1].astype(np.float)
        return img_new

    def Minus(self,x,y,weight):
        img_new = np.zeros(self.img1.shape, np.uint8)
        for i in range(self.img1.shape[0]):
            for j in range(self.img1.shape[1]):
                if j < x or j > self.img2.shape[1] + x or i < y or i > self.img2.shape[0] + y:
                    print(i,j)
                    img_new[i][j] = self.img1[i][j]
                else:
                    a = self.img1[i][j].astype(np.float)/weight - (1-weight)/weight*self.img2[i-y-1][j-x-1].astype(np.float)
                    for ii in range(3):
                        if a[ii] > 255:
                            a[ii] = 255
                        if a[ii] < 0:
                            a[ii] = 0

                    img_new[i][j]  = a
        return img_new

    def CatHor(self):
        newWidth = self.img1.shape[1] + self.img1.shape[1]  # 1000
        newHeight = self.img1.shape[0] if self.img1.shape[0] > self.img2.shape[0] else self.img2.shape[0]  # 307
        shape = (newHeight, newWidth, self.img1.shape[-1])
        img_new = np.zeros(shape, np.uint8)
        for i in range(self.img1.shape[0]):
            for j in range(self.img1.shape[1]):
                img_new[i][j] = self.img1[i][j]
        for i in range(newHeight):
            for j in range(self.img1.shape[1], newWidth):
                img_new[i][j] = self.img2[i][j - self.img1.shape[1]]
        return img_new

    def CatVer(self):
        newHeight = self.img1.shape[0] + self.img2.shape[0]
        newWidth = self.img1.shape[1] if self.img1.shape[1] > self.img2.shape[1] else self.img2.shape[1]
        shape = (newHeight, newWidth, self.img1.shape[-1])
        img_new = np.zeros(shape, np.uint8)
        for i in range(self.img1.shape[0]):
            for j in range(self.img1.shape[1]):
                img_new[i][j] = self.img1[i][j]
        for i in range(self.img1.shape[0], newHeight):
            for j in range(self.img2.shape[1]):
                img_new[i][j] = self.img2[i - self.img1.shape[0]][j]
        return img_new

#　基本变换
class Transform():
    def __init__(self, img):
        self.img = img
        # self.img = cv2.imread(img)
        self.imgWidth = self.img.shape[1]
        self.imgHeight = self.img.shape[0]

    def VerInverse(self):
        img_new = np.zeros(self.img.shape, np.uint8)
        for i in range(self.imgHeight):
            for j in range(self.imgWidth):
                img_new[i][j] = self.img[self.imgHeight - i - 1][j]
                # x,y,_ = np.dot(Cm,[i,j,1])
                # img_new[self.imgWidth-1+x][y] = self.img[i][j]
        return img_new

    def HorInverse(self):
        img_new = np.zeros(self.img.shape, np.uint8)
        for i in range(self.imgHeight):
            for j in range(self.imgWidth):
                img_new[i][j] = self.img[i][self.imgWidth - j - 1]
        return img_new

    # def Rotate(self, theta):
    #     # # newSize: self.width*cos(theta)+self.height*sin(theta),self.width*sin(theta)+self.height*cos(theta)
    #     # angel = np.pi * theta/180
    #     # shape = (int(self.imgWidth*abs(np.cos(theta)) + self.imgHeight*abs(np.sin(theta))+1) ,
    #     #          int(self.imgWidth*abs(np.sin(theta)) + self.imgHeight*abs(np.cos(theta))+1),
    #     #          self.img.shape[2])
    #     # img_new = np.zeros(shape,np.uint8)
    #     # M = [[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]]
    #     # # M = np.array([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]])
    #     # with open('look.txt', 'w') as wf:
    #     #     for y in range(self.imgHeight):
    #     #         for x in range(self.imgWidth):
    #     #             # img_new[self.imgWidth - 1 - i][self.imgHeight - 1 - j] = self.img[i][j]
    #     #             # coodinate = np.array([x, y, 1])
    #     #             # coodinate = coodinate.dot(rotate_matrix)
    #     #             # xnew, ynew, _ = coodinate
    #     #             # xnew = int(xnew + self.imgHeight*abs(np.sin(theta)))
    #     #             # ynew = shape[1] -1 - int(ynew)
    #     #             # if ynew >= shape[0]:
    #     #             #     # or xnew < 0 or xnew > shape[1]-1:
    #     #             #     ynew = shape[0]-1
    #     #             # if xnew >= shape[1]:
    #     #             #     xnew = shape[1] - 1
    #     #             # img_new[ynew][xnew] = self.img[x][y]
    #     #
    #     #             x1, y1, _ = np.dot([x, self.imgHeight - 1 - y, 1], M)
    #     #             # x1,y1,_=np.array([x,self.imgHeight-1-y,1]).dot(M)
    #     #             xnew = x1 + self.imgHeight*abs(np.sin(theta))
    #     #             ynew = shape[1] -1 - y1
    #     #             img_new[int(ynew)][int(xnew)] = self.img[y][x]
    #     #
    #     #             # wf.write("{},{} => {},{} ==> {},{}\n".format(x,self.imgHeight-1-y,int(x1),int(y1),xnew,ynew))
    #     img_new = np.zeros(self.img.shape, dtype=np.uint8)
    #     dsize = self.img.shape
    #     def _rotate_coodinate(x, y, angel):
    #         angel = angel / 180 * math.pi
    #         coodinate = np.array([x, y, 1])
    #         rotate_matrix = np.array(
    #             [[math.cos(angel), -math.sin(angel), 0], [math.sin(angel), math.cos(angel), 0], [0, 0, 1]])
    #         rotate_center_first = np.array([[1, 0, 0], [0, -1, 0], [-0.5 * dsize[1], 0.5 * dsize[0], 1]])
    #         rotate_center_last = np.array([[1, 0, 0], [0, -1, 0], [0.5 * dsize[1], 0.5 * dsize[0], 1]])
    #         coodinate = coodinate.dot(rotate_center_first).dot(rotate_matrix).dot(rotate_center_last)
    #         x, y, _ = coodinate
    #         return int(x), int(y)
    #
    #     for row in range(self.imgHeight):  # 由 x',y' 求 xy
    #         for col in range(self.imgWidth):
    #             dst_x, dst_y = _rotate_coodinate(col, row, theta)
    #             if dst_x < 0 or dst_x >= self.imgWidth or dst_y < 0 or dst_y >= self.imgHeight:
    #                 pass
    #             else:
    #                 img_new[col][row] = self.img[dst_x][dst_y]
    #
    #     return img_new

    def Rotate(self, theta):
        angle = np.pi * theta / 180
        shape = (int(self.imgWidth * abs(np.sin(theta)) + self.imgHeight * abs(np.cos(theta)) + 1),
                 int(self.imgWidth * abs(np.cos(theta)) + self.imgHeight * abs(np.sin(theta)) + 1),
                 self.img.shape[2])
        img_new = np.zeros(shape, np.uint8)
        M = [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        if angle >= 0 and angle < 90:
            Offset = [[1, 0, 0], [0, 1, 0], [self.imgHeight * np.sin(theta), 0, 1]]
        elif angle >= 90 and angle < 180:
            Offset = [[1, 0, 0], [0, 1, 0], [shape[1], self.imgHeight * np.cos(theta), 1]]
        elif angle >= 180 and angle < 270:
            Offset = [[1, 0, 0], [0, 1, 0], [self.imgWidth * np.cos(theta), 0, 1]]
        elif angle >= 270 and angle <= 360:
            Offset = [[1, 0, 0], [0, 1, 0], [0, self.imgWidth * np.sin(theta), 1]]
        for i in range(shape[0]):
            for j in range(shape[1]):
                Pos = np.array([j, shape[0] - 1 - i, 1])
                x, y, _ = Pos.dot(M).dot(np.linalg.inv(Offset))  # 原图像中 （x,y）
                if (x < 0) or (x > self.imgWidth - 1) or (y < 0) or (y > self.imgHeight - 1):
                    continue  # 要计算的点不在源图范围内，直接返回255
                img_new[i][j] = self.img[int(x)][int(y)]
        return img_new

    def RotateBetter(self, angle):
        theta = np.pi * angle / 180
        shape = (int(self.imgWidth * abs(np.sin(theta)) + self.imgHeight * abs(np.cos(theta)) + 1),
                 int(self.imgWidth * abs(np.cos(theta)) + self.imgHeight * abs(np.sin(theta)) + 1),
                 self.img.shape[2])
        img_new = np.zeros(shape, np.uint8)
        M = [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        if angle >= 0 and angle < 90:
            Offset = [[1, 0, 0], [0, 1, 0], [-self.imgHeight * np.sin(theta), 0, 1]]
        elif angle >= 90 and angle < 180:
            Offset = [[1, 0, 0], [0, 1, 0], [-shape[1], self.imgHeight * np.cos(theta), 1]]
        elif angle >= 180 and angle < 270:
            Offset = [[1, 0, 0], [0, 1, 0], [self.imgWidth * np.cos(theta), 0, 1]]
        elif angle >= 270 and angle <= 360:
            Offset = [[1, 0, 0], [0, 1, 0], [0, self.imgWidth * np.sin(theta), 1]]
        for i in range(shape[0]):
            for j in range(shape[1]):
                Pos = np.array([j, shape[0] - 1 - i, 1])
                x, y, _ = Pos.dot(Offset).dot(np.linalg.inv(M))  # 原图像中 （x,y）
                if (x < 0) or (x > self.imgWidth - 1) or (y < 0) or (y > self.imgHeight - 1):
                    continue  # 要计算的点不在源图范围内，直接返回255
                img_new[i][j] = self.Interpolation(x, y)
        return img_new

    def Interpolation(self, x, y):
        lWidth = self.imgWidth
        lHeight = self.imgHeight
        # 计算四个最临近像素的坐标(i1, j1), (i2, j1), (i1, j2), (i2, j2)
        i1 = int(x)
        i2 = i1 + 1
        j1 = int(y)
        j2 = j1 + 1
        EXP = 0.0001
        if np.fabs(x - lWidth + 1) <= EXP:  # 右边界
            if np.fabs(y) <= EXP:
                return self.img[lHeight - 1][lWidth - 1]
            elif np.fabs(y - lHeight + 1) <= EXP:
                return self.img[0][lWidth - 1]
            else:
                f1 = self.img[lHeight - 1 - j1][lWidth - 1]
                f3 = self.img[lHeight - 1 - j2][lWidth - 1]
                return f1.astype(np.float) + (y - j1) * (f3.astype(np.float) - f1.astype(np.float))
        elif np.fabs(x) <= EXP:  # 左边界
            if np.fabs(y) <= EXP:
                return self.img[lHeight - 1][0]
            elif np.fabs(y - lHeight + 1) <= EXP:
                return self.img[0][0]
            else:
                f1 = self.img[lHeight - 1 - j1][0]
                f3 = self.img[lHeight - 1 - j2][0]
                return f1.astype(np.float) + (y - j1) * (f3.astype(np.float) - f1.astype(np.float))
        elif np.fabs(y - lHeight + 1) <= EXP:  # 下边界
            f1 = self.img[lHeight - 1][i1]
            f2 = self.img[lHeight - 1][i2]
            return f1.astype(np.float) + (x - i1) * (f2.astype(np.float) - f1.astype(np.float))
        elif np.fabs(y) <= EXP:  # 上边界
            f1 = self.img[0][i1]
            f2 = self.img[0][i2]
            return f1.astype(np.float) + (x - i1) * (f2.astype(np.float) - f1.astype(np.float))
        else:
            f1 = self.img[lHeight - 1 - j1][i1]
            f2 = self.img[lHeight - 1 - j2][i1]
            f3 = self.img[lHeight - 1 - j1][i2]
            f4 = self.img[lHeight - 1 - j2][i2]
            f12 = f1.astype(np.float) + (y - j1) * (f2.astype(np.float) - f1.astype(np.float))
            f34 = f3.astype(np.float) + (y - j1) * (f4.astype(np.float) - f3.astype(np.float))
            return f12.astype(np.float) + (x - i1) * (f34.astype(np.float) - f12.astype(np.float))

    def ZoomIn(self, rate_w,rate_h):
        newHeight = int(rate_h * self.imgHeight)
        newWidth = int(rate_w * self.imgWidth)
        channels = self.img.shape[-1]
        bilinear_img: ndarray = np.zeros(shape=(newHeight, newWidth, channels), dtype=np.uint8)
        for i in range(0, newHeight):
            for j in range(0, newWidth):
                row = (i / newHeight) * self.img.shape[0]
                col = (j / newWidth) * self.img.shape[1]
                row_int = int(row)
                col_int = int(col)
                u = row - row_int
                v = col - col_int
                if row_int == self.img.shape[0] - 1 or col_int == self.img.shape[1] - 1:
                    row_int -= 1
                    col_int -= 1
                bilinear_img[i][j] = (1 - u) * (1 - v) * self.img[row_int][col_int] + (1 - u) * v * self.img[row_int][
                    col_int + 1] + u * (1 - v) * self.img[row_int + 1][col_int] + u * v * self.img[row_int + 1][
                                         col_int + 1]
        return bilinear_img

    def ZoomOut(self, rate_w,rate_h):
        newHeight = int(rate_h * self.imgHeight)
        newWidth = int(rate_w * self.imgWidth)
        shape = (newHeight, newWidth, self.img.shape[-1])
        img_new = np.zeros(shape, np.uint8)
        for i in range(newHeight):
            for j in range(newWidth):
                x = j / rate_w  # 分别对x，y进行缩放
                y = (newHeight - 1 -i) / rate_h
                img_new[i][j] = self.Interpolation(x,y)
        return img_new

    def spNoise(self, prob):
        '''
        添加椒盐噪声
        prob:噪声比例
        '''
        output = np.zeros(self.img.shape, np.uint8)
        thres = 1 - prob
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = self.img[i][j]
        return output

    def gasussNoise(self, mean=0, var=0.001):
        '''
            添加高斯噪声
            mean : 均值
            var : 方差
        '''
        image = np.array(self.img / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        # cv.imshow("gasuss", out)
        return out

