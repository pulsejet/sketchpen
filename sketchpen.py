import numpy
from PIL import Image, ImageDraw
from tqdm import tqdm
import itertools
import turtle
from sklearn.cluster import KMeans
import random

def centercrop(im):
    width, height = im.size
    nd = min(width, height)
    left = (width - nd)/2
    top = (height - nd)/2
    right = (width + nd)/2
    bottom = (height + nd)/2

    return im.crop((left, top, right, bottom))

def getarray(file):
    pic = Image.open(file).convert('L')
    pic = centercrop(pic)
    return numpy.array(pic)

def putarray(arr, file):
    result = Image.fromarray(arr).convert('RGB')
    result.save(file)

class Edge:
    def __init__(self, x, y, col):
        self.xs = []
        self.ys = []
        self.col = []
        self.add(x, y, col)

    def add(self, x, y, col):
        self.xs.append(int(x))
        self.ys.append(int(y))
        self.col.append(int(col))

    def draw(self, d, w):
        for i in range(1, len(self.xs)):
            tpl = (self.xs[i-1], self.ys[i-1], self.xs[i], self.ys[i])
            fll = int(self.col[i])
            d.line(tpl, fill=fll, width=w)

THRESHOLD = 60
SKIP = 4

def gettrace(arr):
    edges = []

    print('Consturcting traces')
    for j in tqdm(range(2 * arr.shape[0])):
        curr = None
        edge = None
        cc = 0
        ptc = 0
        for i in range(j):
            x = i
            y = j - i
            if y >= arr.shape[0] or x >= arr.shape[1]:
                continue

            if edge is None:
                curr = arr[y][x]
                edge = Edge(0, j, curr)

            cavg = None
            if ptc > 0:
                cavg = cc // ptc

            if abs(int(arr[y][x]) - curr) > THRESHOLD or (cavg and abs(int(arr[y][x]) - cavg) > THRESHOLD):
                curr = arr[y][x]
                if ptc > 0:
                    curr = cc // ptc
                edge.add(x, y, curr)
                cc = 0
                ptc = 0
            else:
                cc += arr[y][x]
                ptc += 1

        if edge:
            x = min(j, arr.shape[0] - 1)
            y = abs(min((arr.shape[0] - 1) - j, 0))
            edge.add(x, y, arr[y][x])
            edges.append(edge)

    return edges

def turtledraw(edges, arr):
    wn = turtle.Screen()
    turtle.setup(arr.shape[1] * 1.1, arr.shape[0] * 1.1)
    wn.bgcolor("white")
    wn.title("Turtle")
    t = turtle.Turtle()
    t.pensize(SKIP)
    t.speed(0)

    def setcolor(c):
        c = c/255; c = (c,c,c)
        t.color(c, (1,0,0))

    setcolor(0)

    def gotok(edge, k):
        gotoxy(edge.xs[k], edge.ys[k])

    def gotoxy(x, y):
        t.goto(-arr.shape[1]//2 + x, arr.shape[0]//2 - y)

    traces = {}

    for edge in tqdm(itertools.islice(edges, None, None, SKIP)):
        for k in range(1, len(edge.col)):
            cc = edge.col[k] // 10
            if cc not in traces:
                traces[cc] = []
            e = [edge.xs[k-1], edge.ys[k-1], edge.xs[k], edge.ys[k], edge.col[k]]
            traces[cc].append(e)

    keys = list(traces.keys())
    keys.sort(reverse=True)

    traces_clustered = {}
    print('Performing K Means Clustering')
    for cc in tqdm(keys):
        X = numpy.array([[(e[0] + e[2])/2, (e[1] + e[2])/2] for e in traces[cc]])
        kmeans = KMeans(n_clusters=min(len(traces[cc]), 10), random_state=0).fit(X)
        labels = list(kmeans.labels_)

        traces_clustered[cc] = []
        lit = list(range(max(labels) + 1))
        random.shuffle(lit)
        for lbl in lit:
            traces_clustered[cc] += [traces[cc][i] for i, x in enumerate(labels) if x == lbl]

    print('Drawing')
    for cc in tqdm(keys):
        turtle.tracer(0, 0)
        i = 0
        for line in traces_clustered[cc]:
            t.up()
            gotoxy(line[0], line[1])
            t.down()
            setcolor(line[4])
            gotoxy(line[2], line[3])

            i += 1
            if i > 1:
                turtle.update()
                i = 0

    t.up()
    gotoxy(1000,1000)
    turtle.update()

    print('All Done!')

    t.getscreen().getcanvas().postscript(file='turtle.ps')
    print('Printed!')

    turtle.done()

if __name__ == '__main__':
    arr = getarray('test.png')
    print('Loaded image of size', arr.shape)
    trace = gettrace(arr)
    turtledraw(trace, arr)
