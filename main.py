##### OpenCV #####
from keras.models import Sequential
from keras.models import load_model
from sklearn.externals import joblib
import cv2
import numpy

model = Sequential()
model = load_model('Skin_CNN.h5', compile=False)

result_map = [
        "鱗狀細胞癌",
        "基底細胞癌",
        "脂漏性角化症",
        "皮膚纖維瘤",
        "黑素細胞痣",
        "血管病變",
        "黑色素瘤",
        "請輸入正確年齡數字"
]

def analysis(temp_part_path, age):
	#讀圖片
    img = cv2.imread(temp_part_path)
    img = splitimage(img)

	#resize版本
    img=img[...,::-1]
    resImg=cv2.resize(img,(28,28))
    resImg=resImg.reshape(1,28,28,3)
    #預測
    result = list()
    if int(age) < 0:
        result = [7,7,7,7]
        for i in range(len(result)):
            result[i] = result_map[result[i]]
        return result
        
    Probability=model.predict(resImg)
    Probability=Probability.tolist()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    Probability[0].append(int(age))
    Probability[0].append(model.predict_classes(resImg).tolist()[0])

    clftree = joblib.load('DecisionTreeClassifier.pkl')#決策樹
    result.append(clftree.predict(Probability)[0])

    clf = joblib.load('LinearSVC.pkl')#支援向量機
    result.append(clf.predict(Probability)[0])

    knn = joblib.load('KNeighbors.pkl')#k鄰近演算法
    result.append(knn.predict(Probability)[0])

    log = joblib.load('LogisticRegression.pkl')#羅吉斯
    result.append(log.predict(Probability)[0])

    print('result = ' + str(result))

    #index轉換成字串
    for i in range(len(result)):
        result[i] = result_map[result[i]]

    return result

##### Flask ####
import os
import time
from flask import Flask, request, render_template, Response, send_from_directory

app = Flask(__name__)


# 頁面宣告
@app.route('/<path:path>')
def send_html(path):
	return send_from_directory('static/', path)

@app.route('/js/<path:path>')
def send_js(path):
	return send_from_directory('static/js', path)

@app.route('/css/<path:path>')
def send_css(path):
	return send_from_directory('static/css', path)

@app.route('/img/<path:path>')
def send_img(path):
	return send_from_directory('static/img', path)

@app.route('/')
def root():
	return send_from_directory('static/', 'index.html')

# 分析頁面宣告
@app.route('/run', methods=['GET','POST'])
def run():
    # 取得年齡
    
    if request.method == 'POST':
        age = request.form["age"]
        print('age = ' + age)
    
        # 處理檔名
        temp_name = str(int(time.time())) + '.jpg'
        temp_part_path = 'static/temp/' + temp_name
        temp_full_path = os.getcwd() + '/' + temp_part_path
    
        # 處理上傳檔案
        request.files['file']
        file = request.files['file']
        if file:
    		 # 儲存檔案
             file.save(temp_full_path)
             rimg = cv2.imread(temp_part_path)
             testimg=cv2.dilate(cv2.erode(rimg, None, iterations=7), None, iterations=7)
             testimg=testimg.tolist()
             j_r=[]
             j_g=[]
             j_b=[]
             for k in range(len(testimg)):
                 for i in testimg[k]:
                     j_g.append(i[0])
                     j_b.append(i[1])
                     j_r.append(i[2])
             if(numpy.array(j_r).std() > 90.06 or numpy.array(j_r).std() < 1.47 or numpy.array(j_g).std() > 89.96 or numpy.array(j_g).std() < 5.88 or numpy.array(j_b).std() > 85.26 or numpy.array(j_b).std() < 5.13):
                result = analysis(temp_part_path, -1)
             else:
    		# 癌症分析
                result = analysis(temp_part_path, age)
    
    	# 宣告回覆 http body (html) 與 header
        resp = Response(render_template('run.html', age = age, result = result, temp_name = 'temp/' + temp_name))
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        resp.headers['Cache-Control'] = 'public, max-age=0'
        return resp
    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5000',debug=True)

def splitimage(rimg):#切割圖片黑色素部分
    gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)

    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)


    blurred = cv2.blur(gradient, (5, 5))
    (_, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
 
    closed = cv2.erode(closed, None, iterations=11)
    closed = cv2.dilate(closed, None, iterations=11)
    

    
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)==0:
        return rimg
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = numpy.int0(cv2.boxPoints(rect))
    

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = rimg[y1:y1+hight, x1:x1+width]

    return cropImg
