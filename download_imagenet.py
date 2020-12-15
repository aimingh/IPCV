from cv2 import cv2 as cv
import os, requests, wget

def store_raw_images():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04552348' # using Above URI
    res = requests.get(neg_images_link)
    neg_image_urls = res.content.decode("utf-8")
    for geturl in neg_image_urls.split('\r\n'):
        try:
            image_name = os.getcwd() + '/neg/' + geturl.split('/')[-1]
            retname = wget.download(url=geturl, out=image_name)
        except Exception as e:
            print(str(e))

if __name__ == "__main__":
    store_raw_images()