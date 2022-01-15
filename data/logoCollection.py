import time
import urllib.request
from selenium import webdriver

STORAGE_FILE = ''

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
dr = webdriver.Chrome('chromedriver', chrome_options=chrome_options)

def get_img(link, path, name):
  dr.get(link)
  time.sleep(1)
  scroll = 500
  dr.save_screenshot('a.png')
  for i in range(1, 200, 3):
    dr.execute_script("window.scrollTo(0, {});".format(scroll))
    scroll+=50
    try:
#      if name.split('_')[1] == '2020':
      try:
        element = dr.find_element_by_xpath('//*[@id="mw-content-text"]/div/table/tbody/tr[{}]/td[2]/div/div/a/img'.format(i))
      except:
        element = dr.find_element_by_xpath('//*[@id="mw-content-text"]/div/table[{}]/tbody/tr[1]/td[2]/div/div/a/img'.format(i))

    except: break
    img_link = element.get_attribute('src')
    urllib.request.urlretrieve(img_link, "{}/{}_{}.png".format(path, name, (i-1)/3))
  dr.get('https://logos.fandom.com/wiki/Logopedia:Recent_logos/Archive')

dr.get('https://logos.fandom.com/wiki/Logopedia:Recent_logos/Archive')
for x in range(1, 11):
  for y in range(1, 3):
    for z in range(1, 7):
      element = dr.find_element_by_xpath('//*[@id="mw-content-text"]/div/table[{}]/tbody/tr[{}]/th[{}]/a'.format(x, y, z))
      link = element.get_attribute('href')
#      print(link)
      get_img(link, STORAGE_FILE, link.split('/')[-1])
