import requests  # 数据请求模块
import parsel  # 数据解析模块

'''
url:链接,
title:标题,
type:租赁方式,
layout:户型,
rent_area:面积,
rent_price:租金,
distance:距最近地铁站距离,
dist:所在区,
bc:二级区（商圈）,
location:地址,
lng:经度,
lat:纬度
# '''
f = open('xablzufang.csv', 'a', encoding='utf-8-sig')
f.write('url,title,type,layout,rent_area,rent_price,distance,dist,bc,location,lng,lat\n')

headers = {

}


def spider(url):
    htmldata = requests.get(url=url, headers=headers).text
    # print(htmldata)
    select = parsel.Selector(htmldata)
    info = {}
    info['url'] = url
    info['title'] = (select.xpath('/html/body/div[3]/div[1]/div[3]/p/text()').get()).replace("\n", '').strip()  # 标题
    info['type'] = select.xpath('//*[@id="aside"]/ul/li[1]/text()').get()  # 租赁方式
    info['layout'] = select.xpath('//*[@id="aside"]/ul/li[2]/text()').get()[0:5]  # 户型
    info['rent_area'] = select.xpath('//*[@id="info"]/ul[1]/li[2]/text()').get()[3:-1]  # 面积
    info['rent_price'] = select.xpath('/html/body/div[3]/div[1]/div[3]/div[2]/div[2]/div[1]/span/text()').get()  # 租金
    info['distance'] = select.xpath('/html/body/div[3]/div[1]/div[4]/ul[2]/li[1]/span[2]/text()').get()  # 最近的地铁站距离
    dist = select.xpath('/html/body/div[3]/div[1]/div[10]/p[1]/a[2]/text()').get()[:-2]  # 所在区
    info['bc'] = select.xpath('/html/body/div[3]/div[1]/div[10]/p[1]/a[3]/text()').get()[:-2]  # 二级区域
    xq = select.xpath('/html/body/div[3]/div[1]/div[10]/h1/a/text()').get()[:-2]  # 小区名称
    location = '西安市' + dist + '区' + xq
    info['dist'] = dist
    info['xq'] = xq
    info['location'] = location
    # 调用百度api获得地址对应经纬度
    base = "http://api.map.baidu.com/geocoder?address=" + location + "&output=json&key=----------------"  # key在百度地图开放平台获取
    response = requests.get(base)
    answer = response.json()
    info['lng'] = answer['result']['location']['lng']
    info['lat'] = answer['result']['location']['lat']
    try:
        f.write(
            '{url},{title},{type},{layout},{rent_area},{rent_price},{distance},{dist},{bc},{location},{lng},{lat}\n'.format(
                **info))
    except:
        print("写入错误！")
    print('正在抓取：', info['url'], info['title'])


# 在一级页面获取链接
hreflist = []
for i in range(1, 101):
    ljurl = 'https://xa.lianjia.com/zufang/pg' + str(i) + 'rs%E7%A2%91%E6%9E%97%E5%8C%BA/?showMore=1'
    res = requests.get(url=ljurl, headers=headers)
    selector = parsel.Selector(res.text)
    href = selector.css('.content__list--item--title a::attr(href)').getall()
    hreflist.extend(href)
for index in hreflist:
    index = 'https://xa.lianjia.com' + index
    try:
        spider(index)
    except:
        print('链接访问不成功')
