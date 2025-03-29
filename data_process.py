import geopandas as gpd
import pandas as pd
shapefile = './res/grouped/forest_new.grouped_3.shp'
data = gpd.read_file(shapefile)

 ############################
# Index(['OBJECTID', '市町村名', '森林簿面積', '森林簿材積', '樹種名_1', '林齢_1', 'distance_y',
#        'RASTERVALU', 'cumsum', 'group', 'geometry'],
#       dtype='object')
# array(['三島市', '富士宮市', '富士市', '小山町', '御殿場市', '沼津市', '裾野市', '長泉町'],
#       dtype=object)



###查看单个市每个group的总材积和总面积
# mijima=data[data['市町村名']=='三島市']
# for i in mijima['group'].unique():
#     print(f"{i}area:{mijima[mijima['group']==i]['森林簿面積'].sum()}")
#     print(f"{i}volume:{mijima[mijima['group']==i]['森林簿材積'].sum()}")
###查看每个group有多少林龄
# for i in mijima['group'].unique():
#     print(f"{i}:{len(mijima[mijima['group']==i]['林齢_1'].unique())}")