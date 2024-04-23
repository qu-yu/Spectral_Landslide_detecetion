#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import os
import glob
from datetime import datetime
from tqdm import tqdm
from Proj import addproj_2shp


def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

    del dataset  # 关闭对象，文件dataset
    return im_proj, im_geotrans, im_data, im_width, im_height


def write_img(filename, im_proj, im_geotrans, im_data):

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype) #options=["INTERLEAVE=BAND"]

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

def Tif_Shp(result_path, save_path):
    """
    :param result_path: 预测结果添加投影坐标
    :param save_path:    保存路径
    :return:
    """
   # 这里就是你的批量栅格存储的文件夹。文件夹里最好除了你的目标栅格数据不要有其他文件了。
    os.chdir(result_path)  # 设置默认路径
    for raster in os.listdir():  # 遍历路径中每一个文件，如果存在gdal不能打开的文件类型，则后续代码可能会报错。
        print(raster)
        save_file = os.path.basename(raster)

        print("{} ，{} 正在读取影像".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), save_file))

        inraster = gdal.Open(raster)  # 读取路径中的栅格数据
        inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段，如果是单波段数据的话那就都是1
        prj = osr.SpatialReference()
        prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息，用来为后面生成的矢量做准备

        outshp = os.path.join(save_path, save_file[:-4] + '.shp')
        #outshp = raster[:-4] + ".shp"  # 给后面生成的矢量准备一个输出文件名，这里就是把原栅格的文件名后缀名改成shp了
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outshp):  # 若文件已经存在，则删除它继续重新做一遍
            driver.DeleteDataSource(outshp)
        Polygon = driver.CreateDataSource(outshp)  # 创建一个目标文件
        Poly_layer = Polygon.CreateLayer("Dust", srs=prj, geom_type=ogr.wkbMultiPolygon)  # 对shp文件创建一个图层，定义为多个面类
        newField = ogr.FieldDefn('value', ogr.OFTReal)  # 给目标shp文件添加一个字段，用来存储原始栅格的pixel value
        Poly_layer.CreateField(newField)

        gdal.FPolygonize(inband, None, Poly_layer, 0)  # 核心函数，执行的就是栅格转矢量操作
        Polygon.SyncToDisk()
        Polygon = None
        print("{} ，{} 读取影像完成".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), save_file))

def Delete_zero(path):
        # 注册所有驱动
        gdal.AllRegister()
        # 解决中文路径乱码问题
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
        driver = ogr.GetDriverByName("ESRI Shapefile")
        fileslist = glob.glob(os.path.join(path, '*.shp'))
        print(len(fileslist))
        for file in fileslist:
            print(file)
            datasource = driver.Open(file, 1)
            layer = datasource.GetLayer(0)
            defn = layer.GetLayerDefn()
            # 删除为0的要素
            strFilter = "value = '" + str(0) + "'"
            layer.SetAttributeFilter(strFilter)
            pFeatureDef = layer.GetLayerDefn()
            pLayerName = layer.GetName()
            pFieldName = "value"
            pFieldIndex = pFeatureDef.GetFieldIndex(pFieldName)
            #print(pFieldIndex)
            for pFeature in layer:
                pFeatureFID = pFeature.GetFID()
                layer.DeleteFeature(int(pFeatureFID))
            strSQL = "REPACK " + str(layer.GetName())
            datasource.ExecuteSQL(strSQL, None, "")
            layer = None
            datasource = None


if __name__ == '__main__':
    result_path = 'E:/landslide_depah/Landslide_Detection/image/'
    refpath = 'E:/landslide_depah/Landslide_Detection/result/'  # 原始影像的路径
    respath = 'E:/landslide_depah/Landslide_Detection/result_shp/'  # result的路径
    addproj_2shp(result_path,refpath)
    Tif_Shp(refpath, respath)
    Delete_zero(respath)




