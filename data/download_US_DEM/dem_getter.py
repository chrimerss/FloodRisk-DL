
#%%
import numpy as np    
import requests
import sys
import os
from osgeo import gdal
import urllib
import pyproj #Used for transforming / reprojecting points
### GLOBAL VARIABLES ###

#DATASETS_DICT is a dictionary that maps a simple, shorthand name for a dataset hosted by TNM to its full API query name
#e.g., DATASETS_DICT[<shorthand>] = <full_query_name>

DATASETS_DICT = {
                 'DEM_1m': 'Digital Elevation Model (DEM) 1 meter',
                 'DEM_5m': 'Alaska IFSAR 5 meter DEM',
                 'NED_1-9as': 'National Elevation Dataset (NED) 1/9 arc-second',
                 'NED_1-3as': 'National Elevation Dataset (NED) 1/3 arc-second',
                 'NED_1as': 'National Elevation Dataset (NED) 1 arc-second',
                 'NED_2as': 'National Elevation Dataset (NED) Alaska 2 arc-second',
                 'LPC': 'Lidar Point Cloud (LPC)',
                 'OPR': 'Original Product Resolution (OPR) Digital Elevation Model (DEM)'
                 }

LIDARDATAYPES = {'LAS','LAZ', 'LAS,LAZ',''}

#Expected EPSG is the EPSG authority code for coordinates expected by the TNM query (currently WGS84 geographic coordinates)
EXPECTED_EPSG=4326

#QUAD_EPSG is the EPSG authority code for the coordinates returned for USGS 7.5'' quad boundaries
QUAD_EPSG=3857

#Path and default queries for API requests based on quad name
QUAD_URL_SMPL='https://index.nationalmap.gov/arcgis/rest/services/USTopoAvailability/MapServer/0/query?'
QUAD_QUERY_TEMPLATE = {'where':"CELL_NAME = '{quadname}' and PRIMARY_STATE='{statename}'",
                        'f':'json','geometryType':'esriGeometryEnvelope'}


#MAXITEMS is the maximum number of datasets the code will return
MAXITEMS = 500

#Path and default queries for elevation data served by The National Map
BASEURL = 'https://tnmaccess.nationalmap.gov/api/v1/products?'
BASEURL+'datasets={}&&bbox={},{},{},{}&max={}&prodFormats={}'
path = BASEURL+"polygon={}&datasets={}&max={}&prodFormats={}"
TNM_QUERY_TEMPLATE = {'datasets':'Digital Elevation Model (DEM) 1 meter','max':str(MAXITEMS),'prodFormats':''}



#%%
def reprojectXYPoints(xyPoints:list, inEPSG:int, outEPSG:int)->list:
    """Takes a list of x,y points and projects them from one coordinate reference system 
    to another based on EPSG authority code

        Args:
            xyPoints (list): List of x,y points [(x1,y1),(x2,y2),...] describing longitude, latitude or northing, easting values
            inEPSG (int): Source coordinate reference system
            outEPSG (int): Target coordinate reference system
        Returns:
            outPoints (tuple): The x,y vectors transformed into the target coordinate reference system
    """

    fromCRS = pyproj.CRS("EPSG:{}".format(inEPSG))
    toCRS = pyproj.CRS("EPSG:{}".format(outEPSG))

    transformer = pyproj.Transformer.from_crs(fromCRS,toCRS,always_xy = True)

    x_prime = [None for i in range(len(xyPoints))]
    y_prime = [None for i in range(len(xyPoints))]
    for i,pt in enumerate(xyPoints):
        x_prime[i],y_prime[i] = transformer.transform(pt[0],pt[1])

    return (x_prime,y_prime)


#%%
def _check_tnm_dataset_datatype_compatibility(dataset:str, dataType:str):
    """Checks the user-input dataset against the list of available datasets

    Args:
        dataset (str): Dataset name; must match key in DATASETS_DICT 
        dataType (str): Datatype to be queried

    Returns:
        dataset_fullname (str): The name of the dataset formatted to be passed to the National Map API

    Raises:    
        Exception: Input dataset is LPC and datatype is not LAS, LAZ, LAS,LAZ or left blank
        KeyError: Input dataset is not in DATASETS_DICT

    """

    if (dataset == 'LPC') and not(dataType in LIDARDATAYPES):
      raise Exception('Warning, {} is not available. Available datatypes for LPC are LAS, LAZ, or LAS,LAZ'.format(dataType))

    try:
        dataset_fullname = DATASETS_DICT[dataset]
    except:
        raise KeyError('Warning, {} is not available. Available datasets are: {}'.format(dataset,list(DATASETS_DICT.keys())))
     
    return(dataset_fullname)


#%%
def _execute_api_request(api_url:str,template_query_params:dict,
                    specific_query_params:dict)-> requests.Response:
    """This function executes a request using python's requests.get(api,params=query) method.

    template_query_params contains standard query params stored as a dictionary, specific_query_params are
    values of that dictionary that should be updated/added for this request 

    Args:
        api_url (str): Path to to api that we are querying
        template_query_params (dict): _description_
        specific_query_params (dict): _description_

    Raises:
        SystemExit: A non-success https code was recieved. The API may be invalid or temporarily down.
        SystemExit: A general error occured.

    Returns:
        requests.Response: The result of the call to request.get
    """
    
    #Copy the template query so as not to update it directly
    query = {k:template_query_params[k] for k in template_query_params}

    #Add any new specifics for thie query
    for key in specific_query_params:
        query[key] = specific_query_params[key]

    #Preset res to None to avoid not returning a result
    res = None

    try:
        res = requests.get(api_url,query)
        res.raise_for_status() #If something failed about the query, but the request went through
    except requests.exceptions.HTTPError as e:
        #404'd!!!
        raise e
    except requests.exceptions.RequestException as e:
        raise e
    
    '''
    Also seem to be able to get:
    {'error': {'code': 400,
    'message': 'Unable to complete operation.',
    'details': []}}
    '''

    if 'error' in res.json():
        raise Exception('The API request completed with an error, '+
                        ' there is likely a problem in the parameters of the request. url: {}'.format(res.url))
    
    #The requests succeeded, though might not have produced any results
    return res

#%%
def _execute_TNM_api_query(apiURL:str,templateQueryParams:dict,specificQueryParams:dict,
                            filePath:str, doExcludeRedundantData:bool):
    """Queries the National Map API and returns list of web-hosted datasets

    Args:
        api_url (str): The API URL
        templateQueryParams (dict): Generic parameters to pass to the query for all requests. 
        specificQueryParams (dict): Specific parameters to pass to the query for this request.
        filePath (str): Path to save output to
        doExcludeRedundantData (bool): When the retrieved data has the same spatial boundary, 
                this option downloads only the latest version

    Returns:
        aws_url (list): List of TNM download paths
    """

    r=_execute_api_request(apiURL,templateQueryParams,specificQueryParams)
    
    try:
        x=r.json()
        items=x['items'] #dicts of all the products
        total=x['total']
        aws_url = [item['downloadURL'] for item in items]

        if doExcludeRedundantData and len(aws_url)>0:
            split_urls=[]
            for lst in aws_url:
                split=urllib.parse.urlparse(lst).path.rsplit("_",1) #separate the base url (contains lat/long data) from the date
                split_urls.append(split)

            drop_items = [False for i in range(len(split_urls))]
            for i,item_i in enumerate(split_urls):
                if not(drop_items[i]):
                    for j,item_j in enumerate(split_urls):
                        if not(i==j):
                            if(item_i[0]== item_j[0]): #if the base url is the same, compare the dates and keep the most recent
                                if item_j[1]<item_i[1] or item_j[1]==item_i[1]:
                                    drop_items[j]=True
                                else:
                                    drop_items[i]=True

            aws_url=[aws_url[i] for i in range(len(aws_url)) if not drop_items[i]]

        if total > len(items):
            print('{} products are available; {} have been fetched.'.format(total, MAXITEMS))

        if len(aws_url)>0 and not(filePath is None) :
            try: 
            #If the query returned products AND filePath was specified, write to it
                with open(filePath,'a') as outputs_file:
                    for line in aws_url:
                        outputs_file.write(line + '\n')

            except: 
                savepath=os.path.join(filePath,"awsPaths.txt")
                with open(savepath,'a') as outputs_file:
                    for line in aws_url:
                        outputs_file.write(line + '\n')

        if not aws_url:
            print('No products available API request to: {}, with parameters: {}'.format(apiURL,specificQueryParams))
            aws_url=None

    except: #bad requests return none
        print('No products available API request to: {}, with parameters: {}'.format(apiURL,specificQueryParams))

        aws_url=None

    return aws_url #return output as a list



#%%
def get_aws_paths(dataset:str, xMin:float, yMin:float, xMax:float, yMax:float, filePath:str = None,
    dataType:str = '', inputEPSG:int=EXPECTED_EPSG, doExcludeRedundantData:bool=True):
    """Retrieves paths to geospatial products from TNM of a user-requested 
    dataset type, delineated by a bounding box of x,y coordinates

    Args:
        dataset (str):Dataset name; must match key in DATASETS_DICT
        xMin, yMin, xMax, yMax (int OR float): longitude/latitude or easting/northing values expressed in the
            coordinate system supplied as inputEPSG
        filePath (str, optional): Path to save output to. Defaults to None, in which case paths
            are only returned as a list
        dataType (str, optional): Format of to be queried. Defaults to ''
        inputEPSG (int, optional): The source EPSG authority code for the bounding box coordinates. 
            Defaults to EXPECTED_EPSG
        doExcludeRedundantData (bool, optional): When retrieved data has the same spatial boundary, this option downloads only the latest version.
            Defaults to True

    Returns:
        aws_urls: list of urls to download
    """
    
    #the way the url for OPR datasets is structured, can't use exclude_redundant
    if dataset=='OPR':
        doExcludeRedundantData=False

    datasetFullName = _check_tnm_dataset_datatype_compatibility(dataset, dataType)
 
    if inputEPSG != EXPECTED_EPSG:
        x=[xMin,xMax]
        y=[yMin,yMax]
        geom=list(zip(x,y))
        geom_proj=reprojectXYPoints(geom,inputEPSG,EXPECTED_EPSG)
        xMin=geom_proj[0][0]
        xMax=geom_proj[0][1]
        yMin=geom_proj[1][0]
        yMax=geom_proj[1][1]

    specificQueryParams = {'prodFormats':dataType,
                           'bbox':'{},{},{},{}'.format(xMin,yMin,xMax,yMax),
                           'datasets':datasetFullName
    }
    return _execute_TNM_api_query(BASEURL,TNM_QUERY_TEMPLATE,specificQueryParams, filePath, doExcludeRedundantData)

#%%   
def get_aws_paths_polygon(dataset, x, y, filePath = None, dataType = '', inputEPSG=EXPECTED_EPSG, doExcludeRedundantData=True):
    """Retrieves paths to geospatial products from TNM of a user-requested 
    dataset type, delineated by a polygon defined by a set of x,y coordinates. Polygons are limited 
    to 11 vertices. If a more complex polygon is provided, it will be generalized to 11 vertices in a very simplified manner.

    Args:
        dataset (str): Dataset name; must match key in DATASETS_DICT
        x (list): List of polygon longitude values expressed in decimal degrees
        y (list): List of polygon latitude values expressed in decimal degrees
        filePath (str, optional): Path to save output to. Defaults to None, in which case paths
            are only returned as a list.
        dataType (str, optional): Format for data to be queried. Defaults to ''.
        inputEPSG (int, optional): The source EPSG authority code for the polygon coordinates. Defaults to EXPECTED_EPSG.
        doExcludeRedundantData (bool, optional): When retrieved data has the same spatial boundary, 
            this option downloads only the latest version. Defaults to True.

    Returns:
        aws_urls: list of urls to download
    """

    #the way the url for OPR datasets is structured, can't use exclude_redundant
    if dataset=='OPR':
        doExcludeRedundantData=False

    datasetFullName= _check_tnm_dataset_datatype_compatibility(dataset, dataType)

    if inputEPSG != EXPECTED_EPSG:
        geom=list(zip(x,y))
        geom_proj=reprojectXYPoints(geom,inputEPSG,EXPECTED_EPSG)
        x=geom_proj[0]
        y=geom_proj[1]
    
    #If there are too many vertices, trim out evenly spaced vertices
    n_max_items=11
    if len(x) > n_max_items:
        new_indices = np.linspace(0,len(x)-1,n_max_items,dtype = int)
        x=[x[idx] for idx in new_indices]
        y=[y[idx] for idx in new_indices]

    xy_entries = ['{} {},'.format(x_i,y_i) for x_i,y_i in zip(x,y)]
    polygonString = ' '.join(xy_entries)[:-1] #take away the last comma

    specificQueryParams = {'prodFormats':dataType,
                            'polygon':polygonString,
                            'datasets':datasetFullName
        }
    return _execute_TNM_api_query(BASEURL,TNM_QUERY_TEMPLATE,specificQueryParams, filePath, doExcludeRedundantData)

#%%
def get_aws_paths_from_geodataframe(dataset, gdf, rowIdx=0, filePath = None, dataType = '', doExcludeRedundantData=True):
    """Retrieves paths to geospatial products from TNM of a user-requested 
    dataset type, delineated by a single feature from a pandas geodataframe 
    identified by its row index number

    Args:
        dataset (str): Dataset name; must match key in DATASETS_DICT
        gdf (GeoDataFrame): Geodataframe with geometry formatted as polygons
        rowIdx (int, optional): Row index of the desired polygon within the geodataframe. Defaults to 0.
        filePath (str, optional): Path to save output to. Defaults to None, in which case paths
            are only returned as a list
        dataType (str, optional): Format for data to be queried. Defaults to ''.
        doExcludeRedundantData (bool, optional): When retrieved data has the same spatial boundary, 
            this option downloads only the latest version. Defaults to True.

    Returns:
       aws_urls: list of urls to download
    """    
    
    #project polygon to correct CRS and get coords for the geometry
    poly_proj = gdf.to_crs(epsg=EXPECTED_EPSG)
    geom=poly_proj['geometry'][rowIdx]
    x=[]
    y=[]
    for idx in geom.exterior.coords:
        x.append(idx[0])
        y.append(idx[1])

    return get_aws_paths_polygon(dataset,x,y, filePath, dataType,EXPECTED_EPSG, doExcludeRedundantData)


#%%
def _get_24kQuad_geom(quadName:str,stateName:str,EPSG:int = None)->tuple:
    """Get the spatial boundary of the specified quad

    Args:
        quadName (str): Name of the quad, case sensitive
        stateName (str): State where the quad is located, case sensitive
        EPSG (int): EPSG used to project the spatial boundary. Defaults to None, in which case
        no projection is done and native geometry of 24 K quad feature service is used (3857)

    Returns:
       geom: tuple of x,y coordinates describing geometry of quad coordinates ([x1,x2, x3,....],[y1, y2, y3,...]). 
    """

    name=quadName #quad name not titlized, needs to be spelled correctly
    state=stateName.title()

    specificQueryParams = {'where':QUAD_QUERY_TEMPLATE['where'].format(quadname=name,statename=state)}
    r=_execute_api_request(QUAD_URL_SMPL,QUAD_QUERY_TEMPLATE,specificQueryParams)
    b=r.json()
    try:
        feat=b['features'][0]
    except:
        return print('Warning: Quad name {} in {} is not available. Check spelling or try another name.'.format(name, state))

    geom = None
    if feat:
        geom=feat['geometry']['rings'][0]

        if EPSG:
            geom=reprojectXYPoints(geom,QUAD_EPSG,EPSG)
        else:
            #During reprojection we transform shape of geometry array for other more common operations, match that here if reprojection didn't occur
            geom = ([pt[0] for pt in geom],[pt[1] for pt in geom])

    return geom
#%%
def get_aws_paths_from_24kQuadName(dataset, quadName, stateName, filePath = None, dataType = '', doExcludeRedundantData=True):
    """Retrieves paths to geospatial products from TNM within the bounds of a USGS 7.5'' quadrangle map

        Args:
            dataset (str): Dataset name; must match key in DATASETS_DICT
            quadName (str): Name of a valid USGS 7.5'' quad map. More information at:
                https://www.usgs.gov/faqs/where-can-i-find-indexes-usgs-topographic-maps
            filePath (str, optional): Path to save output to. Defaults to None, in which case paths
            are only returned as a list.
            dataType (str, optional): Format for data to be queried. Defaults to ''
            doExcludeRedundantData (bool, optional): When retrieved data has the same spatial boundary, 
                this option downloads only the latest version. Defaults to True

        Returns:
            aws_urls: list of urls to download

        Raises:
            Ecxeption: Invalid quad, state, or quad/state combination 
    """

    geom = _get_24kQuad_geom(quadName,stateName,EXPECTED_EPSG)

    return get_aws_paths_polygon(dataset,geom[0],geom[1],filePath, dataType,EXPECTED_EPSG, doExcludeRedundantData)


#%%
def batch_download(dlList, folderName, doForceDownload = False):
    """Download from TNM the data from a list of paths defined in dlList and save to input path

    Args:
        dlList (list):  List of web-hosted datasets (e.g., obtained from one of the get_aws_paths_... functions)
        folderName (str): Name of folder to save downloads in; Defaults to in the current working directory. 
            Directory will be made if it doesn't already exist
        doForceDownload (bool, optional): Should the download commence no matter how large (default is False). If this is false,
            the code queries the user for input. If true, the download proceeds without prompting the
            user. Defaults to False.

    Returns:
        fileList: List of the directory paths to the downloaded data
    """

    size=0
    for lst in dlList:
        req=urllib.request.Request(lst, method='HEAD')
        f = urllib.request.urlopen(req)
        size+=int(f.headers['Content-Length'])

    #Make the size of the download nicely legible
    sizeConvs = ['B','kB','MB','GB','TB'] #list of conversions
    pos = 0
    while (size > 1e3) & (pos < (len(sizeConvs)-1)):
        size/=1e3
        pos+=1

    warning_string = 'WARNING: you are attempting to download {:.2f} {} of data. Continue? y/n'.format(size, sizeConvs[pos])

    if doForceDownload:
        answer = 'y'
    else:
        answer=input(warning_string)
    
    if (str(answer) != 'y'):
         sys.exit('Aborting downloads based on input: {}'.format(answer))
    
    dir = os.getcwd()
    savePath = os.path.join(dir,folderName)
    if not(os.path.isdir(savePath)):
        os.mkdir(savePath)
    fileList=[]
    for line in dlList:
        #Strip off any whitespace
        #Get the 'name' of this file as the end of the filepath, we'll use this to save the file
        line = line.strip()
        name = line.split('/')[-1]
        
        #Keep track of progress
        print('Working on: {}'.format(name))
        downloadPath = os.path.join(savePath,name)
        urllib.request.urlretrieve(line,downloadPath)
        fileList.append(downloadPath)
    return fileList

#%%
def merge_warp_dems(inFileNames, outFileName, outExtent = None, outEPSG = None, pixelSize=None, doReturnGdalSourceResult = False,
                    resampleAlg = 'cubic', noDataValue = None, format = 'GTiff'):
    """Wrapper for gdal.Warp, an image mosaicing, reprojection and cropping function

    Args:
        inFileNames (list): A list of all the filenames to merge
        outFileName (str): the output path to save the file as
        outExtent (list OR tuple, optional): ([minx, maxx], [miny, maxy]). Defaults to None.
        outEPSG (int, optional): EPSG code for the coordinate system of the specified output extent (also sets the output
            coordinate system). Defaults to None.
        pixelSize (float, optional):  Dimension of the output pixels (x and y direction) in the native units of the
            output coordinate system. Defaults to None.
        doReturnGdalSourceResult (bool, optional): If True returns the gdal source object for the newly created dataset. 
            If False (the default) returns none and closes the connection to the newly created dataset. Defaults to False.
        resampleAlg (str, optional): The resampling algorithm to use in reprojecting and merging the raster. Can be
            any option allowed by GDAL. Prefered options will likely be: 'near', 'bilinear', 'cubic', 'cubicspline',
            'average'. Defaults to 'cubic'.
        noDataValue (float, optional): No data value to use for the input and output data. Defaults to None.
        format (str, optional): File format to save the output dataset as. Defaults to 'GTiff'.

    Returns:
        gridSource (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the warp.
    """

    #In some ArcPro created virtual environments the path to the PROJ library (opensource projections) is not always created on startup
    #This will cause a gdal error when trying to transform raster bounding box coordinates to a new CRS.
    #This statement attempts to guess what the path should be, though there is no gurantee this will work w/ all environments
    if not('PROJ_LIB' in os.environ):
        env_path = os.environ['PATH'].split(';')[1] #The second item in the path in arcpro environments is <directory>\\environment\\Libray\\bin
        env_path = os.path.abspath(os.path.join(env_path ,os.pardir)) #Get the path on directory up (to library)
        proj_path = os.path.join(env_path,'share','proj')
        os.environ['PROJ_LIB'] = proj_path

    if not(outExtent is None):
        outExtent = [outExtent[0][0], outExtent[1][0], outExtent[0][1], outExtent[1][1]]


        #If an output coordinate system was specified, format it for gdal
    if not(outEPSG is None):
        outEPSG = 'EPSG:{}'.format(outEPSG)
        #If an output bounding box was specified, format it for gdal. Leave as none if there won't be any clipping


    wrpOptions = gdal.WarpOptions(
        outputBounds=outExtent,
        outputBoundsSRS=outEPSG,
        format=format,
        xRes=pixelSize, yRes=pixelSize,
        resampleAlg=resampleAlg,
        dstSRS=outEPSG,
        dstNodata=noDataValue,
        srcNodata=noDataValue
    )
    gridSource = gdal.Warp(outFileName,inFileNames, options=wrpOptions)

   
    if not(doReturnGdalSourceResult):
        gridSource = None

    return gridSource

#%%

def compute_derivatives(rasterPath, attributeNames, filePath, zFactor = None):
    """Calculates derivates of an input raster and saves the results with the name of the derivative calculated
    as a suffix.

    Args:
        rasterPath (str): The path to a raster dataset
        attributeNames (list OR str): Either a list of attribute names or a single attribute name.
        filePath (str): Path to save output to
        zFactor (float, optional): A number that represents the conversion necessary from the vertical units of the raster
            to the horizontal units defined by the projection. Defaults to None, in which case a z_factor will be estimated
            if necessary.

    Returns:
        fileList (list): List of path(s) to the generated derivate files.
    """

    #is executing these things...
    from .derivative_defaults import derivative_functions,derivative_params
    from .dem_derivatives import estimate_z_factor

    #Open the raster dataset
    raster =  gdal.Open(rasterPath)

    #Get the z_factor of this dataset (this will be 1 for projected data)
    if zFactor is None:
        zFactor = estimate_z_factor(raster)

    #get raster dataset name
    rasName=raster.GetDescription()
    sourceName,sourceExt=os.path.splitext(os.path.split(rasName)[-1])

    fileList=[]

    #Transform attribute names to a list if it isn't already
    if not isinstance(attributeNames,list):
        attributeNames = [attributeNames]

    for name in attributeNames:
        
        thisDict = derivative_params[name].copy()
        savePath=os.path.join(filePath,sourceName +'_{}'.format(name) + sourceExt)
        fileList.append(savePath)

        thisFunction = derivative_functions[name]
        if 'zFactor' in thisDict:
            thisDict['zFactor'] = zFactor

        thisFunction(savePath,rasterPath,**thisDict)

    #Close the gdal dataset to 'disconnect' it from this python session
    raster = None

    return(fileList)

if __name__=='__main':
    pass
