"""
waporact package

structure class (support class)

used to automatically organise the data when running the retrieval class or example pipeline in wapor_pai.py
"""
##########################
# import packages
from logging import exception
import os
from datetime import datetime, timedelta
from parse import parse
from timeit import default_timer


#################################
def check_windows_file_length(path: str):
    """
    Description:
        check the length of a file path to make sure it does not exceed the windows
        limit of 256 characters.

    Args:
        path: path to check

    Return:
        int: 0

    Raise:
        AttributeError: if the file path is too long
    """
    if len(path) > 255:
        raise AttributeError('length of the file path: {}, is too long for windows. please adjust your wapor directory folder location or make some other adjustment to reduce total path length')

    return 0

#################################
class WaporStructure(object):
    """
    Description:
        samll class used to structure the waporact projects in a
        standard way for future use called by
        all other class to tools file retrieval and storage
    
    Args:
        waporact_directory: location to store the projects
        project: name of the project folder
        wapor_level: wapor_level integer to download data for either 1,2, or 3
        period_start: datetime object specifying the start of the period
        period_end: datetime object specifying the end of the period
        return_period: return period code of the component to be downloaded (D (Dekadal) etc.)
        country_code: country code used when running for level 3 data

    Return:
        provides a project and date dict that can be used to retrieve and store
        wapor data from and in the project folders that have been setup
        in a standard way
    """
    # initiate class variables 

    # cube code template (datacomponent folder)
    cube_code_template = 'L{wapor_level}_{component}_{return_period}'
    
    #  input file name template
    input_filename_template = '{raster_id}'

    # output file name template
    output_filename_template = 'L{wapor_level}_{description}_{period_start_str}_{period_end_str}'

    def __init__(
        self,
        waporact_directory: str,
        wapor_level: int,
        country_code: str='notyetinitialised',
        project_name: str = 'test',
        return_period: str = 'D',
        period_end: datetime = datetime.now(),
        period_start: datetime = datetime.now() - timedelta(days=1),
    ):

        self.project_name = project_name
        self.waporact_directory = waporact_directory
        self.wapor_level = wapor_level
        self.period_start = period_start
        self.period_end = period_end
        self.return_period = return_period
        self.country_code = country_code

        # setup the project structure dict
        project = {}
        # setup metadata and temp dirs
        project['meta'] = os.path.join(self.waporact_directory, 'metadata')

        # setup the project dir
        project_dir = os.path.join(self.waporact_directory, self.project_name)
        
        data_dir = os.path.join(project_dir, 'L{}'.format(self.wapor_level))
        
        # setup sub dirs
        project['reference'] = os.path.join(data_dir, '00_reference')
        project['temp'] = os.path.join(data_dir, '01_temp')
        project['download'] = os.path.join(data_dir, '02_download')
        project['masked'] = os.path.join(data_dir, '03_masked')
        project['analysis'] = os.path.join(data_dir, '04_analysis')
        project['results'] = os.path.join(data_dir, '05_results')
        project['images'] = os.path.join(data_dir, '06_images')
        #project['reports'] = os.path.join(data_dir, '08_reports')

        # create the required dirs if not yet existing:
        for _dir in project.values():
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        # output the project dict to self
        self.project = project

    #################################
    # properties
    #################################
    @property
    def wapor_level(self):
        return self._wapor_level

    @wapor_level.setter
    def wapor_level(self,value: int):
        """
        Description
            checks if the wapor level is correct

        Args:
            value: wapor level to check

        Raise:
            AttributeError: If wapor level is incorrect
        """
        if value not in [1,2,3]:
            raise AttributeError("wapor_level (int) needs to be either 1, 2 or 3")
        else:
            self._wapor_level = value

    #################################
    @property
    def project_name(self):
        return self._project_name

    @project_name.setter
    def project_name(self,value: int):
        """
        Description
            sets the project name and checks it exists

        Args:
            value: project name

        Raise:
            AttributeError: If no name is provided or is not a string
        """
        if not isinstance(value, str):
            raise AttributeError("please provide a project name as str")
        else:
            if value=='test':
                print('using standard project name test')
            
            self._project_name = value

    #################################
    @property
    def waporact_directory(self):
        return self._waporact_directory

    @waporact_directory.setter
    def waporact_directory(self,value):
        """
        Description:
            checks for the existance of the waporact dir
            and creates it if it does not exist

        Args:
            value: path to the waporact directory

        Return:
            str: path to the waporact directory

        Raise: 
            AttributeError: if no value or not a string is provided
        """
        if not value:
            raise AttributeError('please provide a waporact_directory')
        if isinstance(value, str):
            if not os.path.exists(value):
                print('waporact_directory provided does not exist attempting to make it now')
                try:
                    os.mkdir(value)
                except Exception as e:
                    print('failed to make the waporact directory: {}'.format(value))
                    raise e

            self._waporact_directory = value

        else:
            raise AttributeError
    
    #################################
    @property
    def input_folder(self):
        return self._input_folder

    @input_folder.setter
    def input_folder(self,value):
        """
        Description:
            checks if the given input matches any of the folder keywords
            and if so provides back the path to the specified folder otherwise raising an error

        Args:
            value: folder keyword to check
        
        Return:
            str: path to the input folder matching the keyword
        """
        if value not in ['temp','download']:
            raise KeyError('input: input_folder must be one of [temp,download]')

        else:
            self._input_folder = self.project[value]

    #################################
    @property
    def output_folder(self):
        return self._output_folder

    @output_folder.setter
    def output_folder(self,value):
        """
        Description:
            checks if the given input matches any of the folder keywords
            and if so provides back the path to the specified folder otherwise raising an error

        Args:
            value: folder keyword to check
        
        Return:
            str: path to the output folder matching the keyword
        """
        if value not in ['masked','analysis','images', 'reference', 'results']:
            raise KeyError('input: output_folder must be one of [masked, analysis, results, images, reports,reference]')

        else:
            self._output_folder = self.project[value]

    #################################
    def generate_dates_dict(
        self,
        period_start: datetime=None,
        period_end: datetime=None,
        return_period: str=None):
        """
        Description:
            generates a dictionary of date related values used by wapor retrieval to retrieve data
            from WAPOR

            NOTE: if no values are provided it uses the class defaults
        
        Args:
            period_start: datetime object containg the start of the analysis period
            period_end: datetime object containg the end of the analysis period
            return_period: string code representing the analysis return period/interval

        Return:
            dict: dictionary of date related values
        """
        if not period_start:
            period_start = self.period_start
        if not period_end:
            period_end = self.period_end
        if not return_period:
            return_period = self.return_period

        assert self.return_period in ['D','A','I','M','S'] , "return period (str) needs to be either D, A, I, M, S"
        
        # setup date dict
        dates = {}

        # prepare the date info
        dates['return_period'] = return_period
        dates['period_end'] = period_end
        dates['period_start'] = period_start
        dates['per_end_str'] = period_end.strftime("%Y%m%d")
        dates['per_start_str'] = period_start.strftime("%Y%m%d")
        dates['per_str'] = f"{dates['per_start_str']}_{ dates['per_end_str']}"
        dates['api_per_end'] = period_end.strftime("%Y-%m-%d")
        dates['api_per_start'] = period_start.strftime("%Y-%m-%d")
        dates['api_period'] = '{},{}'.format(dates['api_per_start'], dates['api_per_end'])

        return dates

    #################################
    def generate_wapor_cube_code(
        self,
        component: str,
        return_period:str
        ):
        """
        format and return the cube code for querying wapor adding the region code for l3 if needed
        """
        if self.wapor_level == 3:
            component = '{}_{}'.format(self.country_code, component)
        cube_code=f"L{self.wapor_level}_{component}_{return_period}"

        return cube_code
  
    #################################
    def create_standardised_datacomponent_folder(
        self,
        component: str,
        return_period:str,
        waporact_folder: str
        ):
        """
        Description:
            create the path to a standardised folder for a wapor datacomponent for the user based on the inputs provided in the
            class and a few remaining arguments.

        Args:
            waporact_folder: keyword argument used to retrieve the path to a main waporact folder
            process_name: name of the process used in creating the file/datacomponent/result in the file
            return_period: return period of the wapor data being downloaded

        """
        cube_code = self.cube_code_template.format(
            wapor_level=self.wapor_level,
            component=component,
            return_period=return_period)

        folder_path = os.path.join(waporact_folder, cube_code)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        check_windows_file_length(path=folder_path)

        return folder_path

    #################################
    def generate_output_file_path(
        self,
        description: str,
        output_folder: str,
        aoi_name: str,
        ext: str,
        period_start: datetime=None,
        period_end: datetime=None):
        """
        Description:
            generate standardised file paths for the user based of the inputs provided in the
            class and a few remaining arguments.

        Args:
            output_folder: keyword argument used to retrieve the path to a main waporact folder
            description: file content one word description used in creating the file name and a subfolder
            period_start: period the analysis covers (start) (if none mainly for the mask it is set to na)
            period_end: period the analysis covers (end) (if none mainly for the mask it is set to na)
            aoi_name: area of interest (aoi) name to use for the mask folder auto set to nomask if not provided
            ext: ext in the file (with the .)
        
        Return:
            str: path to the new output file
        """
        # retrieve the output folder path
        self.output_folder = output_folder

        if period_start:
            period_start_str = period_start.strftime("%Y%m%d")
        else: 
            period_start_str = 'na'

        if period_end:
            period_end_str = period_end.strftime("%Y%m%d")
        else: 
            period_end_str = 'na'

        # check for _ in description and fix as needed for standardised file formating
        if '_' in description:
            description = description.replace('_', '-')
            print(' _ found in the description, replacing with - : {}'.format(description))

        # check for ' ' in description and fix as needed for standardised file formating
        if ' ' in description:
            description = description.replace(' ', '-')
            print('\' \' found in the description, replacing with - : {}'.format(description))

        # check for ' ' in mask_folder and fix as needed for standardised file formating
        if ' ' in aoi_name:
            aoi_name = aoi_name.replace(' ', '_')
            print('\' \' found in the aoi (mask) folder name, replacing with _ : {}'.format(aoi_name))

        output_filename = WaporStructure.output_filename_template.format(
            wapor_level=self.wapor_level,
            description=description,
            period_start_str=period_start_str,
            period_end_str=period_end_str)

        output_folder_path = os.path.join(self.output_folder, aoi_name)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        output_file_path = os.path.join(output_folder_path,output_filename + ext)

        check_windows_file_length(path=output_file_path)

        return output_file_path

    #################################
    @staticmethod
    def deconstruct_output_file_path(
        output_file_path:str
        ):
        """
        Description:
            deconstructs the standardised output file path using the known file construct
            extracting all parts of the name and storing in a dictionary, using the inputs for
            generate_output_file_path as the keys.

        Args:
            file_path: file path to deconstruct

        Return:
            dict: dict of deconstructed parts of the output_file_path
        """  
        #retrieve the file name
        file_name, ext = os.path.splitext(os.path.basename(output_file_path))

        # desconstruct the file name
        path_dict = WaporStructure.deconstruct_output_file_name(
            output_file_name=file_name)

        path_dict['ext'] = ext

        # retrieve the directory paths
        directory = os.path.dirname(output_file_path)
        path_dict['output_folder'], path_dict['mask_folder'] = os.path.split(directory)

        return path_dict

    #################################
    @staticmethod
    def deconstruct_output_file_name(
        output_file_name:str
        ):
        """
        Description:
            deconstructs the standardised output file name using the known file construct
            extracting all parts of the name and storing in a dictionary, using the inputs for
            generate_output_file_path as the keys.

        Args:
            output_file_name: file name to deconstruct

        Return:
            dict: dict of deconstructed parts of the output_file_name
        """
        file_dict = {}

        # desconstruct the file name
        results = parse(WaporStructure.output_filename_template, output_file_name).named
        
        file_dict['description'] = results.get('description')
        
        if results.get('period_start_str') != 'na':
            file_dict['period_end_str'] = 'na'
            file_dict['period_start_str'] = results.get('period_start_str')
            file_dict['period_start'] = datetime.strptime(results.get('period_start_str'),'%Y%m%d')
        else:
            file_dict['period_start'] = None

        if results.get('period_end_str') != 'na':
            file_dict['period_end_str'] = results.get('period_end_str')
            file_dict['period_end'] = datetime.strptime(results.get('period_end_str'),'%Y%m%d')
        else:
            file_dict['period_end_str'] = 'na'
            file_dict['period_end'] = None

        file_dict['wapor_level'] = results.get('wapor_level')

        return file_dict

    #################################
    def generate_input_file_path(
        self,
        component: str,
        raster_id: str,
        return_period: str,
        input_folder: str,
        ext: str):
        """
        Description:
            generate standardised file paths for input (downloaded) files retrieved from wapor for the user
            based of the inputs provided in the class and a few remaining arguments.

        Args:
            input_folder: keyword argument used to retrieve the path to a main waporact inputs folder
            return_period: return period of the wapor data being downloaded
            component: datacomponent being retrieved use din the folder name
            raster_id: wapor raster id
            ext: ext in the file (with the .)

        Return:
            str: path for the new input file
        """
        # retrieve the input folder path
        self.input_folder = input_folder

        # create the full folder path and folder as needed
        cube_code= WaporStructure.cube_code_template.format(
            wapor_level=self.wapor_level,
            component=component,
            return_period=return_period)

        input_folder_path = os.path.join(self.input_folder, cube_code)
        if not os.path.exists(input_folder_path):
            os.makedirs(input_folder_path)

        # create file name
        input_filename = self.input_filename_template.format(raster_id=raster_id)
        if self.wapor_level == 3:
            #remove country code from the name
            country_code_in_name = '_{}'.format(self.country_code)
            input_filename = input_filename.replace(country_code_in_name,'')

        input_file_path = os.path.join(input_folder_path, input_filename + ext)

        check_windows_file_length(path=input_file_path)

        return input_file_path

    #################################
    @staticmethod
    def deconstruct_input_file_path(
        input_file_path:str
        ):
        """
        Description:
            deconstructs the standardised input file path using the known file construct
            extracting all parts of the name and storing in a dictionary, using the inputs for
            generate_input_file_path as the keys.

        Args:
            file_path: file path to deconstruct

        Return:
            dict: dict of deconstructed parts of the input_file_path
        """  
        #retrieve the file name
        file_name, ext = os.path.splitext(os.path.basename(input_file_path))

        # deconstruct the file name
        path_dict = WaporStructure.deconstruct_input_file_name(
            input_file_name=file_name)

        path_dict['ext'] = ext

        # retrieve the directory paths
        directory = os.path.dirname(input_file_path)
        path_dict['input_folder'], path_dict['cube_code'] = os.path.split(directory)

        return path_dict

    #################################
    @staticmethod
    def deconstruct_input_file_name(
        input_file_name:str
        ):
        """
        Description:
            deconstructs the standardised input file name using the known file construct
            extracting all parts of the name and storing in a dictionary, using the inputs for
            generate_input_file_path as the keys.

        Args:
            output_file_name: file name to deconstruct

        Return:
            dict: dict of deconstructed parts of the input_file_name
        """
        file_dict = {}

        # desconstruct the file name
        results = parse(WaporStructure.input_filename_template, input_file_name).named
        
        file_dict['raster_id'] = results.get('raster_id')

        return file_dict

if __name__ == "__main__":
    start = default_timer()

