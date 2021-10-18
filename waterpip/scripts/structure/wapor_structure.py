"""
script for the structuring WAPOR projects per area in a standard way
"""

import os
from datetime import datetime, time, timedelta

class WaporStructure(object):
    """
    Description:
        samll class used to structure the waterpip projects in a 
        standard way for future use called by 
        all other class to support file retrieval and storage
    
    Args:
        waterpip_directory: location to store the projects
        project: name of the project folder

    Return:
        provides a project and date dict that can be used to retrieve and store
        wapor data from and in the project folders that have been setup 
        in a standard way
    """
    def __init__(
        self,
        waterpip_directory: str,
        wapor_level: int,
        project_name: str = 'test',
        return_period: str = 'D',
        period_end: datetime = datetime.now(),
        period_start: datetime = datetime.now() - timedelta(days=1),
    ):

        self.project_name = project_name
        self.waterpip_directory = waterpip_directory
        self.wapor_level = wapor_level
        self.period_start = period_start 
        self.period_end = period_end
        self.return_period = return_period

        #super().__init__(**kwargs)
        
        assert self.wapor_level in [1,2,3] , "wapor_level (int) needs to be either 1, 2 or 3"
        assert isinstance(self.project_name, str), 'please provide a project name'

        # setup the project structure dict
        project = {}
        # setup metadata and temp dirs
        project['meta'] = os.path.join(self.waterpip_directory, 'metadata')
        project['temp'] = os.path.join(self.waterpip_directory, 'temp')

        # setup the project dir
        project_dir = os.path.join(self.waterpip_directory, self.project_name)
        
        data_dir = os.path.join(project_dir, 'L{}'.format(self.wapor_level))
        
        # setup sub dirs
        project['download'] = os.path.join(data_dir, '01_download')
        project['processed'] = os.path.join(data_dir, '02_processed')
        project['analysis'] = os.path.join(data_dir, '03_analysis')
        project['results'] = os.path.join(data_dir, '04_results')
        project['reference'] = os.path.join(data_dir, '00_reference')
        project['images'] = os.path.join(project_dir, 'images')
        project['reports'] = os.path.join(project_dir, 'reports')

        # cereate the required dirs if not yet existing:
        for dir in project.values():
            if not os.path.exists(dir):
                os.makedirs(dir)

        # output the project dict to self
        self.project = project

        # setup date dict
        self.dates = self.generate_dates_dict()

    @property
    def waterpip_directory(self):
        return self._project_directory

    @waterpip_directory.setter
    def waterpip_directory(self,value):
        """
        Quick description:
            checks for the existance of the projects dir 
            and creates it if it does not exist
        """
        if not value:
            raise AttributeError('please provide a projects directory')
        if isinstance(value, str):
            if not os.path.exists(value):
                print('projects dir created does not exist attempting to make it now')
                os.mkdir(value)

            self._project_directory = value

        else:
            raise AttributeError

    def generate_dates_dict(
        self, 
        period_start: datetime=None, 
        period_end: datetime=None,
        return_period: datetime=None):
        """
        dates dict is updatable now
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

    def generate_file_path(self, datacomponent, folder: str = 'processed', return_period: str=None):
        """
        needs work
    
        """
    
        assert folder in ['download','processed','analysis','images','reports', 'reference'], 'input: folder must be one of [download, processed, analysis, images, reports,reference]' 

        component_dir = 'L{}_{}_{}'.format(self.wapor_level, datacomponent, return_period)

        folder = os.path.join(self.project['folder'],component_dir)

        return folder
