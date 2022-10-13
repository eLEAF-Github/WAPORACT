![alt text](https://github.com/eLEAF-Github/WAPORACT/blob/master/images/wapor_banner.png?raw=true)

# Water Productivity in practice: WAPORACT  
_Version 0.2_   
_2022/2/10_
_Authors: Roeland de Koning (Roeland.de.Koning@eleaf.com), Abdur Rahim Safi (Abdur.Rahim@eleaf.com), Annemarie Klaasse (annemarie.klaasse@eleaf.com)_

Increasing competition for and limited availability of water and land resources puts a serious constraint on agricultural production systems. Sustainable land and water management practices will be critical to expand production efficiently and address food insecurity while limiting the impact on the ecosystem. This requires a good understanding of how agricultural systems are performing, what are the underlying causes of spatio-temporal performance variations and the existing potential for improvement. Therefore, building satellite observations based tools to analyse and compare agriculture and irrigation performances is vital. 

This repo hosts tools to extract, interpret, analyse and visualize open-access geodata to improve water productivity. Currently, the focus is on data such as available in the FAO WaPOR database but also other open-access datasets will be considered in future. The tools in this repo can be used as building blocks to create customized services to target specific user needs.  

For further details see the [wiki](https://github.com/eLEAF-Github/WAPORACT/wiki) 

waporact videos will soo be available at: https://www.youtube.com/c/WaterPIPproject 


WARNING: the wiki is currently outdated since the 0.2.2 release of the package, we are aiming to have it updated by 26/8/2022

Views: [![HitCount](https://hits.dwyl.com/operations@eleafcom/https://githubcom/eLEAF-Github/WAPORACT.svg?style=flat-square)](http://hits.dwyl.com/operations@eleafcom/https://githubcom/eLEAF-Github/WAPORACT)

## Installation:

see: https://github.com/eLEAF-Github/WAPORACT/wiki/1.-Introduction-4.-WaporAct-Install-Instructions

an installation video will soon be available at: https://www.youtube.com/watch?v=SRl9dMZ6lbI 

## Release Notes

#### 0.2 2022/2/10

- first version of the WaPORAct package as part of the waterpip project. Includes: 
- basic framework for producing actionable data in a repeatable and clear manner based of the available WAPOR datasets using generic tools. 
- retreival functions
- vector and raster support functions
- statistics functions for analysis
- basic plotting functions for visualisation
- basic tutorials
- PAI pipeline example


#### 0.2 2022/2/15

- improved install instructions including a pdf copy, see wiki for detailed view 
- new install .txt file replacing .yml file creating an up to date and stable env
- added missing screenshots to the wiki

#### 0.2 2022/2/15

- improved wiki and install instructions clarity further and included exit python advice

#### 0.2 2022/2/18

- added running waporact instructions to the wiki and as pdf 

#### 0.2 2022/3/11

- added 01c step by step yield calculation tutorial to waporact basics tutorials
- bug fix: made axis argument for numpy multiple array calculation explicit
- all functions producing a file now return the output file name not just success 0
- in a function where a file is produced auto directory existance and build if missing added


#### 0.2.1 2022/8/15

- WaporRetrieval class rewrite to inherit from WaporStructure instead of WaporAPI (improves use and logic)
- WaporRetrieval class rewrite of download process to organise/ group them by year. limits downloads to a 
  max of 36 per group (36 dekads in a year) 
- plots script updated and completed, including improving all existing plots and adding static plots
- WaporPAI script and class updated to include plotting 
- WaporPAI script and class fixed temporal relative evapotranspiration function (was incorrect previously)
- mask_folder argument replaced with aoi_name (area of interest name) in all places as it is more legible/ user friendly
- Add option to set compression for output raster, default set to: LZW compression
- Add error message for maximum characters reached in automated folder path (255, due to windows)
- changed folder names of 01_download -> 01_temp,  02_processed -> 02_download (there was confusion amongst users , 
  names changed to make it more legible/user friendly)


#### 0.2.2 2022/8/19

  - made period_start and period_end mandatory arguments for Wapor_retrieval class when first inititated (feedback from users showed that the automated version led to confusion)
  - doc strings updated for all functions and classes across all scripts
  - rewrite of multiple scripts: specifically WaporRetrieval and WaporAPI for a more logical flow and interaction
  - debug of multiple errors across all scripts:
    - inf in array set to nan
    - gdal warp no output fixed by setting output geotransform
    - etc
  -  improved error messages in case there was an error during running, specifically retrieval/ wapor database error messages improved
  - Performance Assesment Indicator tutorial notebook released
  - 01B_basic_statistical_analysis renamed to 01B_basic_statistical_analysis_and_plotting to better reflect content
  - tutorial notebooks reorganised to better reflect their difficulty:
    - waporact\tutorials\01_waporact_basics\01A_downloading_from_wapor.ipynb
    - waporact\tutorials\01_waporact_basics\01B_basic_statistical_analysis_and_plotting.ipynb
    - waporact\tutorials\02_waporact_advanced\02A_step_by_step_yield_calculation.ipynb
    - waporact\tutorials\02_waporact_advanced\02B_waporact_calculating_PAIs.ipynb
  - updated waporact_install.txt made to reflect their new packages used by the scripts
  - environment.yml included for a more flexible build if the user wishes

#### 0.2.3 2022/8/25

- removal of install txt file and update of environment.yml install method
- install tutorial video: https://www.youtube.com/watch?v=SRl9dMZ6lbI
- running tutorial 1A video: https://www.youtube.com/watch?v=pRh1BG_PGjQ
- bug fix pandas to_csv index=false instead of read_csv index=false

## In Development 

- running tutorial 1B video
- running tutorial 2A video
- running tutorial 2B video
- update of wiki content to match the 0.2.3 waporact package 
- automated reporting functionality
- crop yield factor pipeline and notebook (possibility)


## Acknowledgement  
This repo was developed by [eLEAF](https://www.eleaf.com) under the [WaterPIP project](https://waterpip.un-ihe.org/welcome-waterpip). The Water Productivity Improvement in Practice (WaterPIP) project is supported by the Directorate-General for International Cooperation (DGIS) of the Ministry of Foreign Affairs of the Netherlands under the IHE Delft Partnership Programme for Water and Development (DUPC2). WaterPIP aims to guide countries and water projects in Water Productivity concepts to reach 25% WP improvement in the agricultural sector using WaPOR. It is lead by [IHE Delft](https://www.un-ihe.org/) in partnership with [Wageningen University and Research Center (WUR)](https://www.wur.nl/), [MetaMeta](https://metameta.nl), [eLEAF](https://www.eleaf.com) and [FAO](https://www.fao.org).

## Links  
- [WaterPIP project](https://waterpip.un-ihe.org/welcome-waterpip)
- [FAO WaPOR database](https://wapor.apps.fao.org/home/WAPOR_2/1)
- [Water accounting repository](https://github.com/wateraccounting/WAPORWP)
- [WaPOR v2 methodology document](http://www.fao.org/3/ca9894en/CA9894EN.pdf)
- [WaPOR application catalogue](http://www.fao.org/in-action/remote-sensing-for-water-productivity/use-casesresources/en/)
- [WaPOR master classes](https://thewaterchannel.tv/videos/june-10-2020-monitoring-water-productivity-using-wapor-part-1/)

