# Water Productivity in practice - tools  
_Version 1.0_  
_2 October 2021_   
_Authors: Roeland de Koning (Roeland.de.Koning@eleaf.com), Abdur Rahim Safi (Abdur.Rahim@eleaf.com), Annemarie Klaasse (annemarie.klaasse@eleaf.com)_

## Introduction  

Increasing competition for and limited availability of water and land resources puts a serious constraint on agricultural production systems. Sustainable land and water management practices will be critical to expand production efficiently and address food insecurity while limiting the impact on the ecosystem. This requires a good understanding of how agricultural systems are performing, what are the underlying causes of spatio-temporal performance variations and the existing potential for improvement. Therefore, building satellite observations based tools to analyse and compare agriculture and irrigation performances is vital. This repo hosts tools to extract, interpret, analyse and visualize open-access geodata to improve water productivity. Currently, the focus is on data such as available in the FAO WaPOR database but also other open-access datasets will be considered in future.

## Toolset 

The developed tools are based on satellite observations of the actual crop production and water consumption from the WaPOR database. Three types of tools are introduced here: 

- Descriptive Statistics (DS) 
- Performance Assessment Indicators (PAI)
- Crop Yield Factors (YF)

PAIs are used to understand how agricultural systems are performing and their potential for improvement. YF helps identify the underlying causes of spatio-temporal variations in crop yield and growth. DS quantitatively describe WaPOR data and provide meaningful information to the users. The tools focus on the actual performance of the agriculture system and the underlying biophysical factors, but as a satellite-based system, it cannot provide information on underlying socio-ecological variables. 

NOTE: For users that have not used Github repositories and Conda/Python packages before: [Getting started](GettingStarted)  


**Download**  

**Descriptive Statistics**  
The purpose of descriptive statistics is to quantitatively describe WaPOR data and provide meaningful information to the users. For detailed information on descriptive statistics [click here](Basic Statistics)

**Performance Assessment Indicators**  

Performance Assessment Indicators assess the performance of an agricultural system in terms of equity, adequacy, reliability, efficiency and productivity. For detailed information on Performance Assessment Indicators [click here](Performance-Assessment-Indicators)

**Crop Yield Factors**

Crop Yield Factors help us understand and identify the causes of yield variability. For detailed information on Crop Yield Factors [click here](Crop-Yield-Factors)

**Visualisation**  

**Reporting**  
[Report Generator](ReportGenerator) Automatic dissemination of data in report format

## Acknowledgement  
This repo was developed by [eLEAF](https://www.eleaf.com) under the [WaterPIP project](https://waterpip.un-ihe.org/welcome-waterpip). The Water Productivity Improvement in Practice (WaterPIP) project is supported by the Directorate-General for International Cooperation (DGIS) of the Ministry of Foreign Affairs of the Netherlands under the IHE Delft Partnership Programme for Water and Development (DUPC2). WaterPIP aims to guide countries and water projects in Water Productivity concepts to reach 25% WP improvement in the agricultural sector using WaPOR. It is lead by [IHE Delft](https://www.un-ihe.org/) in partnership with [Wageningen University and Research Center (WUR)](https://www.wur.nl/), [MetaMeta](https://metameta.nl), [eLEAF](https://www.eleaf.com) and [FAO](https://www.fao.org).

## Links  
- [WaterPIP project](https://waterpip.un-ihe.org/welcome-waterpip)
- [FAO WaPOR database](https://wapor.apps.fao.org/home/WAPOR_2/1)
- [Water accounting repository](https://github.com/wateraccounting/WAPORWP)
- [WaPOR v2 methodology document](http://www.fao.org/3/ca9894en/CA9894EN.pdf)
- [WaPOR application catalogue](http://www.fao.org/in-action/remote-sensing-for-water-productivity/use-casesresources/en/)
- [WaPOR master classes](https://thewaterchannel.tv/videos/june-10-2020-monitoring-water-productivity-using-wapor-part-1/)

