"""
Global definitions needed across the whole project
"""

modelRunsDir = 'model-runs'

TravelToWorkFilename = 'wu03ew_msoa.csv'
ZoneCodesFilename = 'ZoneCodesText.csv'
#raw data training files - these are used to create the PyCijRoad, PyTObsRoad etc
TrainingDataRoadCSVFilename = 'trainingdata_road.csv'
TrainingDataBusCSVFilename = 'trainingdata_bus.csv'
TrainingDataGBRailCSVFilename = 'trainingdata_gbrail.csv'
#Python matrices used for training (min=minutes, Cij=cost, TObs=people flow)
#these are Python pickle
PyCijRoadMinFilename = 'Py_Cij_road.bin' #mode 1
PyCijBusMinFilename = 'Py_Cij_bus.bin' #mode 2
PyCijGBRailMinFilename = 'Py_Cij_gbrail.bin' #mode 3
PyTObsAllFilename = 'Py_TObs_all.bin' #TObs.bin
PyTObsRoadFilename = 'Py_TObs_road.bin' #TObs3_1.bin
PyTObsBusFilename = 'Py_TObs_bus.bin' #TObs3_2.bin
PyTObsGBRailFilename = 'Py_TObs_gbrail.bin' #TObs3_3.bin

#matrix filenames - these are C# matrix dumps direct from QUANT
TObsFilename = 'TObs.bin' #1 mode
TObs21Filename = 'TObs2_1.bin' #2 mode
TObs22Filename = 'TObs2_2.bin'
TObs31Filename = 'TObs3_1.bin' #3 mode
TObs32Filename = 'TObs3_2.bin'
TObs33Filename = 'TObs3_3.bin'
#cost matrix names
QUANTCijRoadMinFilename = 'dis_roads_min.bin' #these are C# matrix dumps
QUANTCijBusMinFilename = 'dis_bus_min.bin'
QUANTCijGBRailMinFilename = 'dis_gbrail_min.bin'
#CijRoadMinFilename = 'Cij_road_min.bin' #these are Python pickle
#CijBusMinFilename = 'Cij_bus_min.bin'
#CijRailMinFilename = 'Cij_gbrail_min.bin'