#movidius tests
from mvnc import mvncapi

from models.MovidiusSingleDest import MovidiusSingleDest

###############################################################################

def testMovidius():
    # set the logging level for the NC API
    mvncapi.SetGlobalOption(mvncapi.GlobalOption.LOG_LEVEL, 0)

    # get a list of names for all the devices plugged into the system
    ncs_names = mvncapi.EnumerateDevices()
    if (len(ncs_names) < 1):
        print("Error - no NCS devices detected, verify an NCS device is connected.")
        quit() 


    # get the first NCS device by its name.  For this program we will always open the first NCS device.
    device = mvncapi.Device(ncs_names[0])

    
    # try to open the device.  this will throw an exception if someone else has it open already
    try:
        device.OpenDevice()
    except:
        print("Error - Could not open NCS device.")
        quit()


    print("Hello NCS! Device opened normally.")
    

    try:
        device.CloseDevice()
    except:
        print("Error - could not close NCS device.")
        quit()

###############################################################################

def testBuildMovidiusGraph():
    mvModel = MovidiusSingleDest()
    mvModel.writeGraph('graphRunModel.txt')

###############################################################################

def testRunMovidiusGraph():
    mvModel = MovidiusSingleDest()
    mvModel.run()