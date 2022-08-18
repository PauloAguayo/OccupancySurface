from measurements import Measurements

class Outcomes(object):
    def __init__(self):
        print('-------------------------------------------------------------------------')
        self.standard_area = float(input('MEASURE OF THE POLYGON AREA? (m2) ' ))
        self.measures = Measurements(self.standard_area)

    def Telling(self):
        return(self.standard_area)

    def People_density(self,people):
        people_density = people/self.standard_area
        return(people_density)
