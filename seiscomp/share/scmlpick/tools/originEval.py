#!/usr/bin/env python3


import sys
import seiscomp.core, seiscomp.datamodel, seiscomp.client, seiscomp.logging
import datetime
from seiscomp import geo



class Listener(seiscomp.client.Application):

    def __init__(self, argc, argv):
        seiscomp.client.Application.__init__(self, argc, argv)
        self.setDatabaseEnabled(True, True)
        self.setPrimaryMessagingGroup(seiscomp.client.Protocol.LISTENER_GROUP)
        self.setMessagingEnabled(True)
        self.setPrimaryMessagingGroup("LOCEQCCT")

    def init(self):
        """
        Initialization.
        """
        if not seiscomp.client.Application.init(self):
            return False
        return True
    
    def run(self):
        OID = 'Origin/20241007222446.756432.101309'
        evID = 'texnet2024tthe'
        # region = '31.15,31.45,-104.62,-103.93'
        # region = '0,90,-180.62,180.93'
        region = '/home/cam/EQCCT/sceqcct/eqcct-dev/share/sceqcct/tools/regions/texas.bna'
        # region = '/home/cam/EQCCT/sceqcct/eqcct-dev/share/sceqcct/tools/regions/snyder_eqcct.bna'


        dbq = seiscomp.datamodel.DatabaseQuery(self.database())
        orgs = dbq.getOrigins(evID)
        # orgs = seiscomp.datamodel.Origin.Find(OID)
        for org in orgs:
            orgObj = seiscomp.datamodel.Origin.Cast(org)
        
        # print(orgObj.evaluationStatus())

        checkQc = self.qualityCheckOrigin(orgObj)       
        if region != 'none':
            checkReg = self.geographicCheckOrigin(region, orgObj)
        else:
            checkReg = True

        print(checkQc)
        print(checkReg)

        if checkQc and checkReg:
            orgObj = self.changeOrigStatus(orgObj, 5)
        else:
            orgObj = self.changeOrigStatus(orgObj,4)

        op=seiscomp.datamodel.OP_UPDATE
        # op=seiscomp.datamodel.OP_ADD

        if orgObj:
            msg = seiscomp.datamodel.NotifierMessage()
            n = seiscomp.datamodel.Notifier("EventParameters", op, orgObj)
            msg.attach(n)
            out = self.connection().send(msg)
            print(out)

        return seiscomp.client.Application.run(self)


    def changeOrigStatus(self, origin, EvalStat):
        try:
            origin.setEvaluationStatus(EvalStat)
        except:
            print("Error changing origin evaluation status %s" % origin.publicID())
        return origin


    def qualityCheckOrigin(self, origin):

        self.latUncTHR,self.lonUncTHR,self.depthUncTHR,self.depthTHR = 20,20,20,20
        self.azGapTHR = 270

        latUnc = origin.latitude().uncertainty()
        lonUnc = origin.longitude().uncertainty()
        try:
            depthUnc = origin.depth().uncertainty()
        except ValueError:
            depthUnc = 0
        depth = origin.depth().value()
        azGap = origin.quality().azimuthalGap()

        thresholds = {'latUnc': self.latUncTHR, 'lonUnc': self.lonUncTHR, 'depthUnc': self.depthUncTHR, 'depth': self.depthTHR, 'azGap': self.azGapTHR}
        values = {'latUnc': latUnc, 'lonUnc': lonUnc, 'depthUnc': depthUnc, 'depth': depth, 'azGap': azGap}

        exceeding = {key: value for key, value in values.items() if value > thresholds[key]}
        print(exceeding)

        if exceeding:
            return False
        else:
            return True

    def geographicCheckOrigin(self, region, origin):
        if isinstance(region, str):
            if region.split('.')[-1] == 'bna':
                try:
                    within = self.inPolygon(region, origin.latitude().value(), origin.longitude().value())
                    return within
                except FileNotFoundError:
                    print(f'\n\n\t {region} file not found\n\n')
                    return False
            elif len(region.split(',')) == 4:
                quadrant = region.split(',')
                quadrant = tuple(map(float, quadrant))
                return self.inQuadrant(quadrant, origin)
            else:
                print(f'Provide a quadrant or bna file')
                return False
        elif isinstance(region, tuple):
            return self.inQuadrant(quadrant, origin)
        else:
            print(f'Provide a quadrant or bna file')
            return False

    def inPolygon(self, fileName, lat, lon) -> bool:
        self.fs = geo.GeoFeatureSet()
        if not self.fs.readBNAFile(fileName, None):
            print('Impossible to open the bna file %s'%fileName)
            return False
        
        feature = self.fs.features()[0]
        closed = feature.closedPolygon()
        if not closed:
            print('Please fix this: Polygon %s does not not exist or is not closed'%fileName)
        
        coordinates = geo.GeoCoordinate(lat,lon)
        within = feature.contains(coordinates)
        if not within:
            print('Origin (lat: %s and lon: %s) is not within polygon %s.' % (lat,lon,fileName))

        return(within)

    def inQuadrant(self, quadrant: tuple, origin) -> bool:
        """Check if event location is in a quadrant

        Parameters
        ----------
        quadrant : Tuple
            Quadrant to check in format (lat_min, lat_max, lon_min,  lon_max)

        Returns
        -------
        bool
            True if event is in quadrant, False if not
        """
        assert len(quadrant) == 4, 'Quadrant must be a tuple with 4 elements'
        assert quadrant[0] < quadrant[1], 'The minimum latitude must be less than the maximum latitude'
        assert quadrant[2] < quadrant[3], 'The minimum longitude must be less than the maximum longitude'
        return (quadrant[0] <= origin.latitude().value() <= quadrant[1]
                and quadrant[2] <= origin.longitude().value() <= quadrant[3])

if __name__ == '__main__':
    import sys
    app = Listener(len(sys.argv), sys.argv)
    sys.exit(app())