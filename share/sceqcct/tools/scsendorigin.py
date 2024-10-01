#!/usr/bin/env python3

import sys
import seiscomp.core, seiscomp.datamodel, seiscomp.client, seiscomp.logging
import datetime


class Listener(seiscomp.client.Application):

    def __init__(self, argc, argv):
        seiscomp.client.Application.__init__(self, argc, argv)
        self.setDatabaseEnabled(False, False)
        self.setPrimaryMessagingGroup(seiscomp.client.Protocol.LISTENER_GROUP)
        self.setMessagingEnabled(True)
        self.setPrimaryMessagingGroup("PICK")

    def init(self):
        if not seiscomp.client.Application.init(self):
            return False
        return True

    def run(self):
        data = [{'trace_name': 'TX.PB29.00.HH1', 'network_name': '0 ', 'station_name': 'PB29', 'instrument_type': '0 ', 'station_lat': 0, 
                'station_lon': 0, 'station_elv': 0, 'PdateTime': None, 'p_prob': None, 'SdateTime': None, 's_prob': None}, 
                {'trace_name': 'TX.PB26.00.HHE', 'network_name': '0 ', 'station_name': 'PB26', 'instrument_type': '0 ', 'station_lat': 0, 
                'station_lon': 0, 'station_elv': 0, 'PdateTime': datetime.datetime(2024, 9, 25, 17, 15, 14, 471000), 'p_prob': 0.137, 
                'SdateTime': datetime.datetime(2024, 9, 25, 17, 15, 49, 361000), 's_prob': 0.037}]

        for pickData in data:
            if pickData['PdateTime']:
                pick = seiscomp.datamodel.Pick.Create()
                Ptime = pickData['PdateTime']
                scPtime = Ptime.strftime('%Y-%m-%d %H:%M:%S.%f')
                scPtime = seiscomp.core.Time()
                scPtime.set(Ptime.year,Ptime.month,Ptime.day,Ptime.hour,Ptime.minute,Ptime.second,Ptime.microsecond)
                timeQ =  seiscomp.datamodel.TimeQuantity()
                timeQ.setValue(scPtime)
                timeQ.setUncertainty(pickData['p_prob'])
                pick.setTime(timeQ)
                # print(pick.time().value())
                # print(pick.time().uncertainty())
                wfID = seiscomp.datamodel.WaveformStreamID()
                wfID.setNetworkCode(pickData['trace_name'].split('.')[0])
                wfID.setStationCode(pickData['trace_name'].split('.')[1])
                wfID.setLocationCode(pickData['trace_name'].split('.')[2])
                wfID.setChannelCode(pickData['trace_name'].split('.')[3])
                pick.setWaveformID(wfID)
                phase = seiscomp.datamodel.Phase()
                phase.setCode('P')
                pick.setPhaseHint(phase)

                # pick.setMethodID('EQCCT')
                pick.setMethodID('AIC')
                # ci = CreationInfo()
                # ci.setCreationTime(scPtime)
                # pick.setCreationInfo(ci)
                pick.setFilterID('ITAPER(4)>>BW(4,4,8)')

                # print(pick.waveformID().networkCode())
                # print(pick.phaseHint().code())
                # print(pick.methodID())
                print(pick.publicID())
                print(pick.time())
                print(pick.waveformID())
                print(pick.filterID())
                print(pick.methodID())
                self.sendPick(pick)
        return seiscomp.client.Application.run(self)

    def sendPick(self,pick):
        op = seiscomp.datamodel.OP_ADD
        # op = OP_UPDATE
        if pick:
            ep = seiscomp.datamodel.EventParameters()
            ep.add(pick)
            msg = seiscomp.datamodel.ArtificialEventParametersMessage(ep)
            a = self.connection().send(msg)
            print(a)


if __name__ == '__main__':
    import sys
    app = Listener(len(sys.argv), sys.argv)
    sys.exit(app())