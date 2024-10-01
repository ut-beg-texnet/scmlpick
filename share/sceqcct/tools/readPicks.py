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
        self.setPrimaryMessagingGroup("PICK") ### IMPORTANTE!!!!!! ESTE ERA EL ERROR!
        self.addMessagingSubscription("PICK")

    def init(self):
        """
        Initialization.
        """
        if not seiscomp.client.Application.init(self):
            return False
        return True
    def run(self):
        """
        Start the main loop.
        """
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
                timeQ = seiscomp.datamodel.TimeQuantity()
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
                # print(pick.publicID())
                # print(pick.time())
                # print(pick.waveformID())
                # print(pick.filterID())
                # print(pick.methodID())

                pickRef = seiscomp.datamodel.PickReference()
                pickRef.setPickID(pick.publicID())
                # print(pickRef)
                # print(pickRef.pickID())
                self.sendPick(pick)
                self.sendPickRef(pickRef)
        return seiscomp.client.Application.run(self)

    def sendPickRef(self, pickRef):
        op = seiscomp.datamodel.OP_ADD
        n = seiscomp.datamodel.Notifier("EventParameters", op, pickRef)
        m = seiscomp.datamodel.NotifierMessage()
        m.attach(n)
        a = self.connection().send(m)
        print(a)


    def sendPick(self, pick):
        # op = seiscomp.datamodel.OP_ADD
        # # op = OP_UPDATE
        # if pick:
        #     ep = seiscomp.datamodel.EventParameters()
        #     ep.add(pick)
        #     msg = seiscomp.datamodel.ArtificialEventParametersMessage(ep)
        #     a = self.connection().send(msg)
        #     print(a)
        op = seiscomp.datamodel.OP_ADD
        # op = OP_UPDATE
        if pick:
            print(pick.publicID())
            # pickObj = Pick.Cast(pick)
            # msg = NotifierMessage()
            # # print(msg.size())
            # Notifier.SetEnabled(True)
            # n = Notifier("EventParameters", op, pickObj)
            # print(n)
            # print(n.meta())
            # print(n.meta().property(2).name())
            # print(n.meta().property(2).type())
            # print(n.GetMessage())
            # print(n.Size())
            # msg.attach(n)
            # # print(msg.size())

            # pickObj = Pick.Cast(pick)
            # ep = EventParameters()
            # a = ep.add(pickObj)
            # print(a)
            # print(ep.pickCount())
            # print(ep.pick(0))
            # msg = ArtificialEventParametersMessage(ep)
            # print(msg.TypeInfo())
            # con = Connection()
            # print(con.source())
            # print(con.isConnected())
            # print(con.connect('CAM','PICK'))
            # print(con.isConnected())
            # print(con.subscribe('PICK'))
            # print(con.isConnected())
            # print(con.sendMessage('PICK', msg))
            # print(con.send('LOCAON', msg))

            # pickObj = Pick.Cast(pick)
            # ep = EventParameters()
            # ep.add(pickObj)
            # msg = NotifierMessage()
            # # msg = ArtificialEventParametersMessage(ep)
            # Notifier.SetEnabled(True)
            # n = Notifier("Pick", op, pickObj)
            # msg.attach(n)
            # a = Connection().send(msg)
            # print(Connection().state())
            # print(a)

            pickObj = seiscomp.datamodel.Pick.Cast(pick)
            # print(self.isMessagingEnabled())
            # print(seiscomp.datamodel.Notifier.IsEnabled())
            seiscomp.datamodel.Notifier.Enable()
            # print(seiscomp.datamodel.Notifier.IsEnabled())
            n = seiscomp.datamodel.Notifier("EventParameters", op, pickObj)
            # n = Notifier("EventParameters", op, pick)
            m = seiscomp.datamodel.NotifierMessage()
            m.attach(n)
            a = self.connection().send(m)
            print(a)

if __name__ == '__main__':
    import sys
    app = Listener(len(sys.argv), sys.argv)
    sys.exit(app())