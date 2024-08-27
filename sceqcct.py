#!/usr/bin/env python3

"""
Copyright (C) by TexNet/CISR
Created on Jul 31 2024
Author of the Software: Camilo Munoz
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from seiscomp.client import StreamApplication, Inventory, Protocol
from seiscomp import core
from obspy import Stream, Trace, UTCDateTime
from pathlib import Path
from tools.predictor import mseed_predictor, load_model
from tools.picks2xml import main_picks as picks2xml

class fetchWf(StreamApplication):
    def __init__(self):
        StreamApplication.__init__(self, len(sys.argv), sys.argv)
        # datamodel.Notifier.SetEnabled(False)
        # self.setPrimaryMessagingGroup(Protocol.LISTENER_GROUP)
        self.setMessagingEnabled(True)
        self.setLoadConfigModuleEnabled(True)
        self.setLoadInventoryEnabled(True)
        self.initStreams = []
        self.streams = []
        self.stream = Stream()
        self.streamPick = Stream()
        self.oneMinuteStream = Stream()
        self.stasDict = {}
        self.stasPick = set()
        self.eqcctWindow = 60 # Duration of eqcct window (in seconds)
        self.time_shift = 10 # Time in seconds shift to aply when slice a trace which was send to pick and an update of start time is requiered.
        self.remTr = 600 # Duration in seconds to remove traces from the processing stream
        self.bulkSts = 2 # Number of stations to send each time to the picker
        self.staSend = []
        self.eqcctTHR = 0.001
        # self.stas = ['TX.PB01','TX.PB03','TX.PB04','TX.PB05','TX.PB06','TX.PB07',
        #              'TX.PB08','TX.PB09','TX.PB10','TX.PB11','TX.PB12','TX.PB13',
        #              'TX.PB14','TX.PB16','TX.PB17','TX.PB17','TX.PB18','TX.PB19',
        #              'TX.PB20','TX.PB21','TX.PB22','TX.PB23','TX.PB24','TX.PB25',
        #              'TX.PB26','TX.PB28','TX.PB29','TX.PB30','TX.PB31','TX.PB32',
        #              'TX.PB33','TX.PB34','TX.PB35','TX.PB36','TX.PB37','TX.PB38',
        #              'TX.PB39','TX.PB40','TX.PB42','TX.PB43','TX.PB44','TX.PB46',
        #              'TX.PB47','TX.PB51','TX.PB52','TX.PB53','TX.PB54','TX.PB55',
        #              'TX.PB56','TX.PB57','TX.PB58','TX.PB59']
        self.stas = ['TX.DG01','TX.DG04','TX.DG05','TX.DG09','TX.PB03','TX.PB04',
                     'TX.PB05','TX.PB14','TX.PB15','TX.PB16','TX.PB17','TX.PB18',
                     'TX.PB19','TX.PB21','TX.PB22','TX.PB30','TX.PB32']
        # self.stas = ['TX.DG01','TX.DG04','TX.DG05','TX.DG09']
        self.instTypes = ['HH','CH']

    def init(self):
        if not StreamApplication.init(self): return False
        # self.enableTimer(self.latencyPeriod)
        # logging.info("Starting timer with %d seconds" % self.latencyPeriod)
        self.updateStreams()
        return True

    def updateStreams(self):
        # streamIDs = self.configStreams()
        streamIDs = self.currentStreams()
        for streamID in streamIDs:
            for sta in self.stas:
                if (streamID[0] == sta.split('.')[0] and streamID[1] == sta.split('.')[1]):
                    if streamID[1] in self.stasDict:
                        self.stasDict[streamID[1]].append(streamID[3])
                    else:
                        self.stasDict[streamID[1]] = [streamID[3]]
                    self.initStreams.append(streamID)
                    self.recordStream().addStream(*streamID)
        return

    def configStreams(self):
        streamIDs=[]
        cfg = self.configModule()
        inv = Inventory.Instance().inventory()
        for i in range(cfg.configStationCount()):
            cfg_sta = cfg.configStation(i)
            net = cfg_sta.networkCode()
            sta = cfg_sta.stationCode()
            nets = inv.networkCount()
            for n in range(nets):
                network = inv.network(n)
                if network.code() == net:
                    stas = network.stationCount()
                    for s in range(stas):
                        station = network.station(s)
                        if station.code() == sta:
                            locs = station.sensorLocationCount()
                            for l in range(locs):
                                location = station.sensorLocation(l)
                                loc = location.code()
                                chans = location.streamCount()
                                for c in range(chans):
                                    channel = location.stream(c)
                                    chan = channel.code()
                                    if chan[0:2] in self.instTypes:
                                        streamIDs.append([net, sta, loc, chan])
        return streamIDs

    def currentStreams(self):
        now = core.Time.GMT()
        streamIDs=[]
        inv = Inventory.Instance()
        nets = inv.inventory().networkCount()
        for n in range(nets):
            network = inv.inventory().network(n)
            net = network.code()
            stas = network.stationCount()
            for s in range(stas):
                station = network.station(s)
                sta = station.code()
                station_now = False
                station_now = inv.getStation(network.code(),sta,now)
                if station_now:
                    staCode = station_now.code()
                    locs = station_now.sensorLocationCount()
                    for l in range(locs):
                        location = station_now.sensorLocation(l)
                        loc = location.code()
                        loc_now = False
                        loc_now = inv.getSensorLocation(network.code(),staCode,loc,now)
                        if loc_now:
                            locCode = loc_now.code()
                            chans = location.streamCount()
                            for c in range(chans):
                                channel = location.stream(c)
                                chan = channel.code()
                                if chan[0:2] in self.instTypes:
                                    chan_now = False
                                    chan_now = inv.getStream(network.code(),staCode,locCode,chan,now)
                                    if chan_now:
                                        chanCode = chan_now.code()
                                        sid = [net, staCode, locCode, chanCode]
                                        if sid not in streamIDs:
                                            streamIDs.append([net, staCode, locCode, chanCode])
        return streamIDs
                
    def handleRecord(self, rec):
        # now = core.Time.GMT()
        # diftime = diftime.seconds()
        # streamid = "%s.%s.%s.%s"%tuple(rec.streamID().split('.'))
        startTime = rec.startTime()
        endTime =  rec.endTime()
        sampFreq = rec.samplingFrequency()
        # freq = str(1000/sampFreq)+'ms'
        net = rec.networkCode()
        sta = rec.stationCode()
        net_sta = f"{net}.{sta}"
        loc = rec.locationCode()
        chan = rec.channelCode()
        start_time = pd.to_datetime(str(startTime))
        end_time = pd.to_datetime(str(endTime))
        # time_range = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive='left')
        # time_array = np.array(time_range)
        # time_array_utc = [UTCDateTime(t) for t in time_array]
        if rec.data():
            # print(rec.data().size())
            # print(len(time_array))
            dataObj = core.FloatArray.Cast(rec.data())
            if dataObj:
                data = [dataObj.get(i) for i in range(dataObj.size())]
            else:
                print("  no data")
        data = np.array(data)
        header = {
            'network': net, 
            'station': sta, 
            'location': loc, 
            'channel': chan, 
            'npts': len(data), 
            'sampling_rate': sampFreq,
            'starttime': UTCDateTime(start_time)
        }
        tr = Trace(data=data, header=header)

        if len(self.stream) == 0:
            self.stream.append(tr)
        else:
            inSt = False
            for i,trace in enumerate(self.stream):
                if trace.stats.network == net and trace.stats.station == sta and trace.stats.channel == chan:
                    inSt = True
                    tempSt = Stream()
                    tempSt.append(trace)
                    tempSt.append(tr)
                    tempSt.merge(method=1, fill_value=0)
                    self.stream[i] = tempSt[0]
                    # trace.data = np.concatenate((trace.data, tr.data))
                    # trace.stats.endtime = tr.stats.endtime
                    print(self.stream[i].stats.endtime - self.stream[i].stats.starttime)
                    if self.stream[i].stats.endtime - self.stream[i].stats.starttime >= self.eqcctWindow:
                        traceID = f'{net}.{sta}.{loc}.{chan}'
                        trace_found = any(tr.id == traceID for tr in self.streamPick)
                        # print(traceID)
                        # print(any(tr.id == traceID for tr in self.streamPick))
                        # print(tr.id for tr in self.streamPick)
                        if not trace_found:
                            self.streamPick.append(self.stream[i])
                            self.stasPick.add(self.stream[i].stats.station)
                        else:
                            pass
            if not inSt:
                    self.stream.append(tr)

        for trace in reversed(self.stream):
            lenght = trace.stats.endtime - trace.stats.starttime
            if lenght > self.remTr:
                self.stream.remove(trace)

        # print("stream")
        # print(self.stream)
        # print("streamPick")
        # print(self.streamPick)

        # print('staSend')
        # print(self.staSend)
        # print('stasPick')
        # print(self.stasPick)

        for sta in self.stasPick:
            checkSta = True
            # print('stasPick')
            # print(sta)
            for recChann in self.stasDict[sta]:
                # print('recChann')
                # print(recChann)
                checkChann = False
                for i,trace in enumerate(self.streamPick):
                    if trace.stats.station == sta:
                        if recChann == trace.stats.channel:
                            # print('streamPick')
                            # print(trace.stats.channel)
                            checkChann = True
                if not checkChann:
                    checkSta = False
            if checkSta and sta not in self.staSend:
                # print('checkSta')
                # print(sta)
                self.staSend.append(sta)

        [self.stasPick.remove(sta) for sta in self.staSend if sta in self.stasPick]

        # print('staSend')
        # print(self.staSend)
        # print('stasPick')
        # print(self.stasPick)

        toRemove = []
        if len(self.staSend) >= self.bulkSts:
            # Process the one-minute stream
            for i,trace in enumerate(self.streamPick):
                if self.streamPick[i].stats.station in self.staSend:
                    startMinute = self.streamPick[i].stats.starttime
                    endMinute = self.streamPick[i].stats.starttime + self.eqcctWindow
                    traceMin = trace.slice(
                                        starttime=startMinute,
                                        endtime=endMinute
                                        )
                    self.oneMinuteStream.append(traceMin)
                    toRemove.append(i)
            
            for i in sorted(toRemove, reverse=True): del self.streamPick[i]

            print("one_minute_stream")
            print(self.oneMinuteStream)

            for i,trace in enumerate(self.stream):
                for tr in self.oneMinuteStream:
                    if trace.stats.network == tr.stats.network and trace.stats.station == tr.stats.station and trace.stats.channel == tr.stats.channel:
                        # print(self.stream[i])
                        # print(self.stream[i].stats.starttime)
                        # print(self.stream[i].stats.endtime)
                        try:
                            self.stream[i] = trace.slice(
                                starttime=self.stream[i].stats.starttime + self.eqcctWindow - self.time_shift,
                                endtime=self.stream[i].stats.endtime
                                )
                        except:
                            print(self.stream[i])
                            print(self.stream[i].stats.starttime)
                            print(self.stream[i].stats.endtime)
                            print('Error when slice the trace')
                        # print(self.stream[i])
            
            # print("stream after slice")
            # print(self.stream)
            
            # Run EQCCT picker
            self.run_picker(startMinute,endMinute,self.oneMinuteStream)
            self.oneMinuteStream = Stream()
            self.staSend = []

    def run_picker(self,start_time,end_time,stream):
        # Create working dirs
        pathResutls = "results"
        Path(f"./{pathResutls}/picks/").mkdir(exist_ok=True, parents=True)
        Path(f"./{pathResutls}/logs/picker").mkdir(exist_ok=True, parents=True)

        # Load EQCCT model
        eqcct_Pmodel = '/home/cam/EQCCT/sceqcct/eqcct-dev/model/ModelPS/test_trainer_024.h5'
        eqcct_Smodel = '/home/cam/EQCCT/sceqcct/eqcct-dev/model/ModelPS/test_trainer_021.h5'
        
        model = load_model(eqcct_Pmodel, eqcct_Smodel, f"./{pathResutls}/logs/model.log")

        # Get picker tasks
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)
        chunk_time = [[start_time, end_time]]

        # Prepare tasks for EQCCT picking
        stations = []
        for trace in stream:
            station = f'{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}'
            # station = f'{trace.stats.network}.{trace.stats.station}'
            if station not in stations:
                stations.append(station)

        print(stations)
        tasks = [[f"({i+1}/{len(chunk_time)})", chunk_time[i][0], chunk_time[i][1], stations, [], model] for i in range(len(chunk_time))]

        # Run picker
        for task in tasks:
            self.picker(task,stream)

    def picker(self,task,stream):
        pathResutls = "results"
        eqcct_P_threshold = 0.001
        eqcct_S_threshold = 0.02
        eqcct_overlap = 0.0
        eqcct_batch_size = 1
        eqcct_gpu_id = 0
        eqcct_gpu_limit = 1

        pos, starttime, endtime, stations, params, model = task
        begin = starttime.strftime(format="%Y%m%dT%H%M%SZ")
        end = endtime.strftime(format="%Y%m%dT%H%M%SZ")
        log_file = f"{pathResutls}/logs/picker/{begin}_{end}.log"
        print(f"[{datetime.now()}] Running EQCCT picker {pos}. Log file: {log_file}")
        output_eqcct = f"{pathResutls}/eqcct/{begin}_{end}/"
        
        mseed_predictor(stream            = stream,
                        output_dir    = output_eqcct,
                        stations2use  = stations,
                        P_threshold   = eqcct_P_threshold,
                        S_threshold   = eqcct_S_threshold,
                        normalization_mode = 'std',
                        overlap       = eqcct_overlap,
                        batch_size    = eqcct_batch_size,
                        overwrite     = True,
                        log_file      = log_file,
                        model         = model,
                        gpu_id        = eqcct_gpu_id,
                        gpu_limit     = eqcct_gpu_limit)
        picks2xml(input_file=output_eqcct, output_file=f"./{pathResutls}/picks/picks_{begin}_{end}.xml", ai='eqcc', thr_dict=self.eqcctTHR)

    def run(self):
        return StreamApplication.run(self)

app = fetchWf()
sys.exit(app())