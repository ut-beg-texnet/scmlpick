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
        self.eqcctWindow = 60 # Duration of eqcct window (in seconds)

    def init(self):
        if not StreamApplication.init(self): return False
        # self.enableTimer(self.latencyPeriod)
        # logging.info("Starting timer with %d seconds" % self.latencyPeriod)
        self.updateStreams()
        return True

    def updateStreams(self):
        streamIDs = self.configStreams()
        for streamID in streamIDs:
            # print(streamID)
            if (streamID[0] == "TX" and streamID[1] == "PB35"):
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
                                    if chan[0:2] == "HH": # OJO!!! prueba pendiente por quitar
                                        streamIDs.append([net, sta, loc, chan])
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
            i = 0
            for trace in self.stream:
                if trace.stats.station == sta and trace.stats.channel == chan:
                    inSt = True
                    tempSt = Stream()
                    tempSt.append(trace)
                    tempSt.append(tr)
                    tempSt.merge(method=1)
                    self.stream[i] = tempSt[0]
                    # trace.data = np.concatenate((trace.data, tr.data))
                    # trace.stats.endtime = tr.stats.endtime
                i+=1
            if not inSt:
                    self.stream.append(tr)

        print(self.stream[0].stats.endtime - self.stream[0].stats.starttime)
        if self.stream[0].stats.endtime - self.stream[0].stats.starttime >= self.eqcctWindow:
            
            # Process the one-minute stream
            startMinute = self.stream[0].stats.starttime
            endMinute = self.stream[0].stats.starttime + self.eqcctWindow
            one_minute_stream = self.stream.slice(
                starttime=startMinute,
                endtime=endMinute
            )
            # print(one_minute_stream)
            
            self.stream = self.stream.slice(
                starttime=self.stream[0].stats.starttime + self.eqcctWindow,
                endtime=self.stream[0].stats.endtime
            )
            # Run EQCCT picker
            self.run_picker(startMinute,endMinute,net_sta,one_minute_stream)

        # print(self.stream)

    def run_picker(self, start_time, end_time, net_sta,stream):
        # Create working dirs
        pathResutls = "results"
        Path(f"./{pathResutls}/picks/").mkdir(exist_ok=True, parents=True)
        Path(f"./{pathResutls}/logs/picker").mkdir(exist_ok=True, parents=True)

        # Load EQCCT model
        eqcct_Pmodel = '/home/cam/eqcct-dev/model/ModelPS/test_trainer_024.h5'
        eqcct_Smodel = '/home/cam/eqcct-dev/model/ModelPS/test_trainer_021.h5'
        
        model = load_model(eqcct_Pmodel, eqcct_Smodel, f"./{pathResutls}/logs/model.log")

        # Get picker tasks
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)
        chunk_time = [[start_time, end_time]]

        # Prepare tasks for EQCCT picking
        station = [net_sta.split('.')[1]]
        tasks = [[f"({i+1}/{len(chunk_time)})", chunk_time[i][0], chunk_time[i][1], station, [], model] for i in range(len(chunk_time))]

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
        picks2xml(input_file=output_eqcct, output_file=f"./{pathResutls}/picks/picks_{begin}_{end}.xml", ai='eqcc')

    def run(self):
        return StreamApplication.run(self)
	
app = fetchWf()
sys.exit(app())