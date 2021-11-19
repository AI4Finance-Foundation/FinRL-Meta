from flask_socketio import SocketIO, emit
from flask import session, sessions

from .WtLogger import WtLogger

def get_param(json_data, key:str, type=str, defVal = ""):
    if key not in json_data:
        return defVal
    else:
        return type(json_data[key])

class PushServer:

    def __init__(self, app, dataMgr, logger:WtLogger = None):
        sockio:SocketIO = SocketIO(app)
        self.sockio = sockio
        self.app = app
        self.dataMgr = dataMgr
        self.logger = logger

        @sockio.on('connect', namespace='/')
        def on_connect():
            usrInfo = session.get("userinfo")
            if usrInfo is not None:
                self.logger.info("%s connected" % usrInfo["loginid"])

        @sockio.on('disconnect', namespace='/')
        def on_disconnect():
            usrInfo = session.get("userinfo")
            if usrInfo is not None:
                self.logger.info("%s disconnected" % usrInfo["loginid"])

        @sockio.on('setgroup', namespace='/')
        def set_group(data):
            groupid = get_param(data, "groupid")
            if len(groupid) == 0:
                emit('setgroup', {"result":-2, "message":"组合ID不能为空"})
            else:
                session["groupid"] = groupid         

    def run(self, port:int, host:str):
        self.sockio.run(self.app, host, port)

    def notifyGrpLog(self, groupid, tag:str, time:int, message):
        self.sockio.emit("notify", {"type":"gplog", "groupid":groupid, "tag":tag, "time":time, "message":message}, broadcast=True)

    def notifyGrpEvt(self, groupid, evttype):
        self.sockio.emit("notify", {"type":"gpevt", "groupid":groupid, "evttype":evttype}, broadcast=True)

    def notifyGrpChnlEvt(self, groupid, chnlid, evttype, data):
        self.sockio.emit("notify", {"type":"chnlevt", "groupid":groupid, "channel":chnlid, "data":data, "evttype":evttype}, broadcast=True)